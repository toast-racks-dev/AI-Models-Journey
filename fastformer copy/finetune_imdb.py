
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from tqdm import tqdm
from nltk.tokenize import wordpunct_tokenize
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from fastformer_config import FastformerConfig
from fastformer_modules import FastformerSelfOutput, FastformerIntermediate, FastformerOutput


PRETRAINED_PATH = "fastformer_pretrained.pt"
VOCAB_PATH      = "wikitext_vocab.pkl"

BATCH_SIZE  = 16
LR          = 1e-4                                                            
EPOCHS      = 3
SEQ_LEN     = 256
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
config = FastformerConfig(**checkpoint["config"])
print("Loaded config from checkpoint.")

with open(VOCAB_PATH, "rb") as f:
    word_dict = pickle.load(f)
print(f"Loaded WikiText vocab: {len(word_dict)} tokens")

DATA_PATHS = [
    '/kaggle/input/datasets/ykmmit/imdb-review/data1.json',
    '/kaggle/input/datasets/ykmmit/imdb-review/data2.json',
    '/kaggle/input/datasets/ykmmit/imdb-review/data3.json',
    '/kaggle/input/datasets/ykmmit/imdb-review/data4.json',
]

print("Loading IMDb dataset...")

all_reviews = []
for path in DATA_PATHS:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            all_reviews.extend(json.load(f))

valid_reviews = [r for r in all_reviews if r.get('review') and r.get('rating')]
texts  = [str(r['review']) for r in valid_reviews]
labels = [int(r['rating']) - 1 for r in valid_reviews]

combined = list(zip(texts, labels))
random.seed(42)
random.shuffle(combined)
total     = len(combined)
train_end = int(0.8 * total)
val_end   = int(0.9 * total)
train_data, val_data, test_data = combined[:train_end], combined[train_end:val_end], combined[val_end:]

train_texts, train_labels = zip(*train_data)
val_texts,   val_labels   = zip(*val_data)
test_texts,  test_labels  = zip(*test_data)

def tokenize_list(text_list, desc):
    return [wordpunct_tokenize(str(t).lower()) for t in tqdm(text_list, desc=desc)]

train_tokens = tokenize_list(train_texts, "Tokenizing train")
val_tokens   = tokenize_list(val_texts,   "Tokenizing val")
test_tokens  = tokenize_list(test_texts,  "Tokenizing test")

                                   
print("Extending vocab with IMDb-specific words...")
for tokens in tqdm(train_tokens, desc="Extending vocab"):
    for t in tokens:
        if t not in word_dict:
            word_dict[t] = len(word_dict)

VOCAB_SIZE = len(word_dict)
print(f"Extended vocab size: {VOCAB_SIZE}")

def build_tensors(tokenized_list, labels, word_dict, seq_len):
    UNK = word_dict.get("[UNK]", 2)
    encoded = []
    for tokens in tqdm(tokenized_list, desc="Encoding"):
        ids = [word_dict.get(t, UNK) for t in tokens][:seq_len]
        ids += [0] * (seq_len - len(ids))
        encoded.append(ids)
    return torch.LongTensor(encoded), torch.LongTensor(labels)

train_enc, train_labs = build_tensors(train_tokens, train_labels, word_dict, SEQ_LEN)
val_enc,   val_labs   = build_tensors(val_tokens,   val_labels,   word_dict, SEQ_LEN)
test_enc,  test_labs  = build_tensors(test_tokens,  test_labels,  word_dict, SEQ_LEN)

train_loader = DataLoader(TensorDataset(train_enc, train_labs), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_enc,   val_labs),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(test_enc,  test_labs),  batch_size=BATCH_SIZE)

                                            
                                                  
                                            

class AttentionPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)

    def forward(self, x, attn_mask=None):
        bz    = x.shape[0]
        alpha = self.att_fc2(torch.tanh(self.att_fc1(x)))
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        return torch.bmm(x.permute(0, 2, 1), alpha).reshape(bz, -1)

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.n_heads   = config.num_attention_heads
        self.all_size  = config.hidden_size
        self.query     = nn.Linear(config.hidden_size, self.all_size)
        self.query_att = nn.Linear(self.all_size, self.n_heads)
        self.key       = nn.Linear(config.hidden_size, self.all_size)
        self.key_att   = nn.Linear(self.all_size, self.n_heads)
        self.transform = nn.Linear(self.all_size, self.all_size)
        self.softmax   = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        return x.view(x.size()[:-1] + (self.n_heads, self.head_size)).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        bsz, seq_len, _ = hidden_states.shape
        mq = self.query(hidden_states)
        mk = self.key(hidden_states)
        qs = self.query_att(mq).transpose(1, 2) / self.head_size ** 0.5 + attention_mask
        qw = self.softmax(qs).unsqueeze(2)
        ql = self.transpose_for_scores(mq)
        pq = torch.matmul(qw, ql).transpose(1, 2).view(-1, 1, self.all_size).repeat(1, seq_len, 1)
        mqk  = mk * pq
        qks  = self.key_att(mqk).transpose(1, 2) / self.head_size ** 0.5 + attention_mask
        qkw  = self.softmax(qks).unsqueeze(2)
        kl   = self.transpose_for_scores(mqk)
        pk   = torch.matmul(qkw, kl)
        wv   = (pk * ql).transpose(1, 2).reshape(bsz, seq_len, self.all_size)
        return self.transform(wv) + mq

class FastAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self   = FastSelfAttention(config)
        self.output = FastformerSelfOutput(config)

    def forward(self, x, mask):
        return self.output(self.self(x, mask), x)

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention    = FastAttention(config)
        self.intermediate = FastformerIntermediate(config)
        self.output       = FastformerOutput(config)

    def forward(self, x, mask):
        a = self.attention(x, mask)
        return self.output(self.intermediate(a), a)

class FastformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoders            = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm           = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout             = nn.Dropout(config.hidden_dropout_prob)
        self.pooler              = AttentionPooling(config)

    def forward(self, embs, attention_mask):
        ext_mask = (1.0 - attention_mask.unsqueeze(1).float()) * -10000.0
        bsz, seq_len, _ = embs.shape
        pos_ids = torch.arange(seq_len, device=embs.device).unsqueeze(0).expand(bsz, -1)
        x = self.dropout(self.LayerNorm(embs + self.position_embeddings(pos_ids)))
        for layer in self.encoders:
            x = layer(x, ext_mask)
        return self.pooler(x, attention_mask)

class FastformerClassifier(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super().__init__()
        self.word_embedding    = nn.Embedding(vocab_size, config.hidden_size, padding_idx=0)
        self.fastformer_model  = FastformerEncoder(config)
        self.classifier        = nn.Linear(config.hidden_size, num_classes)
        self.criterion         = nn.CrossEntropyLoss()

    def forward(self, input_ids, targets=None):
        mask   = input_ids.bool().float()
        embs   = self.word_embedding(input_ids)
        vec    = self.fastformer_model(embs, mask)
        logits = self.classifier(vec)
        if targets is not None:
            return self.criterion(logits, targets), logits
        return logits

                                            
                            
                                            
model = FastformerClassifier(config, VOCAB_SIZE, NUM_CLASSES).to(device)

print("Loading pretrained encoder weights...")

                                                                    
                                                                      
pretrained_emb = checkpoint["word_embedding"]["weight"]                         
with torch.no_grad():
    model.word_embedding.weight[:pretrained_emb.size(0)] = pretrained_emb
print(f"  Loaded embedding: {pretrained_emb.size(0)} / {VOCAB_SIZE} rows from pretraining")

                                                         
                                                                       
encoder_state = checkpoint["encoder"]
missing, unexpected = model.fastformer_model.load_state_dict(encoder_state, strict=False)
print(f"  Encoder loaded | Missing keys: {missing} | Unexpected keys: {unexpected}")
                                                                                        

                                            
                     
                                            
optimizer = optim.Adam(model.parameters(), lr=LR)

def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, logits = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(-1) == y).sum().item()
        total      += y.size(0)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total

def evaluate(model, loader, desc):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(y.numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds, average='macro')

print("\nStarting fine-tuning on IMDb...")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc    = train_epoch(model, train_loader, optimizer, epoch)
    val_acc, val_f1    = evaluate(model, val_loader,  f"Epoch {epoch} [Val]")
    print(f"Epoch {epoch} | Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

test_acc, test_f1 = evaluate(model, test_loader, "Final Test")
print(f"\nFinal Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
