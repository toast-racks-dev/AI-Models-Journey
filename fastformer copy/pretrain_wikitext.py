import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
import random
import pickle

from fastformer_config import FastformerConfig
from fastformer_modules import FastformerSelfOutput, FastformerIntermediate, FastformerOutput


config = FastformerConfig(
    hidden_size=128,                                                  
    num_hidden_layers=2,
    num_attention_heads=4,                                     
    intermediate_size=512,
    max_position_embeddings=256,
    hidden_dropout_prob=0.1,
    layer_norm_eps=1e-12,
    initializer_range=0.02,
)

SEQ_LEN       = 256
MASK_PROB     = 0.15                                
BATCH_SIZE    = 16                     
EPOCHS        = 5
LR            = 1e-3
SAVE_PATH     = "fastformer_pretrained.pt"
VOCAB_PATH    = "wikitext_vocab.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading WikiText-2...")

raw = load_dataset("wikitext", "wikitext-2-raw-v1")

def get_lines(split):
    return [
        line.strip()
        for line in raw[split]["text"]
        if line.strip() and not line.strip().startswith("=")
    ]

train_lines = get_lines("train")
val_lines   = get_lines("validation")
print(f"Train lines: {len(train_lines)} | Val lines: {len(val_lines)}")

print("Tokenizing...")
def tokenize_lines(lines):
    return [wordpunct_tokenize(line.lower()) for line in lines]

train_tokens = tokenize_lines(train_lines)
val_tokens   = tokenize_lines(val_lines)


PADDING_IDX = 0
MASK_IDX    = 1
UNK_IDX     = 2

word_dict = {"PADDING": PADDING_IDX, "[MASK]": MASK_IDX, "[UNK]": UNK_IDX}
for tokens in train_tokens:
    for t in tokens:
        if t not in word_dict:
            word_dict[t] = len(word_dict)

VOCAB_SIZE = len(word_dict)
print(f"Vocab size: {VOCAB_SIZE}")


with open(VOCAB_PATH, "wb") as f:
    pickle.dump(word_dict, f)
print(f"Vocab saved to {VOCAB_PATH}")
#Without saving, you'd have to:
# 1. Run pretrain → vocab created in memory
#2. Copy-paste vocab into every other script
#3. Rebuild vocab from scratch every tim


def tokens_to_ids(token_lists, word_dict):
    ids = []
    for tokens in token_lists:
        ids.extend([word_dict.get(t, UNK_IDX) for t in tokens])
    return ids

train_ids = tokens_to_ids(train_tokens, word_dict)
val_ids   = tokens_to_ids(val_tokens,   word_dict)

class MLMDataset(Dataset):
    """
    Chunks a flat token-id stream into fixed-length windows.
    For each window, randomly masks MASK_PROB of tokens.

    Returns:
        input_ids  — the masked sequence (what the model sees)
        labels     — original ids at masked positions, -100 everywhere else
                     (-100 is ignored by CrossEntropyLoss)
        attn_mask  — 1 for real tokens, 0 for padding
    """
    def __init__(self, token_ids, seq_len, mask_prob, mask_idx, vocab_size, pad_idx=0):
        self.seq_len    = seq_len
        self.mask_prob  = mask_prob
        self.mask_idx   = mask_idx
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx

                                            
        n_chunks = len(token_ids) // seq_len
        self.chunks = [
            token_ids[i * seq_len: (i + 1) * seq_len]
            for i in range(n_chunks)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        original = list(self.chunks[idx])
        input_ids = original.copy()
        labels    = [-100] * self.seq_len                                         

        for pos in range(self.seq_len):
            if random.random() < self.mask_prob:
                labels[pos] = original[pos]                            

                r = random.random()
                if r < 0.80:
                                                          
                    input_ids[pos] = self.mask_idx
                elif r < 0.90:
                                                                  
                    input_ids[pos] = random.randint(3, self.vocab_size - 1)
                                                                                 

        attn_mask = [1] * self.seq_len                                       

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attn_mask),
            torch.LongTensor(labels),
        )

train_dataset = MLMDataset(train_ids, SEQ_LEN, MASK_PROB, MASK_IDX, VOCAB_SIZE)
val_dataset   = MLMDataset(val_ids,   SEQ_LEN, MASK_PROB, MASK_IDX, VOCAB_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

                                            
                                            
                                            

class AttentionPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = torch.tanh(self.att_fc1(x))
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).reshape(bz, -1)
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size       = config.hidden_size

        self.query     = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key       = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_att   = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax   = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query = self.query(hidden_states)
        mixed_key   = self.key(hidden_states)

        q_score = self.query_att(mixed_query).transpose(1, 2) / self.attention_head_size ** 0.5
        q_score += attention_mask
        q_weight = self.softmax(q_score).unsqueeze(2)
        q_layer  = self.transpose_for_scores(mixed_query)
        pooled_q = torch.matmul(q_weight, q_layer).transpose(1, 2).view(-1, 1, self.all_head_size)

                                                                  
        mixed_qk = mixed_key * pooled_q.repeat(1, seq_len, 1)

                                   
        qk_score  = self.key_att(mixed_qk).transpose(1, 2) / self.attention_head_size ** 0.5
        qk_score += attention_mask
        qk_weight = self.softmax(qk_score).unsqueeze(2)
        k_layer   = self.transpose_for_scores(mixed_qk)
        pooled_k  = torch.matmul(qk_weight, k_layer)

                                                                          
        weighted_value = (pooled_k * q_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(weighted_value.size()[:-2] + (self.all_head_size,))
        return self.transform(weighted_value) + mixed_query

class FastAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self   = FastSelfAttention(config)
        self.output = FastformerSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        return self.output(self.self(input_tensor, attention_mask), input_tensor)

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention    = FastAttention(config)
        self.intermediate = FastformerIntermediate(config)
        self.output       = FastformerOutput(config)

    def forward(self, hidden_states, attention_mask):
        attn_out  = self.attention(hidden_states, attention_mask)
        inter_out = self.intermediate(attn_out)
        return self.output(inter_out, attn_out)

class FastformerEncoder(nn.Module):
    """
    Shared encoder — same as your IMDb model, just extracted cleanly
    so both the MLM model and the classifier can use it.
    """
    def __init__(self, config):
        super().__init__()
        self.encoders            = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm           = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout             = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embs, attention_mask):
                                                
        ext_mask = (1.0 - attention_mask.unsqueeze(1).float()) * -10000.0

        bsz, seq_len, _ = input_embs.shape
        pos_ids  = torch.arange(seq_len, device=input_embs.device).unsqueeze(0).expand(bsz, -1)
        pos_embs = self.position_embeddings(pos_ids)

        x = self.dropout(self.LayerNorm(input_embs + pos_embs))
        for layer in self.encoders:
            x = layer(x, ext_mask)
        return x                                                              

class FastformerForMLM(nn.Module):
    """
    Encoder + word embedding + MLM prediction head.
    The encoder weights here are what we save and reuse for fine-tuning.
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=PADDING_IDX)
        self.encoder        = FastformerEncoder(config)
                                                            
        self.mlm_head       = nn.Linear(config.hidden_size, vocab_size)
        self.loss_fn        = nn.CrossEntropyLoss(ignore_index=-100)                                

    def forward(self, input_ids, attention_mask, labels=None):
        embs         = self.word_embedding(input_ids)
        hidden       = self.encoder(embs, attention_mask)              
        logits       = self.mlm_head(hidden)                           

        if labels is not None:
                                                              
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits

model = FastformerForMLM(config, VOCAB_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")



def run_epoch(model, loader, optimizer, device, train=True, desc=""):
    model.train() if train else model.eval()
    total_loss = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for input_ids, attn_mask, labels in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels    = labels.to(device)

            loss, _ = model(input_ids, attn_mask, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)

print("\nStarting MLM pretraining on WikiText-2...")
for epoch in range(1, EPOCHS + 1):
    tr_loss  = run_epoch(model, train_loader, optimizer, device, train=True,  desc=f"Epoch {epoch} [Train]")
    val_loss = run_epoch(model, val_loader,   optimizer, device, train=False, desc=f"Epoch {epoch} [Val]  ")
    print(f"Epoch {epoch} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")


torch.save({
    "word_embedding": model.word_embedding.state_dict(),
    "encoder":        model.encoder.state_dict(),
    "config":         config.__dict__,
}, SAVE_PATH)
print(f"\nPretrained weights saved to {SAVE_PATH}")
print("Next step: load these weights in your fine-tuning script.")
