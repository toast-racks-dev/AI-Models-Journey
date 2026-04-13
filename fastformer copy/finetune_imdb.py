
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random

from nltk.tokenize import wordpunct_tokenize
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from fastformer_config import FastformerConfig
from fastformer_modules import (
    FastformerSelfOutput,
    FastformerIntermediate,
    FastformerOutput,
    AttentionPooling,
    FastSelfAttention,
    FastAttention,
    FastformerLayer,
    FastformerEncoder
)


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
    print(f"{desc}...")
    tokens = []
    for i, t in enumerate(text_list):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i}/{len(text_list)}...")
        tokens.append(wordpunct_tokenize(str(t).lower()))
    return tokens

train_tokens = tokenize_list(train_texts, "Tokenizing train")
val_tokens   = tokenize_list(val_texts,   "Tokenizing val")
test_tokens  = tokenize_list(test_texts,  "Tokenizing test")

                                   
print("Extending vocab with IMDb-specific words...")
print("Extending vocab with IMDb-specific words...")
for i, tokens in enumerate(train_tokens):
    if i % 5000 == 0 and i > 0:
        print(f"  Processed {i}/{len(train_tokens)} vocabs...")
    for t in tokens:
        if t not in word_dict:
            word_dict[t] = len(word_dict)

VOCAB_SIZE = len(word_dict)
print(f"Extended vocab size: {VOCAB_SIZE}")

def build_tensors(tokenized_list, labels, word_dict, seq_len, desc):
    print(f"{desc}...")
    UNK = word_dict.get("[UNK]", 2)
    encoded = []
    for i, tokens in enumerate(tokenized_list):
        if i % 5000 == 0 and i > 0:
            print(f"  Encoded {i}/{len(tokenized_list)} sequences...")
        ids = [word_dict.get(t, UNK) for t in tokens][:seq_len]
        ids += [0] * (seq_len - len(ids))
        encoded.append(ids)
    return torch.LongTensor(encoded), torch.LongTensor(labels)

train_enc, train_labs = build_tensors(train_tokens, train_labels, word_dict, SEQ_LEN, "Encoding train")
val_enc,   val_labs   = build_tensors(val_tokens,   val_labels,   word_dict, SEQ_LEN, "Encoding val")
test_enc,  test_labs  = build_tensors(test_tokens,  test_labels,  word_dict, SEQ_LEN, "Encoding test")

train_loader = DataLoader(TensorDataset(train_enc, train_labs), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_enc,   val_labs),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(test_enc,  test_labs),  batch_size=BATCH_SIZE)

                                            


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
    print(f"Starting Epoch {epoch} [Train]...")
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, logits = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(-1) == y).sum().item()
        total      += y.size(0)
        
        if i > 0 and i % 100 == 0:
            print(f"  Batch {i}/{len(loader)} | Loss: {total_loss/(i+1):.4f} | Acc: {correct/total:.4f}")
            
    return total_loss / len(loader), correct / total

def evaluate(model, loader, desc):
    model.eval()
    preds, trues = [], []
    print(f"Starting {desc}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(y.numpy())
            
            if i > 0 and i % 100 == 0:
                print(f"  Eval Batch {i}/{len(loader)}...")
                
    return accuracy_score(trues, preds), f1_score(trues, preds, average='macro')

print("\nStarting fine-tuning on IMDb...")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc    = train_epoch(model, train_loader, optimizer, epoch)
    val_acc, val_f1    = evaluate(model, val_loader,  f"Epoch {epoch} [Val]")
    print(f"Epoch {epoch} | Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

test_acc, test_f1 = evaluate(model, test_loader, "Final Test")
print(f"\nFinal Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
