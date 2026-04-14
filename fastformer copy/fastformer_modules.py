import torch
import torch.nn as nn

class FastformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FastformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class FastformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

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
        # It takes every word vector and multiplies it by its corresponding weight from alpha.
        x = torch.bmm(x.permute(0, 2, 1), alpha).reshape(bz, -1)
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size       = config.hidden_size

        # Project input hidden states into Query, Key, and Value spaces
        self.query     = nn.Linear(config.hidden_size, self.all_head_size)
        self.key       = nn.Linear(config.hidden_size, self.all_head_size)
        self.value     = nn.Linear(config.hidden_size, self.all_head_size)

        # to calculate a single score for that specific head.
        self.query_att = nn.Linear(self.attention_head_size, 1)
        self.key_att   = nn.Linear(self.attention_head_size, 1)

        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax   = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # view splits the work into heads.
        # permute moves heads to the front so they can be processed in parallel.
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Project to Q, K, V
        mixed_query = self.query(hidden_states)
        mixed_key   = self.key(hidden_states)
        mixed_value = self.value(hidden_states)

        # 2. SPLIT FIRST (Move to Multi-Head space: [Batch, Heads, Seq, HeadSize])
        q_layer = self.transpose_for_scores(mixed_query)
        
        # 3. CALCULATE WEIGHTS INDEPENDENTLY PER HEAD
        # We pass the pre-sliced q_layer directly into query_att. 
        # It calculates 1 score for every word in every head.
        q_score = self.query_att(q_layer).squeeze(-1) / self.attention_head_size ** 0.5
        q_score += attention_mask # Apply padding mask
        q_weight = self.softmax(q_score).unsqueeze(2) # [Batch, Heads, 1, Seq]
        
        # 4. Global Query Pooling (Weighted sum of words in each head)
        # pooled_q becomes the "Summary" of what each head cares about.
        pooled_q = torch.matmul(q_weight, q_layer) # [Batch, Heads, 1, HeadSize]

        # 5. KEY INTERACTION
        # Move keys to Multi-Head space
        k_layer = self.transpose_for_scores(mixed_key)
        # Element-wise product of Keys and the Global Query Summary
        mixed_qk = k_layer * pooled_q

        # 6. CALCULATE BETA WEIGHTS INDEPENDENTLY PER HEAD
        qk_score  = self.key_att(mixed_qk).squeeze(-1) / self.attention_head_size ** 0.5
        qk_score += attention_mask
        qk_weight = self.softmax(qk_score).unsqueeze(2)
        
        # 7. Global Key Pooling
        pooled_k  = torch.matmul(qk_weight, k_layer) # [Batch, Heads, 1, HeadSize]

        # 8. VALUE INTERACTION
        # Move values to Multi-Head space
        v_layer = self.transpose_for_scores(mixed_value)
        # Final weighted sum using the Global Key summary
        weighted_value = (pooled_k * v_layer).transpose(1, 2) # [Batch, Seq, Heads, HeadSize]
        
        # 9. RECONSTRUCT (Glue heads back together)
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
    def __init__(self, config):
        super().__init__()
        self.encoders            = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm           = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout             = nn.Dropout(config.hidden_dropout_prob)
        self.pooler              = AttentionPooling(config)

    def forward(self, input_embs, attention_mask):
        ext_mask = (1.0 - attention_mask.unsqueeze(1).float()) * -10000.0
        bsz, seq_len, _ = input_embs.shape
        pos_ids  = torch.arange(seq_len, device=input_embs.device).unsqueeze(0).expand(bsz, -1)
        pos_embs = self.position_embeddings(pos_ids)
        x = self.dropout(self.LayerNorm(input_embs + pos_embs))
        for layer in self.encoders:
            x = layer(x, ext_mask)
        return self.pooler(x, attention_mask)
