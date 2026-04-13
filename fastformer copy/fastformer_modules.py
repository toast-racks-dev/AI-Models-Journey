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
