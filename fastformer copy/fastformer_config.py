import json

class FastformerConfig:
    def __init__(self, **kwargs):
        # Set default values
        self.hidden_size = 256
        self.hidden_dropout_prob = 0.2
        self.num_hidden_layers = 2
        self.hidden_act = "gelu"
        self.num_attention_heads = 16
        self.intermediate_size = 1024
        self.max_position_embeddings = 1024
        self.type_vocab_size = 2
        self.vocab_size = 100000
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02
        self.pooler_type = "weightpooler"
        self.attention_probs_dropout_prob = 0.1
        
        # Overwrite defaults with any passed keyword arguments
        self.__dict__.update(kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
