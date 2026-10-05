from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50_257
    max_seq_len: int = 1024

    d_model: int = 768
    n_layers: int = 6
    n_loops: int = 2
    n_heads: int = 12

    d_ff: int = 3072

    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    bias: bool = True

    max_pos_emb: int = 1024
    rope_theta: float = 10_000.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
