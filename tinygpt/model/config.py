from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8          # < n_heads => grouped-query attention
    mlp_hidden: int = 1408       # SwiGLU hidden size (~8/3 * d_model, rounded to a multiple of 64)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_params(self) -> int:
        """Analytic parameter count (matches sum(p.numel()) of GPT(cfg))."""
        d, hd = self.d_model, self.head_dim
        attn = d * self.n_heads * hd + 2 * d * self.n_kv_heads * hd + self.n_heads * hd * d
        mlp = 3 * d * self.mlp_hidden
        per_layer = attn + mlp + 2 * d
        total = self.n_layers * per_layer + d + self.vocab_size * d
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total

    def kv_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_elem

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
