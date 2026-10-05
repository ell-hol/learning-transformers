from config import GPT2Config
import math
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# ============================================================
# Configuration
# ============================================================

SEQ_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

TRAIN_STORIES = 100_000
VAL_STORIES = 1_000

GENERATE_EVERY = 500
GENERATION_PROMPT = "There once was"

warmup_ratio = 0
min_lr_ratio = 0.1


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ):
        super().__init__()

        assert head_dim % 2 == 0, "RoPE requires an even head_dim"

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # [head_dim / 2]

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )
        # [T]

        freqs = torch.outer(
            positions,
            inv_freq,
        )
        # [T, head_dim / 2]

        # LLaMA-style layout
        freqs = torch.cat(
            [freqs, freqs],
            dim=-1,
        )
        # [T, head_dim]

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, Dh]

        seq_len = x.size(1)

        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)

        return x * cos + rotate_half(x) * sin

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # x: [B, H, T, Dh]

    #     seq_len = x.size(-2)

    #     cos = self.cos[:seq_len][None, None, :, :]
    #     sin = self.sin[:seq_len][None, None, :, :]

    #     return x * cos + rotate_half(x) * sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)

    return torch.cat(
        [-x2, x1],
        dim=-1,
    )


# ============================================================
# Attention
# ============================================================


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    return weights @ v


# def scaled_dot_product_attention(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     mask: torch.Tensor | None = None,
# ) -> torch.Tensor:
#     d_k = q.size(-1)

#     scores = torch.bmm(
#         q,
#         k.transpose(-1, -2),
#     ) / (d_k**0.5)

#     if mask is not None:
#         scores = scores.masked_fill(
#             mask == 0,
#             float("-inf"),
#         )

#     weights = F.softmax(scores, dim=-1)

#     return torch.bmm(weights, v)


# class AttentionHead(nn.Module):
#     def __init__(
#         self, embed_dim: int, head_dim: int, max_seq_len: int, rope_theta: float = 10_000.0
#     ):
#         super().__init__()

#         self.q = nn.Linear(embed_dim, head_dim, bias=False)
#         self.k = nn.Linear(embed_dim, head_dim, bias=False)
#         self.v = nn.Linear(embed_dim, head_dim, bias=False)

#         self.rope = RotaryEmbedding(
#             head_dim=head_dim,
#             max_seq_len=max_seq_len,
#             theta=rope_theta,
#         )

#         # One scalar gate per token for this head
#         self.gate = nn.Linear(
#             embed_dim,
#             1,
#             bias=True,
#         )

#     def forward(
#         self,
#         q_states,
#         kv_states=None,
#         mask=None,
#     ):
#         if kv_states is None:
#             kv_states = q_states

#         q = self.q(q_states)
#         k = self.k(kv_states)
#         v = self.v(kv_states)

#         q = self.rope(q)
#         k = self.rope(k)

#         # attn_out = scaled_dot_product_attention(q, k, v, mask)
#         attn_out = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=mask,
#             dropout_p=0.0,
#         )

#         gate = torch.sigmoid(self.gate(q_states))

#         return gate * attn_out


class AttentionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()

        self.q = nn.Linear(
            embed_dim,
            head_dim,
            bias=False,
        )

        self.k = nn.Linear(
            embed_dim,
            head_dim,
            bias=False,
        )

        self.v = nn.Linear(
            embed_dim,
            head_dim,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # One scalar gate per token for this head.
        self.gate = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

    def forward(
        self,
        q_states: torch.Tensor,
        kv_states: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if kv_states is None:
            kv_states = q_states

        q = self.q(q_states)
        k = self.k(kv_states)
        v = self.v(kv_states)

        q = self.rope(q)
        k = self.rope(k)

        # attn_out = F.scaled_dot_product_attention(
        #     q,
        #     k,
        #     v,
        #     attn_mask=mask,
        #     dropout_p=0.0,
        # )

        attn_out = scaled_dot_product_attention(q, k, v, mask)

        gate = torch.sigmoid(self.gate(q_states))

        return gate * attn_out


# class MultiHeadAttention(nn.Module):
#     def __init__(self, config: GPT2Config):
#         super().__init__()

#         self.d_model = config.d_model
#         self.n_heads = config.n_heads

#         assert self.d_model % self.n_heads == 0

#         self.head_dim = self.d_model // self.n_heads

#         self.q_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )

#         self.k_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )

#         self.v_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )

#         # Match the old output_linear exactly.
#         self.output_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=True,
#         )

#         self.rope = RotaryEmbedding(
#             head_dim=self.head_dim,
#             max_seq_len=config.max_seq_len,
#             theta=config.rope_theta,
#         )

#         self.gate_proj = nn.Linear(
#             self.d_model,
#             self.n_heads,
#             bias=True,
#         )

#         self.dropout = nn.Dropout(config.dropout_prob)

#     def forward(
#         self,
#         q_states: torch.Tensor,
#         kv_states: torch.Tensor | None = None,
#         mask: torch.Tensor | None = None,
#     ) -> torch.Tensor:

#         if kv_states is None:
#             kv_states = q_states

#         B, Tq, _ = q_states.shape
#         Tk = kv_states.size(1)

#         q = self.q_proj(q_states)
#         k = self.k_proj(kv_states)
#         v = self.v_proj(kv_states)

#         q = q.reshape(
#             B,
#             Tq,
#             self.n_heads,
#             self.head_dim,
#         ).transpose(1, 2)

#         k = k.reshape(
#             B,
#             Tk,
#             self.n_heads,
#             self.head_dim,
#         ).transpose(1, 2)

#         v = v.reshape(
#             B,
#             Tk,
#             self.n_heads,
#             self.head_dim,
#         ).transpose(1, 2)

#         q = self.rope(q)
#         k = self.rope(k)

#         x = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=mask,
#             dropout_p=0.0,
#         )

#         gate = torch.sigmoid(self.gate_proj(q_states))
#         # [B, Tq, H]

#         gate = gate.transpose(1, 2).unsqueeze(-1)
#         # [B, H, Tq, 1]

#         x = x * gate

#         x = (
#             x.transpose(1, 2)
#             .contiguous()
#             .reshape(
#                 B,
#                 Tq,
#                 self.d_model,
#             )
#         )

#         x = self.output_proj(x)

#         return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        embed_dim = config.d_model
        num_heads = config.n_heads

        assert embed_dim % num_heads == 0

        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embed_dim=embed_dim,
                    head_dim=head_dim,
                    max_seq_len=config.max_seq_len,
                    rope_theta=config.rope_theta,
                )
                for _ in range(num_heads)
            ]
        )

        self.output_linear = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = torch.cat(
            [
                head(
                    q_states=q,
                    kv_states=kv,
                    mask=mask,
                )
                for head in self.heads
            ],
            dim=-1,
        )

        x = self.output_linear(x)

        return self.dropout(x)


# ============================================================
# Feed-forward network
# ============================================================


class FeedForward(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.d_model,
            config.d_ff,
        )

        self.linear_2 = nn.Linear(
            config.d_ff,
            config.d_model,
        )

        # Close to GPT-2's GELU variant.
        self.gelu = nn.GELU(approximate="tanh")

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)

        return self.dropout(x)


# ============================================================
# Embeddings
# ============================================================


class Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.tok_emb = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.tok_emb(input_ids)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


# ============================================================
# Encoder
# ============================================================


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_up_proj = nn.Linear(
            config.d_model,
            2 * config.d_ff,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=False,
        )

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)

        x = F.silu(gate) * up
        x = self.down_proj(x)

        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.layer_norm1 = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.layer_norm2 = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.layer_norm1(x),
            mask=mask,
        )

        x = x + self.feed_forward(self.layer_norm2(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.embeddings = Embeddings(config)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(
                x,
                mask,
            )

        return x


# ============================================================
# Decoder
#
# add_cross_attention=False:
#     decoder-only / GPT
#
# add_cross_attention=True:
#     encoder-decoder Transformer
# ============================================================


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        add_cross_attention: bool = False,
    ):
        super().__init__()

        self.add_cross_attention = add_cross_attention

        self.layer_norm1 = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.layer_norm2 = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.self_attention = MultiHeadAttention(config)

        if add_cross_attention:
            self.cross_attention = MultiHeadAttention(config)

            self.layer_norm3 = RMSNorm(
                config.d_model,
                eps=config.layer_norm_eps,
            )

        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        self_attention_mask: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ------------------------------------
        # Causal self-attention
        # ------------------------------------

        x = x + self.self_attention(
            q=self.layer_norm1(x),
            mask=self_attention_mask,
        )

        # ------------------------------------
        # Encoder-decoder cross-attention
        # ------------------------------------

        if self.add_cross_attention:
            if encoder_output is None:
                raise ValueError("encoder_output is required " "when cross-attention is enabled")

            x = x + self.cross_attention(
                q=self.layer_norm2(x),
                kv=encoder_output,
                mask=cross_attention_mask,
            )

            # FFN gets its own pre-norm.
            x = x + self.feed_forward(self.layer_norm3(x))

        # ------------------------------------
        # Decoder-only case
        # ------------------------------------

        else:
            x = x + self.feed_forward(self.layer_norm2(x))

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        add_cross_attention: bool = False,
    ):
        super().__init__()

        self.embeddings = Embeddings(config)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    config,
                    add_cross_attention=add_cross_attention,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.n_loops = config.n_loops

    def forward(
        self,
        input_ids: torch.Tensor,
        self_attention_mask: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)

        for _ in range(self.n_loops):
            for layer in self.layers:
                x = layer(
                    x,
                    self_attention_mask=self_attention_mask,
                    encoder_output=encoder_output,
                    cross_attention_mask=cross_attention_mask,
                )

        return x


# ============================================================
# Decoder-only language model
# ============================================================


class TransformerLM(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config

        self.decoder = TransformerDecoder(
            config,
            add_cross_attention=False,
        )

        self.final_norm = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        self_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.decoder(
            input_ids,
            self_attention_mask=self_attention_mask,
        )

        x = self.final_norm(x)

        return self.lm_head(x)


# ============================================================
# GPT-2-style initialization
# ============================================================


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


# ============================================================
# Dataset
# ============================================================


class LanguageModelDataset(Dataset):
    """
    Converts many stories into one continuous token stream:

    story 1 <EOS> story 2 <EOS> story 3 <EOS> ...

    The stream is then divided into fixed-length chunks.

    Each example is:

        inputs  = tokens[t : t + seq_len]
        targets = tokens[t + 1 : t + seq_len + 1]
    """

    def __init__(
        self,
        texts,
        tokenizer,
        seq_len: int,
        tokenization_batch_size: int = 256,
    ):
        self.seq_len = seq_len

        tokens = []

        # Batch tokenization is substantially faster than
        # tokenizer.encode() once per story.
        for start in range(
            0,
            len(texts),
            tokenization_batch_size,
        ):
            text_batch = texts[start : start + tokenization_batch_size]

            encoded = tokenizer(
                text_batch,
                add_special_tokens=False,
            )["input_ids"]

            for ids in encoded:
                tokens.extend(ids)
                tokens.append(tokenizer.eos_token_id)

        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        print(
            f"Dataset: {len(texts):,} stories, "
            f"{len(self.tokens):,} tokens, "
            f"{len(self):,} sequences"
        )

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len

        chunk = self.tokens[start : start + self.seq_len + 1]

        input_ids = chunk[:-1]
        targets = chunk[1:]

        return input_ids, targets


# ============================================================
# Masks
# ============================================================


def get_causal_mask(
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=device,
        )
    )


# ============================================================
# Generation
# ============================================================


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
):
    was_training = model.training
    model.eval()

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)

    for _ in range(max_new_tokens):

        # Keep generation inside the positional embedding limit.
        model_input = input_ids[
            :,
            -model.config.max_pos_emb :,
        ]

        seq_len = model_input.size(1)

        causal_mask = get_causal_mask(
            seq_len,
            device,
        )

        logits = model(
            model_input,
            self_attention_mask=causal_mask,
        )

        logits = logits[:, -1, :] / 0.8

        top_k = 50

        values, indices = torch.topk(
            logits,
            top_k,
            dim=-1,
        )

        probs = F.softmax(values, dim=-1)

        sample = torch.multinomial(
            probs,
            num_samples=1,
        )

        next_token = indices.gather(
            -1,
            sample,
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    if was_training:
        model.train()

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def evaluate(
    model,
    loader,
    causal_mask,
    device,
):
    model.eval()

    total_loss = 0.0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(
            input_ids,
            self_attention_mask=causal_mask,
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            targets.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Main
# ============================================================


def main():
    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # No padding is necessary because we're packing a
    # continuous token stream into fixed-size chunks.

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        max_pos_emb=SEQ_LEN,
        d_model=256,
        n_heads=4,
        n_layers=4,
        n_loops=2,
        d_ff=1024,
        dropout_prob=0.0,
    )

    assert tokenizer.vocab_size == config.vocab_size

    # --------------------------------------------------------
    # TinyStories
    # --------------------------------------------------------

    print("Loading TinyStories...")

    train_texts = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{TRAIN_STORIES}]",
    )["text"]

    val_texts = load_dataset(
        "roneneldan/TinyStories",
        split=f"validation[:{VAL_STORIES}]",
    )["text"]

    print("Tokenizing training set...")

    train_dataset = LanguageModelDataset(
        train_texts,
        tokenizer,
        seq_len=SEQ_LEN,
    )

    print("Tokenizing validation set...")

    val_dataset = LanguageModelDataset(
        val_texts,
        tokenizer,
        seq_len=SEQ_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"device: {device}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TransformerLM(config)

    model.apply(init_weights)

    effective_depth = config.n_layers * config.n_loops  # 8 here
    scaled_std = 0.02 / math.sqrt(2 * effective_depth)

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            # nn.init.normal_(module.output_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(
                module.output_linear.weight,
                mean=0.0,
                std=scaled_std,
            )
        if isinstance(module, SwiGLU):
            nn.init.normal_(module.down_proj.weight, mean=0.0, std=scaled_std)
        if isinstance(module, AttentionHead):
            nn.init.zeros_(module.gate.weight)
            nn.init.constant_(module.gate.bias, 2.0)

    # GPT-2-style weight tying.
    model.lm_head.weight = model.decoder.embeddings.tok_emb.weight

    model = model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"parameters: " f"{num_parameters / 1e6:.2f}M")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=LEARNING_RATE,
    #     weight_decay=0.01,
    # )

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps

        progress = (step - warmup_steps) / (total_steps - warmup_steps)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    # Every training sequence has exactly SEQ_LEN tokens,
    # so this mask can be constructed once.
    causal_mask = get_causal_mask(
        SEQ_LEN,
        device,
    )

    global_step = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_train_loss = 0.0

        for step, (
            input_ids,
            targets,
        ) in enumerate(train_loader):

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # [B, T, V]
            logits = model(
                input_ids,
                self_attention_mask=causal_mask,
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if global_step % 50 == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={step} "
                    f"global_step={global_step} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"loss={loss.item():.4f}"
                )

            if global_step > 0 and global_step % GENERATE_EVERY == 0:
                generated = generate(
                    model,
                    tokenizer,
                    prompt=GENERATION_PROMPT,
                    device=device,
                    max_new_tokens=50,
                )

                print()
                print(f"--- generation " f"@ step {global_step} ---")
                print(generated)
                print("---------------------------")
                print()

            global_step += 1

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        train_loss = total_train_loss / len(train_loader)

        val_loss = evaluate(
            model,
            val_loader,
            causal_mask,
            device,
        )

        print()
        print(f"epoch={epoch + 1} complete")
        print(f"train_loss={train_loss:.4f}")
        print(f"val_loss={val_loss:.4f}")

        generated = generate(
            model,
            tokenizer,
            prompt=GENERATION_PROMPT,
            device=device,
            max_new_tokens=75,
        )

        print()
        print("--- generation ---")
        print(generated)
        print("------------------")
        print()


if __name__ == "__main__":
    main()
