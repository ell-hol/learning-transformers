from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Experiment configuration
# ============================================================

SEED = 42

TRAIN_STORIES = 1_000_000
VAL_STORIES = 10_000

VOCAB_SIZE = 8_192
TOKENIZER_DIR = Path("tinystories_tokenizer_8k")
TOKEN_CACHE_DIR = Path("token_cache")
EOS_TOKEN = "<|endoftext|>"

SEQ_LEN = 513
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
# 8 * 256 * 4 = 8192 tokens / optimizer update.

NUM_EPOCHS = 10

# Auxiliary AdamW: embedding + RMSNorm scales.
ADAMW_LR = 1e-3
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01

# Reference-style Muon learning-rate convention.
MUON_LR = 0.02
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5

# Match approximately the per-step decay coefficient of AdamW:
#   ADAMW_LR * ADAMW_WEIGHT_DECAY == MUON_LR * MUON_WEIGHT_DECAY
MUON_WEIGHT_DECAY = ADAMW_LR * ADAMW_WEIGHT_DECAY / MUON_LR

# WSD with zero warmup: stable for 85%, cosine decay for final 15%.
STABLE_RATIO = 0.85
MIN_LR_RATIO = 0.10

MAX_GRAD_NORM = 1.0

LOG_EVERY = 25
GENERATE_EVERY = 250
CHECKPOINT_EVERY = 250

GENERATION_PROMPT = "There once was"
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 50

CHECKPOINT_DIR = Path("checkpoints_tinystories_muon")
# Example: CHECKPOINT_DIR / "latest.pt"
RESUME_FROM: Path | None = None


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = SEQ_LEN
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    n_loops: int = 2
    d_ff: int = 1408
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0


# ============================================================
# Tokenizer
# ============================================================


def get_or_train_tokenizer(train_texts) -> ByteLevelBPETokenizer:
    vocab_file = TOKENIZER_DIR / "vocab.json"
    merges_file = TOKENIZER_DIR / "merges.txt"

    if vocab_file.exists() and merges_file.exists():
        print(f"Loading tokenizer from {TOKENIZER_DIR}")
        tokenizer = ByteLevelBPETokenizer(
            str(vocab_file),
            str(merges_file),
        )
    else:
        print(f"Training {VOCAB_SIZE:,}-token byte-level BPE tokenizer...")
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            train_texts,
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=[EOS_TOKEN],
        )
        tokenizer.save_model(str(TOKENIZER_DIR))

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise RuntimeError(f"Tokenizer is missing special token {EOS_TOKEN!r}")

    actual_vocab = tokenizer.get_vocab_size()
    print(f"tokenizer vocab size: {actual_vocab:,}")
    print(f"EOS token id: {eos_id}")

    return tokenizer


# ============================================================
# Compact tokenized corpus cache
# ============================================================


def tokenize_corpus(
    texts,
    tokenizer: ByteLevelBPETokenizer,
    cache_path: Path,
    batch_size: int = 256,
):
    """
    Store all documents in one compact int32 token tensor plus offsets.

    For document i:
        tokens[offsets[i] : offsets[i + 1]]

    Every document ends in EOS.
    """

    if cache_path.exists():
        print(f"Loading token cache: {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        return cached["tokens"], cached["offsets"]

    print(f"Building token cache: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    token_chunks: list[torch.Tensor] = []
    offsets = [0]
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encodings = tokenizer.encode_batch(batch)

        flat_batch: list[int] = []
        for encoding in encodings:
            ids = encoding.ids
            flat_batch.extend(ids)
            flat_batch.append(eos_id)

            total_tokens += len(ids) + 1
            offsets.append(total_tokens)

        token_chunks.append(torch.tensor(flat_batch, dtype=torch.int32))

        if start > 0 and start % 10_000 == 0:
            print(f"  tokenized {start:,}/{len(texts):,} stories " f"({total_tokens:,} tokens)")

    tokens = torch.cat(token_chunks)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)

    torch.save(
        {
            "tokens": tokens,
            "offsets": offsets_tensor,
        },
        cache_path,
    )

    print(f"Cached {len(texts):,} stories, " f"{tokens.numel():,} tokens")

    return tokens, offsets_tensor


# ============================================================
# Randomized, document-aware packing
# ============================================================


class PackedStoryDataset(Dataset):
    """
    Packs multiple TinyStories into fixed SEQ_LEN training examples.

    Important properties:

    1. Stories are reshuffled every epoch.
       Therefore pack boundaries, and the split points of long documents,
       naturally change every epoch.

    2. Each returned token has a next-token target from the SAME document.
       EOS is a target, but the final EOS itself is never asked to predict
       the beginning of another story.

    3. segment_ids let attention construct a block-diagonal causal mask, so
       two unrelated stories sharing one physical sequence cannot attend to
       each other.

    4. position_ids reset to zero for each packed document fragment.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        offsets: torch.Tensor,
        seq_len: int,
        seed: int,
        randomize: bool,
    ):
        self.tokens = tokens
        self.offsets = offsets
        self.seq_len = seq_len
        self.seed = seed
        self.randomize = randomize
        self.num_documents = offsets.numel() - 1

        self.packs: list[list[tuple[int, int, int]]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        order = list(range(self.num_documents))

        if self.randomize:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(order)

        packs: list[list[tuple[int, int, int]]] = []
        current_pack: list[tuple[int, int, int]] = []
        used = 0

        for doc_idx in order:
            doc_len = int(self.offsets[doc_idx + 1] - self.offsets[doc_idx])

            # Need at least input + target.
            predictable_tokens = doc_len - 1
            if predictable_tokens <= 0:
                continue

            doc_pos = 0

            while doc_pos < predictable_tokens:
                space = self.seq_len - used
                take = min(
                    space,
                    predictable_tokens - doc_pos,
                )

                current_pack.append((doc_idx, doc_pos, take))

                used += take
                doc_pos += take

                if used == self.seq_len:
                    packs.append(current_pack)
                    current_pack = []
                    used = 0

        # Deliberately drop the one final incomplete pack.
        # Everything else is densely packed with no padding.
        self.packs = packs

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        pack = self.packs[idx]

        input_ids = torch.empty(
            self.seq_len,
            dtype=torch.long,
        )
        targets = torch.empty(
            self.seq_len,
            dtype=torch.long,
        )
        segment_ids = torch.empty(
            self.seq_len,
            dtype=torch.long,
        )
        position_ids = torch.empty(
            self.seq_len,
            dtype=torch.long,
        )

        cursor = 0

        for segment_id, (doc_idx, doc_pos, take) in enumerate(pack):
            doc_start = int(self.offsets[doc_idx])
            start = doc_start + doc_pos

            # take inputs + one extra token for their targets.
            chunk = self.tokens[start : start + take + 1].long()

            input_ids[cursor : cursor + take] = chunk[:-1]
            targets[cursor : cursor + take] = chunk[1:]
            segment_ids[cursor : cursor + take] = segment_id

            # Reset RoPE coordinates for every independent fragment.
            position_ids[cursor : cursor + take] = torch.arange(
                take,
                dtype=torch.long,
            )

            cursor += take

        if cursor != self.seq_len:
            raise RuntimeError(f"Internal packing bug: packed {cursor}, expected {self.seq_len}")

        return {
            "input_ids": input_ids,
            "targets": targets,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
        }


# ============================================================
# RMSNorm
# ============================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()

        rms = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x_norm = x_float * torch.rsqrt(rms + self.eps)

        return x_norm.to(input_dtype) * self.weight


# ============================================================
# RoPE
# ============================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(positions, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        # q, k: [B, H, T, Dh]
        # position_ids: [B, T]

        cos = self.cos[position_ids].unsqueeze(1)
        sin = self.sin[position_ids].unsqueeze(1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        return q, k


# ============================================================
# Manual, vectorized, FP32 attention + QK-Norm
# ============================================================


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5

        # One large projection rather than separate Python AttentionHead modules.
        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False,
        )

        self.output_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        # QK-Norm operates on each query/key vector's final dimension.
        # The learned scale is shared across heads, which is the usual
        # RMSNorm-last-dimension behavior.
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_eps,
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rope(
            q,
            k,
            position_ids,
        )

        # Explicitly keep attention-logit math and softmax in FP32.
        scores = (
            torch.matmul(
                q.float(),
                k.float().transpose(-2, -1),
            )
            * self.scale
        )

        scores = scores.masked_fill(
            ~attention_mask,
            float("-inf"),
        )

        weights = F.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        )

        out = torch.matmul(
            weights,
            v.float(),
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.d_model,
            )
        )

        out = out.to(x.dtype)
        out = self.output_proj(out)

        return self.dropout(out)


# ============================================================
# SwiGLU
# ============================================================


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
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

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return self.dropout(x)


# ============================================================
# Transformer
# ============================================================


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.d_model,
            eps=config.rms_eps,
        )
        self.ffn_norm = RMSNorm(
            config.d_model,
            eps=config.rms_eps,
        )

        self.attention = MultiHeadAttention(config)
        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attn_norm(x),
            attention_mask,
            position_ids,
        )

        x = x + self.feed_forward(self.ffn_norm(x))

        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # Four PHYSICAL layers. Their weights are reused on loop 2.
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])

        self.final_norm = RMSNorm(
            config.d_model,
            eps=config.rms_eps,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        causal = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal[None, None, :, :],
            persistent=False,
        )

    def build_attention_mask(
        self,
        seq_len: int,
        segment_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        causal = self.causal_mask[
            :,
            :,
            :seq_len,
            :seq_len,
        ]

        if segment_ids is None:
            return causal

        same_document = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]

        return causal & same_document

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max {self.config.max_seq_len}")

        if position_ids is None:
            position_ids = (
                torch.arange(
                    seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        attention_mask = self.build_attention_mask(
            seq_len,
            segment_ids,
        )

        x = self.token_embedding(input_ids)

        # Fixed recurrent depth: 4 shared layers x 2 loops = 8 logical passes.
        for _ in range(self.config.n_loops):
            for layer in self.layers:
                x = layer(
                    x,
                    attention_mask,
                    position_ids,
                )

        x = self.final_norm(x)
        return self.lm_head(x)


# ============================================================
# Initialization
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

    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def initialize_model(
    model: TransformerLM,
    config: ModelConfig,
):
    model.apply(init_weights)

    # GPT-2-style residual projection scaling, but use logical depth because
    # the same physical layers are traversed twice.
    effective_depth = config.n_layers * config.n_loops
    residual_std = 0.02 / math.sqrt(2 * effective_depth)

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            nn.init.normal_(
                module.output_proj.weight,
                mean=0.0,
                std=residual_std,
            )

        elif isinstance(module, SwiGLU):
            nn.init.normal_(
                module.down_proj.weight,
                mean=0.0,
                std=residual_std,
            )

    # Weight tying. This is one shared Parameter object.
    model.lm_head.weight = model.token_embedding.weight


# ============================================================
# Muon for PyTorch 2.2.2 / MPS
# ============================================================


def zeropower_via_newton_schulz5_fp32(
    grad: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """
    Reference Muon quintic Newton-Schulz iteration, deliberately kept FP32.

    The public Muon reference converts this operation to bfloat16 for modern
    CUDA accelerators. For an older AMD MPS machine / PyTorch 2.2.2, FP32 is
    the safer choice.
    """

    if grad.ndim != 2:
        raise ValueError("Muon expects 2D hidden weight matrices")

    a, b, c = 3.4445, -4.7750, 2.0315

    x = grad.float()

    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        aa = x @ x.transpose(-2, -1)
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if transposed:
        x = x.transpose(-2, -1)

    return x


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    ns_steps: int,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - beta)

    # Nesterov form used by the reference implementation.
    update = grad.lerp(momentum_buffer, beta)

    update = zeropower_via_newton_schulz5_fp32(
        update,
        steps=ns_steps,
    )

    update *= (
        max(
            1.0,
            update.size(-2) / update.size(-1),
        )
        ** 0.5
    )

    return update


class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = MUON_LR,
        weight_decay: float = MUON_WEIGHT_DECAY,
        momentum: float = MUON_MOMENTUM,
        ns_steps: int = MUON_NS_STEPS,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim != 2:
                    raise ValueError(f"Muon got non-2D parameter with shape {tuple(p.shape)}")

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                )

                p.mul_(1.0 - group["lr"] * group["weight_decay"])

                p.add_(
                    update.to(p.dtype),
                    alpha=-group["lr"],
                )

        return loss


# ============================================================
# Optimizer parameter split
# ============================================================


def create_optimizers(model: TransformerLM):
    embedding_id = id(model.token_embedding.weight)

    muon_params = []
    adam_decay = []
    adam_no_decay = []

    muon_names = []
    adam_names = []

    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if id(p) in seen:
            continue
        seen.add(id(p))

        # Hidden matrix weights only.
        use_muon = p.ndim == 2 and id(p) != embedding_id

        if use_muon:
            muon_params.append(p)
            muon_names.append(name)
        else:
            # Embedding / tied LM head is 2D and gets AdamW decay.
            # RMSNorm scales are 1D and get no decay.
            if p.ndim >= 2:
                adam_decay.append(p)
            else:
                adam_no_decay.append(p)

            adam_names.append(name)

    print("\nMuon parameters:")
    for name in muon_names:
        print(f"  {name}")

    print("\nAdamW parameters:")
    for name in adam_names:
        print(f"  {name}")
    print()

    muon = SingleDeviceMuon(
        muon_params,
        lr=MUON_LR,
        momentum=MUON_MOMENTUM,
        weight_decay=MUON_WEIGHT_DECAY,
        ns_steps=MUON_NS_STEPS,
    )

    adamw = torch.optim.AdamW(
        [
            {
                "params": adam_decay,
                "weight_decay": ADAMW_WEIGHT_DECAY,
            },
            {
                "params": adam_no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
    )

    return muon, adamw


# ============================================================
# Zero-warmup WSD schedule
# ============================================================


def create_wsd_scheduler(
    optimizer,
    total_updates: int,
):
    stable_updates = int(total_updates * STABLE_RATIO)

    def lr_lambda(update: int):
        if update < stable_updates:
            return 1.0

        decay_updates = max(
            1,
            total_updates - stable_updates,
        )

        progress = (update - stable_updates) / decay_updates

        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ============================================================
# Generation
# ============================================================


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
):
    was_training = model.training
    model.eval()

    ids = tokenizer.encode(prompt).ids

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    for _ in range(max_new_tokens):
        model_input = input_ids[
            :,
            -model.config.max_seq_len :,
        ]

        logits = model(model_input)

        next_logits = logits[:, -1, :] / GENERATION_TEMPERATURE

        k = min(
            GENERATION_TOP_K,
            next_logits.size(-1),
        )

        values, indices = torch.topk(
            next_logits,
            k=k,
            dim=-1,
        )

        probs = F.softmax(
            values,
            dim=-1,
        )

        sample = torch.multinomial(
            probs,
            num_samples=1,
        )

        next_token = indices.gather(
            -1,
            sample,
        )

        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1,
        )

        if next_token.item() == eos_id:
            break

    if was_training:
        model.train()

    return tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens=True,
    )


# ============================================================
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        position_ids = batch["position_ids"].to(device)

        logits = model(
            input_ids,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


# ============================================================
# Checkpointing
# ============================================================


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    muon,
    adamw,
    muon_scheduler,
    adamw_scheduler,
    epoch: int,
    batch_idx: int,
    global_step: int,
    epoch_loss_sum: float,
    epoch_batches_seen: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": model.state_dict(),
        "muon": muon.state_dict(),
        "adamw": adamw.state_dict(),
        "muon_scheduler": muon_scheduler.state_dict(),
        "adamw_scheduler": adamw_scheduler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_batches_seen": epoch_batches_seen,
        "config": model.config.__dict__,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
    }

    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: TransformerLM,
    muon,
    adamw,
    muon_scheduler,
    adamw_scheduler,
    device: torch.device,
):
    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if checkpoint["seq_len"] != SEQ_LEN:
        raise ValueError("SEQ_LEN differs from checkpoint")

    if checkpoint["micro_batch_size"] != MICRO_BATCH_SIZE:
        raise ValueError("MICRO_BATCH_SIZE differs from checkpoint")

    if checkpoint["grad_accum_steps"] != GRAD_ACCUM_STEPS:
        raise ValueError("GRAD_ACCUM_STEPS differs from checkpoint")

    model.load_state_dict(checkpoint["model"])
    muon.load_state_dict(checkpoint["muon"])
    adamw.load_state_dict(checkpoint["adamw"])
    muon_scheduler.load_state_dict(checkpoint["muon_scheduler"])
    adamw_scheduler.load_state_dict(checkpoint["adamw_scheduler"])

    return (
        checkpoint["epoch"],
        checkpoint["batch_idx"] + 1,
        checkpoint["global_step"],
        checkpoint.get("epoch_loss_sum", 0.0),
        checkpoint.get("epoch_batches_seen", 0),
    )


# ============================================================
# Main
# ============================================================


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Raw TinyStories
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

    # --------------------------------------------------------
    # Custom 8k tokenizer
    # --------------------------------------------------------

    tokenizer = get_or_train_tokenizer(train_texts)
    vocab_size = tokenizer.get_vocab_size()

    # --------------------------------------------------------
    # Token caches
    # --------------------------------------------------------

    train_cache = TOKEN_CACHE_DIR / f"train_{TRAIN_STORIES}_v{vocab_size}.pt"

    val_cache = TOKEN_CACHE_DIR / f"val_{VAL_STORIES}_v{vocab_size}.pt"

    train_tokens, train_offsets = tokenize_corpus(
        train_texts,
        tokenizer,
        train_cache,
    )

    val_tokens, val_offsets = tokenize_corpus(
        val_texts,
        tokenizer,
        val_cache,
    )

    # Raw text is no longer needed after tokenization.
    del train_texts
    del val_texts

    # --------------------------------------------------------
    # Packed datasets
    # --------------------------------------------------------

    train_dataset = PackedStoryDataset(
        train_tokens,
        train_offsets,
        seq_len=SEQ_LEN,
        seed=SEED,
        randomize=True,
    )

    val_dataset = PackedStoryDataset(
        val_tokens,
        val_offsets,
        seq_len=SEQ_LEN,
        seed=SEED,
        randomize=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Pack count is invariant to story ordering because we densely pack the
    # same number of predictable tokens and drop only one final partial pack.
    train_dataset.set_epoch(0)
    train_batches_per_epoch = math.ceil(len(train_dataset) / MICRO_BATCH_SIZE)
    updates_per_epoch = math.ceil(train_batches_per_epoch / GRAD_ACCUM_STEPS)
    total_updates = NUM_EPOCHS * updates_per_epoch

    print(f"train packs/epoch: {len(train_dataset):,}")
    print(f"train microbatches/epoch: {train_batches_per_epoch:,}")
    print(f"optimizer updates/epoch: {updates_per_epoch:,}")
    print(f"total optimizer updates: {total_updates:,}")
    print("effective tokens/update: " f"{MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS:,}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda")

    print(f"device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    config = ModelConfig(
        vocab_size=vocab_size,
    )

    model = TransformerLM(config)
    initialize_model(model, config)
    model = model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"parameters: {num_parameters / 1e6:.2f}M")
    print(
        f"architecture: d_model={config.d_model}, "
        f"heads={config.n_heads}, "
        f"physical_layers={config.n_layers}, "
        f"loops={config.n_loops}, "
        f"d_ff={config.d_ff}, "
        f"context={config.max_seq_len}"
    )

    # --------------------------------------------------------
    # Muon + AdamW
    # --------------------------------------------------------

    muon, adamw = create_optimizers(model)

    muon_scheduler = create_wsd_scheduler(
        muon,
        total_updates,
    )

    adamw_scheduler = create_wsd_scheduler(
        adamw,
        total_updates,
    )

    print(
        f"Muon: lr={MUON_LR:g}, "
        f"momentum={MUON_MOMENTUM}, "
        f"wd={MUON_WEIGHT_DECAY:g}, "
        f"NS steps={MUON_NS_STEPS}"
    )

    print(f"AdamW: lr={ADAMW_LR:g}, " f"betas={ADAMW_BETAS}, " f"wd={ADAMW_WEIGHT_DECAY:g}")

    print(f"WSD: warmup=0, stable={STABLE_RATIO:.0%}, " f"final_lr_ratio={MIN_LR_RATIO}")

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 0
    start_batch = 0
    global_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_batches_seen = 0

    if RESUME_FROM is not None:
        (
            start_epoch,
            start_batch,
            global_step,
            resumed_epoch_loss_sum,
            resumed_epoch_batches_seen,
        ) = load_checkpoint(
            RESUME_FROM,
            model,
            muon,
            adamw,
            muon_scheduler,
            adamw_scheduler,
            device,
        )

        if start_batch >= train_batches_per_epoch:
            start_epoch += 1
            start_batch = 0
            resumed_epoch_loss_sum = 0.0
            resumed_epoch_batches_seen = 0

        print(
            f"Resuming epoch={start_epoch}, "
            f"microbatch={start_batch}, "
            f"global_step={global_step}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_dataset.set_epoch(epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=MICRO_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        if len(train_loader) != train_batches_per_epoch:
            raise RuntimeError("Unexpected train-loader length change")

        model.train()

        if epoch == start_epoch and start_batch > 0:
            epoch_loss_sum = resumed_epoch_loss_sum
            epoch_batches_seen = resumed_epoch_batches_seen
        else:
            epoch_loss_sum = 0.0
            epoch_batches_seen = 0

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        accum_loss_sum = 0.0
        accum_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            position_ids = batch["position_ids"].to(device)

            logits = model(
                input_ids,
                segment_ids=segment_ids,
                position_ids=position_ids,
            )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            epoch_loss_sum += loss.item()
            epoch_batches_seen += 1

            accum_loss_sum += loss.item()
            accum_count += 1

            # Correct scaling even for the short final accumulation group.
            group_start = (batch_idx // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS

            group_size = min(
                GRAD_ACCUM_STEPS,
                len(train_loader) - group_start,
            )

            (loss / group_size).backward()

            end_of_group = (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or batch_idx + 1 == len(
                train_loader
            )

            if not end_of_group:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            muon.step()
            adamw.step()

            muon_scheduler.step()
            adamw_scheduler.step()

            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)

            mean_group_loss = accum_loss_sum / accum_count
            accum_loss_sum = 0.0
            accum_count = 0

            # Deliberately print optimizer step 0.
            if global_step % LOG_EVERY == 0:
                print(
                    f"epoch={epoch} "
                    f"global_step={global_step} "
                    f"microbatch={batch_idx} "
                    f"muon_lr={muon_scheduler.get_last_lr()[0]:.2e} "
                    f"adamw_lr={adamw_scheduler.get_last_lr()[0]:.2e} "
                    f"loss={mean_group_loss:.4f} "
                    f"grad_norm={float(grad_norm):.3f}"
                )

            global_step += 1

            if global_step % GENERATE_EVERY == 0:
                generated = generate(
                    model,
                    tokenizer,
                    prompt=GENERATION_PROMPT,
                    device=device,
                    max_new_tokens=80,
                )

                print()
                print(f"--- generation @ step {global_step} ---")
                print(generated)
                print("--------------------------------------")
                print()

            if global_step % CHECKPOINT_EVERY == 0:
                numbered_path = CHECKPOINT_DIR / f"step_{global_step:08d}.pt"

                save_checkpoint(
                    numbered_path,
                    model,
                    muon,
                    adamw,
                    muon_scheduler,
                    adamw_scheduler,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_batches_seen=epoch_batches_seen,
                )

                save_checkpoint(
                    CHECKPOINT_DIR / "latest.pt",
                    model,
                    muon,
                    adamw,
                    muon_scheduler,
                    adamw_scheduler,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_batches_seen=epoch_batches_seen,
                )

                print(f"checkpoint saved: {numbered_path}")

        start_batch = 0

        train_loss = epoch_loss_sum / epoch_batches_seen
        val_loss = evaluate(
            model,
            val_loader,
            device,
        )

        print()
        print(f"epoch={epoch} complete")
        print(f"train_loss={train_loss:.4f}")
        print(f"val_loss={val_loss:.4f}")

        generated = generate(
            model,
            tokenizer,
            prompt=GENERATION_PROMPT,
            device=device,
            max_new_tokens=120,
        )

        print()
        print("--- generation ---")
        print(generated)
        print("------------------")
        print()

        save_checkpoint(
            CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt",
            model,
            muon,
            adamw,
            muon_scheduler,
            adamw_scheduler,
            epoch=epoch,
            batch_idx=len(train_loader) - 1,
            global_step=global_step,
            epoch_loss_sum=epoch_loss_sum,
            epoch_batches_seen=epoch_batches_seen,
        )

        save_checkpoint(
            CHECKPOINT_DIR / "latest.pt",
            model,
            muon,
            adamw,
            muon_scheduler,
            adamw_scheduler,
            epoch=epoch,
            batch_idx=len(train_loader) - 1,
            global_step=global_step,
            epoch_loss_sum=epoch_loss_sum,
            epoch_batches_seen=epoch_batches_seen,
        )


if __name__ == "__main__":
    main()
