from config import GPT2Config

import math
from pathlib import Path

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
WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.0
MIN_LR_RATIO = 0.1

TRAIN_STORIES = 100_000
VAL_STORIES = 1_000

GENERATE_EVERY = 500

CHECKPOINT_EVERY = 500
CHECKPOINT_DIR = Path("checkpoints")

# Set to e.g. "checkpoints/latest.pt" to resume.
RESUME_FROM = None

GENERATION_PROMPT = "There once was"

SEED = 42


# ============================================================
# Architecture
# ============================================================

# Four physical layers, recursively reused.
N_LAYERS = 4

# Maximum token recursion depth.
MAX_RECURSIONS = 4

# Hybrid positional encoding:
#
# layer 0 -> RoPE
# layer 1 -> NoPE
# layer 2 -> RoPE
# layer 3 -> NoPE
#
ROPE_LAYERS = {0, 2}


# ============================================================
# MoR router
# ============================================================

# Switch-style load balancing coefficient.
ROUTER_BALANCE_COEF = 0.01

# Router logit stabilization.
ROUTER_Z_LOSS_COEF = 1e-3

ROUTER_TEMPERATURE = 1.0


# ============================================================
# RMSNorm
# ============================================================


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))

        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        rms = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        return x * torch.rsqrt(rms + self.eps) * self.weight


# ============================================================
# Rotary embeddings
# ============================================================


def rotate_half(
    x: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = x.chunk(
        2,
        dim=-1,
    )

    return torch.cat(
        (-x2, x1),
        dim=-1,
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ):
        super().__init__()

        assert head_dim % 2 == 0

        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        # LLaMA-style layout.
        freqs = torch.cat(
            (freqs, freqs),
            dim=-1,
        )

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
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # x:
        # [B, T, Dh]
        #
        # position_ids:
        # [B, T]

        cos = self.cos[position_ids]
        sin = self.sin[position_ids]

        return x * cos + rotate_half(x) * sin


# ============================================================
# Attention
# ============================================================


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.size(-1)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    weights = F.softmax(
        scores,
        dim=-1,
    )

    return weights @ v


class AttentionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        max_seq_len: int,
        use_rope: bool,
        norm_eps: float,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()

        self.use_rope = use_rope

        self.q_proj = nn.Linear(
            embed_dim,
            head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            embed_dim,
            head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            embed_dim,
            head_dim,
            bias=False,
        )

        # ------------------------------------
        # QK-Norm
        # ------------------------------------

        self.q_norm = RMSNorm(
            head_dim,
            eps=norm_eps,
        )

        self.k_norm = RMSNorm(
            head_dim,
            eps=norm_eps,
        )

        # ------------------------------------
        # Optional RoPE
        # ------------------------------------

        if use_rope:
            self.rope = RotaryEmbedding(
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                theta=rope_theta,
            )

        # Attention-output gate.
        self.gate = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # QK-Norm BEFORE RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            q = self.rope(
                q,
                position_ids,
            )

            k = self.rope(
                k,
                position_ids,
            )

        attn_out = scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
        )

        gate = torch.sigmoid(self.gate(x))

        return gate * attn_out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        use_rope: bool,
    ):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        head_dim = config.d_model // config.n_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embed_dim=config.d_model,
                    head_dim=head_dim,
                    max_seq_len=config.max_pos_emb,
                    use_rope=use_rope,
                    norm_eps=config.layer_norm_eps,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.n_heads)
            ]
        )

        self.output_proj = nn.Linear(
            config.d_model,
            config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                head(
                    x,
                    mask,
                    position_ids,
                )
                for head in self.heads
            ],
            dim=-1,
        )

        x = self.output_proj(x)

        return self.dropout(x)


# ============================================================
# SwiGLU
# ============================================================


class SwiGLU(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
    ):
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(
            2,
            dim=-1,
        )

        x = F.silu(gate) * up

        x = self.down_proj(x)

        return self.dropout(x)


# ============================================================
# Transformer layer
# ============================================================


class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.use_rope = layer_idx in ROPE_LAYERS

        self.attn_norm = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.ffn_norm = RMSNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.attention = MultiHeadAttention(
            config,
            use_rope=self.use_rope,
        )

        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attn_norm(x),
            mask,
            position_ids,
        )

        x = x + self.feed_forward(self.ffn_norm(x))

        return x


# ============================================================
# Token-choice Mixture-of-Recursions
# ============================================================


class TokenChoiceRouter(nn.Module):
    """
    Each token chooses ONE recursion depth upfront.

    Output index:
        0 -> stop after recursion 1
        1 -> stop after recursion 2
        ...
        R-1 -> stop after recursion R
    """

    def __init__(
        self,
        d_model: int,
        num_recursions: int,
    ):
        super().__init__()

        self.num_recursions = num_recursions

        self.router = nn.Linear(
            d_model,
            num_recursions,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        logits = self.router(x / ROUTER_TEMPERATURE)

        # Router math in FP32.
        probs = F.softmax(
            logits.float(),
            dim=-1,
        )

        # Hard top-1 routing.
        depth_index = probs.argmax(dim=-1)

        # 1 .. num_recursions
        assigned_depth = depth_index + 1

        selected_prob = probs.gather(
            dim=-1,
            index=depth_index.unsqueeze(-1),
        ).squeeze(-1)

        # ------------------------------------
        # Switch-style balancing loss
        # ------------------------------------

        mean_prob = probs.mean(dim=(0, 1))

        hard_fraction = torch.stack(
            [(depth_index == i).float().mean() for i in range(self.num_recursions)]
        )

        balance_loss = self.num_recursions * (mean_prob * hard_fraction).sum()

        # ------------------------------------
        # Router z-loss
        # ------------------------------------

        z_loss = (
            torch.logsumexp(
                logits.float(),
                dim=-1,
            )
            .square()
            .mean()
        )

        return (
            assigned_depth,
            selected_prob,
            balance_loss,
            z_loss,
        )


class TransformerLM(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
    ):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # These are the PHYSICAL layers.
        # The same weights are reused every recursion.
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config,
                    layer_idx=i,
                )
                for i in range(config.n_layers)
            ]
        )

        self.router = TokenChoiceRouter(
            d_model=config.d_model,
            num_recursions=config.n_loops,
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

    def run_shared_stack(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs ONE recursion through all physical layers.

        x contains only the tokens still active in this recursion.
        """

        seq_len = x.size(1)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        for layer in self.layers:
            x = layer(
                x,
                mask,
                position_ids,
            )

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
    ):
        # [B, T, D]
        base = self.token_embedding(input_ids)

        batch_size, seq_len, _ = base.shape

        # ------------------------------------
        # Route every token ONCE, upfront.
        # ------------------------------------

        (
            assigned_depth,
            selected_prob,
            balance_loss,
            z_loss,
        ) = self.router(base)

        # Current recurrent states.
        x = base

        # Each token will be written here once
        # when it reaches its selected depth.
        #
        # Starting from base also reproduces the
        # residual-style weighted routing used
        # by MoR.
        final_x = base.clone()

        # ------------------------------------
        # Recursive computation
        # ------------------------------------

        for recursion in range(
            1,
            self.config.n_loops + 1,
        ):
            # Tokens whose chosen depth is >=
            # this recursion are still active.
            active = assigned_depth >= recursion

            x_next = x.clone()

            # We deliberately keep this version
            # simple rather than building padded
            # variable-length batches.
            #
            # Each sequence gathers its active
            # tokens independently.
            for b in range(batch_size):
                active_indices = torch.where(active[b])[0]

                if active_indices.numel() == 0:
                    continue

                # [1, T_active, D]
                active_x = x[
                    b : b + 1,
                    active_indices,
                    :,
                ]

                # IMPORTANT:
                # preserve ORIGINAL positions.
                #
                # If positions 1, 5 and 9 survive,
                # RoPE sees 1, 5 and 9,
                # not 0, 1 and 2.
                position_ids = active_indices.unsqueeze(0)

                active_x = self.run_shared_stack(
                    active_x,
                    position_ids,
                )

                # Tokens continuing into a deeper
                # recursion receive the new state.
                x_next[
                    b,
                    active_indices,
                ] = active_x[0]

                # Which active tokens finish here?
                finishes_here = (
                    assigned_depth[
                        b,
                        active_indices,
                    ]
                    == recursion
                )

                finished_local = torch.where(finishes_here)[0]

                if finished_local.numel() == 0:
                    continue

                finished_global = active_indices[finished_local]

                finished_x = active_x[
                    0,
                    finished_local,
                ]

                # Router probability associated
                # with the selected depth.
                routing_weight = (
                    selected_prob[
                        b,
                        finished_global,
                    ]
                    .unsqueeze(-1)
                    .to(finished_x.dtype)
                )

                # Weighted residual routing.
                #
                # This is important because the
                # task loss then has a gradient
                # path into the router probability.
                final_x[
                    b,
                    finished_global,
                ] = (
                    base[
                        b,
                        finished_global,
                    ]
                    + routing_weight * finished_x
                )

            x = x_next

        x = self.final_norm(final_x)

        logits = self.lm_head(x)

        # ------------------------------------
        # Routing metrics
        # ------------------------------------

        depth_counts = torch.stack(
            [
                (assigned_depth == depth).sum()
                for depth in range(
                    1,
                    self.config.n_loops + 1,
                )
            ]
        )

        mean_depth = assigned_depth.float().mean()

        router_info = {
            "balance_loss": balance_loss,
            "z_loss": z_loss,
            "depth_counts": depth_counts,
            "mean_depth": mean_depth,
        }

        return (
            logits,
            router_info,
        )


# ============================================================
# Initialization
# ============================================================


def init_weights(
    module: nn.Module,
):
    if isinstance(
        module,
        nn.Linear,
    ):
        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(
        module,
        nn.Embedding,
    ):
        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=0.02,
        )

    elif isinstance(
        module,
        RMSNorm,
    ):
        nn.init.ones_(module.weight)


def initialize_model(
    model: TransformerLM,
    config: GPT2Config,
):
    model.apply(init_weights)

    # Scale according to maximum effective
    # residual depth.
    effective_depth = config.n_layers * config.n_loops

    residual_std = 0.02 / math.sqrt(2 * effective_depth)

    for module in model.modules():

        if isinstance(
            module,
            MultiHeadAttention,
        ):
            nn.init.normal_(
                module.output_proj.weight,
                mean=0.0,
                std=residual_std,
            )

        elif isinstance(
            module,
            SwiGLU,
        ):
            nn.init.normal_(
                module.down_proj.weight,
                mean=0.0,
                std=residual_std,
            )

        elif isinstance(
            module,
            AttentionHead,
        ):
            # Initially mostly-open attention gate.
            nn.init.zeros_(module.gate.weight)

            nn.init.constant_(
                module.gate.bias,
                2.0,
            )

    # GPT-style weight tying.
    model.lm_head.weight = model.token_embedding.weight


# ============================================================
# Dataset
# ============================================================


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        seq_len: int,
        tokenization_batch_size: int = 256,
    ):
        self.seq_len = seq_len

        tokens = []

        for start in range(
            0,
            len(texts),
            tokenization_batch_size,
        ):
            batch = texts[start : start + tokenization_batch_size]

            encoded = tokenizer(
                batch,
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
            f"Dataset: "
            f"{len(texts):,} stories, "
            f"{len(self.tokens):,} tokens, "
            f"{len(self):,} sequences"
        )

    def __len__(
        self,
    ):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(
        self,
        idx,
    ):
        start = idx * self.seq_len

        chunk = self.tokens[start : start + self.seq_len + 1]

        return (
            chunk[:-1],
            chunk[1:],
        )


# ============================================================
# Router metrics
# ============================================================


def format_depth_distribution(
    depth_counts: torch.Tensor,
) -> str:
    counts = depth_counts.detach().cpu().float()

    total = counts.sum()

    if total.item() == 0:
        return "[]"

    ratios = counts / total

    return "[" + ", ".join(f"{i + 1}:{ratio.item():.2f}" for i, ratio in enumerate(ratios)) + "]"


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
        model_input = input_ids[
            :,
            -model.config.max_pos_emb :,
        ]

        (
            logits,
            _,
        ) = model(model_input)

        logits = logits[:, -1, :] / 0.8

        values, indices = torch.topk(
            logits,
            k=50,
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
            (
                input_ids,
                next_token,
            ),
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
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    total_lm_loss = 0.0

    depth_counts = torch.zeros(
        model.config.n_loops,
        dtype=torch.long,
    )

    total_mean_depth = 0.0

    for (
        input_ids,
        targets,
    ) in loader:
        input_ids = input_ids.to(device)

        targets = targets.to(device)

        (
            logits,
            router_info,
        ) = model(input_ids)

        lm_loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            targets.reshape(-1),
        )

        total_lm_loss += lm_loss.item()

        depth_counts += router_info["depth_counts"].detach().cpu()

        total_mean_depth += router_info["mean_depth"].item()

    return {
        "lm_loss": (total_lm_loss / len(loader)),
        "mean_depth": (total_mean_depth / len(loader)),
        "depth_counts": depth_counts,
    }


# ============================================================
# Optimizer
# ============================================================


def create_optimizer(
    model: nn.Module,
):
    decay = []
    no_decay = []

    for (
        _,
        param,
    ) in model.named_parameters():

        if not param.requires_grad:
            continue

        # Matrices get weight decay.
        #
        # Biases and all RMSNorm scales,
        # including QK-Norm,
        # do not.
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=LEARNING_RATE,
    )


# ============================================================
# LR scheduler
# ============================================================


def create_scheduler(
    optimizer,
    total_steps: int,
):
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        decay_steps = max(
            1,
            total_steps - warmup_steps,
        )

        progress = (step - warmup_steps) / decay_steps

        progress = min(
            max(
                progress,
                0.0,
            ),
            1.0,
        )

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ============================================================
# Checkpointing
# ============================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    epoch_lm_loss_sum: float,
    epoch_total_loss_sum: float,
    epoch_batches_completed: int,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "epoch_lm_loss_sum": (epoch_lm_loss_sum),
        "epoch_total_loss_sum": (epoch_total_loss_sum),
        "epoch_batches_completed": (epoch_batches_completed),
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "n_layers": N_LAYERS,
        "max_recursions": MAX_RECURSIONS,
        "rope_layers": sorted(ROPE_LAYERS),
    }

    temp_path = Path(str(path) + ".tmp")

    torch.save(
        state,
        temp_path,
    )

    temp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
):
    path = Path(path)

    print(f"Loading checkpoint: " f"{path}")

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if checkpoint["seq_len"] != SEQ_LEN:
        raise ValueError("SEQ_LEN differs from checkpoint")

    if checkpoint["batch_size"] != BATCH_SIZE:
        raise ValueError("BATCH_SIZE differs from checkpoint")

    if checkpoint["n_layers"] != N_LAYERS:
        raise ValueError("N_LAYERS differs from checkpoint")

    if checkpoint["max_recursions"] != MAX_RECURSIONS:
        raise ValueError("MAX_RECURSIONS differs from checkpoint")

    if checkpoint["rope_layers"] != sorted(ROPE_LAYERS):
        raise ValueError("ROPE_LAYERS differs from checkpoint")

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint["epoch"]

    next_step = checkpoint["step_in_epoch"] + 1

    global_step = checkpoint["global_step"]

    epoch_lm_loss_sum = checkpoint.get(
        "epoch_lm_loss_sum",
        0.0,
    )

    epoch_total_loss_sum = checkpoint.get(
        "epoch_total_loss_sum",
        0.0,
    )

    epoch_batches_completed = checkpoint.get(
        "epoch_batches_completed",
        0,
    )

    print(f"Resuming from " f"epoch={epoch}, " f"batch={next_step}, " f"global_step={global_step}")

    return (
        epoch,
        next_step,
        global_step,
        epoch_lm_loss_sum,
        epoch_total_loss_sum,
        epoch_batches_completed,
    )


# ============================================================
# Main
# ============================================================


def main():
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        max_pos_emb=SEQ_LEN,
        d_model=256,
        n_heads=4,
        # Physical shared layers.
        n_layers=N_LAYERS,
        # Maximum recursion depth.
        n_loops=MAX_RECURSIONS,
        d_ff=1024,
        dropout_prob=0.0,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("Loading TinyStories...")

    train_texts = load_dataset(
        "roneneldan/TinyStories",
        split=(f"train[:" f"{TRAIN_STORIES}]"),
    )["text"]

    val_texts = load_dataset(
        "roneneldan/TinyStories",
        split=(f"validation[:" f"{VAL_STORIES}]"),
    )["text"]

    print("Tokenizing training set...")

    train_dataset = LanguageModelDataset(
        train_texts,
        tokenizer,
        SEQ_LEN,
    )

    print("Tokenizing validation set...")

    val_dataset = LanguageModelDataset(
        val_texts,
        tokenizer,
        SEQ_LEN,
    )

    train_generator = torch.Generator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
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

    initialize_model(
        model,
        config,
    )

    model = model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"parameters: " f"{num_parameters / 1e6:.2f}M")

    print(f"maximum recursions: " f"{MAX_RECURSIONS}")

    for i, layer in enumerate(model.layers):
        mode = "RoPE" if layer.use_rope else "NoPE"

        print(f"physical layer {i}: " f"{mode}")

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    optimizer = create_optimizer(model)

    total_steps = NUM_EPOCHS * len(train_loader)

    scheduler = create_scheduler(
        optimizer,
        total_steps,
    )

    # --------------------------------------------------------
    # Resume state
    # --------------------------------------------------------

    start_epoch = 0
    start_step = 0
    global_step = 0

    resumed_lm_loss_sum = 0.0
    resumed_total_loss_sum = 0.0
    resumed_batches = 0

    if RESUME_FROM is not None:
        (
            start_epoch,
            start_step,
            global_step,
            resumed_lm_loss_sum,
            resumed_total_loss_sum,
            resumed_batches,
        ) = load_checkpoint(
            RESUME_FROM,
            model,
            optimizer,
            scheduler,
            device,
        )

        # Checkpoint was at end of epoch.
        if start_step >= len(train_loader):
            start_epoch += 1
            start_step = 0

            resumed_lm_loss_sum = 0.0
            resumed_total_loss_sum = 0.0
            resumed_batches = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):
        model.train()

        # Reconstruct the exact shuffle for
        # this epoch when resuming.
        train_generator.manual_seed(SEED + epoch)

        if epoch == start_epoch and start_step > 0:
            epoch_lm_loss_sum = resumed_lm_loss_sum

            epoch_total_loss_sum = resumed_total_loss_sum

            epoch_batches_completed = resumed_batches

        else:
            epoch_lm_loss_sum = 0.0
            epoch_total_loss_sum = 0.0
            epoch_batches_completed = 0

        for (
            step,
            (
                input_ids,
                targets,
            ),
        ) in enumerate(train_loader):
            # --------------------------------------------
            # Resume inside an epoch
            # --------------------------------------------

            if epoch == start_epoch and step < start_step:
                continue

            input_ids = input_ids.to(device)

            targets = targets.to(device)

            # --------------------------------------------
            # Forward
            # --------------------------------------------

            (
                logits,
                router_info,
            ) = model(input_ids)

            lm_loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

            balance_loss = router_info["balance_loss"]

            z_loss = router_info["z_loss"]

            loss = lm_loss + ROUTER_BALANCE_COEF * balance_loss + ROUTER_Z_LOSS_COEF * z_loss

            # --------------------------------------------
            # Backward
            # --------------------------------------------

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            epoch_lm_loss_sum += lm_loss.item()

            epoch_total_loss_sum += loss.item()

            epoch_batches_completed += 1

            # --------------------------------------------
            # Logging
            #
            # Intentionally logs global_step=0.
            # --------------------------------------------

            if global_step % 50 == 0:
                depth_dist = format_depth_distribution(router_info["depth_counts"])

                print(
                    f"epoch={epoch} "
                    f"step={step} "
                    f"global_step={global_step} "
                    f"lr="
                    f"{scheduler.get_last_lr()[0]:.2e} "
                    f"lm_loss={lm_loss.item():.4f} "
                    f"loss={loss.item():.4f} "
                    f"grad_norm="
                    f"{float(grad_norm):.3f} "
                    f"mean_depth="
                    f"{router_info['mean_depth'].item():.2f} "
                    f"depths={depth_dist}"
                )

            # This now means:
            #
            # "number of completed optimizer steps".
            global_step += 1

            # --------------------------------------------
            # Generation
            # --------------------------------------------

            if global_step % GENERATE_EVERY == 0:
                generated = generate(
                    model,
                    tokenizer,
                    prompt=GENERATION_PROMPT,
                    device=device,
                    max_new_tokens=50,
                )

                print()

                print(f"--- generation " f"@ step " f"{global_step} ---")

                print(generated)

                print("---------------------------")

                print()

            # --------------------------------------------
            # Periodic checkpoint
            # --------------------------------------------

            if global_step % CHECKPOINT_EVERY == 0:
                checkpoint_path = CHECKPOINT_DIR / (f"step_" f"{global_step:08d}" f".pt")

                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step_in_epoch=step,
                    global_step=global_step,
                    epoch_lm_loss_sum=(epoch_lm_loss_sum),
                    epoch_total_loss_sum=(epoch_total_loss_sum),
                    epoch_batches_completed=(epoch_batches_completed),
                )

                save_checkpoint(
                    CHECKPOINT_DIR / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step_in_epoch=step,
                    global_step=global_step,
                    epoch_lm_loss_sum=(epoch_lm_loss_sum),
                    epoch_total_loss_sum=(epoch_total_loss_sum),
                    epoch_batches_completed=(epoch_batches_completed),
                )

                print(f"checkpoint saved: " f"{checkpoint_path}")

        # Following epochs start normally.
        start_step = 0

        # ----------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------

        train_lm_loss = epoch_lm_loss_sum / epoch_batches_completed

        train_total_loss = epoch_total_loss_sum / epoch_batches_completed

        val = evaluate(
            model,
            val_loader,
            device,
        )

        print()

        print(f"epoch={epoch} complete")

        print(f"train_lm_loss=" f"{train_lm_loss:.4f}")

        print(f"train_total_loss=" f"{train_total_loss:.4f}")

        print(f"val_loss=" f"{val['lm_loss']:.4f}")

        print(f"val_mean_depth=" f"{val['mean_depth']:.2f}")

        print("val_depths=" + format_depth_distribution(val["depth_counts"]))

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # End-of-epoch checkpoint
        # ----------------------------------------------------

        epoch_checkpoint = CHECKPOINT_DIR / (f"epoch_" f"{epoch:03d}" f".pt")

        save_checkpoint(
            epoch_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            step_in_epoch=(len(train_loader) - 1),
            global_step=global_step,
            epoch_lm_loss_sum=(epoch_lm_loss_sum),
            epoch_total_loss_sum=(epoch_total_loss_sum),
            epoch_batches_completed=(epoch_batches_completed),
        )

        save_checkpoint(
            CHECKPOINT_DIR / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            step_in_epoch=(len(train_loader) - 1),
            global_step=global_step,
            epoch_lm_loss_sum=(epoch_lm_loss_sum),
            epoch_total_loss_sum=(epoch_total_loss_sum),
            epoch_batches_completed=(epoch_batches_completed),
        )


if __name__ == "__main__":
    main()
