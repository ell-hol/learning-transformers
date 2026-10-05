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

# Set this to a checkpoint path to resume.
#
# Example:
#
# RESUME_FROM = "checkpoints/latest.pt"
#
RESUME_FROM = None

GENERATION_PROMPT = "There once was"

SEED = 42

# Hybrid NoPE/RoPE:
#
# physical layer 0 -> RoPE
# physical layer 1 -> NoPE
# physical layer 2 -> RoPE
# physical layer 3 -> NoPE
#
ROPE_LAYERS = {0, 2}


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
    ) -> torch.Tensor:
        seq_len = x.size(1)

        cos = self.cos[:seq_len].unsqueeze(0)

        sin = self.sin[:seq_len].unsqueeze(0)

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

        # QK-Norm.
        self.q_norm = RMSNorm(
            head_dim,
            eps=norm_eps,
        )

        self.k_norm = RMSNorm(
            head_dim,
            eps=norm_eps,
        )

        if use_rope:
            self.rope = RotaryEmbedding(
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                theta=rope_theta,
            )

        # Scalar attention gate per token/head.
        self.gate = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ------------------------------------
        # QK-Norm
        # ------------------------------------

        q = self.q_norm(q)
        k = self.k_norm(k)

        # ------------------------------------
        # Hybrid RoPE / NoPE
        # ------------------------------------

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Otherwise this physical layer is NoPE.

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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                head(
                    x,
                    mask,
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
# Transformer
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
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attn_norm(x),
            mask,
        )

        x = x + self.feed_forward(self.ffn_norm(x))

        return x


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

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config,
                    layer_idx=i,
                )
                for i in range(config.n_layers)
            ]
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
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embedding(input_ids)

        # Same physical layers are reused
        # across multiple logical passes.
        for _ in range(self.config.n_loops):
            for layer in self.layers:
                x = layer(
                    x,
                    attention_mask,
                )

        x = self.final_norm(x)

        return self.lm_head(x)


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

    effective_depth = config.n_layers * config.n_loops

    residual_std = 0.02 / math.sqrt(2 * effective_depth)

    for module in model.modules():

        # Residual projection scaling.
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
            # Start attention gates mostly open.
            nn.init.zeros_(module.gate.weight)

            nn.init.constant_(
                module.gate.bias,
                2.0,
            )

    # Weight tying.
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

    def __len__(self):
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
        model_input = input_ids[
            :,
            -model.config.max_pos_emb :,
        ]

        mask = get_causal_mask(
            model_input.size(1),
            device,
        )

        logits = model(
            model_input,
            attention_mask=mask,
        )

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
    causal_mask,
    device,
):
    model.eval()

    total_loss = 0.0

    for (
        input_ids,
        targets,
    ) in loader:
        input_ids = input_ids.to(device)

        targets = targets.to(device)

        logits = model(
            input_ids,
            attention_mask=causal_mask,
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
# Optimizer
# ============================================================


def create_optimizer(
    model: nn.Module,
):
    decay = []
    no_decay = []

    for (
        name,
        param,
    ) in model.named_parameters():
        if not param.requires_grad:
            continue

        # Matrices -> weight decay.
        #
        # 1D parameters:
        #   RMSNorm scales
        #   biases
        #
        # -> no weight decay.
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
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        # Epoch currently being processed.
        "epoch": epoch,
        # Last COMPLETED batch in this epoch.
        "step_in_epoch": step_in_epoch,
        # Number of completed optimizer steps.
        "global_step": global_step,
        # Useful for detecting accidental config changes.
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
    }

    # Write temporary file first so an interrupted write
    # does not destroy the previous checkpoint.
    temp_path = path.with_suffix(".tmp")

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

    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if checkpoint["seq_len"] != SEQ_LEN:
        raise ValueError("SEQ_LEN differs from checkpoint")

    if checkpoint["batch_size"] != BATCH_SIZE:
        raise ValueError("BATCH_SIZE differs from checkpoint")

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint["epoch"]

    next_step_in_epoch = checkpoint["step_in_epoch"] + 1

    global_step = checkpoint["global_step"]

    print(
        f"Resuming from "
        f"epoch={epoch + 1}, "
        f"batch={next_step_in_epoch}, "
        f"global_step={global_step}"
    )

    return (
        epoch,
        next_step_in_epoch,
        global_step,
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
        n_layers=4,
        n_loops=2,
        d_ff=704,
        dropout_prob=0.0,
    )

    assert tokenizer.vocab_size == config.vocab_size

    assert (
        max(
            ROPE_LAYERS,
            default=-1,
        )
        < config.n_layers
    )

    # --------------------------------------------------------
    # Dataset
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

    # We explicitly control the generator seed
    # per epoch so that mid-epoch resume can
    # reconstruct the exact same shuffle.
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

    for i, layer in enumerate(model.layers):
        positional_mode = "RoPE" if layer.use_rope else "NoPE"

        print(f"layer {i}: " f"{positional_mode}")

    # --------------------------------------------------------
    # Optimizer + scheduler
    # --------------------------------------------------------

    optimizer = create_optimizer(model)

    total_steps = NUM_EPOCHS * len(train_loader)

    scheduler = create_scheduler(
        optimizer,
        total_steps,
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    causal_mask = get_causal_mask(
        SEQ_LEN,
        device,
    )

    # --------------------------------------------------------
    # Resume state
    # --------------------------------------------------------

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0

    if RESUME_FROM is not None:
        (
            start_epoch,
            start_step_in_epoch,
            global_step,
        ) = load_checkpoint(
            RESUME_FROM,
            model,
            optimizer,
            scheduler,
            device,
        )

        # If the checkpoint was taken after the
        # final batch of an epoch, continue from
        # the following epoch.
        if start_step_in_epoch >= len(train_loader):
            start_epoch += 1
            start_step_in_epoch = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):
        model.train()

        # Deterministic shuffle for this epoch.
        #
        # This lets us regenerate exactly the
        # same order after a restart.
        train_generator.manual_seed(SEED + epoch)

        total_train_loss = 0.0
        completed_batches = 0

        for step, (
            input_ids,
            targets,
        ) in enumerate(train_loader):
            # --------------------------------------------
            # Mid-epoch resume
            # --------------------------------------------

            if epoch == start_epoch and step < start_step_in_epoch:
                continue

            input_ids = input_ids.to(device)

            targets = targets.to(device)

            # --------------------------------------------
            # Forward
            # --------------------------------------------

            logits = model(
                input_ids,
                attention_mask=causal_mask,
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

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

            total_train_loss += loss.item()

            completed_batches += 1

            # --------------------------------------------
            # Logging
            # --------------------------------------------

            if global_step % 50 == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={step} "
                    f"global_step={global_step} "
                    f"lr="
                    f"{scheduler.get_last_lr()[0]:.2e} "
                    f"loss={loss.item():.4f} "
                    f"grad_norm="
                    f"{float(grad_norm):.3f}"
                )

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
                )

                save_checkpoint(
                    CHECKPOINT_DIR / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step_in_epoch=step,
                    global_step=global_step,
                )

                print(f"checkpoint saved: " f"{checkpoint_path}")

        # Once the resumed epoch has finished,
        # subsequent epochs begin normally.
        start_step_in_epoch = 0

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if completed_batches > 0:
            train_loss = total_train_loss / completed_batches
        else:
            train_loss = float("nan")

        val_loss = evaluate(
            model,
            val_loader,
            causal_mask,
            device,
        )

        print()
        print(f"epoch={epoch + 1} complete")
        print(f"train_loss=" f"{train_loss:.4f}")
        print(f"val_loss=" f"{val_loss:.4f}")

        # ----------------------------------------------------
        # End-of-epoch generation
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

        epoch_checkpoint = CHECKPOINT_DIR / (f"epoch_" f"{epoch + 1:03d}" f".pt")

        save_checkpoint(
            epoch_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            step_in_epoch=(len(train_loader) - 1),
            global_step=global_step,
        )

        save_checkpoint(
            CHECKPOINT_DIR / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            step_in_epoch=(len(train_loader) - 1),
            global_step=global_step,
        )


if __name__ == "__main__":
    main()
