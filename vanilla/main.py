# from transformers import AutoTokenizer

# model_id = "LiquidAI/LFM2.5-350M"

# tokenizer = AutoTokenizer.from_pretrained(model_id)

from config import GPT2Config
from torch import nn
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

config = GPT2Config(
    vocab_size=50_257, max_pos_emb=128, d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout_prob=0
)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = k.size(-1)
    scores = torch.bmm(q, k.transpose(-1, -2)) / (d_k**0.5)
    # print("scores: ", scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # print("scores with mask: ", scores)
    weights = F.softmax(scores, dim=-1)
    # print("weights: ", weights)
    attn_out = torch.bmm(weights, v)
    # print("attn_output: ", attn_out)
    return attn_out


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, q, kv=None, mask=None):
        # self-attention if no kv_states provided
        if kv is None:
            kv = q
        return scaled_dot_product_attention(self.q(q), self.k(kv), self.v(kv), mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        embed_dim = config.d_model
        num_heads = config.n_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = torch.cat([h(q, kv, mask) for h in self.heads], dim=-1)
        return self.output_linear(x)


class FeedForward(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_ff)
        self.linear_2 = nn.Linear(config.d_ff, config.d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x), mask=mask)
        return x + self.feed_forward(self.layer_norm2(x))


class Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_pos_emb, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(pos_ids)
        return self.dropout(self.layer_norm(tok_emb + pos_emb))


class TransformerEncoder(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        self_attention_mask: torch.Tensor,
        encoder_output: torch.Tensor = None,
        cross_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = x + self.self_attention(q=self.layer_norm1(x), mask=self_attention_mask)

        if encoder_output is not None:
            x = x + self.cross_attention(
                q=self.layer_norm2(x), kv=encoder_output, mask=cross_attention_mask
            )
            x = x + self.feed_forward(self.layer_norm3(x))

        else:
            x = x + self.feed_forward(self.layer_norm2(x))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.embeddings = Embeddings(config)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        self_attention_mask: torch.Tensor,
        encoder_output: torch.Tensor = None,
        cross_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(
                x,
                self_attention_mask=self_attention_mask,
                encoder_output=encoder_output,
                cross_attention_mask=cross_attention_mask,
            )

        return x


class TransformerLM(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.decoder = TransformerDecoder(config)
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # GPT 2 style weight tying
        self.lm_head.weight = self.decoder.embeddings.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, self_attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.decoder(input_ids, self_attention_mask=self_attention_mask)
        x = self.final_norm(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 15,
):
    was_training = model.training
    model.eval()

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)

    for _ in range(max_new_tokens):
        seq_len = input_ids.size(1)

        mask = get_causal_mask(
            seq_len,
            input_ids.device,
        )

        logits = model(
            input_ids,
            self_attention_mask=mask,
        )

        # greedy decoding
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    if was_training:
        model.train()

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )


def get_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 15,
):
    was_training = model.training
    model.eval()

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)

    for _ in range(max_new_tokens):
        seq_len = input_ids.size(1)

        mask = get_causal_mask(
            seq_len,
            input_ids.device,
        )

        logits = model(
            input_ids,
            self_attention_mask=mask,
        )

        # greedy decoding
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    if was_training:
        model.train()

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )


model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

texts = [
    "Hello.",
    "This sentence is somewhat longer.",
    "Transformers are cool.",
    "Introduce yourself and tell us a little more about yourself",
]

assert (
    tokenizer.vocab_size == config.vocab_size
), f"vocab size mismatch: tokenizer {tokenizer.vocab_size}, but config {config.vocab_size}"

batch = tokenizer(texts, padding=True, return_tensors="pt")

input_ids, attn_mask = batch.input_ids, batch.attention_mask  # both [B, N]


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = TransformerLM(config).to(device)

input_ids = batch.input_ids.to(device)
padding_mask = batch.attention_mask.bool().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-3,
)

model.train()

for step in range(1000):
    seq_len = input_ids.size(1)

    # causal mask: [T, T]
    causal_mask = get_causal_mask(
        seq_len,
        device,
    )

    # padding mask: [B, 1, T]
    # combined mask: [B, T, T]
    self_attention_mask = causal_mask & padding_mask[:, None, :]

    # [B, T, V]
    logits = model(
        input_ids,
        self_attention_mask=self_attention_mask,
    )

    # predict token t+1 from position t
    pred = logits[:, :-1, :]  # [B, T-1, V]
    targets = input_ids[:, 1:].clone()  # [B, T-1]

    # don't compute loss on padding
    target_mask = padding_mask[:, 1:]
    targets[~target_mask] = -100

    loss = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"{step}: {loss.item():.4f}")

    if step % 25 == 0:
        text = generate(
            model,
            tokenizer,
            prompt="Tell us",
            device=device,
        )

        print(f"\nstep {step} | loss {loss.item():.4f}")
        print(f"> {text}")
