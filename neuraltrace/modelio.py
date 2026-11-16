"""Model loading, batched generation, activation extraction and steering hooks.

Layer indexing: layer l (0-based) means the residual stream *after* decoder
block l, i.e. hidden_states[l + 1] in HuggingFace output, and the output of
model.model.layers[l]. Steering at layer l edits exactly what is probed at l.
"""
from __future__ import annotations

import numpy as np


def load(name: str, dtype: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype), device_map="auto")
    model.eval()
    return tok, model


def render(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def entity_pos_from_end(tok, text: str, entity: str) -> int:
    """Distance (in tokens) from the end of the prompt to the last token of the
    entity name. Robust to left padding in batches."""
    start = text.rfind(entity)
    if start < 0:
        raise ValueError(f"entity {entity!r} not found in prompt")
    end = start + len(entity)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    idx = max(i for i, (a, b) in enumerate(offs) if a < end)
    return len(offs) - 1 - idx


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _order_by_length(texts):
    order = np.argsort([len(t) for t in texts], kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return order, inverse


def generate(model, tok, texts, max_new_tokens: int = 24, batch_size: int = 32, steer=None):
    """Greedy generation. steer is an optional context manager (see Steer)."""
    import contextlib

    import torch

    order, inverse = _order_by_length(texts)
    sorted_texts = [texts[i] for i in order]
    outs: list[str] = []
    ctx = steer if steer is not None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        for i in range(0, len(sorted_texts), batch_size):
            enc = tok(sorted_texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
            )
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return [outs[j] for j in inverse]


def extract_activations(model, tok, texts, pos_from_end, batch_size: int = 16):
    """Residual stream at (a) the last entity token and (b) the final prompt token.

    Returns two float16 arrays of shape (n_layers, n_examples, d_model).
    """
    import torch

    order, inverse = _order_by_length(texts)
    st = [texts[i] for i in order]
    sp = [pos_from_end[i] for i in order]
    ent_chunks, last_chunks = [], []
    with torch.no_grad():
        for i in range(0, len(st), batch_size):
            enc = tok(st[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(model.device)
            out = model(**enc, output_hidden_states=True)
            B, T = enc["input_ids"].shape
            idx = T - 1 - torch.tensor(sp[i : i + batch_size])
            ent_l, last_l = [], []
            for h in out.hidden_states[1:]:
                rows = torch.arange(B, device=h.device)
                ent_l.append(h[rows, idx.to(h.device)].to(torch.float16).cpu())
                last_l.append(h[:, -1].to(torch.float16).cpu())
            ent_chunks.append(torch.stack(ent_l, 0))
            last_chunks.append(torch.stack(last_l, 0))
    ent = torch.cat(ent_chunks, 1).numpy()[:, inverse]
    last = torch.cat(last_chunks, 1).numpy()[:, inverse]
    if not np.isfinite(ent.astype(np.float32)).all():
        raise FloatingPointError("non-finite activations (fp16 overflow?)")
    return ent, last


class Steer:
    """Context manager that edits the residual stream after one decoder block.

    mode='add':    h <- h + scale * d       (d is a unit vector)
    mode='ablate': h <- h - (h . d) d       (project the direction out)
    Applied at every token position, including generated tokens.
    """

    def __init__(self, model, layer: int, direction, mode: str = "add", scale: float = 1.0):
        import torch

        self.module = model.model.layers[layer]
        self.d = torch.as_tensor(np.asarray(direction, dtype=np.float32))
        self.mode, self.scale = mode, float(scale)
        self.handle = None

    def _hook(self, module, args, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        d = self.d.to(h.device)
        hf = h.float()
        if self.mode == "add":
            hf = hf + self.scale * d
        elif self.mode == "ablate":
            hf = hf - (hf * d).sum(-1, keepdim=True) * d
        else:
            raise ValueError(self.mode)
        h2 = hf.to(h.dtype)
        return (h2,) + tuple(out[1:]) if is_tuple else h2

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False
