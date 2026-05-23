# -*- coding: utf-8 -*-
"""
Semantic-Qubit: Shared utilities
"""
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os as _os

# Use local snapshot path (local_files_only=True does not work with model name string)
_HF_CACHE = _os.path.expanduser("~/.cache/huggingface/hub")
_SNAP_1B5 = _os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                           "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
_SNAP_0B5 = _os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                           "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")

# Use 1.5B if available, fallback to 0.5B
MODEL_ID = _SNAP_1B5 if _os.path.exists(_SNAP_1B5) else _SNAP_0B5


def load_model(device=None, dtype=None):
    """Load Qwen2.5-1.5B with local_files_only=True."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dtype is None:
        dtype = torch.float16 if device == 'cuda' else torch.float32

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    return model, tok


def inject_hook(model, layer_idx, hook_fn):
    """Register a forward hook on model.model.layers[layer_idx]. Returns handle."""
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    return handle


def replace_last_token_hook(vec):
    """Returns a hook that replaces the hidden state of the last token."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            if h.dim() == 3:
                h[0, -1, :] = vec.to(h.dtype)
            else:
                h[-1, :] = vec.to(h.dtype)
            return (h,) + output[1:]
        else:
            h = output.clone()
            if h.dim() == 3:
                h[0, -1, :] = vec.to(h.dtype)
            else:
                h[-1, :] = vec.to(h.dtype)
            return h
    return hook


def get_logits(model, tok, prompt, device):
    """Single forward pass, return final logits (vocab,)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :]


def top_probs(logits, k=5):
    """Return top-k (token_str, prob) list."""
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    return topk.values.cpu().tolist(), topk.indices.cpu().tolist()
