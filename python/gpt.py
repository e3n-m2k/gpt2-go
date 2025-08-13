import numpy as np
import random


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(
        x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask)
                 for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, lm_head, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    if lm_head:
        return layer_norm(x, **ln_f) @ lm_head.T
    return layer_norm(x, **ln_f) @ wte.T


def generate(inputs, params, n_head, n_tokens_to_generate, topk=5):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = random.choice(np.argsort(logits[-1])[-topk:])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]


import torch
import json
from functools import lru_cache
import regex as re

def load_encoder_hparams_and_params(model_path, tokenizer_path, device="cpu"):
    n_layers = 12
    prefifx = ""
    model = torch.load(model_path, map_location=device)
    # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 768]
    wte = model[f"{prefifx}wte.weight"].numpy()
    # context embedding. shape [ctx_len, emb_dim] ex. [1024, 768]
    wpe = model[f"{prefifx}wpe.weight"].numpy()

    blocks = []
    for i in range(n_layers):
        mlp_c_fc = {"w": model[f"{prefifx}h.{i}.mlp.c_fc.weight"].numpy(),
                    "b": model[f"{prefifx}h.{i}.mlp.c_fc.bias"].numpy()}
        mlp_c_proj = {"w": model[f"{prefifx}h.{i}.mlp.c_proj.weight"].numpy(),
                      "b": model[f"{prefifx}h.{i}.mlp.c_proj.bias"].numpy()}
        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}
        c_attn = {"w": model[f"{prefifx}h.{i}.attn.c_attn.weight"].numpy(),
                  "b": model[f"{prefifx}h.{i}.attn.c_attn.bias"].numpy()}
        c_proj = {"w": model[f"{prefifx}h.{i}.attn.c_proj.weight"].numpy(),
                  "b": model[f"{prefifx}h.{i}.attn.c_proj.bias"].numpy()}
        attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_1 = {"g": model[f"{prefifx}h.{i}.ln_1.weight"].numpy(),
                "b": model[f"{prefifx}h.{i}.ln_1.bias"].numpy()}
        ln_2 = {"g": model[f"{prefifx}h.{i}.ln_2.weight"].numpy(),
                "b": model[f"{prefifx}h.{i}.ln_2.bias"].numpy()}
        block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
        blocks.append(block)
    ln_f = {"g": model[f"{prefifx}ln_f.weight"].numpy(
    ), "b": model[f"{prefifx}ln_f.bias"].numpy()}
    if f"{prefifx}lm_head.weight" in model:
        lm_head = model[f"{prefifx}lm_head.weight"].numpy()
    else:
        lm_head = None
    params = {
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f": ln_f,
        "lm_head": lm_head
    }
    hparams = {}
    hparams["n_head"] = 12
    hparams["n_ctx"] = 1024
    return get_encoder(tokenizer_path), hparams, params


"""Byte pair encoding utilities.

Copied from: https://github.com/openai/gpt-2/blob/master/src/encoder.py.
"""


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"),
                                                          ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_data, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b]
                            for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c]
                         for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder(tokenizer_path):
    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)
    return Encoder(encoder=tokenizer["model"]["vocab"], bpe_data=tokenizer["model"]["merges"])


import sys
base_path = sys.argv[1]
encoder, hparams, params = load_encoder_hparams_and_params(
        model_path=f"{base_path}/pytorch_model.bin", tokenizer_path=f"{base_path}/tokenizer.json")


n_tokens_to_generate=10
topk=5
prompt = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."

input_ids = encoder.encode(prompt)
assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
output_ids = generate(
    input_ids, params, hparams["n_head"], n_tokens_to_generate, topk)
output_text = encoder.decode(output_ids)
print(output_text)