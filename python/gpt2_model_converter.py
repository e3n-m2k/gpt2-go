import torch
import sys
import struct
import numpy as np

model_path = sys.argv[1]
output_path = sys.argv[2]

model = torch.load(model_path)
hidden_dim = 768
n_ctx = 1024
n_vocab = 50257
n_head = 12
fc_dim = 3072

# for k, v in model.items():
#     print(k)
#     print(v.shape)

with open(output_path, "wb") as f:
    w = model["wte.weight"].reshape(-1).numpy().tolist()
    f.write(struct.pack("f" * (n_vocab * hidden_dim), *w))

    w = model["wpe.weight"].reshape(-1).numpy().tolist()
    f.write(struct.pack("f" * (n_ctx * hidden_dim), *w))

    w = model["ln_f.weight"].reshape(-1).numpy().tolist()
    f.write(struct.pack("f" * hidden_dim, *w))

    w = model["ln_f.bias"].reshape(-1).numpy().tolist()
    f.write(struct.pack("f" * hidden_dim, *w))

    for i in range(n_head):
        w = model[f"h.{i}.ln_1.weight"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim, *w))

        w = model[f"h.{i}.ln_1.bias"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim, *w))

        w = model[f"h.{i}.ln_2.weight"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim, *w))

        w = model[f"h.{i}.ln_2.bias"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim, *w))

        w = model[f"h.{i}.attn.c_attn.weight"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * (hidden_dim * hidden_dim * 3), *w))

        w = model[f"h.{i}.attn.c_attn.bias"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim * 3, *w))

        w = model[f"h.{i}.attn.c_proj.weight"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * (hidden_dim * hidden_dim), *w))

        w = model[f"h.{i}.attn.c_proj.bias"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim, *w))

        w = model[f"h.{i}.mlp.c_fc.weight"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * (hidden_dim * fc_dim), *w))

        w = model[f"h.{i}.mlp.c_fc.bias"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * fc_dim, *w))

        w = model[f"h.{i}.mlp.c_proj.weight"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * (fc_dim * hidden_dim), *w))

        w = model[f"h.{i}.mlp.c_proj.bias"].reshape(-1).numpy().tolist()
        f.write(struct.pack("f" * hidden_dim, *w))