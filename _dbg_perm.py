import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_unet import FusionUNet

model = FusionUNet(num_images=10, in_channels=1, base_channels=16, fusion_dim=64)
model.eval()

N = 3
x = torch.rand(1, N, 1, 64, 64)
perm = torch.tensor([2, 0, 1])

acts = {}
def grab(name):
    def hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        acts[name] = t.detach()
    return hook

model.stem.register_forward_hook(grab("stem"))
model.down1.register_forward_hook(grab("down1"))
model.down2.register_forward_hook(grab("down2"))
model.down3.register_forward_hook(grab("down3"))
model.down4.register_forward_hook(grab("down4"))
model.bottleneck.register_forward_hook(grab("bn"))
model.aggregator.mha.register_forward_hook(grab("mha_out"))
model.aggregator.pma.register_forward_hook(grab("pma_out"))

with torch.no_grad():
    ref = model(x)
    r1 = {k: v.clone() for k, v in acts.items()}
    out_p = model(x[:, perm])
    r2 = {k: v.clone() for k, v in acts.items()}

# 对 [B*N,...] 形状的激活：按 perm 还原行后与参考顺序比较
for k in ("stem", "down1", "down2", "down3", "down4", "bn"):
    a = r2.get(k)
    if a is None or a.dim() < 2 or a.shape[0] != N:
        continue
    restored = torch.empty_like(a)
    restored[perm] = a
    rr = r1.get(k)
    if rr is None or rr.shape != restored.shape:
        continue
    print(f"{k}: restored-vs-ref maxdiff={(restored - rr).abs().max().item():.6f}")

print("---")
print("depth maxdiff (direct):", (ref[0] - out_p[0]).abs().max().item())