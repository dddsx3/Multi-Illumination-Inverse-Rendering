import numpy as np
from PIL import Image

def oetf(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

for scene in ("003715596f13472c877ee3a622782d5a", "0094c76440e3447a900797ede9bad5bf"):
    d = "D:/data/synthetic_v3/" + scene
    rgb8 = np.asarray(Image.open(d + "/light_001_rgb.png"))
    gray8 = np.asarray(Image.open(d + "/light_001.png"))
    c01 = rgb8 / 255.0
    lin = np.where(c01 <= 0.04045, c01 / 12.92, ((c01 + 0.055) / 1.055) ** 2.4)
    luma_lin = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    h1 = np.round(0.2126 * rgb8[..., 0].astype(np.float32)
                  + 0.7152 * rgb8[..., 1].astype(np.float32)
                  + 0.0722 * rgb8[..., 2].astype(np.float32))
    h2 = np.round(oetf(luma_lin) * 255)
    print(scene[:12],
          "| H1 encoded-luma maxdiff:", int(np.abs(h1 - gray8).max()),
          "| H2 OETF-linear maxdiff:", int(np.abs(h2 - gray8).max()))