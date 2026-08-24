import numpy as np
from PIL import Image
import scipy.io as sio

d = "D:/data/DiLiGenT/pmsData/bearPNG"
# light_intensities: 每行是 RGB 三通道强度（彩色光照！）
li = np.loadtxt(d + "/light_intensities.txt")
ld = np.loadtxt(d + "/light_directions.txt")
print("intensities:", li.shape, "directions:", ld.shape)
print("intensity row0:", li[0])

nt = np.loadtxt(d + "/normal.txt")
print("normal.txt:", nt.shape)
H, W = 512, 612
nmap = nt.reshape(H, W, 3) if nt.size == H*W*3 else None
if nmap is not None:
    print("gt normal z stats: min", nmap[...,2].min(), "max", nmap[...,2].max())
    print("gt norm mean:", np.linalg.norm(nmap, axis=-1).mean())

mat = sio.loadmat(d + "/Normal_gt.mat")
print("Normal_gt keys:", [k for k in mat.keys() if not k.startswith("__")])