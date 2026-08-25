import glob, os
d = sorted(os.listdir("D:/data/synthetic_v3"))[:3]
print("sample dirs:", d)
sd = "D:/data/synthetic_v3/" + d[0]
print("files:", sorted(os.listdir(sd))[:18])

import glob as g
p1 = g.glob(os.path.join(sd, "light_[0-9][0-9][0-9].png"))
p2 = g.glob(os.path.join(sd, "light_*.png"))
print("precise glob:", len(p1), "| star glob:", len(p2))