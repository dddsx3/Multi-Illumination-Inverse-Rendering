import re
raw = open("_train_bf16_log.txt", "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    c = raw.decode("utf-16").splitlines()
else:
    c = raw.decode("utf-8", errors="replace").splitlines()

DONE = "\u5b8c\u6210"
VAL = "\u9a8c\u8bc1\u635f\u5931"
NEWBEST = "\u65b0\u4f73"

ep, traj, best = None, [], []
for ln in c:
    m = re.search(r"Epoch (\d+) " + DONE, ln)
    if m:
        ep = int(m.group(1))
    m2 = re.search(VAL + r": ([0-9.]+)", ln)
    if m2 and ep is not None:
        v = float(m2.group(1))
        if NEWBEST in ln:
            best.append((ep, v))
        else:
            traj.append((ep, v))

print("val trajectory (every 10):")
for e, v in traj[::10]:
    print(f"  ep{e}: {v:.4f}")
print(f"final: ep{traj[-1][0]} = {traj[-1][1]:.4f}")
print("best marks:", best[-3:])