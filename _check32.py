import re
raw = open("_train_fp32_log.txt", "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    text = raw.decode("utf-16")
else:
    text = raw.decode("utf-8", errors="replace")
c = text.splitlines()
eps, ep, nan = [], None, 0
for ln in c:
    m = re.match(r".*Epoch (\d+) \u5b8c\u6210", ln)
    if m:
        ep = int(m.group(1)); eps.append(ep)
    if "nan" in ln.lower():
        nan += 1
print("epochs_done:", sorted(set(eps))[-6:] if eps else "none")
print("nan_mentions:", nan)
vals = [ln.strip() for ln in c if "\u9a8c\u8bc1\u635f\u5931" in ln][-4:]
print("last_vals:", vals)