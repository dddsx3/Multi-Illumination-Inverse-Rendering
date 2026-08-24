import re
raw = open("_train_full_log.txt", "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    text = raw.decode("utf-16")
else:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace")
c = text.splitlines()
ep = None
nan_ep, ok_ep = [], []
for ln in c:
    m = re.match(r".*Epoch (\d+) 完成", ln)
    if m:
        ep = int(m.group(1))
    m2 = re.search(r"\u9a8c\u8bc1\u635f\u5931: ([0-9.nan]+)", ln)
    if m2 and ep is not None and "\u4f73" not in ln:
        v = m2.group(1)
        if v == "nan":
            nan_ep.append(ep)
        else:
            ok_ep.append((ep, float(v)))
print("nan epochs:", nan_ep[:80])
print("last numeric epochs:", ok_ep[-8:])
print("numeric:", len(ok_ep), "/ nan:", len(nan_ep), "/ total:", len(nan_ep)+len(ok_ep))