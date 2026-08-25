import re
raw = open("main.py", "rb").read()
text = raw.decode("utf-8", errors="replace")
lines = text.splitlines()
func = None
for i, ln in enumerate(lines):
    m = re.match(r"def (\w+)\(", ln)
    if m: func = m.group(1)
    if "args." in ln and func and func != "main":
        print(f"L{i+1} in {func}: {ln.strip()[:70]}")