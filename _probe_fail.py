import json
d = json.load(open("D:/data/synthetic_v3/_validation/validation.json", encoding="utf-8"))
items = list(d["issues"].items())[:2]
for k, v in items:
    print("SCENE", k)
    for msg in v[:6]:
        print("   ", msg[:70])