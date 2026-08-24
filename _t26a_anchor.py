import json, subprocess

new = json.load(open("eval_output/p2_t26a_test_phase1recovered/eval_summary.json", encoding="utf-8"))["metrics_mean_std"]
old_raw = subprocess.run(
    ["git", "show", "b7b144d:eval_output/eval_summary.json"],
    capture_output=True).stdout.decode("utf-8")
old = json.loads(old_raw)["metrics_mean_std"]

assert set(new.keys()) == set(old.keys()), "metric keys mismatch"
max_dev, worst = 0.0, ""
rows = []
for k in sorted(new.keys()):
    dm = abs(new[k]["mean"] - old[k]["mean"])
    ds = abs(new[k]["std"] - old[k]["std"])
    d = max(dm, ds)
    if d > max_dev:
        max_dev, worst = d, k
    rows.append((k, old[k]["mean"], old[k]["std"], new[k]["mean"], new[k]["std"], d))
print(f"anchor check over {len(rows)} metrics: max_deviation={max_dev:.2e} ({worst})")
for k, om, os_, nm, ns, d in rows:
    print(f"  {k:22s} old={om:.6f}+-{os_:.6f}  new={nm:.6f}+-{ns:.6f}  dev={d:.2e}")
print("ANCHOR:", "PASS" if max_dev < 1e-9 else ("PASS(tol)" if max_dev < 1e-6 else "FAIL"))