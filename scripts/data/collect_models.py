"""Collect the best trained per-element operator checkpoint into the
package model store (`spexai/models/`), with a stable naming scheme and a
provenance manifest. Re-runnable: rerun as more elements finish training
and it refreshes the store.

Naming: `spexai/models/Z<zz>_<symbol>.pt`  (e.g. Z26_Fe.pt), so files sort
by atomic number and the joint model can map Z -> file. Provenance (source
run, architecture, test metrics) is kept in `spexai/models/manifest.json`
rather than in the filename, so the canonical name stays element-keyed.
"""
import argparse
import glob
import json
import os
import shutil
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
           "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V",
           "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
SYMBOL = {z: SYMBOLS[z - 1] for z in range(1, 31)}

# elements known to need a re-run before they are science-grade (recorded
# in the manifest so the joint model / users are not misled). Na(11)/Al(13)
# were re-trained clean in the production run and are no longer flagged.
KNOWN_ISSUES = {}


def find_checkpoint(runroot, z):
    """Best checkpoint for element z: prefer tier1/reweight_full.pt, else
    the newest reweight_full.pt anywhere under the element dir."""
    cands = glob.glob(os.path.join(runroot, f"element{z}", "tier1",
                                   "reweight_full.pt"))
    if not cands:
        cands = glob.glob(os.path.join(runroot, f"element{z}", "**",
                                       "reweight_full.pt"), recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None


def metrics_for(ckpt_path):
    """Pull test metrics + architecture from the run alongside a checkpoint."""
    d = os.path.dirname(ckpt_path)
    out = {}
    bt = os.path.join(d, "benchmark_test.json")
    if os.path.exists(bt):
        m = next(iter(json.load(open(bt)).values()))
        o = m.get("overall", {})
        out.update(test_mre=round(o.get("mre_mean", float("nan")), 6),
                   yield_1pct=o.get("yield_1pct"),
                   yield_01pct=o.get("yield_01pct"),
                   params=m.get("params"))
    # architecture/provenance: prefer the run history, but fall back to the
    # args the checkpoint carries internally. An interrupted run can leave a
    # perfectly good checkpoint with no history file (element 21 is exactly
    # this case), and without the fallback those fields would silently vanish
    # from the manifest on the next collect.
    a, src = {}, None
    hist = os.path.join(d, "reweight_full_history.json")
    if os.path.exists(hist):
        a, src = json.load(open(hist)).get("args", {}), "history"
    else:
        try:
            import torch
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and isinstance(ck.get("args"), dict):
                a, src = ck["args"], "checkpoint"
        except Exception as exc:                                 # noqa: BLE001
            print(f"  [!] could not read args from {ckpt_path}: {exc}")
    if a:
        out.update(hidden=a.get("hidden"), layers=a.get("layers"),
                   n_freqs=a.get("n_freqs"), use_linehead=a.get("use_linehead"),
                   signal_frac=a.get("signal_frac"), lr=a.get("lr"),
                   args_source=src)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", default=os.path.expanduser(
        "~/work/data/spexai/runs"))
    ap.add_argument("--outdir", default=os.path.join(REPO, "spexai", "models"))
    ap.add_argument("--elements", nargs="+", type=int,
                    default=list(range(1, 31)))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Merge into any existing manifest so previously-shipped elements (e.g.
    # trained from a different runroot) are preserved; only the elements
    # collected in this invocation are refreshed.
    manifest_path = os.path.join(args.outdir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
    else:
        manifest = {"elements": {}}
    manifest["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["runroot"] = args.runroot
    copied = []
    for z in args.elements:
        ckpt = find_checkpoint(args.runroot, z)
        if ckpt is None:
            continue
        name = f"Z{z:02d}_{SYMBOL[z]}.pt"
        shutil.copy2(ckpt, os.path.join(args.outdir, name))
        entry = {"file": name, "Z": z, "symbol": SYMBOL[z],
                 "source": os.path.relpath(ckpt, os.path.expanduser("~")),
                 **metrics_for(ckpt)}
        if z in KNOWN_ISSUES:
            entry["status"] = KNOWN_ISSUES[z]
        manifest["elements"][str(z)] = entry
        copied.append((z, name, entry.get("test_mre")))

    # missing = anything in the full periodic range not present in the store
    present = {int(z) for z in manifest["elements"]}
    missing = sorted(set(range(1, 31)) - present)
    manifest["missing_elements"] = missing
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"collected {len(copied)} models into {args.outdir}")
    for z, name, mre in copied:
        flag = "  [!] " + KNOWN_ISSUES[z] if z in KNOWN_ISSUES else ""
        print(f"  {name:<12} test_mre={mre}{flag}")
    print(f"missing (not yet trained): {missing}")


if __name__ == "__main__":
    main()
