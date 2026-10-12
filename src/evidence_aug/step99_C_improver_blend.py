"""Step 99 (Claude): score-level blend of the step-97 improvers vs the greedy stack S*. Registered: prediction-improver-blend.json."""
import sys, io, json, hashlib, datetime
sys.dont_write_bytecode = True
from pathlib import Path
CODEX = Path("C:/Data/Projects/Money Making/work/poker/Codex Resume")
CV = Path("C:/Data/Projects/Money Making/work/poker/Claude Version")
sys.path[:0] = [str(CV / "src"), str(CODEX / "src")]
import numpy as np, polars as pl
import audit_common as ac
from evidence_aug.paths import check_protected
E97 = CV / "experiments/codex_ideas_sweep_C_20260913"
OUT = CV / "experiments/improver_blend_C_20260913"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
REG = OUT / "prediction-improver-blend.json"
assert sha(REG) == "d7b104e5880f907c5e733141bc37d8d275a4290cd106d7eb6190cf90c597da0b"
assert sha(E97 / "prediction-codex-ideas-sweep.json") == "33903378d56bed3b38b8d77b8a3d709fa07b2b5883fc19092f28bfa9442cd2da"
for f in ("scores-blends.parquet", "blends-frozen.json", "audit-C-improver-blend.json"):
    assert not (OUT / f).exists(), f
protected0 = check_protected()
A97 = json.load(io.open(E97 / "audit-C-codex-ideas-sweep.json", encoding="utf-8"))
SF = json.load(io.open(E97 / "singles-frozen.json", encoding="utf-8"))
GF = json.load(io.open(E97 / "greedy-frozen.json", encoding="utf-8"))
assert sha(E97 / SF["scores_file"]) == SF["scores_sha256"]
d, shared, extra, saved = ac.load_current()
S = pl.read_parquet(E97 / SF["scores_file"])
assert S.select("pair_id", "hand_id", "table_id").equals(d.select("pair_id", "hand_id", "table_id"))
arms = A97["singles"]["arms"]
GROUP = {a: v["idea_group"] for a, v in arms.items()}
CLASS = {a: v["production_class"] for a, v in arms.items()}
improvers = [a for a in A97["singles"]["improvers"] if CLASS[a] in ("A", "B")]
assert improvers, "NOT_APPLICABLE path would apply"
top = (d["oof_rank"] <= 20).to_numpy()
B0 = S["B0"].to_numpy()
assert np.array_equal(top, B0 > 0), "top-20 mask"
pid = d["pair_id"].to_numpy()[top]
def ranks(values):
    return pl.DataFrame({"pair_id": pid, "v": values[top]}).select(
        (-pl.col("v").rank("average", descending=True).over("pair_id")).alias("r"))["r"].to_numpy()
def blend(member_arms):
    groups = {}
    for a in member_arms:
        groups.setdefault(GROUP[a], []).append(a)
    comps = {"B0": ranks(B0)}
    for g in sorted(groups):
        comps[g] = np.mean([ranks(S[a].to_numpy()) for a in sorted(groups[g])], axis=0)
    w = 1.0 / len(comps)
    p = d["oof"].to_numpy().copy() - 1e6
    p[top] = sum(w * c for c in comps.values()) + 1000
    return p, {g: sorted(v) for g, v in groups.items()}, len(comps)
bi, bi_groups, bi_n = blend(improvers)
all_ab = [a for a in arms if CLASS[a] in ("A", "B")]
ba, ba_groups, ba_n = blend(all_ab)
step = GF["steps"][0]
G = pl.read_parquet(E97 / step["scores_file"])
assert sha(E97 / step["scores_file"]) == step["scores_sha256"] and np.array_equal(G["S"].to_numpy(), B0)
SSTAR = G["S_plus_X"].to_numpy()
assert A97["S_star"]["members"] == [step["candidate"]] and len(GF["steps"]) == 1
cols = {"B0": B0, "S_star": SSTAR, "blend_improvers": bi, "blend_all_ab": ba}
sf = OUT / "scores-blends.parquet"
d.select("pair_id", "hand_id", "table_id").with_columns(*[pl.Series(k, v) for k, v in cols.items()]).write_parquet(sf)
back = pl.read_parquet(sf)
assert all(np.array_equal(back[k].to_numpy(), v) for k, v in cols.items())
frozen = {"scores_file": sf.name, "scores_sha256": sha(sf), "frozen_utc": utc(), "evaluations_before_freeze": 0,
          "blend_improvers_components": {"B0": "B0", **bi_groups}, "blend_improvers_n_components": bi_n,
          "blend_all_ab_components": {"B0": "B0", **ba_groups}, "blend_all_ab_n_components": ba_n,
          "S_star": A97["S_star"]["members"], "improvers_class_AB": improvers}
io.open(OUT / "blends-frozen.json", "x", encoding="utf-8").write(json.dumps(frozen, indent=2) + "\n")
started = utc()
vs_b0 = ac.evaluate(d, {"incumbent": B0, "S_star": SSTAR, "blend_improvers": bi, "blend_all_ab": ba})
vs_s = ac.evaluate(d, {"incumbent": SSTAR, "blend_improvers": bi, "blend_all_ab": ba})
keep = ("mean_delta", "ci95", "halves", "by_family", "pairs_better", "pairs_worse", "n")
pick = lambda c: {k: c[k] for k in keep}
BI_S, BI_B = vs_s["comparisons"]["blend_improvers"], vs_b0["comparisons"]["blend_improvers"]
cond = {"blend_minus_Sstar_ge_.003": BI_S["mean_delta"] >= .003,
        "both_halves_blend_minus_Sstar_ge_0": min(BI_S["halves"].values()) >= 0,
        "every_family_blend_minus_B0_ge_-.010": min(BI_B["by_family"].values()) >= -.010}
replace = all(cond.values())
audit = {"step": 99, "author": "Claude (script run by me directly)", "registration_sha256": sha(REG), "evaluation_started_utc": started,
         "frozen": frozen, "maps": {k: vs_b0["sets"][k]["map5"] for k in cols},
         "vs_B0": {k: pick(vs_b0["comparisons"][k]) for k in ("S_star", "blend_improvers", "blend_all_ab")},
         "vs_Sstar": {k: pick(vs_s["comparisons"][k]) for k in ("blend_improvers", "blend_all_ab")},
         "decision_conditions": cond, "blend_replaces_Sstar": replace,
         "step98_candidate": "blend_improvers" if replace else "S_star (B0+" + "+".join(A97["S_star"]["members"]) + ")",
         "blend_all_ab_role": "descriptive only, never promoted", "protected_unchanged": check_protected() == protected0,
         "held_read": False, "upload": False}
io.open(OUT / "audit-C-improver-blend.json", "x", encoding="utf-8").write(json.dumps(audit, indent=2) + "\n")
for k, v in audit["vs_B0"].items():
    print("vs B0   %-16s %+.6f [%+.6f, %+.6f] halves %s fam-min %+.6f" % (k, v["mean_delta"], v["ci95"][0], v["ci95"][1], {h: round(x, 6) for h, x in v["halves"].items()}, min(v["by_family"].values())))
for k, v in audit["vs_Sstar"].items():
    print("vs S*   %-16s %+.6f [%+.6f, %+.6f] halves %s" % (k, v["mean_delta"], v["ci95"][0], v["ci95"][1], {h: round(x, 6) for h, x in v["halves"].items()}))
print("components:", frozen["blend_improvers_components"], "| all_ab n", ba_n)
print("conditions:", cond, "=> replace:", replace, "| candidate:", audit["step98_candidate"], "| protected same:", audit["protected_unchanged"])
print("audit sha", sha(OUT / "audit-C-improver-blend.json"))
