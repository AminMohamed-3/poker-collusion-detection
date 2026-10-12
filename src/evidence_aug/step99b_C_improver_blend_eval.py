"""Step 99b (Claude): resume of step 99 after its audit-assembly crash. Replays the frozen blends bit-exactly, freezes
the registration-faithful per-section descriptive blend before any outcome is read, then evaluates once.
Registered: experiments/improver_blend_C_20260913/prediction-improver-blend.json."""
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
assert sha(OUT / "blends-frozen.json") == "066e273917ba4ab6f4cd6ac4c518e3cd9759b8e788d130fd2ab863257521ecda"
FZ = json.load(io.open(OUT / "blends-frozen.json", encoding="utf-8"))
assert sha(OUT / FZ["scores_file"]) == FZ["scores_sha256"] == "24f5f3024b0ea430911e031ed8df7a068ddb4fb64f863396f03b3fdc81c12b83"
for f in ("scores-blends-addendum.parquet", "blends-addendum-frozen.json", "audit-C-improver-blend.json"):
    assert not (OUT / f).exists(), f
protected0 = check_protected()
A97 = json.load(io.open(E97 / "audit-C-codex-ideas-sweep.json", encoding="utf-8"))
SF = json.load(io.open(E97 / "singles-frozen.json", encoding="utf-8"))
assert sha(E97 / SF["scores_file"]) == SF["scores_sha256"]
d, shared, extra, saved = ac.load_current()
S = pl.read_parquet(E97 / SF["scores_file"])
F = pl.read_parquet(OUT / FZ["scores_file"])
for x in (S, F):
    assert x.select("pair_id", "hand_id", "table_id").equals(d.select("pair_id", "hand_id", "table_id"))
arms = A97["singles"]["arms"]
CLASS = {a: v["production_class"] for a, v in arms.items()}
IDEA = {a: v["idea_group"] for a, v in arms.items()}
TRAIN_SECTION = {"symmetric_censor_half": "sec2", "subset_nll": "sec16"}
assert sorted(a for a in arms if IDEA[a] == "TRAIN") == sorted(TRAIN_SECTION)
SECTION = {a: TRAIN_SECTION.get(a, IDEA[a]) for a in arms}
top = (d["oof_rank"] <= 20).to_numpy()
B0 = S["B0"].to_numpy()
assert np.array_equal(top, B0 > 0) and np.array_equal(F["B0"].to_numpy(), B0)
pid = d["pair_id"].to_numpy()[top]
def ranks(values):
    return pl.DataFrame({"pair_id": pid, "v": values[top]}).select(
        (-pl.col("v").rank("average", descending=True).over("pair_id")).alias("r"))["r"].to_numpy()
def blend(member_arms, key):
    groups = {}
    for a in member_arms:
        groups.setdefault(key[a], []).append(a)
    comps = {"B0": ranks(B0)}
    for g in sorted(groups):
        comps[g] = np.mean([ranks(S[a].to_numpy()) for a in sorted(groups[g])], axis=0)
    w = 1.0 / len(comps)
    p = d["oof"].to_numpy().copy() - 1e6
    p[top] = sum(w * c for c in comps.values()) + 1000
    return p, {g: sorted(v) for g, v in groups.items()}, len(comps)
improvers = [a for a in A97["singles"]["improvers"] if CLASS[a] in ("A", "B")]
all_ab = [a for a in arms if CLASS[a] in ("A", "B")]
bi, _, _ = blend(improvers, SECTION)
ba_idea, _, _ = blend(all_ab, IDEA)
ba_sec, sec_groups, sec_n = blend(all_ab, SECTION)
replay = {"blend_improvers": bool(np.array_equal(bi, F["blend_improvers"].to_numpy())),
          "blend_all_ab_as_frozen": bool(np.array_equal(ba_idea, F["blend_all_ab"].to_numpy()))}
assert all(replay.values()), replay
GF = json.load(io.open(E97 / "greedy-frozen.json", encoding="utf-8"))
step = GF["steps"][0]
G = pl.read_parquet(E97 / step["scores_file"])
assert sha(E97 / step["scores_file"]) == step["scores_sha256"] and len(GF["steps"]) == 1
SSTAR = G["S_plus_X"].to_numpy()
assert np.array_equal(F["S_star"].to_numpy(), SSTAR)
af = OUT / "scores-blends-addendum.parquet"
d.select("pair_id", "hand_id", "table_id").with_columns(pl.Series("blend_all_ab_sections", ba_sec)).write_parquet(af)
assert np.array_equal(pl.read_parquet(af)["blend_all_ab_sections"].to_numpy(), ba_sec)
add = {"scores_file": af.name, "scores_sha256": sha(af), "frozen_utc": utc(), "replay_of_frozen_columns_bit_exact": replay,
       "blend_all_ab_sections_components": {"B0": "B0", **sec_groups}, "blend_all_ab_sections_n_components": sec_n,
       "why": "The registration defines one component per Codex idea section and lists sec2 and sec16 separately. The first run grouped by the step-97 idea_group, which merges sec2 symmetric_censor_half and sec16 subset_nll under TRAIN (9 components, not 10). Descriptive arm only; blend_improvers is unaffected because both improvers are sec14.",
       "outcomes_read_before_this_freeze": 0,
       "crashed_first_run": "step99_C_improver_blend.py froze scores-blends.parquet, completed two in-memory evaluate() calls, then raised KeyError while assembling the audit (evaluate names the reference set incumbent, not B0). Nothing from those calls was printed or saved."}
io.open(OUT / "blends-addendum-frozen.json", "x", encoding="utf-8").write(json.dumps(add, indent=2) + "\n")
started = utc()
NAMES = ("blend_improvers", "blend_all_ab_sections", "blend_all_ab_as_frozen")
cand = {"blend_improvers": bi, "blend_all_ab_sections": ba_sec, "blend_all_ab_as_frozen": ba_idea}
vs_b0 = ac.evaluate(d, {"incumbent": B0, "S_star": SSTAR, **cand})
vs_s = ac.evaluate(d, {"incumbent": SSTAR, **cand})
keep = ("mean_delta", "ci95", "halves", "by_family", "pairs_better", "pairs_worse", "n")
pick = lambda c: {k: c[k] for k in keep if k in c}
BI_S, BI_B = vs_s["comparisons"]["blend_improvers"], vs_b0["comparisons"]["blend_improvers"]
cond = {"blend_minus_Sstar_ge_.003": BI_S["mean_delta"] >= .003,
        "both_halves_blend_minus_Sstar_ge_0": min(BI_S["halves"].values()) >= 0,
        "every_family_blend_minus_B0_ge_-.010": min(BI_B["by_family"].values()) >= -.010}
replace = all(cond.values())
pc = {"blend_improvers_minus_Sstar": {"observed": BI_S["mean_delta"], "point": 0.0, "interval80": [-.008, .008], "inside": -.008 <= BI_S["mean_delta"] <= .008},
      "blend_improvers_minus_B0": {"observed": BI_B["mean_delta"], "point": .008, "interval80": [-.002, .02], "inside": -.002 <= BI_B["mean_delta"] <= .02},
      "p_blend_replaces_Sstar_predicted": .25, "replaced": replace, "p_not_applicable_predicted": .12, "applicable": True}
audit = {"step": 99, "author": "Claude (scripts run by me directly)", "registration_sha256": sha(REG),
         "evaluation_started_utc": started, "frozen_first_run": FZ, "frozen_addendum": add,
         "map5_audit": {"B0": vs_b0["sets"]["incumbent"].get("map5"), "S_star": vs_b0["sets"]["S_star"].get("map5"),
                        **{k: vs_b0["sets"][k].get("map5") for k in NAMES}},
         "vs_B0": {k: pick(vs_b0["comparisons"][k]) for k in ("S_star",) + NAMES},
         "vs_Sstar": {k: pick(vs_s["comparisons"][k]) for k in NAMES},
         "decision_conditions": cond, "blend_replaces_Sstar": replace,
         "step98_candidate": "blend_improvers" if replace else "S_star = B0 + " + " + ".join(A97["S_star"]["members"]),
         "descriptive_arms": "blend_all_ab_sections (registration-faithful) and blend_all_ab_as_frozen (first-run grouping); never promoted",
         "prediction_check": pc, "deviations": [add["crashed_first_run"], add["why"]],
         "protected_unchanged": check_protected() == protected0, "held_read": False, "upload": False,
         "sets_vs_B0": vs_b0["sets"]}
safe = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
io.open(OUT / "audit-C-improver-blend.json", "x", encoding="utf-8").write(json.dumps(audit, indent=2, default=safe) + "\n")
print("audit sha", sha(OUT / "audit-C-improver-blend.json"), "| addendum sha", add["scores_sha256"][:16])
print("MAP5", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in audit["map5_audit"].items()})
for k, v in audit["vs_B0"].items():
    print("vs B0  %-22s %+.6f [%+.6f, %+.6f] halves %+.6f/%+.6f fam %s" % (k, v["mean_delta"], v["ci95"][0], v["ci95"][1],
          v["halves"]["0"], v["halves"]["1"], {f[:5]: round(x, 6) for f, x in v["by_family"].items()}))
for k, v in audit["vs_Sstar"].items():
    print("vs S*  %-22s %+.6f [%+.6f, %+.6f] halves %+.6f/%+.6f better/worse %s/%s" % (k, v["mean_delta"], v["ci95"][0], v["ci95"][1],
          v["halves"]["0"], v["halves"]["1"], v.get("pairs_better"), v.get("pairs_worse")))
print("conditions", cond, "=> replace", replace, "| candidate:", audit["step98_candidate"])
print("prediction inside80:", pc["blend_improvers_minus_Sstar"]["inside"], pc["blend_improvers_minus_B0"]["inside"], "| protected same:", audit["protected_unchanged"])
