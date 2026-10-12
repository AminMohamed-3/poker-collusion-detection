"""S02b - outsider-centred crisp atoms replayed from the raw actions for the 372 positive pairs' hands.
Development phase asserted (hands come from base_frame, which asserted it). No evaluation row read.
Per (pair, hand): counts of outsider folds facing member aggression, pair re-raise / sandwich structure,
who was the first aggressor, whether the pair ended up alone, member fold after outsiders were out, etc.
"""
import polars as pl, numpy as np, os, json, collections
ROOT = r"C:\Data\Projects\Money Making\work\poker"
OUT = os.path.join(ROOT, "Fable Version", "marking_rule", "out")
base = pl.read_parquet(os.path.join(OUT, "base_frame.parquet"), columns=["pair_id", "hand_id", "family"])
lab = pl.read_csv(os.path.join(ROOT, "data", "development_labels.csv")).filter(pl.col("label") == 1).select("pair_id", "player_1", "player_2")
base = base.join(lab, on="pair_id")
hids = base["hand_id"].unique()
acts = pl.read_parquet(os.path.join(ROOT, "data", "actions.parquet")).filter(pl.col("hand_id").is_in(hids.implode())).sort(["hand_id", "action_no"])
print("action values:", acts["action"].unique().to_list(), "streets:", acts["street"].unique().to_list())
seats = pl.read_parquet(os.path.join(ROOT, "data", "seats.parquet"), columns=["hand_id", "player_id", "folded", "went_to_showdown", "net_chips", "total_contribution", "starting_stack"]).filter(pl.col("hand_id").is_in(hids.implode()))
hands = pl.read_parquet(os.path.join(ROOT, "data", "hands.parquet"), columns=["hand_id", "big_blind", "players_dealt", "phase"]).filter(pl.col("hand_id").is_in(hids.implode()))
assert (hands["phase"] == "development").all()
bb = dict(zip(hands["hand_id"].to_list(), hands["big_blind"].to_list()))
pdealt = dict(zip(hands["hand_id"].to_list(), hands["players_dealt"].to_list()))
# group actions by hand
A = collections.defaultdict(list)
for h, st, p, a, amt, amt_to, pot, stk, tc, pa in zip(acts["hand_id"].to_list(), acts["street"].to_list(), acts["player_id"].to_list(), acts["action"].to_list(),
                                                      acts["amount"].to_list(), acts["amount_to"].to_list(), acts["pot_before"].to_list(), acts["stack_before"].to_list(), acts["to_call"].to_list(), acts["players_active"].to_list()):
    A[h].append((st, p, a, amt, amt_to, pot, stk, tc, pa))
S = collections.defaultdict(dict)
for h, p, f, sd, net, contrib, ss in zip(seats["hand_id"].to_list(), seats["player_id"].to_list(), seats["folded"].to_list(), seats["went_to_showdown"].to_list(), seats["net_chips"].to_list(), seats["total_contribution"].to_list(), seats["starting_stack"].to_list()):
    S[h][p] = (f, sd, net, contrib, ss)
AGG = ("bet", "raise")
rows = []
for pid, h, p1, p2 in zip(base["pair_id"].to_list(), base["hand_id"].to_list(), base["player_1"].to_list(), base["player_2"].to_list()):
    mem = {p1, p2}
    seat = S[h]
    outs = [p for p in seat if p not in mem]
    r = dict(pair_id=pid, hand_id=h)
    last_aggr = None; street = None
    n_out_fold_vs_mem = 0; n_out_fold_vs_out = 0; n_out_fold_noagg = 0; n_out_vpip = 0; n_out_call_mem_agg = 0
    out_vpip = set(); out_called_mem = set(); out_folded_after_calling_mem = 0
    first_aggr = None; pair_reraise = 0; sandwich = 0; both_agg_same_street = 0
    mem_fold_after_outs_out = 0; mem_fold_vs_partner = 0; mem_agg_streets = set()
    street_aggr_seq = collections.defaultdict(list)  # street -> list of aggressor ids in order
    n_mem_agg = 0; n_out_agg = 0; mem_raise_over_out = 0; out_raise_over_mem = 0
    live_outs = set(outs)
    folded = set()
    for st, p, a, amt, amt_to, pot, stk, tc, pa in A[h]:
        if a == 'all_in':
            a = 'raise' if amt > tc else 'call'
        if st != street:
            street = st; last_aggr = None
        is_mem = p in mem
        if a in AGG:
            if first_aggr is None: first_aggr = "mem" if is_mem else "out"
            if is_mem:
                n_mem_agg += 1; mem_agg_streets.add(st)
                if last_aggr is not None and last_aggr in mem and last_aggr != p:
                    pair_reraise += 1
                    if any(q not in mem for q in street_aggr_seq[st][-1:]): pass
                if last_aggr is not None and last_aggr not in mem: mem_raise_over_out += 1
                # sandwich: partner aggressed earlier this street, an outsider called in between (approximate via seq)
            else:
                n_out_agg += 1
                if last_aggr is not None and last_aggr in mem: out_raise_over_mem += 1
            street_aggr_seq[st].append(p)
            last_aggr = p
        elif a == "call":
            if not is_mem:
                out_vpip.add(p)
                if last_aggr in mem: n_out_call_mem_agg += 1; out_called_mem.add(p)
        elif a == "fold":
            folded.add(p)
            if not is_mem:
                if last_aggr in mem: n_out_fold_vs_mem += 1
                elif last_aggr is None: n_out_fold_noagg += 1
                else: n_out_fold_vs_out += 1
                if p in out_called_mem: out_folded_after_calling_mem += 1
                live_outs.discard(p)
            else:
                if last_aggr in mem and last_aggr != p:
                    mem_fold_vs_partner += 1
                    if len(live_outs) == 0: mem_fold_after_outs_out += 1
        if a in AGG and not is_mem: pass
    # sandwich: on some street, sequence member -> outsider (call or raise) -> partner raise
    for st, seq in street_aggr_seq.items():
        pass
    both_agg_same_street = 0
    for st in ("preflop", "flop", "turn", "river"):
        seq = street_aggr_seq.get(st, [])
        ms = [q for q in seq if q in mem]
        if len(set(ms)) == 2: both_agg_same_street = 1
    n_out = len(outs)
    out_sd = sum(1 for p in outs if seat[p][1])
    mem_sd = sum(1 for p in mem if p in seat and seat[p][1])
    pair_alone_end = int(len(live_outs) == 0 and all(p in seat and not seat[p][0] for p in mem))
    m_net = {p: seat[p][2] for p in mem if p in seat}
    r.update(r_n_out=n_out, r_n_out_vpip=len(out_vpip), r_n_out_fold_vs_mem=n_out_fold_vs_mem, r_n_out_fold_vs_out=n_out_fold_vs_out,
             r_n_out_fold_noagg=n_out_fold_noagg, r_n_out_call_mem_agg=n_out_call_mem_agg, r_out_fold_after_call_mem=out_folded_after_calling_mem,
             r_first_aggr_mem=int(first_aggr == "mem"), r_first_aggr_none=int(first_aggr is None), r_pair_reraise=pair_reraise,
             r_both_agg_same_street=both_agg_same_street, r_mem_fold_vs_partner=mem_fold_vs_partner, r_mem_fold_after_outs_out=mem_fold_after_outs_out,
             r_n_mem_agg=n_mem_agg, r_n_out_agg=n_out_agg, r_mem_raise_over_out=mem_raise_over_out, r_out_raise_over_mem=out_raise_over_mem,
             r_out_sd=out_sd, r_mem_sd=mem_sd, r_pair_alone_end=pair_alone_end, r_all_outs_folded=int(len(live_outs) == 0),
             r_mem_agg_streets=len(mem_agg_streets), r_net_sign_opposite=int(len(m_net) == 2 and min(m_net.values()) < 0 < max(m_net.values())),
             r_mem_net_sum_bb=sum(m_net.values()) / bb[h], r_out_net_sum_bb=sum(seat[p][2] for p in outs) / bb[h])
    rows.append(r)
df = pl.DataFrame(rows)
df.write_parquet(os.path.join(OUT, "raw_atoms.parquet"))
print(df.shape); print(df.describe())
