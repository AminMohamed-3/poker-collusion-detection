"""Check identity and full payload uniqueness across calibrated and shifted data."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from pathlib import Path
import hashlib,json
import polars as pl
H=Path(__file__).resolve().parent;OLD=H.parent/'history_realism_20260910'
seen=set();hand_ids=set();players=set();pools=set();count=0
for d in [OLD/'persistent_sessions',OLD/'session_confirmation',H/'varied']:
    for folder in sorted((d/'pilot').glob('pool-*')):
        h=pl.read_parquet(folder/'hands.parquet');s=pl.read_parquet(folder/'seats.parquet')
        ids=set(h['hand_id']);assert not hand_ids&ids;hand_ids.update(ids)
        players.update(s['player_id']);pools.update(h['table_id'])
        f=s.sort('hand_id','seat_no').group_by('hand_id',maintain_order=True).agg(pl.struct(pl.exclude('hand_id','player_id')).alias('payload')).join(h.select('hand_id','board_cards','big_blind'),on='hand_id')
        for row in f.iter_rows(named=True):
            digest=hashlib.sha256(json.dumps([row['payload'],row['board_cards'],row['big_blind']],sort_keys=True).encode()).digest()
            assert digest not in seen;seen.add(digest);count+=1
assert len(hand_ids)==count==len(seen)==384000
assert len(players)==3840 and len(pools)==128
(H/'cross-batch-identity.json').write_text(json.dumps({'hands':count,'unique_payloads':len(seen),'unique_hand_ids':len(hand_ids),'players':len(players),'pools':len(pools),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_identifier_or_full_payload_overlap':True,'not_a_claim_of_independent_policies':True},indent=2))
print('All 384000 hand payloads and identifiers are unique across 128 pools.')
