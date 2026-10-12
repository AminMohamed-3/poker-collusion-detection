from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
CURRENT=ROOT/'foundation_pair/full_fast/submission_fast_rank80_20_on_current.csv'
OLD=ROOT/'reproduction/writeup/case_selection.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=json.loads(OLD.read_text()); wanted={r['pair_id']:r for r in selection['cases']}
with CURRENT.open(encoding='utf-8-sig',newline='') as f: rows={r['pair_id']:r for r in csv.DictReader(f) if r['pair_id'] in wanted}
assert set(rows)==set(wanted); checks=[]
for pid,old in wanted.items():
 row=rows[pid]; evidence=[row[f'evidence_hand_{i}'] for i in range(1,6)]; expected=[old[f'evidence_hand_{i}'] for i in range(1,6)]
 assert evidence==expected and row['predicted_behavior']==old['predicted_behavior']
 checks.append({'pair_id':pid,'risk_score':float(row['risk_score']),'predicted_behavior':row['predicted_behavior'],'evidence':evidence,'behavior_exact_vs_91411':True,'evidence_exact_vs_91411':True})
out={'current_csv':str(CURRENT),'current_sha256':sha(CURRENT),'rows':len(rows),'all_exact':True,'checks':checks}
(HERE/'case_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
