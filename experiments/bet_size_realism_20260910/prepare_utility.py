"""Prepare a conditional two-dose test before sizing confirmation outcomes."""
from pathlib import Path
H=Path(__file__).resolve().parent;E=H.parent;N=E/'bet_size_confirmation_20260910'
s=(E/'softplay_causal_style_20260910/build_utility.py').read_text()
s=s.replace("C=E/'softplay_style_confirmation_20260910';N=E/'softplay_style_utility_20260910'","C=E/'bet_size_realism_20260910';N=E/'bet_size_utility_20260910'")
s=s.replace('both_development_realism_gates_passed','all_confirmation_gates_passed').replace("['passes_realism']","['passes_initial_pilot']")
s=s.replace("arms=['b0_style','b1_style','b2_style'];parents=[H/'b0_style',H/'b1_style',C/'fresh']","arms=['b0_sized','b1_sized','b2_sized'];parents=[C/'sized',H/'b1_sized',H/'fresh_sized']")
s=s.replace('softplay_causal_style_20260910/controller-fit.json','softplay_capacity_20260910/controller-fit.json').replace("['style']","['sized']")
s=s.replace('Probability coupling plus causal own-player tendencies','Conditional ordinary bet-size distributions plus the qualified causal soft-action model')
(N/'build_base_utility.py').write_text(s,encoding='utf-8')
print('Conditional base utility builder prepared; no trial or fit started.')
