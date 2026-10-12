"""Prewrite conditional confirmation; running the builder requires all development gates."""
from pathlib import Path
H=Path(__file__).resolve().parent
P=H.parent/'softplay_causal_style_20260910'
s=(P/'build_confirmation.py').read_text()
s=s.replace('softplay_style_confirmation_20260910','softplay_capacity_confirmation_20260910')
s=s.replace('both_development_realism_gates_passed','all_three_realism_gates_passed')
s=s.replace('b0_style','b0_capacity').replace('832+i','896+i').replace('832..863','896..927').replace('1200559','1400753')
s=s.replace('the three style populations','all three development populations and the fresh confirmation')
(H/'build_confirmation.py').write_text(s,encoding='utf-8')
s=(P/'summarize_confirmation.py').read_text().replace('softplay_style_confirmation_20260910','softplay_capacity_confirmation_20260910').replace('b0_style','b0_capacity').replace('causal-style confirmation','capacity-policy confirmation').replace('all three populations','all four populations')
(H/'summarize_confirmation.py').write_text(s,encoding='utf-8')
s=(P/'build_utility.py').read_text()
s=s.replace('softplay_style_confirmation_20260910','softplay_capacity_confirmation_20260910').replace('softplay_style_utility_20260910','softplay_capacity_utility_20260910')
s=s.replace('both_development_realism_gates_passed','all_three_realism_gates_passed')
s=s.replace("arms=['b0_style','b1_style','b2_style'];parents=[H/'b0_style',H/'b1_style',C/'fresh']", "arms=['b0_capacity','b1_capacity','b2_capacity','b3_capacity'];parents=[H/'b0_capacity',H/'b1_capacity',H/'b2_capacity',C/'fresh']")
s=s.replace('two development realism populations','three development realism populations')
s=s.replace('two development populations','three development populations')
s=s.replace('three populations','four populations').replace('three generation populations','four generation populations').replace('three synthetic populations','four synthetic populations')
s=s.replace("'detector_fits':21","'detector_fits':24").replace('all21 fits','all24 fits').replace("'fits':21","'fits':24").replace("['style']","['capacity']").replace('nine fitted synthetic models','twelve fitted synthetic models')
s=s.replace('softplay_causal_style_20260910/controller-fit.json','softplay_capacity_20260910/controller-fit.json')
s=s.replace('Probability coupling plus causal own-player tendencies','A larger300-tree policy with probability coupling and causal own-player tendencies')
(H/'build_utility.py').write_text(s,encoding='utf-8')
print('Conditional confirmation scripts prepared; no generation started.')
