# Synthetic teacher-mechanics pilot

These 12 original synthetic cases contain no real poker records, predictions, labels, or teacher outputs. They form six matched two-case patterns. `design_role` is fixture-only construction metadata and is never placed in a prompt. Observable mechanism tags are not intent or guilt labels. Every case requires abstention from intent; pair 06 is deliberately observationally unidentifiable and requires high uncertainty.

Amounts are additional chips committed by an action. Only a raise has `to_amount`, the resulting street commitment. The reported pot is the called pot: any unmatched terminal bet or raise is emitted and checked as a separate `returned_uncalled` fact. Net chips cover both the return and terminal award; split pots are excluded. Complete card identities, intent, private communications, and outside history are explicitly unavailable.

Run `python teacher_mechanics_fixtures.py --self-test` before inference. The standalone `evaluate_response` function is the factuality gate. It rejects wrong IDs, actions, amounts, returns, full-hand nets, outcomes, unavailable-fact sets, missing abstention, and source/variant factual drift. Evidence prose is checked only for non-empty text and valid citations. `author_mechanism_note` tags are optional descriptions excluded from the factual admission gate; they are neither coordination nor guilt labels.

All cases require intent abstention, so this pilot measures instruction following, chronology, arithmetic, source grounding, and alias invariance. The one-hand examples cannot establish benign-versus-cheating discrimination, calibrated suspicion, or student utility. Those require a separately frozen experiment.

`HASHES.sha256` freezes the fixture, prompt, response schema, and reference responses. Root review of these files and hashes is required before any teacher inference. Use one fixed greedy non-thinking decode, a 1,024-token output cap, and no post-hoc prompt repair or variant sampling.
