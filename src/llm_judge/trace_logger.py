"""STEP264: append-only rollout logger for the GRPO run.  PURE STANDARD LIBRARY (gzip + json).

Runs on the GPU box.  One JSON object per rollout, one line, written to a gzip file opened in
APPEND mode, so a crashed run loses at most the last line and a resumed run keeps the old file.
Log EVERY rollout: the traces are the raw material for the later 0.6B student (see
select_best_traces.py) and for any post-hoc analysis of what the policy actually learned.

Row schema (keys always present, value may be null):
  run_id str            the whole training run
  step int              optimiser step
  checkpoint str        checkpoint tag the rollout was sampled from, e.g. 'step_0150' or 'base'
  uid str               the opaque prompt id from train.jsonl / heldout.jsonl
  fold int              step232 table fold of the row
  source_class str      POS / NEG_RANDOM / NEG_HARD / NEG_WITHIN / AUDIT
  label int             0 / 1
  family str            NONE / directed_transfer / soft_play / coordinated_isolation
  sample_idx int        index inside the GRPO group
  temperature float     sampling temperature
  prompt_sha1 str       sha1 of the prompt text (the prompt itself only if store_prompt=True)
  prompt str|null       full prompt, only when the logger was built with store_prompt=True
  completion str        FULL completion text INCLUDING the thinking
  verdict str|null      parsed verdict
  confidence int|null   parsed confidence
  format_ok bool
  brier / family_bonus / consistency / length_penalty / total   float  (reward components)
  n_think_tokens int|null
  n_answer_tokens int|null
  mean_token_logprob float|null
  verdict_logprobs dict|null   {'NONE': lp, 'DIRECTED_TRANSFER': lp, 'SOFT_PLAY': lp,
                                'COORDINATED_ISOLATION': lp} at the first answer token
  finish_reason str|null       'stop' / 'length' / 'error' / whatever the sampler reports
  wall_s float|null            seconds spent generating this rollout
  logged_at float              unix time the row was written

Usage
    from trace_logger import TraceLogger
    with TraceLogger('traces/run7.jsonl.gz', run_id='run7') as tl:
        tl.log(step=12, checkpoint='step_0000', uid=u, fold=0, source_class='POS', label=1,
               family='soft_play', sample_idx=3, temperature=1.0, prompt=p, completion=c,
               n_think_tokens=512, n_answer_tokens=14, finish_reason='stop', wall_s=2.3)

Self-test:  python trace_logger.py --selftest
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as RW                                                          # noqa: E402

VERDICTS = RW.VERDICTS
FIELDS = ('run_id', 'step', 'checkpoint', 'uid', 'fold', 'source_class', 'label', 'family',
          'sample_idx', 'temperature', 'prompt_sha1', 'prompt', 'completion', 'verdict',
          'confidence', 'format_ok', 'brier', 'family_bonus', 'consistency', 'length_penalty',
          'total', 'n_think_tokens', 'n_answer_tokens', 'mean_token_logprob', 'verdict_logprobs',
          'finish_reason', 'wall_s', 'logged_at')


def sha1(text: str) -> str:
    return hashlib.sha1((text or '').encode('utf-8')).hexdigest()


def approx_tokens(text: str) -> int:
    """Fallback only.  Pass the real counts from the tokenizer whenever you have them."""
    return int(round(len(text or '') / 3.6))


class TraceLogger:
    def __init__(self, path: str, run_id: str, store_prompt: bool = False,
                 flush_every: int = 1) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        self.run_id = run_id
        self.store_prompt = store_prompt
        self.flush_every = max(1, int(flush_every))
        self._n = 0
        self._fh = gzip.open(path, 'at', encoding='utf-8', newline='\n')

    # ------------------------------------------------------------------
    def log(self, *, step, checkpoint, uid, fold, source_class, label, family, sample_idx,
            temperature, completion, prompt=None, prompt_sha1=None, reward=None,
            n_think_tokens=None, n_answer_tokens=None, mean_token_logprob=None,
            verdict_logprobs=None, finish_reason=None, wall_s=None) -> dict:
        if reward is None:
            reward = RW.reward(completion, label, family, n_think_tokens)
        p = {'verdict': reward.get('verdict'), 'confidence': reward.get('confidence')}
        if 'verdict' not in reward:
            p = RW.parse(completion)
        if n_think_tokens is None or n_answer_tokens is None:
            think, answer = RW.split_think(completion)
            n_think_tokens = approx_tokens(think) if n_think_tokens is None else n_think_tokens
            n_answer_tokens = approx_tokens(answer) if n_answer_tokens is None else n_answer_tokens
        if verdict_logprobs is not None:
            verdict_logprobs = {v: (None if verdict_logprobs.get(v) is None
                                    else float(verdict_logprobs[v])) for v in VERDICTS}
        row = {
            'run_id': self.run_id, 'step': None if step is None else int(step),
            'checkpoint': checkpoint, 'uid': uid,
            'fold': None if fold is None else int(fold), 'source_class': source_class,
            'label': None if label is None else int(label), 'family': family,
            'sample_idx': None if sample_idx is None else int(sample_idx),
            'temperature': None if temperature is None else float(temperature),
            'prompt_sha1': prompt_sha1 or (sha1(prompt) if prompt is not None else None),
            'prompt': prompt if self.store_prompt else None,
            'completion': completion,
            'verdict': p.get('verdict'), 'confidence': p.get('confidence'),
            'format_ok': bool(reward.get('format_ok', False)),
            'brier': float(reward.get('brier', 0.0)),
            'family_bonus': float(reward.get('family_bonus', 0.0)),
            'consistency': float(reward.get('consistency', 0.0)),
            'length_penalty': float(reward.get('length_penalty', 0.0)),
            'total': float(reward.get('total', 0.0)),
            'n_think_tokens': None if n_think_tokens is None else int(n_think_tokens),
            'n_answer_tokens': None if n_answer_tokens is None else int(n_answer_tokens),
            'mean_token_logprob': (None if mean_token_logprob is None
                                   else float(mean_token_logprob)),
            'verdict_logprobs': verdict_logprobs,
            'finish_reason': finish_reason,
            'wall_s': None if wall_s is None else float(wall_s),
            'logged_at': time.time(),
        }
        assert set(row) == set(FIELDS), 'trace schema drift'
        self._fh.write(json.dumps(row, ensure_ascii=True) + '\n')
        self._n += 1
        if self._n % self.flush_every == 0:
            self._fh.flush()
        return row

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_traces(paths):
    """Yield every row of one path, a list of paths, or a directory of *.jsonl.gz."""
    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = [os.path.join(paths, f) for f in sorted(os.listdir(paths))
                     if f.endswith('.jsonl.gz')]
        else:
            paths = [paths]
    for p in sorted(paths):
        with gzip.open(p, 'rt', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ---------------------------------------------------------------------------- self-test


def _selftest() -> None:
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix='step264_trace_')
    try:
        p = os.path.join(d, 'sub', 'run.jsonl.gz')
        c = 'reasoning about the flop\nVERDICT: SOFT_PLAY\nCONFIDENCE: 88'
        with TraceLogger(p, run_id='t1', store_prompt=True) as tl:
            r = tl.log(step=1, checkpoint='base', uid='abc', fold=0, source_class='POS', label=1,
                       family='soft_play', sample_idx=0, temperature=1.0, prompt='PROMPT',
                       completion=c, n_think_tokens=800, n_answer_tokens=12,
                       mean_token_logprob=-0.4,
                       verdict_logprobs={'NONE': -3.0, 'SOFT_PLAY': -0.1}, finish_reason='stop',
                       wall_s=1.5)
        assert r['family_bonus'] == .15, '1 reward computed from the completion'
        assert abs(r['length_penalty'] + .1) < 1e-9, '2 length penalty applied'
        assert r['prompt_sha1'] == sha1('PROMPT'), '3 prompt hashed'
        assert r['verdict_logprobs']['COORDINATED_ISOLATION'] is None, '4 all four keys present'
        rows = list(read_traces(p))
        assert len(rows) == 1 and rows[0]['uid'] == 'abc', '5 one row round-trips'
        assert set(rows[0]) == set(FIELDS), '6 schema stable through the file'
        # append mode: a second logger must not truncate
        with TraceLogger(p, run_id='t1') as tl:
            tl.log(step=2, checkpoint='step_0050', uid='def', fold=1, source_class='NEG_HARD',
                   label=0, family='NONE', sample_idx=1, temperature=.7,
                   completion='VERDICT: NONE\nCONFIDENCE: 5')
        rows = list(read_traces(p))
        assert len(rows) == 2, '7 append kept the earlier row'
        assert rows[1]['prompt'] is None, '8 prompt omitted when store_prompt is False'
        assert rows[1]['total'] > .99, '9 a confident correct NONE scores near 1'
        assert len(list(read_traces(os.path.dirname(p)))) == 2, '7b a directory reads its files'
        bad = TraceLogger(p, run_id='t1')
        rb = bad.log(step=3, checkpoint='base', uid='ghi', fold=2, source_class='NEG_RANDOM',
                     label=0, family='NONE', sample_idx=0, temperature=1.0,
                     completion='no answer here', finish_reason='length')
        bad.close()
        assert rb['total'] == 0.0 and rb['format_ok'] is False, '10 malformed rollout scores 0'
        assert rb['n_think_tokens'] is not None, '11 token counts estimated when not supplied'
        print('trace_logger.py selftest: 11 assertions passed')
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
