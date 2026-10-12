"""STEP264: verifiable reward for the hand-level collusion judge.  PURE STANDARD LIBRARY.

This file runs on the GPU box inside the GRPO loop, so it imports nothing outside the standard
library and has no project paths in it.

parse(completion) -> {'verdict', 'confidence', 'format_ok', 'think', 'answer'}
    tolerant of a <think>...</think> block and of surrounding prose: the LAST well-formed
    VERDICT line and the LAST well-formed CONFIDENCE line win, and they must both be outside the
    thinking block (a model that only names a verdict inside <think> has not answered).

reward(completion, label, family, n_think_tokens=None) -> dict with the components
    brier          1 - (confidence/100 - label)**2                     in [0, 1]
    family_bonus   +.15 if label == 1 and verdict == family.upper()
    consistency    -.2 if (verdict == 'NONE') != (confidence < 50)
    length_penalty -.1 per 250 thinking tokens beyond 700 (the caller supplies the count)
    format         if either answer line is missing or malformed the TOTAL is 0 and every
                   component is reported as 0 with format_ok False

Self-test:  python reward.py --selftest
"""
from __future__ import annotations

import re
import sys

VERDICTS = ('NONE', 'DIRECTED_TRANSFER', 'SOFT_PLAY', 'COORDINATED_ISOLATION')
FAMILY_TO_VERDICT = {'directed_transfer': 'DIRECTED_TRANSFER', 'soft_play': 'SOFT_PLAY',
                     'coordinated_isolation': 'COORDINATED_ISOLATION', 'none': 'NONE',
                     'NONE': 'NONE'}

_THINK = re.compile(r'<think>.*?(?:</think>|\Z)', re.DOTALL | re.IGNORECASE)
# STEP264 r3: a pre-rendered prompt can already end with the opening <think> tag, so the
# completion carries only the CLOSING tag.  Used only when no opening tag is present.
_THINK_CLOSE = re.compile(r'</think>', re.IGNORECASE)
_VERDICT = re.compile(r'^[^\S\n]*\**VERDICT\**[^\S\n]*:[^\S\n]*\**([A-Za-z_]+)\**[^\S\n]*\.?'
                      r'[^\S\n]*$', re.MULTILINE | re.IGNORECASE)
_CONF = re.compile(r'^[^\S\n]*\**CONFIDENCE\**[^\S\n]*:[^\S\n]*(\d{1,3})[^\S\n]*%?[^\S\n]*\.?'
                   r'[^\S\n]*$', re.MULTILINE | re.IGNORECASE)

import os as _os
FAMILY_BONUS = .15
CONSISTENCY_PENALTY = -.2
# STEP264 r1e (2026-09-18, Amin: 'Go'): length-penalty constants overridable by environment; defaults = the delivered v1
LEN_FREE = int(_os.environ.get('STEP264_LEN_FREE', 700))
LEN_STEP = int(_os.environ.get('STEP264_LEN_STEP', 250))
LEN_RATE = float(_os.environ.get('STEP264_LEN_RATE', -.1))


def split_think(completion: str) -> tuple[str, str]:
    """Return (thinking, answer_region).  An unterminated <think> swallows the rest."""
    if completion is None:
        return '', ''
    m = _THINK.search(completion)
    if not m:
        # No opening tag: the prompt supplied it.  Everything up to and including the first
        # </think> is thinking; the rest is the answer region.  (STEP264 r3)
        c = _THINK_CLOSE.search(completion)
        if c:
            return completion[:c.end()], completion[c.end():]
        return '', completion
    return completion[m.start():m.end()], completion[:m.start()] + completion[m.end():]


def parse(completion: str) -> dict:
    think, answer = split_think(completion)
    vs = _VERDICT.findall(answer)
    cs = _CONF.findall(answer)
    verdict = None
    if vs:
        cand = vs[-1].strip().upper()
        if cand in VERDICTS:
            verdict = cand
    confidence = None
    if cs:
        val = int(cs[-1])
        if 0 <= val <= 100:
            confidence = val
    return {'verdict': verdict, 'confidence': confidence,
            'format_ok': verdict is not None and confidence is not None,
            'think': think, 'answer': answer}


def reward(completion: str, label, family=None, n_think_tokens=None) -> dict:
    p = parse(completion)
    out = {'verdict': p['verdict'], 'confidence': p['confidence'], 'format_ok': p['format_ok'],
           'brier': 0.0, 'family_bonus': 0.0, 'consistency': 0.0, 'length_penalty': 0.0,
           'total': 0.0}
    if not p['format_ok']:
        return out
    y = float(label)
    conf = p['confidence'] / 100.0
    out['brier'] = 1.0 - (conf - y) ** 2
    if y == 1.0 and family is not None:
        want = FAMILY_TO_VERDICT.get(str(family), str(family).upper())
        if p['verdict'] == want:
            out['family_bonus'] = FAMILY_BONUS
    if (p['verdict'] == 'NONE') != (p['confidence'] < 50):
        out['consistency'] = CONSISTENCY_PENALTY
    if n_think_tokens is not None and n_think_tokens > LEN_FREE:
        over = int(n_think_tokens) - LEN_FREE
        out['length_penalty'] = LEN_RATE * ((over + LEN_STEP - 1) // LEN_STEP)
    out['total'] = (out['brier'] + out['family_bonus'] + out['consistency']
                    + out['length_penalty'])
    return out


# ---------------------------------------------------------------------------- self-test


def _selftest() -> None:
    ok = 'VERDICT: NONE\nCONFIDENCE: 10'
    r = reward(ok, 0)
    assert r['format_ok'] is True, '1 well-formed answer parses'
    assert abs(r['brier'] - (1 - .01)) < 1e-9, '2 brier on a confident NONE'
    assert r['consistency'] == 0.0, '3 NONE with conf < 50 is consistent'
    assert abs(r['total'] - .99) < 1e-9, '4 total equals brier here'

    r = reward('VERDICT: NONE\nCONFIDENCE: 80', 0)
    assert r['consistency'] == -.2, '5 NONE with conf >= 50 is inconsistent'
    assert abs(r['total'] - (1 - .64 - .2)) < 1e-9, '6 penalty enters the total'

    r = reward('VERDICT: SOFT_PLAY\nCONFIDENCE: 90', 1, 'soft_play')
    assert r['family_bonus'] == .15, '7 family bonus on the right family'
    assert abs(r['total'] - (1 - .01 + .15)) < 1e-9, '8 bonus enters the total'

    r = reward('VERDICT: DIRECTED_TRANSFER\nCONFIDENCE: 90', 1, 'soft_play')
    assert r['family_bonus'] == 0.0, '9 no bonus for the wrong family'

    r = reward('VERDICT: SOFT_PLAY\nCONFIDENCE: 90', 0, None)
    assert r['family_bonus'] == 0.0, '10 no family bonus on a negative'
    assert r['consistency'] == 0.0, '11 non-NONE with conf >= 50 is consistent'

    r = reward('VERDICT: SOFT_PLAY\nCONFIDENCE: 20', 1, 'soft_play')
    assert r['consistency'] == -.2, '12 non-NONE with conf < 50 is inconsistent'

    assert reward('I think it is fine.', 0)['total'] == 0.0, '13 no answer lines -> 0'
    assert reward('VERDICT: NONE', 0)['total'] == 0.0, '14 missing CONFIDENCE -> 0'
    assert reward('CONFIDENCE: 10', 0)['total'] == 0.0, '15 missing VERDICT -> 0'
    assert reward('VERDICT: MAYBE\nCONFIDENCE: 10', 0)['total'] == 0.0, '16 unknown verdict -> 0'
    assert reward('VERDICT: NONE\nCONFIDENCE: 140', 0)['total'] == 0.0, '17 conf out of range -> 0'
    assert reward(None, 0)['total'] == 0.0, '18 None completion -> 0'

    t = '<think>\nVERDICT: SOFT_PLAY\nCONFIDENCE: 99\nnow the answer\n</think>\n' + ok
    r = reward(t, 0)
    assert r['verdict'] == 'NONE' and r['confidence'] == 10, '19 thinking block is ignored'
    t2 = '<think>VERDICT: NONE\nCONFIDENCE: 10</think>\nI am not sure.'
    assert reward(t2, 0)['total'] == 0.0, '20 an answer only inside <think> does not count'
    t3 = '<think>unterminated reasoning\nVERDICT: NONE\nCONFIDENCE: 10'
    assert reward(t3, 0)['total'] == 0.0, '21 unterminated <think> swallows the answer'

    r = reward(ok, 0, None, 700)
    assert r['length_penalty'] == 0.0, '22 no penalty at exactly 700 thinking tokens'
    r = reward(ok, 0, None, 701)
    assert abs(r['length_penalty'] + .1) < 1e-9, '23 first bracket costs .1'
    r = reward(ok, 0, None, 950)
    assert abs(r['length_penalty'] + .1) < 1e-9, '24 950 is still the first bracket'
    r = reward(ok, 0, None, 951)
    assert abs(r['length_penalty'] + .2) < 1e-9, '25 951 enters the second bracket'
    r = reward(ok, 0, None, 1700)
    assert abs(r['length_penalty'] + .4) < 1e-9, '26 1700 is four brackets'

    p = parse('prose\nverdict : soft_play\nConfidence: 73 %\ntrailing prose')
    assert p['verdict'] == 'SOFT_PLAY' and p['confidence'] == 73, '27 case and spacing tolerated'
    p = parse('VERDICT: NONE\nCONFIDENCE: 5\nVERDICT: SOFT_PLAY\nCONFIDENCE: 95')
    assert p['verdict'] == 'SOFT_PLAY' and p['confidence'] == 95, '28 the last answer wins'
    p = parse('The VERDICT: NONE is my call.')
    assert p['verdict'] is None, '29 an inline mention is not an answer line'

    r = reward('VERDICT: COORDINATED_ISOLATION\nCONFIDENCE: 100', 1, 'coordinated_isolation', 1200)
    assert abs(r['total'] - (1.0 + .15 - .2)) < 1e-9, '30 all components combine'
    print('reward.py selftest: 30 assertions passed')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
