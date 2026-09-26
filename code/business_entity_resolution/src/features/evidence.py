"""Alternate evidence views using Python Unicode metadata and supplied records only.

Romanization is deliberately approximate, not a language or transliteration model.
It never replaces primary text or indexed digits. Ambiguous alphanumeric address
identifiers remain available through the legacy numeric features.
"""

import re
import unicodedata
from functools import lru_cache

from ..preprocessing.normalize import accent_fold

VOWEL_SIGNS = {'AA': 'a', 'I': 'i', 'II': 'i', 'U': 'u', 'UU': 'u', 'VOCALIC R': 'ri', 'E': 'e', 'AI': 'ai', 'O': 'o',
               'AU': 'au', 'CANDRA E': 'e', 'CANDRA O': 'o', 'SHORT E': 'e', 'SHORT O': 'o'}
CONS_FIX = {'tt': 't', 'dd': 'd', 'nn': 'n', 'ss': 'sh', 'll': 'l', 'rr': 'r', 'tth': 'th', 'ddh': 'dh', 'lll': 'l', 'nnn': 'n'}
SCRIPTS = ('DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU', 'KANNADA', 'MALAYALAM')


@lru_cache(maxsize=20_000)
def romanize(text):
    out, pending = [], False
    for ch in unicodedata.normalize('NFC', text):
        name = unicodedata.name(ch, '')
        script = next((s for s in SCRIPTS if name.startswith(s + ' ')), None)
        if not script:
            if pending:
                if ch.isalnum():
                    out.append('a')
                pending = False
            out.append(ch)
            continue
        rest = name[len(script) + 1:]
        if rest.startswith('LETTER '):
            letter = rest[7:].lower().split()[0]
            if pending:
                out.append('a')
            if letter[-1] == 'a' and len(letter) > 1 and letter not in ('aa', 'ai', 'au'):
                base = letter[:-1]
                out.append(CONS_FIX.get(base, base))
                pending = True
            else:
                out.append(VOWEL_SIGNS.get(letter.upper(), letter))
                pending = False
        elif rest.startswith('VOWEL SIGN '):
            out.append(VOWEL_SIGNS.get(rest[11:], rest[11:].lower()))
            pending = False
        elif rest in ('SIGN VIRAMA', 'SIGN HALANT'):
            pending = False
        elif rest in ('SIGN ANUSVARA', 'SIGN CANDRABINDU'):
            if pending:
                out.append('a')
                pending = False
            out.append('n')
        elif rest == 'SIGN VISARGA':
            out.append('h')
        elif rest.startswith('DIGIT '):
            out.append(str(unicodedata.digit(ch)))
        # nukta and other signs are dropped
    return ''.join(out)  # word-final inherent vowel omitted (schwa deletion)


LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a'})


@lru_cache(maxsize=20_000)
def phonetic(text):
    tokens = []
    for t in romanize(accent_fold(text)).split():
        if re.search('[a-z]', t) and re.search('[0-9]', t):
            t = t.translate(LEET)
        t = t.replace('ph', 'f').replace('x', 'ks').replace('q', 'k').replace('w', 'v').replace('z', 'j')
        t = re.sub('c(?!h)', 'k', t).replace('y', 'i')
        t = re.sub('([bcdfghjklmnpqrstv])h', r'\1', t)
        t = re.sub('ee|ii', 'i', t).replace('oo', 'u').replace('aa', 'a')
        t = re.sub(r'(.)\1+', r'\1', t)
        tokens.append(t)
    return ' '.join(tokens)


def grams(s):
    s = s.replace(' ', '')
    return {s[i:i + 3] for i in range(max(0, len(s) - 2))}


TLD = {'com', 'net', 'org', 'in', 'co', 'www', 'biz', 'info', 'us'}


def nums(text):
    return [x.lstrip('0') or '0' for x in re.findall(r'\d+', text)]


def numeric_relation(a, b):
    A, B = set(a), set(b)
    if not A or not B:
        return 0, 0, 0, len(A), len(B)
    left_only, right_only = A - B, B - A
    containment = any(len(x) >= 2 and len(y) >= 2 and x != y and (x in y or y in x) for x in left_only for y in right_only)
    substitution = any(len(x) == len(y) >= 2 and sum(c != d for c, d in zip(x, y)) == 1 for x in left_only for y in right_only)
    disjoint = not (A & B) and not containment
    return int(containment), int(substitution), int(disjoint), len(left_only), len(right_only)
