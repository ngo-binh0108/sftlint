"""Find exact and near-duplicate records with MinHash and locality-sensitive hashing."""

from __future__ import annotations

import functools
import hashlib
import random
import re

from .formats import Record

try:  # optional: the same signatures, computed about 20 times faster
    import numpy
except ImportError:
    numpy = None

# Words, plus each CJK character on its own, since those scripts do not separate words with spaces.
_TOKEN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]|\w+")
SHINGLE = 3  # words per shingle; short chat records need short shingles
NUM_PERM = 64
BANDS = 16  # 16 bands of 4 rows: pairs above about 0.5 similarity become candidates


def tokens(rec: Record) -> list[str]:
    """Lower-cased words of the record, leaving out system prompts, which many records share."""
    text = "\n".join(t.text for t in rec.turns if t.role not in ("system", "developer"))
    return _TOKEN.findall(text.lower())


def exact_key(words: list[str]) -> str | None:
    return hashlib.sha1(" ".join(words).encode("utf-8")).hexdigest() if words else None


_M64 = (1 << 64) - 1
# Odd 64-bit constants: one multiplier per position in the shingle, then the splitmix64 finaliser.
_POSITION = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD)[:SHINGLE]
_MIX1, _MIX2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


@functools.lru_cache(maxsize=1 << 20)
def _word_hash(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")


def shingle_hashes(words: list[str], use_numpy: bool = True):
    """One 64-bit hash per distinct run of SHINGLE words. numpy and pure Python give the same values."""
    n = len(words) - SHINGLE + 1
    wh = [_word_hash(w) for w in words]
    if numpy is not None and use_numpy:
        w = numpy.array(wh, dtype=numpy.uint64)
        h = numpy.zeros(n, dtype=numpy.uint64)
        for k, c in enumerate(_POSITION):  # uint64 arithmetic wraps, the same as the & _M64 below
            h += w[k : k + n] * numpy.uint64(c)
        h ^= h >> numpy.uint64(30)
        h *= numpy.uint64(_MIX1)
        h ^= h >> numpy.uint64(27)
        h *= numpy.uint64(_MIX2)
        h ^= h >> numpy.uint64(31)
        return numpy.unique(h)
    p = [[x * c for x in wh[k : k + n]] for k, c in enumerate(_POSITION)]
    hs = [sum(t) & _M64 for t in zip(*p)]
    hs = [((h ^ (h >> 30)) * _MIX1) & _M64 for h in hs]
    hs = [((h ^ (h >> 27)) * _MIX2) & _M64 for h in hs]
    return sorted({h ^ (h >> 31) for h in hs})


class MinHasher:
    """MinHash with XOR masks: the signature holds, for each mask, the smallest shingle hash XOR that mask."""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = 1, use_numpy: bool = True):
        rng = random.Random(seed)
        self.masks = [rng.getrandbits(64) for _ in range(num_perm)]
        self.use_numpy = use_numpy and numpy is not None
        self.mask_array = numpy.array(self.masks, dtype=numpy.uint64) if self.use_numpy else None

    def signature(self, words: list[str]) -> tuple[int, ...] | None:
        if len(words) < SHINGLE:
            return None  # too short to compare by shingles; exact matching still covers it
        hashes = shingle_hashes(words, self.use_numpy)
        if self.use_numpy:
            return tuple(int(v) for v in numpy.bitwise_xor.outer(self.mask_array, hashes).min(axis=1))
        return tuple(min(map(mask.__xor__, hashes)) for mask in self.masks)


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Estimated Jaccard similarity of the two records' shingle sets."""
    return sum(x == y for x, y in zip(a, b)) / len(a)


class Index:
    """Records seen so far, kept so that each new record can be compared with them.

    A threshold of 1 turns off near-duplicate matching and leaves only exact matching.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.hasher = MinHasher()
        self.rows = NUM_PERM // BANDS
        self.exact: dict[str, int] = {}
        self.buckets: dict[tuple, list[int]] = {}
        self.signatures: dict[int, tuple[int, ...]] = {}

    def _bands(self, sig):
        return [(band, sig[band * self.rows : (band + 1) * self.rows]) for band in range(BANDS)]

    def match(self, rid: int, words: list[str]) -> tuple[int, float] | None:
        """(earlier record id, similarity) if this record repeats one already seen; otherwise remember it and return None."""
        key = exact_key(words)
        if key is None:
            return None
        if key in self.exact:
            return self.exact[key], 1.0
        sig = self.hasher.signature(words) if self.threshold < 1 else None
        if sig is not None:
            candidates = {i for band in self._bands(sig) for i in self.buckets.get(band, ())}
            best = max(((similarity(sig, self.signatures[i]), -i) for i in candidates), default=None)
            if best and best[0] >= self.threshold:
                self.exact[key] = -best[1]  # a later exact copy points at the same original
                return -best[1], best[0]
        self.exact[key] = rid
        if sig is not None:
            self.signatures[rid] = sig
            for band in self._bands(sig):
                self.buckets.setdefault(band, []).append(rid)
        return None
