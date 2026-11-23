"""Cross-split near-duplicate candidate generation and exact verification."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

TOKEN = re.compile(r"\w+", re.UNICODE)
PRIME = (1 << 61) - 1


@dataclass(frozen=True)
class Record:
    id: str
    split: str
    text: str


def shingles(text: str, width: int = 3) -> frozenset[str]:
    """Lowercase Unicode word shingles; short documents use their whole token sequence."""
    if width < 1:
        raise ValueError("shingle width must be positive")
    words = TOKEN.findall(text.casefold())
    if not words:
        raise ValueError("each document must contain at least one word")
    if len(words) < width:
        return frozenset((" ".join(words),))
    return frozenset(" ".join(words[i:i + width]) for i in range(len(words) - width + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        raise ValueError("empty shingle set")
    return len(a & b) / len(a | b)


def _signature(shingle_set: frozenset[str], n: int, seed: int) -> tuple[int, ...]:
    """Stable 61-bit hashes and deterministic seeded hash permutations."""
    values = [int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") % PRIME
              for s in shingle_set]
    result = []
    for i in range(n):
        coeff = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        a = 1 + int.from_bytes(coeff[:8], "big") % (PRIME - 1)
        b = int.from_bytes(coeff[8:16], "big") % PRIME
        result.append(min((a * value + b) % PRIME for value in values))
    return tuple(result)


def audit(records: list[Record], threshold: float = 0.8, width: int = 3,
          mode: str = "exact", permutations: int = 128, bands: int = 32,
          seed: int = 42) -> dict:
    """Return only cross-split matches; LSH is approximate and may miss matches."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    if mode not in {"exact", "lsh"}:
        raise ValueError("mode must be exact or lsh")
    if permutations < 1 or bands < 1 or permutations % bands:
        raise ValueError("permutations and bands must be positive and divisible")
    ids = [r.id for r in records]
    if any(not r.id.strip() or not r.split.strip() for r in records):
        raise ValueError("id and split must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be globally unique")
    sets = [shingles(r.text, width) for r in records]
    total = sum(1 for i in range(len(records)) for j in range(i + 1, len(records))
                if records[i].split != records[j].split)
    if mode == "exact":
        candidates = {(i, j) for i in range(len(records)) for j in range(i + 1, len(records))
                      if records[i].split != records[j].split}
    else:
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        rows = permutations // bands
        for i, words in enumerate(sets):
            sig = _signature(words, permutations, seed)
            for band in range(bands):
                buckets.setdefault((band, sig[band * rows:(band + 1) * rows]), []).append(i)
        candidates = set()
        for members in buckets.values():
            for k, i in enumerate(members):
                for j in members[k + 1:]:
                    if records[i].split != records[j].split:
                        candidates.add((i, j))
    matches = []
    for i, j in sorted(candidates):
        score = jaccard(sets[i], sets[j])
        if score >= threshold:
            matches.append({"left_id": records[i].id, "left_split": records[i].split,
                            "right_id": records[j].id, "right_split": records[j].split,
                            "jaccard": score})
    return {"mode": mode, "threshold": threshold, "shingle_width": width,
            "records": len(records), "cross_split_pairs": total,
            "candidate_pairs": len(candidates), "matches": matches,
            "lsh": {"permutations": permutations, "bands": bands, "seed": seed}
                   if mode == "lsh" else None}
