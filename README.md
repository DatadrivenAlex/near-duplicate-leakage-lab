# Near-Duplicate Leakage Lab

An offline Python audit for detecting near-duplicate text across ML dataset splits. Exact word-shingle Jaccard is the reference; MinHash locality-sensitive hashing (LSH) is an approximate candidate generator whose candidates are verified using exact Jaccard. Reports contain IDs, split names and similarity scores, not source text.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
leakage-audit examples/synthetic.jsonl --threshold 0.8
leakage-audit examples/synthetic.jsonl --mode lsh --permutations 128 --bands 32
python -m unittest discover -s tests -v
```

Each JSONL row needs string `id`, `split`, and `text`; IDs must be globally unique. Tokenization uses Unicode word tokens and case folding, and shingles are word sequences (default width 3). Short texts use their full token sequence as one shingle. Empty-token texts are rejected. Similarity is intersection over union; matches are at or above the threshold. All split-name pairs are compared, never records in the same split. Blank lines are ignored. Use `--fail-on-match` to exit 1 on a detected match, or 0 when none are detected; malformed input exits 2.

## Method and tradeoffs

- `exact` compares all cross-split pairs. It gives a complete answer for this specific tokenization, shingling and threshold, with quadratic pair costs.
- `lsh` uses deterministic seeded MinHash signatures, splits them into bands, and compares cross-split pairs that collide in at least one band. A candidate match is always checked using exact Jaccard; LSH can miss similar pairs, so a clean LSH report is not a guarantee of no leakage. Tune parameters and compare with exact mode on representative subsets before using it.
- The reported cross-split total is still computed quadratically; this is a teaching and small-data audit, not a distributed solution. Large buckets can also be quadratic.
- Exact lexical shingles miss semantic paraphrases, translations, identifier changes, and leakage not visible in text. A match is a review lead, not proof of improper leakage: templates and legitimate repeated text can trigger it. The example is wholly synthetic; no trained model or real patient data is included.

Output JSON lists counts, settings and verified pairs; it intentionally excludes the source text, but record IDs and scores may still be sensitive. Keep input and reports in an appropriately protected location. Review flagged examples and dataset provenance before changing splits. Python 3.11+; runtime uses only the standard library.

## License

MIT. See `LICENSE`.
