import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from leakage_audit.cli import main
from leakage_audit.core import Record, audit, jaccard, shingles


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.records = [Record("a", "train", "One two three four five"),
                        Record("b", "test", "ONE two three four five"),
                        Record("c", "test", "Unrelated different sample text here"),
                        Record("d", "train", "One two three four five")]

    def test_casefold_and_punctuation(self):
        self.assertEqual(shingles("HELLO, world!", 2), shingles("hello world", 2))

    def test_unicode(self):
        self.assertEqual(shingles("Straße IST gut", 2), shingles("STRASSE ist gut", 2))

    def test_short_text(self):
        self.assertEqual(shingles("Hello world", 3), frozenset({"hello world"}))

    def test_empty_and_width_rejected(self):
        for text, width in [("!!!", 3), ("hi", 0)]:
            with self.assertRaises(ValueError):
                shingles(text, width)

    def test_jaccard_known(self):
        self.assertAlmostEqual(jaccard(frozenset("abc"), frozenset("bcd")), 0.5)

    def test_exact_cross_split_only(self):
        report = audit(self.records)
        self.assertEqual(report["cross_split_pairs"], 4)
        self.assertEqual(report["candidate_pairs"], 4)
        self.assertEqual([(p["left_id"], p["right_id"]) for p in report["matches"]],
                         [("a", "b"), ("b", "d")])
        self.assertTrue(all(p["jaccard"] == 1 for p in report["matches"]))
        self.assertNotIn("text", json.dumps(report))

    def test_lsh_identical_and_reproducible(self):
        report = audit(self.records, mode="lsh", permutations=64, bands=16)
        self.assertEqual(report, audit(self.records, mode="lsh", permutations=64, bands=16))
        self.assertEqual({(p["left_id"], p["right_id"]) for p in report["matches"]},
                         {("a", "b"), ("b", "d")})

    def test_lsh_results_verified_exact(self):
        exact = audit(self.records, threshold=0.5)
        approx = audit(self.records, threshold=0.5, mode="lsh", permutations=32, bands=8)
        self.assertTrue({(m["left_id"], m["right_id"]) for m in approx["matches"]}
                        <= {(m["left_id"], m["right_id"]) for m in exact["matches"]})

    def test_threshold_boundary(self):
        records = [Record("a", "train", "a b c"), Record("b", "test", "b c d")]
        self.assertEqual(len(audit(records, 0.5, width=1)["matches"]), 1)
        self.assertEqual(len(audit(records, 0.51, width=1)["matches"]), 0)

    def test_bad_input_rejected(self):
        bad = [self.records[0], Record("a", "test", "text")]
        for call in [lambda: audit(bad), lambda: audit(self.records, threshold=float("nan")),
                     lambda: audit(self.records, mode="lsh", permutations=10, bands=3),
                     lambda: audit([Record("a", "", "words")])]:
            with self.assertRaises(ValueError):
                call()

    def test_cli_gate_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('\n'.join(json.dumps(r.__dict__) for r in self.records), encoding="utf8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = main([str(path), "--fail-on-match"])
            self.assertEqual(result, 1)
            self.assertEqual(len(json.loads(out.getvalue())["matches"]), 2)

    def test_cli_bad_rows_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"id":"a","split":1,"text":"x"}\n', encoding="utf8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(path)]), 2)
                self.assertEqual(main([str(path) + "absent"]), 2)


if __name__ == "__main__":
    unittest.main()
