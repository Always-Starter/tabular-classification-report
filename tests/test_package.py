import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class PackageTests(unittest.TestCase):
    def test_schema_inspection_suggests_without_claiming_certainty(self):
        data = ROOT / "tests/fixtures/synthetic_train.csv"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "schema.json"
            command = [sys.executable, str(ROOT / "scripts/inspect_training_schema.py"),
                       "--train", str(data), "--output", str(out)]
            subprocess.run(command, check=True, capture_output=True)
            schema = json.loads(out.read_text())
            self.assertEqual(schema["target_resolution"], "ai_inferred")
            self.assertEqual(schema["target"], "label")
            self.assertIn("not confirmation", schema["target_inference_note"])
            selected = next(item for item in schema["target_candidates"] if item["column"] == "label")
            self.assertEqual(selected["name_evidence"], "conventional_target_name")
            self.assertIn("binary_cardinality", selected["inference_reasons"])
            self.assertEqual(schema["target_inference"]["selected_column"], "label")
            self.assertFalse(schema["target_inference"]["semantics_confirmed"])
            self.assertIn("label", schema["columns"])
            self.assertNotIn("class_counts", schema)
            self.assertFalse(schema["test_data_accessed"])
            subprocess.run(command + ["--target", "label"], check=True, capture_output=True)
            explicit = json.loads(out.read_text())
            self.assertEqual(explicit["target"], "label")
            self.assertEqual(explicit["target_resolution"], "explicitly_supplied")
            subprocess.run(command + ["--target", "category"], check=True, capture_output=True)
            self.assertEqual(json.loads(out.read_text())["target"], "category")
            missing = subprocess.run(command + ["--target", "absent"], capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Available columns", missing.stderr)
            self.assertEqual(json.loads(out.read_text())["target"], "category")
            no_target = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_training.py"),
                                        "--train", str(data)], capture_output=True, text=True)
            self.assertNotEqual(no_target.returncode, 0)
            self.assertIn("--target", no_target.stderr)
            bad_diagnosis = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_training.py"),
                                            "--train", str(data), "--target", "absent"],
                                           capture_output=True, text=True)
            self.assertNotEqual(bad_diagnosis.returncode, 0)
            self.assertIn("Available columns", bad_diagnosis.stderr)

    def test_schema_inference_ranks_ambiguity_and_stops_only_without_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "train.csv"
            cases = [
                ("feature,label,outcome\n1,yes,no\n2,no,yes\n3,yes,no\n4,no,yes\n", 2, "ai_inferred", "label"),
                ("feature,flag\n1,yes\n2,no\n3,yes\n4,no\n", 1, "ai_inferred", "flag"),
                ("feature,prior_label\n1,yes\n2,no\n3,yes\n4,no\n", 1, "ai_inferred", "prior_label"),
                ("feature,label\n1,a\n2,b\n3,c\n4,d\n", 0, "unresolved", None),
            ]
            for contents, count, resolution, target in cases:
                with self.subTest(contents=contents.splitlines()[0]):
                    data.write_text(contents, encoding="utf-8")
                    result = subprocess.run([sys.executable, str(ROOT / "scripts/inspect_training_schema.py"),
                                             "--train", str(data)], check=True, capture_output=True, text=True)
                    schema = json.loads(result.stdout)
                    self.assertEqual(schema["target_resolution"], resolution)
                    self.assertEqual(schema["target"], target)
                    self.assertEqual(len(schema["target_candidates"]), count)
                    if count == 2:
                        self.assertEqual(schema["target_inference"]["equally_scored_candidates"],
                                         ["label", "outcome"])
                        self.assertIn("lexical", schema["target_inference"]["tie_breaker"])

    def test_plan_and_diagnosis(self):
        plan=ROOT/"tests/fixtures/example-plan.json"; data=ROOT/"tests/fixtures/synthetic_train.csv"
        subprocess.run([sys.executable,str(ROOT/"scripts/validate_plan.py"),str(plan)],check=True,capture_output=True)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"diagnosis.json"
            subprocess.run([sys.executable,str(ROOT/"scripts/diagnose_training.py"),"--train",str(data),"--target","label","--output",str(out)],check=True,capture_output=True)
            result=json.loads(out.read_text()); self.assertFalse(result["test_data_accessed"]); self.assertEqual(result["shape"],[12,4])

if __name__=="__main__": unittest.main()
