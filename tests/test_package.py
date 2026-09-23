import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class PackageTests(unittest.TestCase):
    def test_schema_inspection_never_guesses_target(self):
        data = ROOT / "tests/fixtures/synthetic_train.csv"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "schema.json"
            command = [sys.executable, str(ROOT / "scripts/inspect_training_schema.py"),
                       "--train", str(data), "--output", str(out)]
            subprocess.run(command, check=True, capture_output=True)
            schema = json.loads(out.read_text())
            self.assertEqual(schema["target_resolution"], "unresolved")
            self.assertIsNone(schema["target"])
            self.assertIn("label", schema["columns"])
            self.assertNotIn("class_counts", schema)
            self.assertFalse(schema["test_data_accessed"])
            subprocess.run(command + ["--target", "label"], check=True, capture_output=True)
            self.assertEqual(json.loads(out.read_text())["target"], "label")
            missing = subprocess.run(command + ["--target", "absent"], capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Available columns", missing.stderr)
            self.assertEqual(json.loads(out.read_text())["target"], "label")
            no_target = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_training.py"),
                                        "--train", str(data)], capture_output=True, text=True)
            self.assertNotEqual(no_target.returncode, 0)
            self.assertIn("--target", no_target.stderr)
            bad_diagnosis = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_training.py"),
                                            "--train", str(data), "--target", "absent"],
                                           capture_output=True, text=True)
            self.assertNotEqual(bad_diagnosis.returncode, 0)
            self.assertIn("Available columns", bad_diagnosis.stderr)

    def test_plan_and_diagnosis(self):
        plan=ROOT/"tests/fixtures/example-plan.json"; data=ROOT/"tests/fixtures/synthetic_train.csv"
        subprocess.run([sys.executable,str(ROOT/"scripts/validate_plan.py"),str(plan)],check=True,capture_output=True)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"diagnosis.json"
            subprocess.run([sys.executable,str(ROOT/"scripts/diagnose_training.py"),"--train",str(data),"--target","label","--output",str(out)],check=True,capture_output=True)
            result=json.loads(out.read_text()); self.assertFalse(result["test_data_accessed"]); self.assertEqual(result["shape"],[12,4])

if __name__=="__main__": unittest.main()
