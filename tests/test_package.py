import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class PackageTests(unittest.TestCase):
    def test_plan_and_diagnosis(self):
        plan=ROOT/"tests/fixtures/example-plan.json"; data=ROOT/"tests/fixtures/synthetic_train.csv"
        subprocess.run([sys.executable,str(ROOT/"scripts/validate_plan.py"),str(plan)],check=True,capture_output=True)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"diagnosis.json"
            subprocess.run([sys.executable,str(ROOT/"scripts/diagnose_training.py"),"--train",str(data),"--target","label","--output",str(out)],check=True,capture_output=True)
            result=json.loads(out.read_text()); self.assertFalse(result["test_data_accessed"]); self.assertEqual(result["shape"],[12,4])

if __name__=="__main__": unittest.main()
