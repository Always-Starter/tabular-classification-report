#!/usr/bin/env python3
"""Create a synthetic demonstration report; never accepts a user's data path."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_json, sha, write_json
from diagnose_training import diagnose
from run_nested_cv import run
from generate_report import generate


def demo(output_dir):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty demonstration directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    x = rng.normal(size=(150, 2))
    d = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "category": np.where(x[:, 1] > 0, "a", "b"),
                      "label": np.where(x[:, 0] + .4 * x[:, 1] > 0, "yes", "no")})
    d.loc[::10, "x1"] = np.nan
    train = out / "synthetic_train.csv"
    d.to_csv(train, index=False)
    plan = read_json(ROOT / "tests/fixtures/example-plan.json")
    write_json(out / "plan.json", plan)
    write_json(out / "diagnosis.json", {"training_source": {"path": str(train), "sha256": sha(train)},
                                         "test_data_accessed": False, **diagnose(d, "label")})
    result = run(train, out / "plan.json", out / "development")
    narrative = read_json(ROOT / "references/narrative-template.json")
    narrative["metadata"]["llm_interface"] = "Synthetic Python smoke test; no LLM analysis run"
    narrative.update(exploration="This generated fixture has two classes, three predictors and missing numeric values. It has no asserted business meaning.",
                     preprocessing="The linear model uses median imputation, indicators, scaling and one-hot encoding. The tree uses the same recipe without scaling; learned statistics stay inside each fold.",
                     features="All fixture predictors are retained. No feature engineering or outcome-dependent exclusion is applied.",
                     model_rationale="A linear model and a shallow tree exercise two different decision boundaries. Each grid contains two candidates. These fixture choices are not prescribed for user data.",
                     findings="This report demonstrates the reporting workflow using saved development results. No separate held-out dataset was read or evaluated.",
                     limitations="Synthetic fixture performance is not an estimate for a real application. Error costs and business meanings are unspecified. Sensitivities are empty for this mechanical smoke test.")
    narrative["evidence_summary"] = {
        "fact": "Nested CV compared two predeclared pipelines on the generated training fixture.",
        "interpretation": "The scores demonstrate workflow execution only, not real-world performance.",
        "limitation_unknown": "Business meaning and FP/FN costs are unspecified.",
        "decision": "Retain the predeclared fixed-threshold development comparison as a software demonstration.",
        "future_work": "Use a new independent run with a real task-specific plan."
    }
    write_json(out / "narrative.json", narrative)
    manifest = generate(out / "development/training_results.json", out / "diagnosis.json", out / "narrative.json", out / "report")
    print(f"Synthetic demo complete: {out / 'report/report.pdf'}; {manifest['main_pages']} main pages")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    demo(parser.parse_args().output_dir)
