from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling.compare_models import compare_and_select
from src.modeling.train_enriched import train_enriched
from src.modeling.train_vanilla import train_vanilla


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BAF vanilla vs enriched experiment")
    parser.add_argument("--data", required=True, help="Path to BAF csv")
    parser.add_argument("--results-dir", default="results", help="Output results directory")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and force CPU training")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    vanilla_dir = results_dir / "vanilla"
    enriched_dir = results_dir / "enriched"

    vanilla = train_vanilla(args.data, vanilla_dir, prefer_gpu=not args.cpu_only)
    enriched = train_enriched(args.data, enriched_dir, prefer_gpu=not args.cpu_only)
    selected = compare_and_select(vanilla_dir=vanilla_dir, enriched_dir=enriched_dir, output_path=results_dir / "model_selection.json")

    summary = {
        "vanilla": vanilla["metrics"],
        "enriched": enriched["metrics"],
        "selection": selected,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
