"""
Standalone validation script

Run this to validate the neoantigen tool against known epitopes:
    python validation/run_validation.py
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os

from src.orchestrator import NeoantigenOrchestrator
from validation.benchmark_itsndb import ValidationBenchmark
from src.layer_6_output.report_generator import ReportGenerator


def main():
    """Run complete validation"""
    
    print("=" * 80)
    print("NEOANTIGEN VALIDATION BENCHMARK")
    print("=" * 80)
    
    # Load environment
    load_dotenv(dotenv_path=".env")
    
    # Initialize orchestrator
    print("\n[1/3] Initializing orchestrator...")
    orchestrator = NeoantigenOrchestrator(use_claude_api=True)
    
    # Run validation benchmark
    print("[2/3] Running validation benchmark...")
    benchmark = ValidationBenchmark(orchestrator)
    metrics = benchmark.run_validation()
    
    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(metrics)
    
    # Generate reports
    print("\n[3/3] Generating reports...")
    
    # Analyze some test peptides for reports
    test_peptides = ["KRASGSDFVQ", "NLVPMVATV", "RMSFVKQFQ"]
    test_hlas = ["HLA-A*02:01"] * 3
    
    batch_results = orchestrator.analyze_batch(
        test_peptides, 
        test_hlas, 
        verbose=False,
        auto_save=True  # This saves reports automatically
    )
    
    print("\n" + "=" * 80)
    print("✓ Validation complete!")
    print("=" * 80)
    print(f"\nResults saved to: results/")
    print("\nMetrics Summary:")
    print(f"  Accuracy: {metrics.accuracy:.2%}")
    print(f"  Precision: {metrics.precision:.2%}")
    print(f"  Recall: {metrics.recall:.2%}")
    print(f"  F1 Score: {metrics.f1_score:.2%}")
    print(f"  Improvement vs baseline: +{metrics.improvement:.2%}")


if __name__ == "__main__":
    main()