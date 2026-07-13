"""
Test suite for validation benchmark
"""

import pytest
import os
from dotenv import load_dotenv
from src.orchestrator import NeoantigenOrchestrator
from src.validation.benchmark import ValidationBenchmark
from src.layer_6_output.report_generator import ReportGenerator


@pytest.fixture
def orchestrator():
    """Create orchestrator"""
    load_dotenv(dotenv_path=".env")
    return NeoantigenOrchestrator(use_claude_api=True)


@pytest.fixture
def benchmark(orchestrator):
    """Create benchmark"""
    return ValidationBenchmark(orchestrator)


def test_validation_data_loads(benchmark):
    """Test that validation data loads correctly"""
    
    immunogenic, non_immunogenic = benchmark.load_validation_data()
    
    assert len(immunogenic) > 0, "Should have immunogenic epitopes"
    assert len(non_immunogenic) > 0, "Should have non-immunogenic epitopes"
    
    print(f"\n✓ Loaded {len(immunogenic)} immunogenic and {len(non_immunogenic)} non-immunogenic epitopes")


def test_validation_benchmark(benchmark):
    """Run full validation benchmark"""
    
    metrics = benchmark.run_validation()
    
    assert metrics.total_epitopes > 0
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1_score <= 1
    
    print(f"\n✓ Validation Results:")
    print(f"  Accuracy: {metrics.accuracy:.2%}")
    print(f"  Precision: {metrics.precision:.2%}")
    print(f"  Recall: {metrics.recall:.2%}")
    print(f"  F1: {metrics.f1_score:.2%}")
    print(f"  Improvement vs baseline: +{metrics.improvement:.2%}")


def test_report_generation(orchestrator):
    """Test report generation"""
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV"]
    hlas = ["HLA-A*02:01"] * 2
    
    batch_results = orchestrator.analyze_batch(peptides, hlas, verbose=False, auto_save=False)
    
    generator = ReportGenerator(output_dir="/tmp/test_reports")
    
    html_path = generator.generate_html_report(batch_results)
    csv_path = generator.generate_csv_report(batch_results)
    json_path = generator.generate_json_report(batch_results)
    
    assert html_path.exists()
    assert csv_path.exists()
    assert json_path.exists()
    
    print(f"\n✓ Reports generated:")
    print(f"  HTML: {html_path}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


def test_auto_save(orchestrator):
    """Test that auto-save works"""
    
    import shutil
    
    # Clean up results directory
    results_dir = Path("results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV"]
    hlas = ["HLA-A*02:01"] * 2
    
    # Run with auto-save enabled
    batch_results = orchestrator.analyze_batch(peptides, hlas, verbose=False, auto_save=True)
    
    # Check that results were saved
    results_dir = Path("results")
    assert results_dir.exists(), "Results directory should exist"
    
    # Check for result files
    html_files = list(results_dir.glob("*.html"))
    csv_files = list(results_dir.glob("*.csv"))
    json_files = list(results_dir.glob("*.json"))
    
    assert len(html_files) > 0, "Should have HTML report"
    assert len(csv_files) > 0, "Should have CSV report"
    assert len(json_files) > 0, "Should have JSON report"
    
    print(f"\n✓ Auto-save working:")
    print(f"  HTML files: {len(html_files)}")
    print(f"  CSV files: {len(csv_files)}")
    print(f"  JSON files: {len(json_files)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])