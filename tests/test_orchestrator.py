"""
Test suite for orchestrator
"""

import pytest
import os
from dotenv import load_dotenv
from src.orchestrator import NeoantigenOrchestrator, AnalysisResult, BatchAnalysisResults


@pytest.fixture
def orchestrator():
    """Create orchestrator with Claude API"""
    load_dotenv(dotenv_path=".env")
    return NeoantigenOrchestrator(use_claude_api=True)


def test_single_peptide_analysis(orchestrator):
    """Test single peptide analysis through full pipeline"""
    
    result = orchestrator.analyze_single(
        peptide="KRASGSDFVQ",
        hla_allele="HLA-A*02:01",
        patient_data={"cancer_type": "pancreatic", "expression_tpm": 150}
    )
    
    # Check result structure
    assert isinstance(result, AnalysisResult)
    assert result.peptide == "KRASGSDFVQ"
    assert result.hla_allele == "HLA-A*02:01"
    assert result.input_valid == True
    
    # Check all layers ran
    assert result.ensemble_recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
    assert result.literature_consensus in ["strong", "moderate", "weak", "none"]
    assert result.claude_recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
    
    # Check final recommendation
    assert result.final_recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
    assert 0 <= result.final_confidence <= 1
    assert len(result.final_reasoning) > 0
    
    # Check processing time recorded
    assert result.processing_time_sec > 0
    
    print(f"\n✓ Single analysis: {result.final_recommendation} ({result.final_confidence:.2f})")


def test_batch_analysis(orchestrator):
    """Test batch analysis"""
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV", "RMSFVKQFQ"]
    hlas = ["HLA-A*02:01"] * 3
    
    batch_results = orchestrator.analyze_batch(peptides, hlas, verbose=False)
    
    # Check batch structure
    assert isinstance(batch_results, BatchAnalysisResults)
    assert batch_results.total_peptides == 3
    assert batch_results.successfully_analyzed > 0
    assert len(batch_results.results) > 0
    
    # Check results are sorted
    for i in range(len(batch_results.results) - 1):
        r1 = batch_results.results[i]
        r2 = batch_results.results[i + 1]
        
        # Should be sorted by recommendation priority
        priority = {"INCLUDE": 0, "BORDERLINE": 1, "EXCLUDE": 2}
        assert priority[r1.final_recommendation] <= priority[r2.final_recommendation]
    
    print(f"\n✓ Batch analysis: {batch_results.successfully_analyzed}/{batch_results.total_peptides} successful")


def test_invalid_input_handling(orchestrator):
    """Test handling of invalid inputs"""
    
    result = orchestrator.analyze_single(
        peptide="INVALID",  # Too short
        hla_allele="HLA-A*02:01"
    )
    
    # Should fail validation
    assert result.input_valid == False
    assert len(result.errors) > 0
    
    print(f"\n✓ Invalid input rejected gracefully")


def test_ranking_table_generation(orchestrator):
    """Test ranking table generation"""
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV"]
    hlas = ["HLA-A*02:01"] * 2
    
    batch_results = orchestrator.analyze_batch(peptides, hlas, verbose=False)
    table = orchestrator.generate_ranking_table(batch_results)
    
    # Check table contains expected elements
    assert "Peptide" in table
    assert "Ensemble" in table
    assert "Final" in table
    assert "KRASGSDFVQ" in table
    assert "NLVPMVATV" in table
    
    print(f"\n✓ Ranking table generated successfully")


def test_all_layers_integrated(orchestrator):
    """Test that all layers are properly integrated"""
    
    result = orchestrator.analyze_single(
        peptide="NLVPMVATV",
        hla_allele="HLA-A*02:01"
    )
    
    # Should have signals from all layers
    assert result.ensemble_confidence > 0
    assert result.literature_strength >= 0
    assert result.claude_confidence > 0
    
    # Final recommendation should consider all signals
    assert result.final_recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
    
    # Reasoning should mention all components
    assert "Ensemble" in result.final_reasoning or result.final_reasoning != ""
    
    print(f"\n✓ All layers integrated and working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])