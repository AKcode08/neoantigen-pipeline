"""
Test Claude synthesis engine
"""

import pytest
import os
from dotenv import load_dotenv
from src.layer_2_predictors.ensemble import EnsemblePredictor
from src.layer_5_synthesis.claude_engine import ClaudeSynthesisEngine, ClaudeAnalysis


@pytest.fixture
def ensemble():
    """Create ensemble predictor"""
    return EnsemblePredictor()


@pytest.fixture
def claude_engine():
    """Create Claude synthesis engine with API key from .env"""
    # Load .env file
    load_dotenv(dotenv_path=".env")
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    
    return ClaudeSynthesisEngine(api_key=api_key)


@pytest.fixture
def ensemble_result(ensemble):
    """Get ensemble result for testing"""
    return ensemble.predict(
        peptide="KRASGSDFVQ",
        hla_allele="HLA-A*02:01"
    )


def test_claude_synthesis_single_peptide(claude_engine, ensemble_result):
    """Test Claude synthesis of ensemble result"""
    
    analysis = claude_engine.synthesize(ensemble_result)
    
    # Check result structure
    assert analysis is not None
    assert isinstance(analysis, ClaudeAnalysis)
    assert analysis.peptide == "KRASGSDFVQ"
    assert analysis.hla_allele == "HLA-A*02:01"
    
    # Check Claude generated content
    assert len(analysis.prediction_interpretation) > 0
    assert len(analysis.predictor_agreement_analysis) > 0
    assert len(analysis.confidence_assessment) > 0
    assert len(analysis.mechanistic_reasoning) > 0
    
    # Check recommendation is valid
    assert analysis.final_recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
    assert 0 <= analysis.final_confidence <= 1
    
    # Check we have key factors and caveats
    assert len(analysis.key_factors) > 0
    assert len(analysis.caveats) > 0
    
    print(f"\n✓ Claude synthesis successful")
    print(f"  Recommendation: {analysis.final_recommendation}")
    print(f"  Confidence: {analysis.final_confidence:.2f}")


def test_claude_recommendation_validity(claude_engine, ensemble_result):
    """Test that Claude's recommendation makes sense"""
    
    analysis = claude_engine.synthesize(ensemble_result)
    
    # If ensemble consensus is high, Claude should be confident
    if ensemble_result.consensus_score > 0.7:
        assert analysis.final_recommendation in ["INCLUDE", "BORDERLINE"]
    
    # If ensemble consensus is low, Claude should recommend caution
    if ensemble_result.consensus_score < 0.3:
        assert analysis.final_recommendation in ["EXCLUDE", "BORDERLINE"]
    
    print(f"\n✓ Claude recommendation is logical")


def test_claude_analysis_references_data(claude_engine, ensemble_result):
    """Test that Claude references actual prediction data"""
    
    analysis = claude_engine.synthesize(ensemble_result)
    
    # Claude should mention consensus score somewhere
    response_lower = analysis.raw_response.lower()
    
    # Should mention predictors
    assert any(pred in response_lower for pred in ["netmhc", "prime", "mixmhc"])
    
    print(f"\n✓ Claude references actual prediction data")


def test_claude_synthesis_multiple_peptides(claude_engine, ensemble):
    """Test Claude synthesis across different peptides"""
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV", "RMSFVKQFQ"]
    
    analyses = []
    for pep in peptides:
        ensemble_result = ensemble.predict(pep, "HLA-A*02:01")
        if ensemble_result:
            analysis = claude_engine.synthesize(ensemble_result)
            analyses.append(analysis)
    
    # Check we got analyses for all peptides
    assert len(analyses) == len(peptides)
    
    # Each should have valid recommendation
    for analysis in analyses:
        assert analysis.final_recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
        print(f"✓ {analysis.peptide}: {analysis.final_recommendation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])