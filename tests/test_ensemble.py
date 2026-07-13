"""
Test suite for ensemble predictor
"""

import pytest
from src.layer_2_predictors.ensemble import EnsemblePredictor, EnsembleResult


@pytest.fixture
def ensemble():
    """Create ensemble predictor"""
    return EnsemblePredictor()


def test_ensemble_single_prediction(ensemble):
    """Test single prediction with ensemble"""
    
    result = ensemble.predict(
        peptide="KRASGSDFVQ",
        hla_allele="HLA-A*02:01"
    )
    
    # Check result structure
    assert result is not None
    assert isinstance(result, EnsembleResult)
    assert result.peptide == "KRASGSDFVQ"
    
    # Check ensemble properties
    assert 0 <= result.consensus_score <= 1
    assert result.agreement_level in ["strong", "moderate", "weak"]
    assert result.disagreement_variance >= 0
    assert result.recommendation in ["INCLUDE", "EXCLUDE", "BORDERLINE"]
    assert 0 <= result.confidence <= 1
    assert len(result.reasoning) > 0
    
    # Check we have multiple predictions
    assert len(result.individual_predictions) >= 2
    
    print(f"\n✓ Ensemble prediction: {result.recommendation} (conf: {result.confidence:.2f})")


def test_ensemble_multiple_predictors(ensemble):
    """Test that ensemble uses multiple predictors"""
    
    result = ensemble.predict(
        peptide="KRASGSDFVQ",
        hla_allele="HLA-A*02:01"
    )
    
    # Check we have multiple predictor results
    predictors = list(result.individual_predictions.keys())
    assert len(predictors) >= 2, f"Expected 2+ predictors, got {len(predictors)}"
    
    # Check each predictor result is valid
    for name, score in result.individual_predictions.items():
        assert 0 <= score.score <= 1, f"{name} score out of range: {score.score}"
        assert score.interpretation in ["strong", "intermediate", "weak"]
        assert 0 < score.confidence <= 1
    
    print(f"\n✓ Multiple predictors: {predictors}")


def test_ensemble_strong_agreement(ensemble):
    """Test that strong binder has strong agreement"""
    
    # CMV epitope - known strong binder for HLA-A*02:01
    result = ensemble.predict(
        peptide="NLVPMVATV",
        hla_allele="HLA-A*02:01"
    )
    
    # Should have high consensus and strong agreement
    assert result.consensus_score > 0.7, f"Expected high consensus, got {result.consensus_score}"
    assert result.disagreement_variance < 0.15, f"Expected strong agreement, got variance {result.disagreement_variance}"
    
    # Should recommend INCLUDE with high confidence
    assert result.recommendation == "INCLUDE"
    assert result.confidence > 0.80
    
    print(f"\n✓ Strong agreement for strong binder")


def test_ensemble_batch_prediction(ensemble):
    """Test batch prediction with ranking"""
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV", "RMSFVKQFQ"]
    hlas = ["HLA-A*02:01"] * 3
    
    results = ensemble.predict_batch(peptides, hlas, verbose=False)
    
    # Check we got results
    assert len(results) > 0
    assert len(results) <= len(peptides)
    
    # Check results are sorted (INCLUDE before EXCLUDE)
    recommendations = [r.recommendation for r in results]
    include_idx = next((i for i, r in enumerate(recommendations) if r == "INCLUDE"), float('inf'))
    exclude_idx = next((i for i, r in enumerate(recommendations) if r == "EXCLUDE"), -1)
    
    if include_idx != float('inf') and exclude_idx != -1:
        assert include_idx < exclude_idx, "Results not properly sorted"
    
    print(f"\n✓ Batch prediction with {len(results)} results")


def test_ensemble_disagreement_detection(ensemble):
    """Test that ensemble detects predictor disagreement"""
    
    # Moderate consensus case
    result = ensemble.predict(
        peptide="RMSFVKQFQ",
        hla_allele="HLA-A*02:01"
    )
    
    # Get all scores
    scores = [p.score for p in result.individual_predictions.values()]
    
    # Check variance is calculated
    assert result.disagreement_variance >= 0
    
    # Manually verify variance calculation
    mean = sum(scores) / len(scores)
    expected_variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    expected_std = expected_variance ** 0.5
    
    assert abs(result.disagreement_variance - expected_std) < 0.01, \
        f"Variance mismatch: {result.disagreement_variance} vs {expected_std}"
    
    print(f"\n✓ Disagreement detection working")


def test_ensemble_recommendation_logic(ensemble):
    """Test that recommendation logic is sound"""
    
    # Test all three test peptides
    test_cases = [
        ("KRASGSDFVQ", "HLA-A*02:01"),  # Moderate
        ("NLVPMVATV", "HLA-A*02:01"),   # Strong
        ("RMSFVKQFQ", "HLA-A*02:01"),   # Borderline
    ]
    
    for peptide, hla in test_cases:
        result = ensemble.predict(peptide, hla)
        
        # Check recommendation is logical
        if result.consensus_score > 0.7 and result.agreement_level == "strong":
            assert result.recommendation == "INCLUDE", f"Should INCLUDE {peptide}"
        elif result.consensus_score < 0.3:
            assert result.recommendation == "EXCLUDE", f"Should EXCLUDE {peptide}"
        
        print(f"✓ {peptide}: {result.recommendation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])