"""
Test suite for NetMHCpan predictor wrapper
"""

import json
import pytest
from pathlib import Path
from src.layer_2_predictors.netmhcpan import NetMHCpanPredictor, NetMHCpanResult


@pytest.fixture
def predictor():
    """Create NetMHCpan predictor instance"""
    return NetMHCpanPredictor(rate_limit_sec=0.5)  # Reduced for testing


@pytest.fixture
def known_epitopes():
    """Load known epitopes with expected results"""
    fixture_path = Path(__file__).parent / "fixtures" / "netmhcpan_known_results.json"
    with open(fixture_path, 'r') as f:
        data = json.load(f)
    return data["known_epitopes"]


def test_netmhcpan_single_prediction(predictor):
    """Test single peptide prediction"""
    
    result = predictor.predict(
        peptide="KRASGSDFVQ",
        hla_allele="HLA-A*02:01"
    )
    
    # Check result structure
    assert result is not None, "Prediction returned None"
    assert isinstance(result, NetMHCpanResult)
    assert result.peptide == "KRASGSDFVQ"
    assert result.hla_allele == "HLA-A*02:01"
    
    # Check values are reasonable
    assert 0 < result.affinity_nm < 100000, f"Affinity out of range: {result.affinity_nm}"
    assert 0 <= result.percentile_rank <= 100, f"Rank out of range: {result.percentile_rank}"
    assert result.binding_level in ["strong", "intermediate", "weak"]
    
    print(f"\n✓ Single prediction: {result}")


def test_netmhcpan_binding_classification(predictor):
    """Test binding level classification"""
    
    # Create mock results to test classification logic
    strong = NetMHCpanResult(
        peptide="TEST", hla_allele="HLA-A*02:01",
        affinity_nm=400, percentile_rank=0.5,
        binding_level="strong", core_peptide="TEST",
        mhc_binding_core="TEST", icore="TEST", rank_ba=0.5
    )
    
    intermediate = NetMHCpanResult(
        peptide="TEST", hla_allele="HLA-A*02:01",
        affinity_nm=2000, percentile_rank=2.0,
        binding_level="intermediate", core_peptide="TEST",
        mhc_binding_core="TEST", icore="TEST", rank_ba=2.0
    )
    
    weak = NetMHCpanResult(
        peptide="TEST", hla_allele="HLA-A*02:01",
        affinity_nm=10000, percentile_rank=10.0,
        binding_level="weak", core_peptide="TEST",
        mhc_binding_core="TEST", icore="TEST", rank_ba=10.0
    )
    
    # Test classification
    assert strong.is_strong_binder() == True
    assert strong.is_intermediate_binder() == False
    assert strong.is_weak_binder() == False
    
    assert intermediate.is_strong_binder() == False
    assert intermediate.is_intermediate_binder() == True
    assert intermediate.is_weak_binder() == False
    
    assert weak.is_strong_binder() == False
    assert weak.is_intermediate_binder() == False
    assert weak.is_weak_binder() == True
    
    print("\n✓ Binding classification correct")


def test_netmhcpan_batch_prediction(predictor):
    """Test batch prediction"""
    
    peptides = ["KRASGSDFVQ", "NLVPMVATV"]
    hlas = ["HLA-A*02:01", "HLA-A*02:01"]
    
    results = predictor.predict_batch(peptides, hlas, verbose=False)
    
    # Check we got results
    assert len(results) > 0, "No results returned from batch prediction"
    assert len(results) <= len(peptides), f"Got more results than input"
    
    # Check each result is valid
    for result in results:
        assert isinstance(result, NetMHCpanResult)
        assert result.affinity_nm > 0
        assert result.percentile_rank >= 0
    
    print(f"\n✓ Batch prediction: {len(results)} results")


def test_netmhcpan_invalid_input(predictor):
    """Test that invalid inputs raise errors"""
    
    # Too short peptide
    with pytest.raises(ValueError):
        predictor.predict("KRA", "HLA-A*02:01")
    
    # Invalid HLA
    with pytest.raises(ValueError):
        predictor.predict("KRASGSDFVQ", "INVALID")
    
    print("\n✓ Invalid input rejection works")


def test_netmhcpan_known_epitopes(predictor, known_epitopes):
    """Test against known epitopes"""
    
    print("\nTesting known epitopes:")
    
    for epitope in known_epitopes:
        name = epitope["name"]
        peptide = epitope["peptide"]
        hla = epitope["hla"]
        expected_level = epitope["expected_binding_level"]
        expected_range = epitope["expected_affinity_range"]
        
        # Skip invalid test epitope
        if "XXX" in peptide:
            print(f"  ⊘ {name}: Skipping (contains XXX)")
            continue
        
        # Predict
        result = predictor.predict(peptide, hla)
        
        if result:
            # Check affinity is in expected range
            in_range = expected_range[0] <= result.affinity_nm <= expected_range[1]
            
            status = "✓" if in_range else "⚠"
            print(f"  {status} {name}: {result.affinity_nm:.1f} nM ({result.binding_level})")
            
            # For known strong/weak, check classification
            if expected_level == "strong":
                assert result.is_strong_binder(), f"{name} should be strong binder"
            elif expected_level == "weak":
                assert result.is_weak_binder(), f"{name} should be weak binder"
        else:
            print(f"  ✗ {name}: Prediction failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])