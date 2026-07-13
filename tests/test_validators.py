"""
Test suite for input validation layer
"""

import json
import pytest
from pathlib import Path

from src.layer_1_input.validator import InputValidationPipeline


@pytest.fixture
def validator():
    """Create validator instance for tests"""
    return InputValidationPipeline()


@pytest.fixture
def test_cases():
    """Load test cases from fixture file"""
    fixture_path = Path(__file__).parent / "fixtures" / "validation_tests.json"
    with open(fixture_path, 'r') as f:
        data = json.load(f)
    return data["test_cases"]


def test_validation_suite(validator, test_cases):
    """Run all validation test cases"""
    
    results = []
    
    for test_case in test_cases:
        name = test_case["name"]
        input_data = test_case["input"]
        expected = test_case["expected"]
        
        # Run validation
        result = validator.validate_single_analysis(
            peptide=input_data["peptide"],
            hla_alleles=input_data["hla_alleles"],
            patient_data=input_data.get("patient_data")
        )
        
        # Check result
        assert result["is_valid"] == expected["is_valid"], \
            f"{name}: is_valid mismatch. Expected {expected['is_valid']}, got {result['is_valid']}"
        
        assert len(result["errors"]) == expected["num_errors"], \
            f"{name}: error count mismatch. Expected {expected['num_errors']}, got {len(result['errors'])}. Errors: {result['errors']}"
        
        assert len(result["warnings"]) == expected["num_warnings"], \
            f"{name}: warning count mismatch. Expected {expected['num_warnings']}, got {len(result['warnings'])}. Warnings: {result['warnings']}"
        
        results.append({
            "test": name,
            "passed": True,
            "errors": result["errors"],
            "warnings": result["warnings"]
        })
        
        print(f"✓ {name}")
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])