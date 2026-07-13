"""
INPUT VALIDATION LAYER

Validates peptide sequences and HLA alleles before processing.
Catches common errors early.
"""

import re
from typing import Dict, List, Tuple
from enum import Enum

# Valid amino acids (standard 20)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# HLA format validation patterns
HLA_PATTERN = re.compile(r"^HLA-[A-C]\*\d{2}:\d{2}$")

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

class ValidationWarning:
    """Store non-fatal issues (warnings)"""
    def __init__(self, field: str, message: str, severity: str = "warning"):
        self.field = field
        self.message = message
        self.severity = severity  # "warning", "caution", "info"
    
    def __repr__(self):
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


class PeptideValidator:
    """Validates peptide sequences"""
    
    # MHC-I typical length
    MIN_LENGTH = 8
    MAX_LENGTH = 15
    
    # Common issues
    SUSPICIOUS_PATTERNS = {
        "all_same_aa": lambda seq: len(set(seq)) == 1,
        "homopolymer_long": lambda seq: any(aa*5 in seq for aa in VALID_AMINO_ACIDS),
        "too_many_prolines": lambda seq: seq.count('P') > 3,
    }
    
    def validate(self, peptide: str) -> Tuple[bool, List[str], List[ValidationWarning]]:
        """
        Validate peptide sequence.
        
        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Clean input
        peptide = peptide.upper().strip()
        
        # Check 1: Length
        if len(peptide) < self.MIN_LENGTH:
            errors.append(f"Peptide too short: {len(peptide)} AA (min {self.MIN_LENGTH})")
        elif len(peptide) > self.MAX_LENGTH:
            warnings.append(
                ValidationWarning(
                    "length",
                    f"Peptide quite long: {len(peptide)} AA (typical max {self.MAX_LENGTH})",
                    severity="caution"
                )
            )
        
        # Check 2: Valid amino acids
        invalid_aas = set(peptide) - VALID_AMINO_ACIDS
        if invalid_aas:
            errors.append(f"Invalid amino acids: {invalid_aas}")
        
        # Check 3: Suspicious patterns
        for pattern_name, pattern_func in self.SUSPICIOUS_PATTERNS.items():
            if pattern_func(peptide):
                warnings.append(
                    ValidationWarning(
                        "sequence_quality",
                        f"Suspicious pattern detected: {pattern_name}. Verify sequence is correct.",
                        severity="warning"
                    )
                )
        
        # Check 4: N/C terminus clarity
        # (Sometimes people add * for stop codon)
        if peptide.startswith('M'):
            warnings.append(
                ValidationWarning(
                    "termini",
                    "Starts with Methionine (M). Is this the N-terminus? Verify if M should be included.",
                    severity="info"
                )
            )
        
        return (len(errors) == 0, errors, warnings)


class HLAValidator:
    """Validates HLA allele names"""
    
    # Known allele prefixes
    VALID_LOCI = ["HLA-A", "HLA-B", "HLA-C", "HLA-DR", "HLA-DQ", "HLA-DP"]
    
    def validate(self, hla_allele: str) -> Tuple[bool, List[str], List[ValidationWarning]]:
        """
        Validate HLA allele name format.
        
        Expected format: HLA-A*02:01
        
        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        hla_allele = hla_allele.upper().strip()
        
        # Check 1: Format matches pattern
        if not HLA_PATTERN.match(hla_allele):
            errors.append(
                f"Invalid HLA format: '{hla_allele}'. Expected format: HLA-A*02:01"
            )
            return (False, errors, warnings)
        
        # Check 2: Known locus
        locus = hla_allele.split('*')[0]
        if locus not in self.VALID_LOCI:
            errors.append(f"Unknown HLA locus: {locus}")
        
        # Check 3: Frequency lookup (optional - warn if very rare)
        # (We'd need a database of HLA frequencies here)
        # For MVP, skip this
        
        return (len(errors) == 0, errors, warnings)


class PatientDataValidator:
    """Validates patient-level data (expression, VAF, etc.)"""
    
    def validate(self, patient_data: Dict) -> Tuple[bool, List[str], List[ValidationWarning]]:
        """
        Validate optional patient data fields.
        
        Args:
            patient_data: {
                "cancer_type": str,
                "expression_tpm": float,
                "vaf": float,
                "mutation_position": int
            }
        
        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        if not patient_data:
            return (True, [], [])  # Optional field
        
        # Check cancer_type
        if "cancer_type" in patient_data:
            cancer = patient_data["cancer_type"]
            valid_cancers = ["melanoma", "pancreatic", "lung", "colorectal", "ovarian"]
            if cancer.lower() not in valid_cancers:
                warnings.append(
                    ValidationWarning(
                        "cancer_type",
                        f"Uncommon cancer type: {cancer}. Supported: {valid_cancers}",
                        severity="info"
                    )
                )
        
        # Check expression_tpm
        if "expression_tpm" in patient_data:
            tpm = patient_data["expression_tpm"]
            if not isinstance(tpm, (int, float)):
                errors.append(f"expression_tpm must be numeric, got {type(tpm)}")
            elif tpm < 0:
                errors.append(f"expression_tpm cannot be negative: {tpm}")
            elif tpm < 1:
                warnings.append(
                    ValidationWarning(
                        "expression_tpm",
                        f"Very low expression: {tpm} TPM. May not be translated into protein.",
                        severity="warning"
                    )
                )
        
        # Check VAF (variant allele frequency)
        if "vaf" in patient_data:
            vaf = patient_data["vaf"]
            if not isinstance(vaf, (int, float)):
                errors.append(f"vaf must be numeric, got {type(vaf)}")
            elif vaf < 0 or vaf > 1:
                errors.append(f"vaf must be between 0 and 1, got {vaf}")
            elif vaf < 0.1:
                warnings.append(
                    ValidationWarning(
                        "vaf",
                        f"Low VAF: {vaf:.2%}. This is a subclonal mutation (found in some tumor cells, not all).",
                        severity="caution"
                    )
                )
        
        return (len(errors) == 0, errors, warnings)


class InputValidationPipeline:
    """Orchestrate all validators"""
    
    def __init__(self):
        self.peptide_validator = PeptideValidator()
        self.hla_validator = HLAValidator()
        self.patient_validator = PatientDataValidator()
    
    def validate_single_analysis(self, 
                                peptide: str,
                                hla_alleles: List[str],
                                patient_data: Dict = None) -> Dict:
        """
        Validate a single neoantigen analysis request.
        
        Returns comprehensive validation report.
        """
        
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "cleaned_peptide": None,
            "cleaned_hla_alleles": None,
        }
        
        # Validate peptide
        peptide_valid, pep_errors, pep_warnings = self.peptide_validator.validate(peptide)
        if not peptide_valid:
            report["is_valid"] = False
            report["errors"].extend(pep_errors)
        else:
            report["cleaned_peptide"] = peptide.upper().strip()
        report["warnings"].extend(pep_warnings)
        
        # Validate each HLA allele
        cleaned_hlas = []
        for hla in hla_alleles:
            hla_valid, hla_errors, hla_warnings = self.hla_validator.validate(hla)
            if not hla_valid:
                report["is_valid"] = False
                report["errors"].extend(hla_errors)
            else:
                cleaned_hlas.append(hla.upper().strip())
            report["warnings"].extend(hla_warnings)
        
        report["cleaned_hla_alleles"] = cleaned_hlas
        
        # Validate patient data
        if patient_data:
            patient_valid, patient_errors, patient_warnings = self.patient_validator.validate(patient_data)
            if not patient_valid:
                report["is_valid"] = False
                report["errors"].extend(patient_errors)
            report["warnings"].extend(patient_warnings)
        
        return report


# Example usage
if __name__ == "__main__":
    validator = InputValidationPipeline()
    
    # Test 1: Good input
    result = validator.validate_single_analysis(
        peptide="KRASGSDFVQ",
        hla_alleles=["HLA-A*02:01"],
        patient_data={"cancer_type": "pancreatic", "expression_tpm": 150, "vaf": 0.45}
    )
    
    print("Test 1 - Good Input:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Errors: {result['errors']}")
    print(f"  Warnings: {result['warnings']}")
    print()
    
    # Test 2: Bad input (too short)
    result = validator.validate_single_analysis(
        peptide="KRAS",
        hla_alleles=["HLA-A*02:01"]
    )
    
    print("Test 2 - Short Peptide:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Errors: {result['errors']}")
    print()
    
    # Test 3: Warnings
    result = validator.validate_single_analysis(
        peptide="KRASGSDFVQ",
        hla_alleles=["HLA-A*02:01"],
        patient_data={"expression_tpm": 0.5, "vaf": 0.05}
    )
    
    print("Test 3 - Warnings:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Warnings: {result['warnings']}")