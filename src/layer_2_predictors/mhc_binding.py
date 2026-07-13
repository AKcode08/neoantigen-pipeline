"""
Real MHC Binding Prediction via MHCflurry

Pure Python, no external dependencies.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

from mhcflurry import Class1AffinityPredictor

logger = logging.getLogger(__name__)


@dataclass
class BindingPrediction:
    """Real MHC binding prediction"""
    peptide: str
    hla_allele: str
    affinity_nm: float
    percentile_rank: float
    binding_level: str  # "strong", "weak", "non-binder"
    prediction_low: float    # 95% confidence interval lower
    prediction_high: float   # 95% confidence interval upper
    prediction_method: str = "mhcflurry"
    
    def __repr__(self):
        return (
            f"Binding({self.peptide}: "
            f"{self.affinity_nm:.1f} nM, "
            f"{self.binding_level})"
        )


class MHCBindingPredictor:
    """
    Real MHC-I binding prediction via MHCflurry.
    
    MHCflurry is a deep learning model trained on MHC binding data.
    Pure Python implementation.
    """
    
    STRONG_BINDER_NM = 500
    WEAK_BINDER_NM = 5000
    
    def __init__(self):
        logger.info("Loading MHCflurry predictor...")
        try:
            self.predictor = Class1AffinityPredictor.load()
            logger.info("✓ MHCflurry loaded successfully")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to load MHCflurry: {e}")
            logger.error("Run: mhcflurry-downloads fetch")
            self.predictor = None
            self.available = False
    
    def predict(self,
               peptide: str,
               hla_allele: str) -> Optional[BindingPrediction]:
        """
        Predict binding for single peptide.
        
        Args:
            peptide: Amino acid sequence (8-15 AA)
            hla_allele: HLA allele (e.g., "HLA-A*02:01")
        
        Returns:
            BindingPrediction with real MHCflurry output, or None if invalid
        """
        
        if not self.available:
            return None
        
        if not peptide or len(peptide) < 8 or len(peptide) > 15:
            logger.warning(f"Invalid peptide length: {peptide}")
            return None
        
        logger.info(f"Predicting: {peptide} + {hla_allele}")
        
        try:
            df = self.predictor.predict_to_dataframe(
                peptides=[peptide],
                alleles=[hla_allele]
            )
            
            if df.empty:
                logger.warning(f"No prediction for {peptide}")
                return None
            
            row = df.iloc[0]
            affinity_nm = float(row["prediction"])
            percentile = float(row["prediction_percentile"])
            pred_low = float(row["prediction_low"])
            pred_high = float(row["prediction_high"])
            
            # Classify
            if affinity_nm < self.STRONG_BINDER_NM:
                binding_level = "strong"
            elif affinity_nm < self.WEAK_BINDER_NM:
                binding_level = "weak"
            else:
                binding_level = "non-binder"
            
            prediction = BindingPrediction(
                peptide=peptide,
                hla_allele=hla_allele,
                affinity_nm=affinity_nm,
                percentile_rank=percentile,
                binding_level=binding_level,
                prediction_low=pred_low,
                prediction_high=pred_high
            )
            
            logger.info(
                f"✓ {affinity_nm:.1f} nM "
                f"({binding_level}, percentile: {percentile:.3f})"
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_batch(self,
                     peptides: List[str],
                     hla_allele: str) -> Dict[str, Optional[BindingPrediction]]:
        """
        Batch prediction (much faster).
        """
        
        if not self.available:
            return {pep: None for pep in peptides}
        
        # Filter valid
        valid_peptides = [
            pep for pep in peptides
            if pep and 8 <= len(pep) <= 15
        ]
        
        if not valid_peptides:
            return {pep: None for pep in peptides}
        
        logger.info(f"Batch predicting {len(valid_peptides)} peptides")
        
        try:
            # MHCflurry needs alleles array matching peptides length
            alleles_array = [hla_allele] * len(valid_peptides)
            
            df = self.predictor.predict_to_dataframe(
                peptides=valid_peptides,
                alleles=alleles_array
            )
            
            results = {}
            
            for peptide in peptides:
                if peptide not in valid_peptides:
                    results[peptide] = None
                    continue
                
                pep_rows = df[df["peptide"] == peptide]
                
                if pep_rows.empty:
                    results[peptide] = None
                    continue
                
                row = pep_rows.iloc[0]
                affinity_nm = float(row["prediction"])
                percentile = float(row["prediction_percentile"])
                
                if affinity_nm < self.STRONG_BINDER_NM:
                    binding_level = "strong"
                elif affinity_nm < self.WEAK_BINDER_NM:
                    binding_level = "weak"
                else:
                    binding_level = "non-binder"
                
                results[peptide] = BindingPrediction(
                    peptide=peptide,
                    hla_allele=hla_allele,
                    affinity_nm=affinity_nm,
                    percentile_rank=percentile,
                    binding_level=binding_level,
                    prediction_low=float(row["prediction_low"]),
                    prediction_high=float(row["prediction_high"])
                )
            
            logger.info(f"✓ Batch complete: {len([r for r in results.values() if r])} successful")
            
            return results
        
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            return {pep: None for pep in peptides}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("REAL MHCFLURRY BINDING PREDICTION")
    print("=" * 80)
    
    predictor = MHCBindingPredictor()
    
    if not predictor.available:
        print("✗ MHCflurry not available")
        exit(1)
    
    # Test 1: Single peptide (known strong binder)
    print("\nTest 1: Known CMV Epitope (Strong Binder)")
    print("-" * 80)
    
    result = predictor.predict("NLVPMVATV", "HLA-A*02:01")
    
    if result:
        print(f"  Peptide: {result.peptide}")
        print(f"  HLA: {result.hla_allele}")
        print(f"  Affinity: {result.affinity_nm:.1f} nM")
        print(f"  95% CI: [{result.prediction_low:.1f}, {result.prediction_high:.1f}] nM")
        print(f"  Percentile: {result.percentile_rank:.3f}")
        print(f"  Level: {result.binding_level}")
    
    # Test 2: Batch prediction
    print("\n" + "=" * 80)
    print("Test 2: Batch Prediction")
    print("-" * 80)
    
    peptides = [
        "NLVPMVATV",      # CMV - strong binder
        "KRASGSDFVQ",     # KRAS mutation
        "RMSFVKQFQ",      # TP53 mutation
        "GILGFVFTL",      # Flu - strong binder
        "YLLEYLQSR",      # Flu epitope
    ]
    
    results = predictor.predict_batch(peptides, "HLA-A*02:01")
    
    print(f"\n{'Peptide':<15} {'Affinity (nM)':<15} {'Level':<15}")
    print("-" * 80)
    
    for pep, pred in results.items():
        if pred:
            print(f"{pep:<15} {pred.affinity_nm:<15.1f} {pred.binding_level:<15}")
        else:
            print(f"{pep:<15} N/A")