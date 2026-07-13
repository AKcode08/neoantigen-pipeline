"""
MHC Presentation Prediction via MHCflurry

Predicts: Probability that peptide will be PRESENTED on cell surface
(NOT immunogenicity - that's a separate step)

Uses MHCflurry's Class1PresentationPredictor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

from mhcflurry import Class1PresentationPredictor

logger = logging.getLogger(__name__)


@dataclass
class PresentationPrediction:
    """MHC presentation prediction"""
    peptide: str
    hla_allele: str
    
    presentation_score: float       # 0-1, probability of cell surface display
    presentation_percentile: float  # Lower % = more likely to be presented
    processing_score: float         # Antigen processing likelihood
    affinity_nm: float              # MHC binding strength
    
    presentation_level: str         # "high", "medium", "low"
    prediction_method: str = "mhcflurry_presentation"
    
    def __repr__(self):
        return (
            f"Presentation({self.peptide}: "
            f"score={self.presentation_score:.3f}, "
            f"{self.presentation_level})"
        )


class MHCPresentationPredictor:
    """
    MHC-I presentation prediction.
    
    This predicts whether a peptide will be presented on the cell surface,
    combining:
    - MHC binding affinity
    - Antigen processing likelihood
    - Overall presentation probability
    
    Trained on mass spectrometry data.
    """
    
    HIGH_PRESENTATION_THRESHOLD = 0.7
    LOW_PRESENTATION_THRESHOLD = 0.3
    
    def __init__(self):
        logger.info("Loading MHCflurry presentation predictor...")
        try:
            self.predictor = Class1PresentationPredictor.load()
            logger.info("✓ Presentation predictor loaded")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            self.predictor = None
            self.available = False
    
    def predict(self,
               peptide: str,
               hla_allele: str) -> Optional[PresentationPrediction]:
        """
        Predict presentation probability.
        
        Args:
            peptide: Amino acid sequence (8-15 AA)
            hla_allele: HLA allele
        
        Returns:
            PresentationPrediction or None if failed
        """
        
        if not self.available:
            return None
        
        if not peptide or len(peptide) < 8 or len(peptide) > 15:
            logger.warning(f"Invalid peptide length: {peptide}")
            return None
        
        logger.info(f"Predicting presentation: {peptide} + {hla_allele}")
        
        try:
            # predict() returns a DataFrame
            df = self.predictor.predict(
                peptides=[peptide],
                alleles=[hla_allele]
            )
            
            if df.empty:
                logger.warning(f"No prediction for {peptide}")
                return None
            
            row = df.iloc[0]
            
            # Extract scores directly from DataFrame columns
            presentation_score = float(row["presentation_score"])
            presentation_percentile = float(row["presentation_percentile"])
            processing_score = float(row["processing_score"])
            affinity_nm = float(row["affinity"])
            
            # Classify presentation level
            if presentation_score >= self.HIGH_PRESENTATION_THRESHOLD:
                presentation_level = "high"
            elif presentation_score >= self.LOW_PRESENTATION_THRESHOLD:
                presentation_level = "medium"
            else:
                presentation_level = "low"
            
            prediction = PresentationPrediction(
                peptide=peptide,
                hla_allele=hla_allele,
                presentation_score=presentation_score,
                presentation_percentile=presentation_percentile,
                processing_score=processing_score,
                affinity_nm=affinity_nm,
                presentation_level=presentation_level
            )
            
            logger.info(
                f"✓ Presentation: {presentation_score:.3f} "
                f"({presentation_level}), "
                f"Processing: {processing_score:.3f}"
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_batch(self,
                     peptides: List[str],
                     hla_allele: str) -> Dict[str, Optional[PresentationPrediction]]:
        """
        Batch prediction (much faster).
        """
        
        if not self.available:
            return {pep: None for pep in peptides}
        
        # Filter valid peptides
        valid_peptides = [
            pep for pep in peptides
            if pep and 8 <= len(pep) <= 15
        ]
        
        if not valid_peptides:
            return {pep: None for pep in peptides}
        
        logger.info(f"Batch prediction: {len(valid_peptides)} peptides")
        
        try:
            # Predict all at once
            df = self.predictor.predict(
                peptides=valid_peptides,
                alleles=[hla_allele] * len(valid_peptides)
            )
            
            results = {}
            
            for peptide in peptides:
                if peptide not in valid_peptides:
                    results[peptide] = None
                    continue
                
                # Find this peptide in results
                pep_rows = df[df["peptide"] == peptide]
                
                if pep_rows.empty:
                    results[peptide] = None
                    continue
                
                row = pep_rows.iloc[0]
                
                presentation_score = float(row["presentation_score"])
                
                if presentation_score >= self.HIGH_PRESENTATION_THRESHOLD:
                    level = "high"
                elif presentation_score >= self.LOW_PRESENTATION_THRESHOLD:
                    level = "medium"
                else:
                    level = "low"
                
                results[peptide] = PresentationPrediction(
                    peptide=peptide,
                    hla_allele=hla_allele,
                    presentation_score=presentation_score,
                    presentation_percentile=float(row["presentation_percentile"]),
                    processing_score=float(row["processing_score"]),
                    affinity_nm=float(row["affinity"]),
                    presentation_level=level
                )
            
            logger.info(f"✓ Batch complete: {len([r for r in results.values() if r])} successful")
            return results
        
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            return {pep: None for pep in peptides}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("REAL MHC PRESENTATION PREDICTION")
    print("=" * 80)
    print("\nNote: This predicts cell surface PRESENTATION")
    print("This is DIFFERENT from immunogenicity (T cell response)\n")
    
    predictor = MHCPresentationPredictor()
    
    if not predictor.available:
        print("✗ Presentation predictor not available")
        exit(1)
    
    test_peptides = [
        ("NLVPMVATV", "HLA-A*02:01"),
        ("GILGFVFTL", "HLA-A*02:01"),
        ("YLLEYLQSR", "HLA-A*02:01"),
        ("KRASGSDFVQ", "HLA-A*02:01"),
        ("RMSFVKQFQ", "HLA-A*02:01"),
    ]
    
    print(f"{'Peptide':<15} {'Pres.':<8} {'Level':<10} {'Proc.':<8} {'Aff.(nM)':<12}")
    print("-" * 80)
    
    for peptide, hla in test_peptides:
        result = predictor.predict(peptide, hla)
        
        if result:
            print(
                f"{result.peptide:<15} "
                f"{result.presentation_score:<8.3f} "
                f"{result.presentation_level:<10} "
                f"{result.processing_score:<8.3f} "
                f"{result.affinity_nm:<12.1f}"
            )
        else:
            print(f"{peptide:<15} FAILED")
    
    # Detailed result
    print("\n" + "=" * 80)
    print("Detailed Result: NLVPMVATV")
    print("=" * 80)
    
    detail = predictor.predict("NLVPMVATV", "HLA-A*02:01")
    if detail:
        print(f"Peptide: {detail.peptide}")
        print(f"Presentation Score: {detail.presentation_score:.3f}")
        print(f"  ↳ Probability peptide is presented on cell surface")
        print(f"Presentation Percentile: {detail.presentation_percentile:.4f}")
        print(f"  ↳ Top {detail.presentation_percentile*100:.2f}% most presented")
        print(f"Processing Score: {detail.processing_score:.3f}")
        print(f"  ↳ Likelihood of antigen processing")
        print(f"Affinity: {detail.affinity_nm:.1f} nM")
        print(f"  ↳ MHC binding strength")
        print(f"Level: {detail.presentation_level}")