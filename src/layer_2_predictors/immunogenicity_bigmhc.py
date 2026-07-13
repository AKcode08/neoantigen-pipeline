"""
Immunogenicity Prediction via BigMHC

Wraps BigMHC's deep learning immunogenicity predictor.
BigMHC IM predicts T cell immunogenicity using a trained neural network.

BigMHC Installation:
    git clone https://github.com/karchinlab/bigmhc.git

BigMHC Output Format (IMPORTANT - actual format from their tool):
    mhc,pep,tgt,len,BigMHC_IM
    NLVPMVATV,HLA-A*02:01,,11,0.7538452
    
    Note: BigMHC's 'mhc' column contains the PEPTIDE,
    and 'pep' column contains the HLA ALLELE (counterintuitive!)
    Score is in 'BigMHC_IM' column.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import subprocess
import tempfile
import os
from typing import Optional, List, Dict
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImmunogenicityPrediction:
    """T cell immunogenicity prediction from BigMHC"""
    peptide: str
    hla_allele: str
    
    immunogenicity_score: float     # 0-1, BigMHC IM output
    immunogenicity_category: str    # "high", "medium", "low"
    
    prediction_method: str = "bigmhc_im"
    
    def __repr__(self):
        return (
            f"Immunogenicity({self.peptide}: "
            f"score={self.immunogenicity_score:.3f}, "
            f"{self.immunogenicity_category})"
        )


class BigMHCImmunogenicityPredictor:
    """
    Immunogenicity prediction using BigMHC.
    
    BigMHC IM (immunogenicity model) predicts T cell response likelihood.
    
    Score interpretation:
    - >= 0.5: high immunogenicity (likely T cell response)
    - 0.1 - 0.5: medium
    - < 0.1: low
    
    Reference:
    Albert et al. Nature Machine Intelligence (2023)
    """
    
    # Score thresholds (calibrated to BigMHC's distribution)
    HIGH_THRESHOLD = 0.5
    LOW_THRESHOLD = 0.1
    
    # BigMHC's actual output column for score
    SCORE_COLUMN = "BigMHC_IM"
    
    def __init__(self, bigmhc_dir: Optional[str] = None):
        if bigmhc_dir is None:
            self.bigmhc_dir = Path(__file__).parent.parent.parent / "bigmhc"
        else:
            self.bigmhc_dir = Path(bigmhc_dir)
        
        self.predict_script = self.bigmhc_dir / "src" / "predict.py"
        
        logger.info(f"BigMHC directory: {self.bigmhc_dir}")
        
        if not self.bigmhc_dir.exists():
            logger.error(f"BigMHC not found at {self.bigmhc_dir}")
            logger.error("Install with: git clone https://github.com/karchinlab/bigmhc.git")
            self.available = False
        elif not self.predict_script.exists():
            logger.error(f"predict.py not found at {self.predict_script}")
            self.available = False
        else:
            logger.info("✓ BigMHC installation detected")
            self.available = True
    
    def _categorize(self, score: float) -> str:
        """Convert numeric score to category"""
        if score >= self.HIGH_THRESHOLD:
            return "high"
        elif score >= self.LOW_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def _run_bigmhc(self, peptides_hla_pairs: List[tuple]) -> Optional[pd.DataFrame]:
        """
        Internal: Run BigMHC on a list of (peptide, hla) pairs.
        
        Returns DataFrame with columns: mhc, pep, tgt, len, BigMHC_IM
        Note: BigMHC swaps mhc/pep — mhc column contains peptides!
        """
        temp_input = None
        temp_output = None
        
        try:
            # Create input CSV
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.csv', delete=False, dir='/tmp'
            ) as f:
                temp_input = f.name
                f.write("peptide,hla\n")
                for peptide, hla in peptides_hla_pairs:
                    f.write(f"{peptide},{hla}\n")
            
            temp_output = temp_input + ".prd"
            
            # Call BigMHC
            cmd = [
                "python",
                str(self.predict_script),
                f"-i={temp_input}",
                "-m=im",
                "-d=cpu"
            ]
            
            logger.info(f"Running BigMHC IM on {len(peptides_hla_pairs)} peptides...")
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180
            )
            
            if result.returncode != 0:
                logger.error(f"BigMHC failed (rc={result.returncode})")
                logger.error(f"STDERR: {result.stderr[:500]}")
                return None
            
            if not os.path.exists(temp_output):
                logger.error(f"BigMHC output file not created: {temp_output}")
                return None
            
            df = pd.read_csv(temp_output)
            
            if df.empty:
                logger.warning("BigMHC returned empty output")
                return None
            
            # Validate expected column exists
            if self.SCORE_COLUMN not in df.columns:
                logger.error(
                    f"Expected column '{self.SCORE_COLUMN}' not found. "
                    f"Got columns: {df.columns.tolist()}"
                )
                return None
            
            return df
            
        except subprocess.TimeoutExpired:
            logger.error("BigMHC prediction timed out")
            return None
        except Exception as e:
            logger.error(f"BigMHC execution failed: {e}")
            return None
        finally:
            # Cleanup
            for f in [temp_input, temp_output]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
    
    def predict(self,
               peptide: str,
               hla_allele: str) -> Optional[ImmunogenicityPrediction]:
        """
        Predict immunogenicity for a single peptide.
        
        Args:
            peptide: Amino acid sequence (8-15 AA)
            hla_allele: HLA allele (e.g., "HLA-A*02:01")
        
        Returns:
            ImmunogenicityPrediction or None if failed
        """
        if not self.available:
            logger.error("BigMHC not available")
            return None
        
        if not peptide or len(peptide) < 8 or len(peptide) > 15:
            logger.warning(f"Invalid peptide length: {peptide}")
            return None
        
        logger.info(f"Predicting immunogenicity: {peptide} + {hla_allele}")
        
        df = self._run_bigmhc([(peptide, hla_allele)])
        
        if df is None or df.empty:
            return None
        
        # BigMHC outputs 'mhc' column with peptide, 'pep' with HLA - confusingly named.
        # We trust the row order and grab the first row since we sent one peptide.
        row = df.iloc[0]
        
        try:
            score = float(row[self.SCORE_COLUMN])
        except (ValueError, TypeError) as e:
            logger.error(f"Could not parse score: {e}")
            return None
        
        # Sanity check
        if pd.isna(score):
            logger.error("BigMHC returned NaN score")
            return None
        
        # Clamp to 0-1 (BigMHC should already be in this range)
        score = max(0.0, min(1.0, score))
        
        category = self._categorize(score)
        
        prediction = ImmunogenicityPrediction(
            peptide=peptide,
            hla_allele=hla_allele,
            immunogenicity_score=score,
            immunogenicity_category=category
        )
        
        logger.info(f"✓ Immunogenicity: {score:.3f} ({category})")
        
        return prediction
    
    def predict_batch(self,
                     peptides: List[str],
                     hla_allele: str) -> Dict[str, Optional[ImmunogenicityPrediction]]:
        """
        Batch prediction - calls BigMHC once for all peptides (more efficient).
        
        Args:
            peptides: List of amino acid sequences
            hla_allele: HLA allele (same for all)
        
        Returns:
            Dict mapping peptide -> ImmunogenicityPrediction (or None)
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
        
        logger.info(f"Batch immunogenicity: {len(valid_peptides)} peptides")
        
        # Run BigMHC
        pairs = [(pep, hla_allele) for pep in valid_peptides]
        df = self._run_bigmhc(pairs)
        
        if df is None or df.empty:
            return {pep: None for pep in peptides}
        
        # Build results - match by order since BigMHC preserves input order
        # The 'mhc' column actually contains the peptide
        results = {pep: None for pep in peptides}
        
        for i, peptide in enumerate(valid_peptides):
            if i >= len(df):
                break
            
            row = df.iloc[i]
            
            try:
                score = float(row[self.SCORE_COLUMN])
                if pd.isna(score):
                    continue
                
                score = max(0.0, min(1.0, score))
                category = self._categorize(score)
                
                results[peptide] = ImmunogenicityPrediction(
                    peptide=peptide,
                    hla_allele=hla_allele,
                    immunogenicity_score=score,
                    immunogenicity_category=category
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to parse score for {peptide}: {e}")
                continue
        
        success_count = sum(1 for r in results.values() if r is not None)
        logger.info(f"✓ Batch complete: {success_count}/{len(peptides)} successful")
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("BIGMHC IMMUNOGENICITY PREDICTION - FIXED VERSION")
    print("=" * 80)
    
    predictor = BigMHCImmunogenicityPredictor()
    
    if not predictor.available:
        print("✗ BigMHC not available")
        print("Install with: git clone https://github.com/karchinlab/bigmhc.git")
        exit(1)
    
    test_peptides = [
        ("NLVPMVATV", "HLA-A*02:01"),     # CMV - should be high
        ("GILGFVFTL", "HLA-A*02:01"),     # Flu - moderate
        ("YLLEYLQSR", "HLA-A*02:01"),     # Should be moderate
        ("KRASGSDFVQ", "HLA-A*02:01"),    # Mutation - varies
        ("RMSFVKQFQ", "HLA-A*02:01"),     # Mutation - varies
        ("AAAAAAAA", "HLA-A*02:01"),      # Non-immunogenic control - should be very low
    ]
    
    print(f"\n{'Peptide':<15} {'Score':<10} {'Category':<10}")
    print("-" * 80)
    
    for peptide, hla in test_peptides:
        result = predictor.predict(peptide, hla)
        
        if result:
            print(
                f"{result.peptide:<15} "
                f"{result.immunogenicity_score:<10.4f} "
                f"{result.immunogenicity_category:<10}"
            )
        else:
            print(f"{peptide:<15} FAILED")
    
    print(f"\n{'='*80}")
    print("Expected calibration (from BigMHC raw output):")
    print(f"{'='*80}")
    print("  NLVPMVATV: ~0.75 (high)")
    print("  GILGFVFTL: ~0.33 (medium)")
    print("  KRASGSDFVQ: ~0.69 (high)")
    print("  AAAAAAAA: ~0.0002 (low) ← Sanity check")
    
    # Batch test
    print(f"\n{'='*80}")
    print("BATCH MODE TEST")
    print(f"{'='*80}")
    
    batch_results = predictor.predict_batch(
        [p for p, _ in test_peptides],
        "HLA-A*02:01"
    )
    
    print(f"\n{'Peptide':<15} {'Score':<10} {'Category':<10}")
    print("-" * 80)
    
    for peptide, result in batch_results.items():
        if result:
            print(
                f"{peptide:<15} "
                f"{result.immunogenicity_score:<10.4f} "
                f"{result.immunogenicity_category:<10}"
            )
        else:
            print(f"{peptide:<15} FAILED")