"""
Real Ensemble Predictor - Complete Pipeline

Combines:
1. Binding (MHCflurry affinity) - Does it stick to MHC?
2. Presentation (MHCflurry presentation) - Will it be displayed?
3. Immunogenicity (BigMHC IM) - Will T cells respond?
4. Structure (NeoaPred PepConf) - What does the 3D structure reveal?
5. Literature (IEDB + PubMed) - What does the published evidence say?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Optional
from dataclasses import dataclass

from src.layer_2_predictors.mhc_binding import MHCBindingPredictor
from src.layer_2_predictors.mhc_presentation import MHCPresentationPredictor
from src.layer_2_predictors.immunogenicity_bigmhc import BigMHCImmunogenicityPredictor
from src.layer_3_structure.neoapred_structure import NeoaPredWrapper
from src.layer_4_literature.literature_evidence import LiteratureAggregator, LiteratureEvidence

# Load environment for NCBI API
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Complete ensemble prediction result with all 5 layers"""
    peptide: str
    hla_allele: str
    
    # Step 1: Binding
    binding_affinity_nm: float
    binding_level: str
    binding_percentile: float
    
    # Step 2: Presentation
    presentation_score: float
    presentation_level: str
    processing_score: float
    
    # Step 3: Immunogenicity
    immunogenicity_score: float
    immunogenicity_level: str
    
    # Step 4: Structure
    structure_score: float
    structure_level: str
    structure_pdb_path: Optional[str]
    structure_atom_count: int
    
    # Step 5: Literature
    literature_evidence_level: str   # none / weak / moderate / strong
    literature_evidence_score: float
    iedb_assay_count: int
    iedb_response_rate: float
    pubmed_paper_count: int
    literature_summary: str
    literature_evidence: Optional[LiteratureEvidence] = None  # Full evidence object
    
    # Ensemble consensus
    consensus_score: float = 0.0
    agreement_level: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    
    # Narrative
    reasoning: str = ""
    
    def __repr__(self):
        return (
            f"Ensemble({self.peptide}: "
            f"bind={self.binding_level}, "
            f"pres={self.presentation_level}, "
            f"immun={self.immunogenicity_level}, "
            f"struct={self.structure_level}, "
            f"lit={self.literature_evidence_level}, "
            f"{self.recommendation})"
        )


class EnsemblePredictor:
    """
    Complete ensemble predictor combining five signals.
    
    Workflow (the biological cascade + evidence):
    1. BINDING: Does peptide stick to MHC? (MHCflurry affinity)
    2. PRESENTATION: Will cell display it? (MHCflurry presentation)
    3. IMMUNOGENICITY: Will T cells respond? (BigMHC IM)
    4. STRUCTURE: What does 3D structure reveal? (NeoaPred PepConf)
    5. LITERATURE: What does published evidence say? (IEDB + PubMed)
    """
    
    def __init__(self, enable_literature: bool = True, enable_structure: bool = True):
        logger.info("Initializing ensemble predictor...")
        self.binding_predictor = MHCBindingPredictor()
        self.presentation_predictor = MHCPresentationPredictor()
        self.immunogenicity_predictor = BigMHCImmunogenicityPredictor()
        
        # Optional layers
        self.enable_structure = enable_structure
        self.enable_literature = enable_literature
        
        if enable_structure:
            self.structure_predictor = NeoaPredWrapper()
        else:
            self.structure_predictor = None
        
        if enable_literature:
            self.literature_aggregator = LiteratureAggregator(verbose=False)
        else:
            self.literature_aggregator = None
        
        # Check availability
        if not self.binding_predictor.available:
            logger.error("Binding predictor unavailable")
            self.available = False
        elif not self.presentation_predictor.available:
            logger.error("Presentation predictor unavailable")
            self.available = False
        elif not self.immunogenicity_predictor.available:
            logger.warning("Immunogenicity predictor (BigMHC) not available")
            self.immunogenicity_available = False
            self.available = True
        else:
            logger.info("✓ Ensemble ready (5 layers)")
            self.available = True
            self.immunogenicity_available = True
    
    def predict(self,
               peptide: str,
               hla_allele: str) -> Optional[EnsembleResult]:
        """
        Run complete ensemble prediction with all 5 layers.
        
        Args:
            peptide: Amino acid sequence
            hla_allele: HLA allele (format: HLA-A*02:01 or A0201)
        
        Returns:
            EnsembleResult with all signals
        """
        
        if not self.available:
            logger.error("Ensemble not available")
            return None
        
        # Normalize HLA allele format for NeoaPred (needs A0201 format)
        hla_neoapred = hla_allele.replace("HLA-", "").replace("*", "").replace(":", "")
        
        logger.info(f"Running ensemble: {peptide} + {hla_allele}")
        
        # Step 1: Binding (REQUIRED)
        logger.info("Step 1: Binding prediction...")
        binding = self.binding_predictor.predict(peptide, hla_allele)
        
        if binding is None:
            logger.warning(f"Binding prediction failed for {peptide}")
            return None
        
        # Step 2: Presentation (REQUIRED)
        logger.info("Step 2: Presentation prediction...")
        presentation = self.presentation_predictor.predict(peptide, hla_allele)
        
        if presentation is None:
            logger.warning(f"Presentation prediction failed for {peptide}")
            return None
        
        # Step 3: Immunogenicity (OPTIONAL)
        logger.info("Step 3: Immunogenicity prediction...")
        immunogenicity = None
        if self.immunogenicity_available:
            immunogenicity = self.immunogenicity_predictor.predict(peptide, hla_allele)
            if immunogenicity is None:
                logger.warning(f"Immunogenicity prediction failed for {peptide}")
        
        # Step 4: Structure (OPTIONAL)
        logger.info("Step 4: Structure prediction...")
        structure = None
        if self.enable_structure and self.structure_predictor:
            try:
                structure = self.structure_predictor.predict(peptide, hla_neoapred)
                if structure is None or not structure.has_structure:
                    logger.warning(f"Structure prediction failed for {peptide}")
            except Exception as e:
                logger.warning(f"Structure prediction error: {e}")
        
        # Step 5: Literature (OPTIONAL)
        logger.info("Step 5: Literature evidence...")
        literature = None
        if self.enable_literature and self.literature_aggregator:
            try:
                literature = self.literature_aggregator.gather_evidence(
                    peptide, hla_allele, max_papers=5
                )
            except Exception as e:
                logger.warning(f"Literature gathering error: {e}")
        
        # Step 6: Calculate consensus
        consensus_score = self._calculate_consensus(
            binding, presentation, immunogenicity, structure, literature
        )
        
        # Step 7: Determine agreement
        agreement_level = self._calculate_agreement(
            binding, presentation, immunogenicity, structure, literature
        )
        
        # Step 8: Generate recommendation
        recommendation, confidence = self._generate_recommendation(
            consensus_score, binding, presentation, immunogenicity, structure, literature
        )
        
        # Step 9: Generate reasoning
        reasoning = self._generate_reasoning(
            binding, presentation, immunogenicity, structure, literature, consensus_score
        )
        
        result = EnsembleResult(
            peptide=peptide,
            hla_allele=hla_allele,
            binding_affinity_nm=binding.affinity_nm,
            binding_level=binding.binding_level,
            binding_percentile=binding.percentile_rank,
            presentation_score=presentation.presentation_score,
            presentation_level=presentation.presentation_level,
            processing_score=presentation.processing_score,
            immunogenicity_score=immunogenicity.immunogenicity_score if immunogenicity else None,
            immunogenicity_level=immunogenicity.immunogenicity_category if immunogenicity else "unknown",
            structure_score=self._get_structure_score(structure),
            structure_level=self._get_structure_level(structure),
            structure_pdb_path=structure.pdb_file_path if structure and structure.has_structure else None,
            structure_atom_count=len(structure.pdb_atoms) if structure and structure.pdb_atoms else 0,
            literature_evidence_level=literature.evidence_level if literature else "unavailable",
            literature_evidence_score=literature.evidence_score if literature else 0.0,
            iedb_assay_count=literature.iedb_assay_count if literature else 0,
            iedb_response_rate=literature.iedb_response_rate if literature else 0.0,
            pubmed_paper_count=literature.pubmed_paper_count if literature else 0,
            literature_summary=literature.summary if literature else "not gathered",
            literature_evidence=literature,
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning
        )
        
        logger.info(
            f"✓ Ensemble: {recommendation} "
            f"(bind={binding.binding_level}, "
            f"pres={presentation.presentation_level}, "
            f"immun={result.immunogenicity_level}, "
            f"struct={result.structure_level}, "
            f"lit={result.literature_evidence_level})"
        )
        
        return result
    
    def _get_structure_score(self, structure) -> float:
        """Convert structure prediction to 0-1 score"""
        if structure is None or not structure.has_structure:
            return 0.0
        if structure.structure_quality == "high":
            return 0.8
        elif structure.structure_quality == "medium":
            return 0.5
        else:
            return 0.2
    
    def _get_structure_level(self, structure) -> str:
        """Convert structure quality to categorical level"""
        if structure is None or not structure.has_structure:
            return "unavailable"
        return structure.structure_quality
    
    def _calculate_consensus(self, binding, presentation, immunogenicity, structure, literature) -> float:
        """
        Combine all signals into consensus score (0-1).
    
        WEIGHTS (updated from CV-validated benchmark on ITSNdb, May 2026):
        - Binding: 45%  — strongest individually-validated signal
        - Presentation: 45% — only layer that empirically lifts AUROC over binding
        - Immunogenicity: 10% — BigMHC kept at low weight (sweep peak at 0.10;
                                at original 0.30 it was hurting AUROC by -0.036)
        - Structure: 0%  — per-feature analysis showed all 4 features at AUROC ~0.50,
                        i.e. zero discriminative signal. Kept in pipeline for
                        Claude's qualitative reasoning, but NOT in consensus score.
    
        LITERATURE: handled separately when available, adds 15% (leaked mode for
        validated peptides with IEDB hits — flagged as optimistic in reports).
    
        These weights replace the prior 0.30/0.30/0.25/0.10/0.15 which were
        intuition-set and demonstrably mis-weighted on the immunogenicity layer.
        """
    
        # Binding score (0-1) — same scoring curve as before
        if binding.affinity_nm < 50:
            binding_score = 0.95
        elif binding.affinity_nm < 500:
            binding_score = 0.85
        elif binding.affinity_nm < 5000:
            binding_score = 0.50
        else:
            binding_score = 0.10
    
        presentation_score = presentation.presentation_score
    
        # Available signals
        immunogenicity_score = immunogenicity.immunogenicity_score if immunogenicity else None
        literature_score = literature.evidence_score if literature else None
        # NOTE: structure no longer contributes to consensus (was zero-signal on ITSNdb).
        # The structure field on EnsembleResult is still populated for reports.
    
        # --- New weighting scheme (L3_tuned from CV benchmark) ---
        if immunogenicity_score is not None and literature_score is not None:
            # All trustworthy signals available (literature flagged as leaky in reports)
            # binding 0.45 → 0.40, presentation 0.45 → 0.40, immuno 0.10 → 0.08, lit 0.12
            # Renormalized so binding+presentation still dominate but literature has a vote.
            consensus = (
                (binding_score * 0.40) +
                (presentation_score * 0.40) +
                (immunogenicity_score * 0.08) +
                (literature_score * 0.12)
            )
        elif immunogenicity_score is not None:
            # Standard live config (no literature available, or honest mode)
            consensus = (
                (binding_score * 0.45) +
                (presentation_score * 0.45) +
                (immunogenicity_score * 0.10)
            )
        else:
            # Fallback: binding + presentation only (L2_only — also CV-validated)
            consensus = (binding_score * 0.50) + (presentation_score * 0.50)
    
        return consensus
    
    def _calculate_agreement(self, binding, presentation, immunogenicity, structure, literature) -> str:
        """Check signal agreement"""
        
        binding_strong = binding.binding_level == "strong"
        presentation_high = presentation.presentation_level == "high"
        immunogenicity_high = (
            immunogenicity is not None and 
            immunogenicity.immunogenicity_category == "high"
        )
        structure_high = (
            structure is not None and 
            structure.has_structure and
            structure.structure_quality == "high"
        )
        literature_strong = (
            literature is not None and
            literature.evidence_level in ["strong", "moderate"]
        )
        
        aligned = sum([
            binding_strong, presentation_high, immunogenicity_high,
            structure_high, literature_strong
        ])
        
        if aligned >= 4:
            return "very_strong"
        elif aligned >= 3:
            return "strong"
        elif aligned >= 2:
            return "moderate"
        else:
            return "weak"
    
    def _generate_recommendation(self, consensus, binding, presentation,
                                immunogenicity, structure, literature) -> tuple:
        """
        Generate INCLUDE/EXCLUDE/BORDERLINE recommendation.
    
        THRESHOLDS (updated from 5-fold CV on ITSNdb, May 2026):
        - consensus ≥ 0.81 → INCLUDE  (CV-derived for L2+presentation, std ±0.026)
        - consensus < 0.60 → EXCLUDE  (CV-derived for L2+presentation, std ±0.004)
        - 0.60 ≤ consensus < 0.81 → BORDERLINE (12% BORD-rate at L2)
    
        Previous thresholds (0.7 / 0.3) were intuition-set and produced
        BORD-rate >40% — too many peptides falling into the uncertain middle.
    
        Literature confidence boost preserved.
        """
    
        if consensus >= 0.81:
            recommendation = "INCLUDE"
            confidence = min(consensus, 1.0)
        elif consensus < 0.60:
            recommendation = "EXCLUDE"
            confidence = 1 - consensus
        else:
            recommendation = "BORDERLINE"
            confidence = 0.5
    
        # Literature confidence boost (unchanged)
        if literature and literature.evidence_level == "strong":
            confidence = min(confidence + 0.1, 1.0)
    
        return recommendation, confidence
    
    def _generate_reasoning(self, binding, presentation, immunogenicity, 
                           structure, literature, consensus) -> str:
        """Generate human-readable reasoning"""
        
        parts = []
        
        # Binding
        if binding.binding_level == "strong":
            parts.append(f"Strong binder ({binding.affinity_nm:.0f} nM, top {binding.percentile_rank:.2f}%)")
        elif binding.binding_level == "weak":
            parts.append(f"Weak binder ({binding.affinity_nm:.0f} nM)")
        else:
            parts.append(f"Non-binder ({binding.affinity_nm:.0f} nM)")
        
        # Presentation
        parts.append(f"{presentation.presentation_level.capitalize()} presentation ({presentation.presentation_score:.2f})")
        
        # Processing
        if presentation.processing_score > 0.5:
            parts.append("Good processing")
        else:
            parts.append("Poor processing")
        
        # Immunogenicity
        if immunogenicity is not None:
            parts.append(f"{immunogenicity.immunogenicity_category.capitalize()} immunogenicity ({immunogenicity.immunogenicity_score:.2f})")
        
        # Structure
        if structure is not None and structure.has_structure:
            parts.append(f"Structure: {structure.structure_quality} ({len(structure.pdb_atoms)} atoms)")
        
        # Literature
        if literature is not None:
            lit_parts = []
            if literature.iedb_assay_count > 0:
                lit_parts.append(
                    f"IEDB: {literature.iedb_assay_count} assays "
                    f"({literature.iedb_response_rate:.0%} positive)"
                )
            if literature.pubmed_paper_count > 0:
                lit_parts.append(f"PubMed: {literature.pubmed_paper_count} papers")
            if not lit_parts:
                lit_parts.append("No literature evidence")
            parts.append(f"Evidence: {literature.evidence_level} ({', '.join(lit_parts)})")
        
        # Consensus
        parts.append(f"Consensus: {consensus:.2f}")
        
        return " | ".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 120)
    print("REAL ENSEMBLE - 5 LAYERS (Binding + Presentation + Immunogenicity + Structure + Literature)")
    print("=" * 120)
    
    predictor = EnsemblePredictor()
    
    if not predictor.available:
        print("✗ Ensemble not available")
        exit(1)
    
    test_peptides = [
        ("NLVPMVATV", "HLA-A*02:01"),
        ("GILGFVFTL", "HLA-A*02:01"),
        ("YLLEYLQSR", "HLA-A*02:01"),
        ("KRASGSDFVQ", "HLA-A*02:01"),
        ("RMSFVKQFQ", "HLA-A*02:01"),
    ]
    
    print(f"\n{'Peptide':<15} {'Bind':<8} {'Pres':<8} {'Immun':<8} {'Struct':<8} {'Lit':<10} {'Cons':<8} {'Rec':<12}")
    print("-" * 120)
    
    for peptide, hla in test_peptides:
        result = predictor.predict(peptide, hla)
        
        if result:
            print(
                f"{result.peptide:<15} "
                f"{result.binding_level:<8} "
                f"{result.presentation_level:<8} "
                f"{result.immunogenicity_level:<8} "
                f"{result.structure_level:<8} "
                f"{result.literature_evidence_level:<10} "
                f"{result.consensus_score:<8.2f} "
                f"{result.recommendation:<12}"
            )
    
    # Detailed result
    print("\n" + "=" * 120)
    print("Detailed Result: NLVPMVATV (CMV epitope)")
    print("=" * 120)
    
    detail = predictor.predict("NLVPMVATV", "HLA-A*02:01")
    if detail:
        print(f"\nPeptide: {detail.peptide}")
        print(f"\n--- Step 1: Binding ---")
        print(f"  Affinity: {detail.binding_affinity_nm:.1f} nM ({detail.binding_level})")
        print(f"  Percentile: {detail.binding_percentile:.3f}")
        print(f"\n--- Step 2: Presentation ---")
        print(f"  Score: {detail.presentation_score:.3f} ({detail.presentation_level})")
        print(f"  Processing: {detail.processing_score:.3f}")
        print(f"\n--- Step 3: Immunogenicity ---")
        if detail.immunogenicity_score is not None:
            print(f"  Score: {detail.immunogenicity_score:.3f} ({detail.immunogenicity_level})")
        else:
            print(f"  (BigMHC not installed)")
        print(f"\n--- Step 4: Structure ---")
        print(f"  Quality: {detail.structure_level}")
        print(f"  Score: {detail.structure_score:.2f}")
        if detail.structure_pdb_path:
            print(f"  PDB File: {detail.structure_pdb_path}")
            print(f"  Atoms: {detail.structure_atom_count}")
        print(f"\n--- Step 5: Literature ---")
        print(f"  Evidence Level: {detail.literature_evidence_level}")
        print(f"  Evidence Score: {detail.literature_evidence_score:.2f}")
        print(f"  IEDB Assays: {detail.iedb_assay_count} ({detail.iedb_response_rate:.1%} positive)")
        print(f"  PubMed Papers: {detail.pubmed_paper_count}")
        print(f"  Summary: {detail.literature_summary}")
        print(f"\n--- Consensus ---")
        print(f"  Consensus Score: {detail.consensus_score:.2f}")
        print(f"  Agreement: {detail.agreement_level}")
        print(f"\n--- Recommendation ---")
        print(f"  {detail.recommendation} (confidence: {detail.confidence:.2f})")
        print(f"\n--- Reasoning ---")
        print(f"  {detail.reasoning}")