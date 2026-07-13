"""
Orchestrator - Main Pipeline

End-to-end neoantigen analysis pipeline.

Modes:
1. SINGLE: Analyze one peptide → generate report
   python -m src.orchestrator NLVPMVATV HLA-A*02:01
   
2. BATCH: Analyze multiple peptides from CSV → generate reports + summary
   python -m src.orchestrator --batch peptides.csv

Pipeline:
   Layer 1: MHC Binding (MHCflurry)
   Layer 2: Antigen Presentation (MHCflurry)
   Layer 3: Immunogenicity (BigMHC)
   Layer 4: 3D Structure (NeoaPred)
   Layer 5: Literature Evidence (IEDB + PubMed)
   Synthesis: Claude Expert Analysis
   Output: HTML + JSON Reports
"""

import sys
import argparse
import csv
import json
import logging
import time
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer_2_predictors.ensemble import EnsemblePredictor, EnsembleResult
from src.layer_5_synthesis.claude_engine import ClaudeSynthesisEngine, ClaudeAnalysis
from src.layer_6_output.report_generator import ReportGenerator, ReportPaths

# Load environment
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result for a single peptide"""
    peptide: str
    hla_allele: str
    patient_id: Optional[str]
    tumor_type: Optional[str]
    
    # Pipeline results
    ensemble_result: Optional[EnsembleResult] = None
    claude_analysis: Optional[ClaudeAnalysis] = None
    report_paths: Optional[ReportPaths] = None
    
    # Status
    success: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    
    def summary_row(self) -> dict:
        """Get summary row for batch CSV output"""
        if not self.success or not self.ensemble_result or not self.claude_analysis:
            return {
                "peptide": self.peptide,
                "hla_allele": self.hla_allele,
                "patient_id": self.patient_id or "",
                "status": "FAILED",
                "error": self.error or "",
                "elapsed_seconds": f"{self.elapsed_seconds:.1f}",
            }
        
        return {
            "peptide": self.peptide,
            "hla_allele": self.hla_allele,
            "patient_id": self.patient_id or "",
            "tumor_type": self.tumor_type or "",
            "status": "SUCCESS",
            "recommendation": self.claude_analysis.final_recommendation,
            "confidence": f"{self.claude_analysis.final_confidence:.2f}",
            "vaccine_priority": self.claude_analysis.vaccine_priority,
            "consensus_score": f"{self.ensemble_result.consensus_score:.3f}",
            "agreement": self.ensemble_result.agreement_level,
            "binding_nm": f"{self.ensemble_result.binding_affinity_nm:.1f}",
            "binding_level": self.ensemble_result.binding_level,
            "presentation_score": f"{self.ensemble_result.presentation_score:.3f}",
            "presentation_level": self.ensemble_result.presentation_level,
            "immunogenicity_score": (
                f"{self.ensemble_result.immunogenicity_score:.3f}" 
                if self.ensemble_result.immunogenicity_score is not None else "N/A"
            ),
            "immunogenicity_level": self.ensemble_result.immunogenicity_level,
            "structure_level": self.ensemble_result.structure_level,
            "structure_atoms": self.ensemble_result.structure_atom_count,
            "literature_level": self.ensemble_result.literature_evidence_level,
            "iedb_assays": self.ensemble_result.iedb_assay_count,
            "iedb_response_rate": f"{self.ensemble_result.iedb_response_rate:.1%}",
            "pubmed_papers": self.ensemble_result.pubmed_paper_count,
            "executive_summary": self.claude_analysis.executive_summary,
            "html_report": self.report_paths.html_path if self.report_paths else "",
            "elapsed_seconds": f"{self.elapsed_seconds:.1f}",
        }


class NeoantigenOrchestrator:
    """
    Main orchestrator coordinating all 6 layers of the analysis pipeline.
    """
    
    def __init__(self, 
                 enable_structure: bool = True,
                 enable_literature: bool = True,
                 output_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize orchestrator.
        
        Args:
            enable_structure: Use NeoaPred for 3D structure (requires Docker)
            enable_literature: Query IEDB + PubMed for evidence
            output_dir: Where to save reports
            verbose: Print progress messages
        """
        self.verbose = verbose
        
        if self.verbose:
            print("=" * 100)
            print("🧬 NEOANTIGEN ANALYSIS ORCHESTRATOR")
            print("=" * 100)
            print(f"\n📋 Configuration:")
            print(f"   Structure prediction (NeoaPred): {'✅ enabled' if enable_structure else '❌ disabled'}")
            print(f"   Literature evidence (IEDB+PubMed): {'✅ enabled' if enable_literature else '❌ disabled'}")
        
        # Initialize ensemble predictor
        if self.verbose:
            print(f"\n🔧 Initializing predictors...")
        
        self.ensemble = EnsemblePredictor(
            enable_structure=enable_structure,
            enable_literature=enable_literature
        )
        
        if not self.ensemble.available:
            raise RuntimeError("Ensemble predictor not available - check dependencies")
        
        # Initialize Claude engine
        if self.verbose:
            print(f"🧠 Initializing Claude synthesis engine...")
        self.claude = ClaudeSynthesisEngine()
        
        # Initialize report generator
        if self.verbose:
            print(f"📄 Initializing report generator...")
        self.report_generator = ReportGenerator(output_dir=output_dir)
        
        if self.verbose:
            print(f"\n✅ Pipeline ready\n")
    
    def analyze(self,
               peptide: str,
               hla_allele: str,
               patient_id: Optional[str] = None,
               tumor_type: Optional[str] = None,
               open_report: bool = False) -> AnalysisResult:
        """
        Analyze a single peptide end-to-end.
        
        Args:
            peptide: Amino acid sequence
            hla_allele: HLA allele (e.g., "HLA-A*02:01")
            patient_id: Optional patient identifier
            tumor_type: Optional tumor type
            open_report: Auto-open the HTML report in browser
        
        Returns:
            AnalysisResult with full pipeline output
        """
        start_time = time.time()
        result = AnalysisResult(
            peptide=peptide,
            hla_allele=hla_allele,
            patient_id=patient_id,
            tumor_type=tumor_type
        )
        
        try:
            # Step 1: Ensemble prediction
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🔬 Analyzing: {peptide} ({hla_allele})")
                print(f"{'─' * 100}")
                print(f"\n[Step 1/3] Running 5-layer ensemble...")
            
            ensemble_result = self.ensemble.predict(peptide, hla_allele)
            
            if ensemble_result is None:
                result.error = "Ensemble prediction failed"
                result.elapsed_seconds = time.time() - start_time
                return result
            
            result.ensemble_result = ensemble_result
            
            if self.verbose:
                print(f"   ✓ Ensemble: {ensemble_result.recommendation} (consensus: {ensemble_result.consensus_score:.2f})")
            
            # Step 2: Claude analysis
            if self.verbose:
                print(f"\n[Step 2/3] Generating expert analysis with Claude...")
            
            claude_analysis = self.claude.synthesize(ensemble_result)
            result.claude_analysis = claude_analysis
            
            if self.verbose:
                print(f"   ✓ Claude: {claude_analysis.final_recommendation} (confidence: {claude_analysis.final_confidence:.2f})")
                print(f"   ✓ Priority: {claude_analysis.vaccine_priority}")
            
            # Step 3: Generate report
            if self.verbose:
                print(f"\n[Step 3/3] Generating HTML report...")
            
            report_paths = self.report_generator.generate(
                ensemble_result=ensemble_result,
                claude_analysis=claude_analysis,
                patient_id=patient_id,
                tumor_type=tumor_type
            )
            result.report_paths = report_paths
            
            if self.verbose:
                print(f"   ✓ HTML: {report_paths.html_path}")
                print(f"   ✓ JSON: {report_paths.json_path}")
            
            result.success = True
            result.elapsed_seconds = time.time() - start_time
            
            if self.verbose:
                print(f"\n✅ Complete in {result.elapsed_seconds:.1f}s")
            
            # Auto-open report
            if open_report and report_paths.html_path:
                if self.verbose:
                    print(f"\n🌐 Opening report in browser...")
                webbrowser.open(f"file://{report_paths.html_path}")
            
            return result
            
        except Exception as e:
            result.error = str(e)
            result.elapsed_seconds = time.time() - start_time
            
            if self.verbose:
                print(f"\n❌ Error: {e}")
            
            logger.exception(f"Analysis failed for {peptide}")
            return result
    
    def batch_analyze(self,
                     peptide_list: List[Tuple[str, str, Optional[str], Optional[str]]],
                     summary_csv: Optional[str] = None) -> List[AnalysisResult]:
        """
        Batch analyze multiple peptides.
        
        Args:
            peptide_list: List of (peptide, hla_allele, patient_id, tumor_type) tuples
            summary_csv: Path to summary CSV (default: reports/batch_summary_TIMESTAMP.csv)
        
        Returns:
            List of AnalysisResult objects
        """
        total = len(peptide_list)
        
        if self.verbose:
            print(f"\n{'=' * 100}")
            print(f"📦 BATCH MODE: Analyzing {total} peptides")
            print(f"{'=' * 100}")
        
        results = []
        batch_start = time.time()
        
        for i, entry in enumerate(peptide_list, 1):
            peptide = entry[0]
            hla = entry[1]
            patient_id = entry[2] if len(entry) > 2 else None
            tumor_type = entry[3] if len(entry) > 3 else None
            
            if self.verbose:
                print(f"\n[{i}/{total}] {peptide} + {hla}")
            
            result = self.analyze(
                peptide=peptide,
                hla_allele=hla,
                patient_id=patient_id,
                tumor_type=tumor_type,
                open_report=False
            )
            results.append(result)
            
            if self.verbose:
                status = "✅" if result.success else "❌"
                if result.success:
                    print(f"   {status} {result.claude_analysis.final_recommendation} "
                          f"({result.elapsed_seconds:.1f}s)")
                else:
                    print(f"   {status} FAILED: {result.error}")
        
        # Generate summary
        batch_elapsed = time.time() - batch_start
        success_count = sum(1 for r in results if r.success)
        
        if self.verbose:
            print(f"\n{'=' * 100}")
            print(f"📊 BATCH COMPLETE")
            print(f"{'=' * 100}")
            print(f"   Total: {total}")
            print(f"   Success: {success_count}")
            print(f"   Failed: {total - success_count}")
            print(f"   Time: {batch_elapsed:.1f}s ({batch_elapsed/total:.1f}s avg per peptide)")
            
            # Recommendation distribution
            recs = {}
            for r in results:
                if r.success:
                    rec = r.claude_analysis.final_recommendation
                    recs[rec] = recs.get(rec, 0) + 1
            
            if recs:
                print(f"\n   Recommendations:")
                for rec, count in sorted(recs.items()):
                    print(f"      {rec}: {count}")
        
        # Save summary CSV
        if summary_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_csv = self.report_generator.output_dir / f"batch_summary_{timestamp}.csv"
        else:
            summary_csv = Path(summary_csv)
        
        self._write_summary_csv(results, summary_csv)
        
        if self.verbose:
            print(f"\n📄 Summary saved to: {summary_csv}")
        
        return results
    
    def _write_summary_csv(self, results: List[AnalysisResult], output_path: Path):
        """Write batch summary CSV"""
        if not results:
            return
        
        rows = [r.summary_row() for r in results]
        
        # Get all field names
        all_fields = set()
        for row in rows:
            all_fields.update(row.keys())
        
        # Order fields consistently
        priority_fields = [
            "peptide", "hla_allele", "patient_id", "tumor_type", "status",
            "recommendation", "confidence", "vaccine_priority",
            "consensus_score", "agreement",
            "binding_nm", "binding_level",
            "presentation_score", "presentation_level",
            "immunogenicity_score", "immunogenicity_level",
            "structure_level", "structure_atoms",
            "literature_level", "iedb_assays", "iedb_response_rate", "pubmed_papers",
            "executive_summary",
            "html_report", "elapsed_seconds", "error"
        ]
        
        ordered_fields = [f for f in priority_fields if f in all_fields]
        remaining = sorted(all_fields - set(ordered_fields))
        ordered_fields.extend(remaining)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def load_batch_csv(csv_path: str) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    """
    Load peptides from CSV file.
    
    Expected CSV format:
        peptide,hla_allele,patient_id,tumor_type
        NLVPMVATV,HLA-A*02:01,PAT001,melanoma
        ...
    
    Minimum required columns: peptide, hla_allele
    Optional: patient_id, tumor_type
    """
    peptides = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Normalize column names (handle variations)
        for row in reader:
            # Try multiple column name variations
            peptide = (row.get('peptide') or row.get('Peptide') or 
                      row.get('sequence') or row.get('Sequence') or
                      row.get('Pep') or row.get('Mut'))
            
            hla = (row.get('hla_allele') or row.get('HLA') or 
                  row.get('hla') or row.get('Allele') or row.get('allele'))
            
            patient_id = (row.get('patient_id') or row.get('PatientID') or 
                         row.get('patient') or row.get('ID') or row.get('id'))
            
            tumor_type = (row.get('tumor_type') or row.get('TumorType') or 
                         row.get('disease') or row.get('cancer_type'))
            
            if peptide and hla:
                peptides.append((peptide.strip(), hla.strip(), patient_id, tumor_type))
    
    return peptides


def main():
    parser = argparse.ArgumentParser(
        description="Neoantigen Analysis Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single peptide analysis
  python -m src.orchestrator NLVPMVATV HLA-A*02:01
  
  # With patient context and auto-open report
  python -m src.orchestrator NLVPMVATV HLA-A*02:01 --patient PAT001 --tumor melanoma --open
  
  # Batch analysis from CSV
  python -m src.orchestrator --batch peptides.csv
  
  # Batch with custom summary location
  python -m src.orchestrator --batch peptides.csv --summary results.csv
  
  # Disable expensive layers (faster for testing)
  python -m src.orchestrator NLVPMVATV HLA-A*02:01 --no-structure --no-literature

CSV Format for batch mode:
  peptide,hla_allele,patient_id,tumor_type
  NLVPMVATV,HLA-A*02:01,PAT001,melanoma
  GILGFVFTL,HLA-A*02:01,PAT002,lung
"""
    )
    
    # Single mode arguments
    parser.add_argument('peptide', nargs='?', help='Peptide sequence (single mode)')
    parser.add_argument('hla', nargs='?', help='HLA allele (single mode)')
    
    # Batch mode
    parser.add_argument('--batch', metavar='CSV', help='Batch mode: process peptides from CSV')
    parser.add_argument('--summary', metavar='CSV', help='Path for batch summary CSV')
    
    # Optional context
    parser.add_argument('--patient', help='Patient ID')
    parser.add_argument('--tumor', help='Tumor type')
    parser.add_argument('--output-dir', help='Report output directory')
    
    # Behavior
    parser.add_argument('--open', action='store_true', help='Auto-open report in browser')
    parser.add_argument('--no-structure', action='store_true', help='Disable NeoaPred (faster)')
    parser.add_argument('--no-literature', action='store_true', help='Disable IEDB+PubMed (faster)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Validate
    if not args.batch and (not args.peptide or not args.hla):
        parser.error("Must provide either: peptide + hla (single mode) OR --batch CSV (batch mode)")
    
    if args.batch and (args.peptide or args.hla):
        parser.error("Cannot use both single mode and batch mode")
    
    # Initialize orchestrator
    try:
        orchestrator = NeoantigenOrchestrator(
            enable_structure=not args.no_structure,
            enable_literature=not args.no_literature,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Execute
    if args.batch:
        # Batch mode
        try:
            peptides = load_batch_csv(args.batch)
        except (IOError, csv.Error) as e:
            print(f"❌ Failed to load CSV: {e}")
            sys.exit(1)
        
        if not peptides:
            print(f"❌ No valid peptides found in {args.batch}")
            sys.exit(1)
        
        print(f"📋 Loaded {len(peptides)} peptides from {args.batch}")
        
        results = orchestrator.batch_analyze(
            peptide_list=peptides,
            summary_csv=args.summary
        )
        
        # Exit code: 0 if all succeeded, 1 if any failed
        sys.exit(0 if all(r.success for r in results) else 1)
    
    else:
        # Single mode
        result = orchestrator.analyze(
            peptide=args.peptide,
            hla_allele=args.hla,
            patient_id=args.patient,
            tumor_type=args.tumor,
            open_report=args.open
        )
        
        if result.success:
            print(f"\n{'=' * 100}")
            print(f"📄 Report ready:")
            print(f"   {result.report_paths.html_path}")
            print(f"{'=' * 100}")
            
            if not args.open:
                print(f"\nTo view: open {result.report_paths.html_path}")
            
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()