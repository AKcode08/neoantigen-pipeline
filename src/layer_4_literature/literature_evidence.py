"""
Literature Evidence Aggregator

Combines IEDB epitope data + PubMed papers into structured evidence
for ensemble use and Claude analysis.
"""

from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layer_4_literature.iedb_query import IEDBQuerier, IEDBQueryResult
from src.layer_4_literature.pubmed_search import PubMedSearcher, PubMedSearchResult


@dataclass
class LiteratureEvidence:
    """
    Structured literature evidence for a peptide.
    Combines IEDB experimental data + PubMed publications.
    """
    peptide: str
    hla_allele: str
    
    # IEDB data
    iedb_result: Optional[IEDBQueryResult] = None
    iedb_found: bool = False
    iedb_assay_count: int = 0
    iedb_response_rate: float = 0.0
    iedb_diseases: list = field(default_factory=list)
    
    # PubMed data
    pubmed_result: Optional[PubMedSearchResult] = None
    pubmed_paper_count: int = 0
    
    # Aggregate metrics
    evidence_level: str = "none"   # none / weak / moderate / strong
    evidence_score: float = 0.0    # 0-1
    
    # Summary for Claude
    summary: str = ""
    
    def __repr__(self):
        return (
            f"LiteratureEvidence({self.peptide}, "
            f"IEDB={self.iedb_assay_count} assays, "
            f"PubMed={self.pubmed_paper_count} papers, "
            f"level={self.evidence_level})"
        )


class LiteratureAggregator:
    """
    Aggregates literature evidence from multiple sources.
    
    Workflow:
    1. Query IEDB for experimental epitope data
    2. Query PubMed for published research
    3. Combine into structured evidence
    4. Score evidence strength
    """
    
    def __init__(self, verbose: bool = True):
        self.iedb = IEDBQuerier(verbose=verbose)
        self.pubmed = PubMedSearcher(verbose=verbose)
        self.verbose = verbose
    
    def gather_evidence(self,
                       peptide: str,
                       hla_allele: str = "",
                       max_papers: int = 5) -> LiteratureEvidence:
        """
        Gather all literature evidence for a peptide.
        
        Args:
            peptide: Amino acid sequence
            hla_allele: HLA allele
            max_papers: Max PubMed papers to retrieve
        
        Returns:
            LiteratureEvidence with combined data
        """
        if self.verbose:
            print(f"\n📚 Gathering literature evidence: {peptide} + {hla_allele}")
        
        # Query IEDB
        iedb_result = self.iedb.query_peptide(peptide, hla_allele)
        
        # Query PubMed
        pubmed_result = self.pubmed.search(peptide, hla_allele, max_results=max_papers)
        
        # Score evidence
        evidence_score = self._score_evidence(iedb_result, pubmed_result)
        evidence_level = self._level_from_score(evidence_score)
        
        # Generate summary
        summary = self._generate_summary(iedb_result, pubmed_result, evidence_level)
        
        evidence = LiteratureEvidence(
            peptide=peptide,
            hla_allele=hla_allele,
            iedb_result=iedb_result,
            iedb_found=iedb_result.found_in_iedb,
            iedb_assay_count=iedb_result.total_assays,
            iedb_response_rate=iedb_result.response_rate,
            iedb_diseases=iedb_result.diseases,
            pubmed_result=pubmed_result,
            pubmed_paper_count=len(pubmed_result.papers),
            evidence_level=evidence_level,
            evidence_score=evidence_score,
            summary=summary
        )
        
        if self.verbose:
            print(f"  ✓ Evidence: {evidence}")
        
        return evidence
    
    def _score_evidence(self, 
                       iedb: IEDBQueryResult, 
                       pubmed: PubMedSearchResult) -> float:
        """
        Score the strength of literature evidence (0-1).
        
        Factors:
        - IEDB direct match (+0.4)
        - IEDB positive T cell assays (+0.3)
        - PubMed papers found (+0.2)
        - Multiple disease contexts (+0.1)
        """
        score = 0.0
        
        # IEDB direct match
        if iedb.found_in_iedb:
            score += 0.4
            
            # Bonus for positive T cell responses
            if iedb.positive_assays > 0:
                score += min(0.3, iedb.response_rate * 0.3)
        
        # PubMed papers
        if pubmed.papers:
            paper_count = len(pubmed.papers)
            if paper_count >= 5:
                score += 0.2
            elif paper_count >= 3:
                score += 0.15
            elif paper_count >= 1:
                score += 0.1
        
        # Multiple disease contexts
        if len(iedb.diseases) >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _level_from_score(self, score: float) -> str:
        """Convert score to categorical level"""
        if score >= 0.7:
            return "strong"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.1:
            return "weak"
        else:
            return "none"
    
    def _generate_summary(self,
                         iedb: IEDBQueryResult,
                         pubmed: PubMedSearchResult,
                         level: str) -> str:
        """Generate human-readable summary of evidence"""
        parts = []
        
        # IEDB summary
        if iedb.found_in_iedb:
            iedb_parts = [f"IEDB: {iedb.total_assays} T cell assays"]
            if iedb.positive_assays > 0:
                iedb_parts.append(
                    f"{iedb.positive_assays} positive ({iedb.response_rate:.0%} response rate)"
                )
            if iedb.diseases:
                iedb_parts.append(f"diseases: {', '.join(iedb.diseases[:3])}")
            parts.append(" | ".join(iedb_parts))
        else:
            parts.append("IEDB: not found")
        
        # PubMed summary
        if pubmed.papers:
            years = [p.year for p in pubmed.papers if p.year]
            year_range = f"({min(years)}-{max(years)})" if years else ""
            parts.append(f"PubMed: {len(pubmed.papers)} papers {year_range}")
        else:
            parts.append("PubMed: no papers")
        
        # Evidence level
        parts.append(f"Evidence: {level}")
        
        return " | ".join(parts)


if __name__ == "__main__":
    # Load .env
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass
    
    print("=" * 80)
    print("LITERATURE EVIDENCE AGGREGATOR - TEST")
    print("=" * 80)
    
    agg = LiteratureAggregator(verbose=True)
    
    test_peptides = [
        ("NLVPMVATV", "HLA-A*02:01"),
        ("GILGFVFTL", "HLA-A*02:01"),
        ("KRASGSDFVQ", "HLA-A*02:01"),
    ]
    
    for peptide, hla in test_peptides:
        print(f"\n{'='*80}")
        evidence = agg.gather_evidence(peptide, hla, max_papers=5)
        
        print(f"\n📊 Summary:")
        print(f"  {evidence.summary}")
        print(f"  Score: {evidence.evidence_score:.2f}")
        print(f"  Level: {evidence.evidence_level}")
        
        if evidence.pubmed_result and evidence.pubmed_result.papers:
            print(f"\n  Top papers:")
            for paper in evidence.pubmed_result.papers[:2]:
                first_author = paper.authors[0] if paper.authors else "Unknown"
                print(f"    - {first_author} et al. {paper.year} | {paper.journal}")
                print(f"      {paper.title[:80]}...")