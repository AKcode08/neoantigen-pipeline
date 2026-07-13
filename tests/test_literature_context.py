"""
Test suite for literature context layer
"""

import pytest
from src.layer_4_literature.pubmed_search import PubMedSearcher
from src.layer_4_literature.iedb_query import IEDBQuerier
from src.layer_4_literature.evidence_extractor import LiteratureEvidenceExtractor


@pytest.fixture
def pubmed_searcher():
    """Create PubMed searcher"""
    return PubMedSearcher()


@pytest.fixture
def iedb_querier():
    """Create IEDB querier"""
    return IEDBQuerier()


@pytest.fixture
def evidence_extractor():
    """Create evidence extractor"""
    return LiteratureEvidenceExtractor()


def test_pubmed_search_finds_papers(pubmed_searcher):
    """Test PubMed search finds similar epitopes"""
    
    papers = pubmed_searcher.search_similar_epitopes("KRASGSDFVQ", "HLA-A*02:01")
    
    assert len(papers) > 0, "Should find at least one paper"
    
    for paper in papers:
        assert paper.epitope is not None
        assert paper.pmid is not None
        assert 0 <= paper.relevance_score <= 1
    
    print(f"\n✓ Found {len(papers)} papers for KRASGSDFVQ")


def test_pubmed_extract_summary(pubmed_searcher):
    """Test PubMed summary extraction"""
    
    papers = pubmed_searcher.search_similar_epitopes("KRASGSDFVQ", "HLA-A*02:01")
    summary = pubmed_searcher.extract_summary(papers)
    
    assert summary["total_papers"] == len(papers)
    assert summary["cd8_responses"] >= 0
    assert 0 <= summary["confidence"] <= 1
    assert isinstance(summary["consensus_immunogenic"], bool)
    
    print(f"\n✓ Summary: {summary['total_papers']} papers, "
          f"{summary['cd8_responses']} CD8+ responses")


def test_iedb_query_finds_epitope(iedb_querier):
    """Test IEDB query finds validated epitopes"""
    
    result = iedb_querier.query_epitope("KRASGSDFVQ", "HLA-A*02:01")
    
    assert result is not None, "Should find KRASGSDFVQ in IEDB"
    assert result.epitope == "KRASGSDFVQ"
    assert result.hla_allele == "HLA-A*02:01"
    assert isinstance(result.t_cell_response, bool)
    
    print(f"\n✓ Found IEDB record: T cell response = {result.t_cell_response}")


def test_iedb_query_hla(iedb_querier):
    """Test IEDB HLA-specific query"""
    
    epitopes = iedb_querier.query_hla_allele("HLA-A*02:01")
    
    assert len(epitopes) > 0, "Should find epitopes for HLA-A*02:01"
    
    for ep in epitopes:
        assert ep.hla_allele == "HLA-A*02:01"
    
    print(f"\n✓ Found {len(epitopes)} epitopes for HLA-A*02:01")


def test_evidence_extraction(evidence_extractor):
    """Test full evidence extraction"""
    
    evidence = evidence_extractor.extract_evidence("KRASGSDFVQ", "HLA-A*02:01")
    
    assert evidence.peptide == "KRASGSDFVQ"
    assert evidence.hla_allele == "HLA-A*02:01"
    assert evidence.literature_consensus in ["strong", "moderate", "weak", "none"]
    assert 0 <= evidence.evidence_strength <= 1
    assert len(evidence.supporting_evidence) > 0
    
    print(f"\n✓ Evidence extraction successful")
    print(f"  Consensus: {evidence.literature_consensus}")
    print(f"  Strength: {evidence.evidence_strength:.2f}")
    print(f"  Supporting: {len(evidence.supporting_evidence)} points")


def test_evidence_with_no_literature(evidence_extractor):
    """Test evidence extraction for unknown epitope"""
    
    evidence = evidence_extractor.extract_evidence("XXXUNKNOWN", "HLA-B*44:02")
    
    # Should handle gracefully
    assert evidence.peptide == "XXXUNKNOWN"
    assert evidence.literature_consensus == "none"
    assert len(evidence.caveats) > 0
    
    print(f"\n✓ Handles unknown epitope gracefully")
    print(f"  Consensus: {evidence.literature_consensus}")
    print(f"  Caveat: {evidence.caveats[0]}")


def test_literature_consensus_scoring(evidence_extractor):
    """Test that consensus scoring makes sense"""
    
    # Known strong epitope
    strong_evidence = evidence_extractor.extract_evidence("NLVPMVATV", "HLA-A*02:01")
    
    # Unknown epitope
    weak_evidence = evidence_extractor.extract_evidence("XXXUNKNOWN", "HLA-B*44:02")
    
    # Strong should have higher evidence strength
    assert strong_evidence.evidence_strength >= weak_evidence.evidence_strength, \
        "Known epitope should have stronger evidence than unknown"
    
    print(f"\n✓ Consensus scoring correct")
    print(f"  NLVPMVATV strength: {strong_evidence.evidence_strength:.2f}")
    print(f"  XXXUNKNOWN strength: {weak_evidence.evidence_strength:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])