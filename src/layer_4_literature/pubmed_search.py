"""
PubMed Search - REAL API via NCBI E-utilities (Improved)

Smarter query construction:
- Multiple HLA format variations (HLA-A*02:01, HLA-A2, A0201, HLA-A*02)
- Multi-strategy search: tries broader queries if narrow ones fail
- Better term combinations

Workflow:
1. Try specific query first (peptide + HLA + context)
2. If few results, try broader queries (peptide + HLA only)
3. If still nothing, try just peptide
"""

import requests
import json
import hashlib
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta


@dataclass
class PubMedPaper:
    """Single PubMed paper record"""
    pmid: str
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""
    keywords: List[str] = field(default_factory=list)
    
    def __repr__(self):
        first_author = self.authors[0] if self.authors else "Unknown"
        return f"PubMedPaper(PMID:{self.pmid}, {first_author} et al., {self.year})"


@dataclass
class PubMedSearchResult:
    """Combined PubMed search result"""
    query: str
    peptide: str
    hla_allele: str
    
    papers: List[PubMedPaper] = field(default_factory=list)
    total_found: int = 0
    
    # Multi-strategy tracking
    strategy_used: str = ""   # "specific" | "broader" | "peptide_only"
    queries_tried: List[str] = field(default_factory=list)
    
    query_timestamp: str = ""
    cached: bool = False
    notes: str = ""
    
    def __repr__(self):
        return (f"PubMedSearchResult({self.peptide}, "
                f"papers={len(self.papers)}/{self.total_found}, "
                f"strategy={self.strategy_used})")


class PubMedSearcher:
    """
    Search PubMed via NCBI E-utilities with smart multi-strategy querying.
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    CACHE_TTL_DAYS = 30
    REQUEST_TIMEOUT = 30
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 email: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        self.api_key = api_key or os.getenv('NCBI_API_KEY', '')
        self.email = email or os.getenv('NCBI_EMAIL', '')
        
        if not self.email and verbose:
            print("⚠️  No NCBI_EMAIL set - NCBI requires email for API usage")
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "neoantigens" / "pubmed"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.rate_limit = 10 if self.api_key else 3
        self.last_request_time = 0
        
        if self.verbose:
            print(f"📁 PubMed cache: {self.cache_dir}")
            print(f"🔑 API key: {'✓ set' if self.api_key else '✗ not set (3 req/sec limit)'}")
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            cached_at = datetime.fromisoformat(cached.get('query_timestamp', '2000-01-01'))
            age = datetime.now() - cached_at
            
            if age > timedelta(days=self.CACHE_TTL_DAYS):
                if self.verbose:
                    print(f"  ⏰ Cache expired ({age.days} days)")
                return None
            
            if self.verbose:
                print(f"  💾 Using cached result ({age.days} days old)")
            
            return cached
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _save_cache(self, cache_key: str, data: Dict) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            if self.verbose:
                print(f"  ⚠️  Cache write failed: {e}")
    
    def _hla_variants(self, hla_allele: str) -> List[str]:
        """
        Generate multiple HLA format variants to search for.
        
        Input: "HLA-A*02:01"
        Output: ["HLA-A*02:01", "HLA-A*02", "HLA-A2", "A0201", "A*0201"]
        
        This handles the variety of HLA notations used in literature.
        """
        if not hla_allele:
            return []
        
        variants = set()
        variants.add(hla_allele)  # Original: HLA-A*02:01
        
        # Parse HLA components
        # HLA-A*02:01 → gene=A, group=02, allele=01
        try:
            # Remove HLA- prefix if present
            clean = hla_allele.replace("HLA-", "")
            
            # Split on *
            if "*" in clean:
                gene_part, allele_part = clean.split("*", 1)
                
                # Split on :
                if ":" in allele_part:
                    group, allele = allele_part.split(":", 1)
                else:
                    group = allele_part
                    allele = ""
                
                # Build variants
                variants.add(f"HLA-{gene_part}*{group}:{allele}" if allele else f"HLA-{gene_part}*{group}")
                variants.add(f"HLA-{gene_part}*{group}")              # HLA-A*02
                variants.add(f"HLA-{gene_part}{group}")               # HLA-A02
                variants.add(f"HLA-{gene_part}{group.lstrip('0')}")   # HLA-A2 (drop leading zero)
                variants.add(f"{gene_part}*{group}:{allele}" if allele else f"{gene_part}*{group}")
                variants.add(f"{gene_part}{group}{allele}" if allele else f"{gene_part}{group}")  # A0201
                variants.add(f"{gene_part}*{group}{allele}" if allele else f"{gene_part}*{group}")  # A*0201
                
                # Common "supertype" notation: HLA-A2 (without zero padding)
                gene_letter = gene_part[0] if gene_part else ""
                group_int = group.lstrip("0") or "0"
                variants.add(f"HLA-{gene_letter}{group_int}")  # HLA-A2
        except (ValueError, IndexError):
            pass
        
        # Remove empty strings
        return [v for v in variants if v]
    
    def _build_query(self, 
                    peptide: str, 
                    hla_allele: str, 
                    strategy: str = "specific") -> str:
        """
        Build PubMed query based on strategy.
        
        Strategies:
        - "specific": peptide + HLA variants + immunology context
        - "broader": peptide + HLA variants (no context filter)
        - "peptide_only": just peptide (last resort)
        """
        # Always include peptide (most specific term)
        terms = [f'"{peptide}"']
        
        if strategy == "peptide_only":
            return terms[0]
        
        # Add HLA variants as OR group
        if hla_allele:
            variants = self._hla_variants(hla_allele)
            if variants:
                hla_clause = " OR ".join(f'"{v}"' for v in variants)
                terms.append(f"({hla_clause})")
        
        if strategy == "specific":
            # Add immunology context
            terms.append('(epitope OR "T cell" OR vaccine OR immunogenic OR neoantigen OR CTL OR "MHC class I" OR "HLA class I")')
        
        return " AND ".join(terms)
    
    def _esearch(self, query: str, max_results: int = 10) -> List[str]:
        """Search PubMed and return PMIDs"""
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email
        
        try:
            response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data.get('esearchresult', {}).get('idlist', [])
        except requests.RequestException as e:
            if self.verbose:
                print(f"  ⚠️  ESearch failed: {e}")
            return []
    
    def _efetch(self, pmids: List[str]) -> List[PubMedPaper]:
        """Fetch full records for PMIDs"""
        if not pmids:
            return []
        
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email
        
        try:
            response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            return self._parse_pubmed_xml(response.text)
        except requests.RequestException as e:
            if self.verbose:
                print(f"  ⚠️  EFetch failed: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[PubMedPaper]:
        """Parse PubMed XML into PubMedPaper objects"""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            if self.verbose:
                print(f"  ⚠️  XML parse error: {e}")
            return []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ''
                
                title_elem = article.find('.//ArticleTitle')
                title = ''.join(title_elem.itertext()) if title_elem is not None else ''
                
                abstract_parts = article.findall('.//AbstractText')
                abstract = ' '.join(
                    ''.join(elem.itertext()) for elem in abstract_parts
                )
                
                authors = []
                for author in article.findall('.//Author'):
                    last = author.find('LastName')
                    fore = author.find('ForeName')
                    if last is not None:
                        name = last.text or ''
                        if fore is not None and fore.text:
                            name = f"{fore.text} {name}"
                        authors.append(name)
                
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''
                
                year_elem = article.find('.//PubDate/Year')
                year = year_elem.text if year_elem is not None else ''
                
                doi = ''
                for article_id in article.findall('.//ArticleId'):
                    if article_id.get('IdType') == 'doi':
                        doi = article_id.text or ''
                        break
                
                keywords = []
                for kw in article.findall('.//Keyword'):
                    if kw.text:
                        keywords.append(kw.text)
                
                papers.append(PubMedPaper(
                    pmid=pmid, title=title, abstract=abstract,
                    authors=authors, journal=journal, year=year,
                    doi=doi, keywords=keywords
                ))
                
            except (AttributeError, TypeError) as e:
                if self.verbose:
                    print(f"  ⚠️  Failed to parse article: {e}")
                continue
        
        return papers
    
    def search(self, 
              peptide: str, 
              hla_allele: str = "", 
              max_results: int = 5) -> PubMedSearchResult:
        """
        Multi-strategy PubMed search.
        
        Strategy:
        1. Try "specific" query (peptide + HLA + context)
        2. If <2 results, try "broader" (peptide + HLA, no context filter)
        3. If still <1, try "peptide_only"
        """
        if self.verbose:
            print(f"🔍 PubMed search: {peptide} + {hla_allele or '(any HLA)'}")
        
        # Check cache with most specific query
        specific_query = self._build_query(peptide, hla_allele, "specific")
        cache_key = self._cache_key(f"{specific_query}_{max_results}")
        cached = self._get_cached(cache_key)
        if cached:
            result = self._dict_to_result(cached)
            result.cached = True
            return result
        
        queries_tried = []
        all_pmids = []
        strategy_used = ""
        
        # Strategy 1: Specific
        if self.verbose:
            print(f"  📡 Strategy 1 (specific): {specific_query[:100]}...")
        
        queries_tried.append(specific_query)
        pmids = self._esearch(specific_query, max_results)
        
        if pmids and len(pmids) >= 1:
            all_pmids = pmids
            strategy_used = "specific"
            if self.verbose:
                print(f"     ✓ Found {len(pmids)} PMIDs")
        else:
            if self.verbose:
                print(f"     ℹ️  Only {len(pmids)} found, trying broader query...")
            
            # Strategy 2: Broader (no context filter)
            broader_query = self._build_query(peptide, hla_allele, "broader")
            queries_tried.append(broader_query)
            
            if self.verbose:
                print(f"  📡 Strategy 2 (broader): {broader_query[:100]}...")
            
            pmids = self._esearch(broader_query, max_results)
            
            if pmids and len(pmids) >= 1:
                all_pmids = pmids
                strategy_used = "broader"
                if self.verbose:
                    print(f"     ✓ Found {len(pmids)} PMIDs")
            else:
                if self.verbose:
                    print(f"     ℹ️  Still {len(pmids)} found, trying peptide-only...")
                
                # Strategy 3: Peptide only (last resort)
                peptide_query = self._build_query(peptide, hla_allele, "peptide_only")
                queries_tried.append(peptide_query)
                
                if self.verbose:
                    print(f"  📡 Strategy 3 (peptide only): {peptide_query}")
                
                pmids = self._esearch(peptide_query, max_results)
                
                if pmids:
                    all_pmids = pmids
                    strategy_used = "peptide_only"
                    if self.verbose:
                        print(f"     ✓ Found {len(pmids)} PMIDs")
                else:
                    strategy_used = "none"
                    if self.verbose:
                        print(f"     ℹ️  No papers found in any strategy")
        
        # Fetch papers
        if all_pmids:
            if self.verbose:
                print(f"  📥 Fetching {len(all_pmids)} paper details...")
            papers = self._efetch(all_pmids)
        else:
            papers = []
        
        result = PubMedSearchResult(
            query=specific_query,
            peptide=peptide,
            hla_allele=hla_allele,
            papers=papers,
            total_found=len(all_pmids),
            strategy_used=strategy_used,
            queries_tried=queries_tried,
            query_timestamp=datetime.now().isoformat(),
            cached=False,
            notes=f"Strategy: {strategy_used}, Retrieved: {len(papers)} of {len(all_pmids)}"
        )
        
        # Save to cache
        self._save_cache(cache_key, asdict(result))
        
        if self.verbose:
            print(f"  ✓ Retrieved {len(papers)} papers (strategy: {strategy_used})")
        
        return result
    
    def _dict_to_result(self, data: Dict) -> PubMedSearchResult:
        """Convert cached dict back to PubMedSearchResult"""
        papers = [PubMedPaper(**p) for p in data.get('papers', [])]
        return PubMedSearchResult(
            query=data['query'],
            peptide=data['peptide'],
            hla_allele=data['hla_allele'],
            papers=papers,
            total_found=data.get('total_found', 0),
            strategy_used=data.get('strategy_used', 'unknown'),
            queries_tried=data.get('queries_tried', []),
            query_timestamp=data.get('query_timestamp', ''),
            cached=False,
            notes=data.get('notes', '')
        )
    
    def clear_cache(self) -> int:
        """Clear all cached results"""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass
    
    print("=" * 80)
    print("PUBMED REAL API TEST - IMPROVED MULTI-STRATEGY")
    print("=" * 80)
    
    searcher = PubMedSearcher(verbose=True)
    
    # First, show HLA variants generated
    print("\n--- HLA Variant Generation Test ---")
    for hla in ["HLA-A*02:01", "HLA-B*07:02"]:
        variants = searcher._hla_variants(hla)
        print(f"\n{hla} →")
        for v in sorted(variants):
            print(f"   {v}")
    
    test_peptides = [
        ("NLVPMVATV", "HLA-A*02:01"),    # CMV - very well studied
        ("GILGFVFTL", "HLA-A*02:01"),    # Flu M1 - was missing in original
        ("KRASGSDFVQ", "HLA-A*02:01"),   # Less studied
    ]
    
    # CLEAR CACHE for clean test
    cleared = searcher.clear_cache()
    print(f"\n🗑️  Cleared {cleared} cached entries for fresh test")
    
    for peptide, hla in test_peptides:
        print(f"\n{'='*80}")
        print(f"Search: {peptide} + {hla}")
        print("="*80)
        
        result = searcher.search(peptide, hla, max_results=5)
        
        print(f"\n📊 Result:")
        print(f"   Strategy: {result.strategy_used}")
        print(f"   Papers: {len(result.papers)}/{result.total_found}")
        print(f"   Cached: {result.cached}")
        
        if result.papers:
            print(f"\n   Top {min(3, len(result.papers))} papers:")
            for i, paper in enumerate(result.papers[:3], 1):
                first_author = paper.authors[0] if paper.authors else "Unknown"
                print(f"\n   {i}. {paper.title[:80]}...")
                print(f"      {first_author} et al. | {paper.journal} | {paper.year}")
                print(f"      PMID: {paper.pmid}")
                if paper.abstract:
                    print(f"      {paper.abstract[:150]}...")