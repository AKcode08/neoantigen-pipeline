"""
IEDB (Immune Epitope Database) Query - REAL API

Queries the IEDB Query API (IQ-API) for validated epitope data.
API Base: https://query-api.iedb.org/
Documentation: https://help.iedb.org/

Endpoints used:
- /epitope_search - Find epitopes by sequence
- /tcell_search - Find T cell assays
- /mhc_search - Find MHC binding assays
"""

import requests
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta


@dataclass
class IEDBEpitope:
    """Single epitope record from IEDB"""
    epitope_id: str
    sequence: str
    epitope_iri: str = ""
    source_organism: str = ""
    source_antigen: str = ""
    structure_type: str = ""
    
    def __repr__(self):
        return f"IEDBEpitope({self.sequence}, id={self.epitope_id})"


@dataclass
class IEDBTcellAssay:
    """T cell assay result from IEDB"""
    epitope_sequence: str
    mhc_allele: str
    assay_type: str           # e.g., "ELISPOT", "tetramer", "proliferation"
    assay_response: str       # "Positive", "Negative", "Positive-Low" etc.
    host_organism: str = ""
    disease: str = ""
    reference_id: str = ""
    
    def __repr__(self):
        return (f"IEDBTcellAssay({self.epitope_sequence}, "
                f"{self.assay_type}, {self.assay_response})")


@dataclass
class IEDBQueryResult:
    """Combined IEDB query result for a peptide"""
    peptide: str
    hla_allele: str
    
    # Direct match
    found_in_iedb: bool
    epitopes: List[IEDBEpitope] = field(default_factory=list)
    tcell_assays: List[IEDBTcellAssay] = field(default_factory=list)
    
    # Statistics
    total_assays: int = 0
    positive_assays: int = 0
    negative_assays: int = 0
    response_rate: float = 0.0
    
    # Diseases & contexts
    diseases: List[str] = field(default_factory=list)
    
    # Metadata
    query_timestamp: str = ""
    cached: bool = False
    notes: str = ""
    
    def __repr__(self):
        return (
            f"IEDBQueryResult({self.peptide}, "
            f"found={self.found_in_iedb}, "
            f"assays={self.total_assays}, "
            f"response_rate={self.response_rate:.1%})"
        )


class IEDBQuerier:
    """
    Query IEDB Query API (IQ-API) for real epitope data.
    
    Features:
    - Real-time queries to IEDB
    - Caching to ~/.cache/neoantigens/iedb/
    - Cache TTL: 30 days
    - Handles direct matches, HLA-specific queries
    """
    
    BASE_URL = "https://query-api.iedb.org"
    CACHE_TTL_DAYS = 30
    REQUEST_TIMEOUT = 30  # seconds
    
    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize IEDB querier.
        
        Args:
            cache_dir: Path to cache directory. Default: ~/.cache/neoantigens/iedb/
            verbose: Print debug info
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "neoantigens" / "iedb"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        if self.verbose:
            print(f"📁 IEDB cache: {self.cache_dir}")
    
    def _cache_key(self, peptide: str, hla_allele: str = "") -> str:
        """Generate cache key from query parameters"""
        key_str = f"{peptide}_{hla_allele}".lower()
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached result if fresh"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check TTL
            cached_at = datetime.fromisoformat(cached.get('query_timestamp', '2000-01-01'))
            age = datetime.now() - cached_at
            
            if age > timedelta(days=self.CACHE_TTL_DAYS):
                if self.verbose:
                    print(f"  ⏰ Cache expired ({age.days} days old)")
                return None
            
            if self.verbose:
                print(f"  💾 Using cached result ({age.days} days old)")
            
            return cached
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _save_cache(self, cache_key: str, data: Dict) -> None:
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            if self.verbose:
                print(f"  ⚠️  Cache write failed: {e}")
    
    def _search_epitopes(self, peptide: str) -> List[Dict]:
        """
        Search IEDB epitope_search endpoint by linear sequence.
        
        Args:
            peptide: Amino acid sequence
            
        Returns:
            List of epitope records as dicts
        """
        url = f"{self.BASE_URL}/epitope_search"
        params = {
            'linear_sequence': f'eq.{peptide}'
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if self.verbose:
                print(f"  ⚠️  IEDB epitope search failed: {e}")
            return []
    
    def _search_tcell_assays(self, peptide: str, hla_allele: str = "") -> List[Dict]:
        """
        Search IEDB tcell_search endpoint for T cell assays.
        
        Args:
            peptide: Amino acid sequence
            hla_allele: Optional HLA filter (e.g., "HLA-A*02:01")
            
        Returns:
            List of T cell assay records
        """
        url = f"{self.BASE_URL}/tcell_search"
        params = {
            'linear_sequence': f'eq.{peptide}'
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            results = response.json()
            
            # Filter by HLA if specified
            if hla_allele and results:
                # IEDB stores HLA in various fields, try to match
                hla_normalized = hla_allele.replace("HLA-", "").replace("*", "").replace(":", "")
                filtered = []
                for r in results:
                    # Check common HLA fields
                    mhc_str = json.dumps(r).lower()
                    if hla_normalized.lower() in mhc_str or hla_allele.lower() in mhc_str:
                        filtered.append(r)
                results = filtered if filtered else results
            
            return results
        except requests.RequestException as e:
            if self.verbose:
                print(f"  ⚠️  IEDB T cell search failed: {e}")
            return []
    
    def query_peptide(self, peptide: str, hla_allele: str = "") -> IEDBQueryResult:
        """
        Query IEDB for a peptide with optional HLA filter.
        
        Args:
            peptide: Amino acid sequence (e.g., "NLVPMVATV")
            hla_allele: Optional HLA allele (e.g., "HLA-A*02:01")
            
        Returns:
            IEDBQueryResult with all matching data
        """
        if self.verbose:
            print(f"🔍 IEDB query: {peptide} + {hla_allele or '(any HLA)'}")
        
        # Check cache
        cache_key = self._cache_key(peptide, hla_allele)
        cached = self._get_cached(cache_key)
        if cached:
            result = self._dict_to_result(cached)
            result.cached = True
            return result
        
        # Query API
        if self.verbose:
            print(f"  📡 Querying IEDB API...")
        
        epitope_records = self._search_epitopes(peptide)
        tcell_records = self._search_tcell_assays(peptide, hla_allele)
        
        # Parse epitopes
        epitopes = []
        for rec in epitope_records:
            # Extract relevant fields safely
            structure_desc = rec.get('structure_descriptions', [])
            sequence = structure_desc[0] if structure_desc else peptide
            
            epitope = IEDBEpitope(
                epitope_id=str(rec.get('structure_id', '')),
                sequence=sequence,
                epitope_iri=rec.get('structure_iri', ''),
                source_organism=str(rec.get('source_organism_name', '')),
                source_antigen=str(rec.get('source_antigen_name', '')),
                structure_type=rec.get('structure_type', '')
            )
            epitopes.append(epitope)
        
        # Parse T cell assays
        tcell_assays = []
        positive_count = 0
        negative_count = 0
        diseases_set = set()
        
        for rec in tcell_records:
            # Extract assay info
            assay_response = str(rec.get('qualitative_measure', 'Unknown'))
            disease = str(rec.get('disease_name', ''))
            
            if 'positive' in assay_response.lower():
                positive_count += 1
            elif 'negative' in assay_response.lower():
                negative_count += 1
            
            if disease:
                diseases_set.add(disease)
            
            assay = IEDBTcellAssay(
                epitope_sequence=peptide,
                mhc_allele=str(rec.get('mhc_allele_name', '')),
                assay_type=str(rec.get('assay_method', '')),
                assay_response=assay_response,
                host_organism=str(rec.get('host_organism_name', '')),
                disease=disease,
                reference_id=str(rec.get('reference_id', ''))
            )
            tcell_assays.append(assay)
        
        total = len(tcell_assays)
        response_rate = positive_count / total if total > 0 else 0.0
        
        result = IEDBQueryResult(
            peptide=peptide,
            hla_allele=hla_allele,
            found_in_iedb=len(epitopes) > 0 or len(tcell_assays) > 0,
            epitopes=epitopes,
            tcell_assays=tcell_assays,
            total_assays=total,
            positive_assays=positive_count,
            negative_assays=negative_count,
            response_rate=response_rate,
            diseases=list(diseases_set),
            query_timestamp=datetime.now().isoformat(),
            cached=False,
            notes=f"Found {len(epitopes)} epitopes, {total} T cell assays"
        )
        
        # Save to cache
        self._save_cache(cache_key, asdict(result))
        
        if self.verbose:
            print(f"  ✓ Found {len(epitopes)} epitopes, "
                  f"{total} assays ({positive_count} positive)")
        
        return result
    
    def _dict_to_result(self, data: Dict) -> IEDBQueryResult:
        """Convert cached dict back to IEDBQueryResult"""
        epitopes = [IEDBEpitope(**e) for e in data.get('epitopes', [])]
        tcell_assays = [IEDBTcellAssay(**a) for a in data.get('tcell_assays', [])]
        
        return IEDBQueryResult(
            peptide=data['peptide'],
            hla_allele=data['hla_allele'],
            found_in_iedb=data['found_in_iedb'],
            epitopes=epitopes,
            tcell_assays=tcell_assays,
            total_assays=data.get('total_assays', 0),
            positive_assays=data.get('positive_assays', 0),
            negative_assays=data.get('negative_assays', 0),
            response_rate=data.get('response_rate', 0.0),
            diseases=data.get('diseases', []),
            query_timestamp=data.get('query_timestamp', ''),
            cached=False,
            notes=data.get('notes', '')
        )
    
    def clear_cache(self) -> int:
        """Clear all cached results. Returns number of files deleted."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


if __name__ == "__main__":
    print("=" * 80)
    print("IEDB REAL API TEST")
    print("=" * 80)
    
    iedb = IEDBQuerier(verbose=True)
    
    test_peptides = [
        ("NLVPMVATV", "HLA-A*02:01"),    # CMV epitope (known)
        ("GILGFVFTL", "HLA-A*02:01"),    # Flu M1 epitope (known)
        ("KRASGSDFVQ", "HLA-A*02:01"),   # KRAS mutation
        ("XXXNOTREAL", "HLA-A*02:01"),   # Should find nothing
    ]
    
    for peptide, hla in test_peptides:
        print(f"\n{'='*80}")
        print(f"Query: {peptide} + {hla}")
        print("="*80)
        
        result = iedb.query_peptide(peptide, hla)
        
        print(f"\n{result}")
        print(f"  Cached: {result.cached}")
        print(f"  Diseases: {result.diseases[:3] if result.diseases else 'None'}")
        
        if result.tcell_assays:
            print(f"\n  Sample T cell assays (first 3):")
            for assay in result.tcell_assays[:3]:
                print(f"    - {assay.assay_type}: {assay.assay_response} "
                      f"({assay.disease or 'no disease'})")