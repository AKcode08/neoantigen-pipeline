"""
Claude Synthesis Engine - Expert-Level Analysis

Synthesizes the 5-layer ensemble output into expert immunologist-grade analysis.

Layers analyzed:
1. Binding (MHCflurry) - Affinity, percentile
2. Presentation (MHCflurry) - Display likelihood, processing
3. Immunogenicity (BigMHC) - T cell response prediction
4. Structure (NeoaPred) - 3D structure features (PDB parsing for deep analysis)
5. Literature (IEDB + PubMed) - Real experimental evidence

Claude provides:
- Expert prediction interpretation
- Deep structural analysis (anchor positions, TCR contact surface, stability)
- Literature contextualization
- Mechanistic reasoning at the biochemical level
- Final recommendation with confidence
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import Counter

import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layer_2_predictors.ensemble import EnsembleResult

# Load environment
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


@dataclass
class ClaudeAnalysis:
    """Claude's expert synthesis of all predictions"""
    peptide: str
    hla_allele: str
    
    # Expert-level interpretation sections
    executive_summary: str = ""
    prediction_interpretation: str = ""
    structural_analysis: str = ""        # NEW - deep structure interpretation
    literature_context: str = ""         # NEW - what does evidence say
    mechanistic_reasoning: str = ""
    confidence_assessment: str = ""
    
    # Final outputs
    final_recommendation: str = "BORDERLINE"
    final_confidence: float = 0.5
    
    # Supporting analysis
    key_strengths: List[str] = field(default_factory=list)
    key_concerns: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    
    # Vaccine context
    vaccine_priority: str = ""           # high / moderate / low / not recommended
    next_steps: List[str] = field(default_factory=list)
    
    # Full response for reference
    raw_response: str = ""
    
    def __repr__(self):
        return (
            f"ClaudeAnalysis({self.peptide}: "
            f"{self.final_recommendation}, "
            f"confidence={self.final_confidence:.2f}, "
            f"priority={self.vaccine_priority})"
        )


class ClaudeSynthesisEngine:
    """
    Expert-level Claude synthesis with deep structural and literature analysis.
    """
    
    DEFAULT_MODEL = "claude-opus-4-1-20250805"
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Claude engine.
        
        Args:
            api_key: Anthropic API key (from environment if not provided)
            model: Claude model to use
        """
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
    
    def synthesize(self, ensemble_result: EnsembleResult) -> ClaudeAnalysis:
        """
        Synthesize ensemble result with expert-level Claude analysis.
        
        Args:
            ensemble_result: Output from 5-layer ensemble predictor
        
        Returns:
            ClaudeAnalysis with deep expert interpretation
        """
        
        # Extract structural features from PDB if available
        structure_features = self._extract_structure_features(ensemble_result)
        
        # Extract literature context
        literature_context = self._extract_literature_context(ensemble_result)
        
        # Build expert prompt
        prompt = self._build_expert_prompt(
            ensemble_result, structure_features, literature_context
        )
        
        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_response = response.content[0].text
        
        # Parse Claude's structured response
        analysis = self._parse_response(raw_response, ensemble_result)
        
        return analysis
    
    def _extract_structure_features(self, result: EnsembleResult) -> Dict:
        """
        Extract structural features from PDB file for Claude analysis.
        
        Returns:
            Dict with structural metrics: anchor residues, TCR contact, etc.
        """
        features = {
            "available": False,
            "atom_count": result.structure_atom_count,
            "pdb_path": result.structure_pdb_path,
            "quality": result.structure_level,
        }
        
        if not result.structure_pdb_path or not os.path.exists(result.structure_pdb_path):
            return features
        
        try:
            atoms = []
            with open(result.structure_pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        atoms.append({
                            'atom_name': line[12:16].strip(),
                            'residue': line[17:20].strip(),
                            'residue_num': int(line[22:26].strip()),
                            'chain': line[21],
                            'x': float(line[30:38]),
                            'y': float(line[38:46]),
                            'z': float(line[46:54]),
                        })
            
            if not atoms:
                return features
            
            # Identify peptide chain (typically chain C or last chain)
            # For NeoaPred output, peptide is usually the shorter chain
            chains = Counter(a['chain'] for a in atoms)
            
            # Peptide chain has fewest residues
            chain_residues = {}
            for chain in chains:
                residues = set(a['residue_num'] for a in atoms if a['chain'] == chain)
                chain_residues[chain] = len(residues)
            
            peptide_chain = min(chain_residues, key=chain_residues.get)
            peptide_atoms = [a for a in atoms if a['chain'] == peptide_chain]
            
            # Get peptide residue sequence with positions
            peptide_residues = {}
            for atom in peptide_atoms:
                if atom['residue_num'] not in peptide_residues:
                    peptide_residues[atom['residue_num']] = atom['residue']
            
            sorted_positions = sorted(peptide_residues.keys())
            
            # Extract sequence from residues (3-letter to 1-letter)
            three_to_one = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
            }
            
            sequence_from_pdb = ''.join(
                three_to_one.get(peptide_residues[p], 'X') 
                for p in sorted_positions
            )
            
            features["available"] = True
            features["peptide_chain"] = peptide_chain
            features["peptide_length"] = len(sorted_positions)
            features["sequence_in_structure"] = sequence_from_pdb
            features["total_chains"] = len(chains)
            features["mhc_chains"] = [c for c in chains if c != peptide_chain]
            
            # Calculate structural properties
            # 1. Center of mass per residue
            residue_centers = {}
            for pos in sorted_positions:
                pos_atoms = [a for a in peptide_atoms if a['residue_num'] == pos]
                if pos_atoms:
                    cx = sum(a['x'] for a in pos_atoms) / len(pos_atoms)
                    cy = sum(a['y'] for a in pos_atoms) / len(pos_atoms)
                    cz = sum(a['z'] for a in pos_atoms) / len(pos_atoms)
                    residue_centers[pos] = (cx, cy, cz)
            
            # 2. Distance to MHC (proxy for burial)
            mhc_atoms = [a for a in atoms if a['chain'] != peptide_chain]
            
            if mhc_atoms:
                burial_per_residue = {}
                for pos, center in residue_centers.items():
                    # Min distance from this residue's center to any MHC atom
                    min_dist = min(
                        ((center[0]-m['x'])**2 + (center[1]-m['y'])**2 + (center[2]-m['z'])**2)**0.5
                        for m in mhc_atoms
                    )
                    burial_per_residue[pos] = min_dist
                
                features["residue_distances_to_mhc"] = {
                    f"P{i+1}({peptide_residues[p]})": round(burial_per_residue[p], 2)
                    for i, p in enumerate(sorted_positions)
                }
                
                # Identify likely anchor (closest to MHC) vs TCR-facing (furthest)
                sorted_by_dist = sorted(burial_per_residue.items(), key=lambda x: x[1])
                
                if len(sorted_by_dist) >= 2:
                    closest = sorted_by_dist[:2]
                    furthest = sorted_by_dist[-2:]
                    
                    features["likely_anchors"] = [
                        f"P{sorted_positions.index(p)+1}({peptide_residues[p]}, {d:.1f}Å)"
                        for p, d in closest
                    ]
                    features["tcr_facing"] = [
                        f"P{sorted_positions.index(p)+1}({peptide_residues[p]}, {d:.1f}Å)"
                        for p, d in furthest
                    ]
            
        except (IOError, ValueError, KeyError) as e:
            features["error"] = str(e)
        
        return features
    
    def _extract_literature_context(self, result: EnsembleResult) -> Dict:
        """Extract literature evidence in a structured way for Claude"""
        context = {
            "evidence_level": result.literature_evidence_level,
            "evidence_score": result.literature_evidence_score,
            "iedb_assays": result.iedb_assay_count,
            "iedb_response_rate": result.iedb_response_rate,
            "pubmed_papers": result.pubmed_paper_count,
            "summary": result.literature_summary,
        }
        
        # Include top papers if available
        if result.literature_evidence and result.literature_evidence.pubmed_result:
            papers = result.literature_evidence.pubmed_result.papers[:3]
            context["top_papers"] = [
                {
                    "title": p.title,
                    "authors": p.authors[:3] if p.authors else [],
                    "year": p.year,
                    "journal": p.journal,
                    "pmid": p.pmid,
                    "abstract_snippet": p.abstract[:300] if p.abstract else "",
                }
                for p in papers
            ]
        
        # Include IEDB context
        if result.literature_evidence and result.literature_evidence.iedb_result:
            iedb = result.literature_evidence.iedb_result
            context["iedb_diseases"] = iedb.diseases[:5] if iedb.diseases else []
            context["iedb_positive_assays"] = iedb.positive_assays
            context["iedb_negative_assays"] = iedb.negative_assays
        
        return context
    
    def _build_expert_prompt(self, 
                              result: EnsembleResult, 
                              structure_features: Dict,
                              literature_context: Dict) -> str:
        """Build expert immunologist prompt with all data layers"""
        
        # Format structure section
        structure_section = self._format_structure_section(structure_features, result.peptide)
        
        # Format literature section
        literature_section = self._format_literature_section(literature_context)
        
        prompt = f"""You are a senior computational immunologist with expertise in peptide-MHC structural biology, T cell immunology, and cancer vaccine development. You have 20+ years of experience designing personalized cancer vaccines and analyzing neoantigen candidates.

A pipeline has analyzed a candidate neoantigen using 5 prediction layers. Your task is to provide an expert-level analysis suitable for a biotech client deciding whether to include this peptide in a personalized cancer vaccine.

════════════════════════════════════════════════════════════════════════════════════
PEPTIDE CANDIDATE
════════════════════════════════════════════════════════════════════════════════════
Sequence: {result.peptide}
Length: {len(result.peptide)} aa
HLA Restriction: {result.hla_allele}

════════════════════════════════════════════════════════════════════════════════════
LAYER 1 — MHC BINDING (MHCflurry)
════════════════════════════════════════════════════════════════════════════════════
Binding Affinity: {result.binding_affinity_nm:.1f} nM
Binding Category: {result.binding_level}
Percentile Rank: {result.binding_percentile:.3f}% (lower = stronger binder)

Reference thresholds:
- <50 nM = very strong binder
- 50-500 nM = strong binder  
- 500-5000 nM = intermediate
- >5000 nM = weak/non-binder

════════════════════════════════════════════════════════════════════════════════════
LAYER 2 — ANTIGEN PRESENTATION (MHCflurry Presentation)
════════════════════════════════════════════════════════════════════════════════════
Presentation Score: {result.presentation_score:.3f} (0-1, higher = more likely displayed)
Presentation Category: {result.presentation_level}
Processing Score: {result.processing_score:.3f} (proteasomal cleavage probability)

This integrates antigen processing + MHC binding to predict actual cell-surface display.

════════════════════════════════════════════════════════════════════════════════════
LAYER 3 — IMMUNOGENICITY (BigMHC IM)
════════════════════════════════════════════════════════════════════════════════════
Immunogenicity Score: {result.immunogenicity_score if result.immunogenicity_score is not None else 'N/A'}
Immunogenicity Category: {result.immunogenicity_level}

BigMHC IM is a deep learning model trained on validated T cell response data.
This predicts the probability of triggering a CD8+ T cell response.

════════════════════════════════════════════════════════════════════════════════════
LAYER 4 — STRUCTURAL ANALYSIS (NeoaPred PepConf 3D Structure)
════════════════════════════════════════════════════════════════════════════════════
{structure_section}

════════════════════════════════════════════════════════════════════════════════════
LAYER 5 — LITERATURE EVIDENCE (IEDB + PubMed)
════════════════════════════════════════════════════════════════════════════════════
{literature_section}

════════════════════════════════════════════════════════════════════════════════════
ENSEMBLE CONSENSUS
════════════════════════════════════════════════════════════════════════════════════
Consensus Score: {result.consensus_score:.3f}/1.0
Signal Agreement: {result.agreement_level}
Ensemble Recommendation: {result.recommendation}
Ensemble Confidence: {result.confidence:.2f}

════════════════════════════════════════════════════════════════════════════════════
YOUR EXPERT ANALYSIS
════════════════════════════════════════════════════════════════════════════════════

Provide a comprehensive expert analysis using EXACTLY this format with these EXACT section headers:

## EXECUTIVE_SUMMARY
A 2-3 sentence executive summary of the candidate's overall quality and recommendation. This is what a biotech CSO would read first.

## PREDICTION_INTERPRETATION
Expert interpretation of the prediction scores (3-4 sentences). What do these numbers mean biologically? Reference specific nM values, percentiles, and scores.

## STRUCTURAL_ANALYSIS
DEEP structural analysis (4-6 sentences). Based on the 3D structure data:
- Comment on anchor residues (P2, P{len(result.peptide)} positions) and their hydrophobicity/suitability for this HLA pocket
- Analyze TCR-facing residues (central positions P4-P6) and their immunogenic potential
- Discuss peptide stability indicators (charged residues, prolines, hydrophobic patches)
- Mention burial vs exposure based on distance metrics if available
- Consider allele-specific preferences (HLA-A*02:01 prefers hydrophobic anchors at P2/P9)

If structure is unavailable, analyze the sequence biochemically instead and note the limitation.

## LITERATURE_CONTEXT
What does the published evidence say (3-5 sentences)? Discuss:
- IEDB assay results: response rate, sample size
- Disease contexts where this epitope appears
- Notable papers from PubMed
- Whether literature confirms or contradicts the predictions

## MECHANISTIC_REASONING
Biological mechanism (3-4 sentences). Explain WHY this peptide will or won't work:
- Source protein context (if recognizable from sequence)
- T cell repertoire considerations
- Self-tolerance concerns
- Tumor-specific expression context

## CONFIDENCE_ASSESSMENT
How confident are we (2-3 sentences)? Address:
- Agreement between predictors
- Strength of literature backing
- Quality of structural data
- Known limitations of the methods

## FINAL_RECOMMENDATION
[INCLUDE | EXCLUDE | BORDERLINE]
(Must be exactly one word)

## FINAL_CONFIDENCE
[0.0-1.0]
(A single decimal number)

## VACCINE_PRIORITY
[high | moderate | low | not_recommended]
(Single word/phrase)

## KEY_STRENGTHS
- Bullet 1 (specific, references data)
- Bullet 2
- Bullet 3

## KEY_CONCERNS
- Bullet 1
- Bullet 2

## CAVEATS
- Limitation 1
- Limitation 2

## NEXT_STEPS
- Action 1 (e.g., "Validate with HLA-binding assay")
- Action 2 (e.g., "Test in HLA-A*02:01 transgenic mice")
- Action 3

CRITICAL GUIDELINES:
- Be SPECIFIC: reference actual nM values, percentiles, residue positions
- Be HONEST: don't overstate; acknowledge uncertainty
- Be EXPERT: use proper immunology terminology
- Be PRACTICAL: this informs real vaccine manufacturing decisions
- Use the EXACT section headers above (with ## prefix) so the parser works
"""
        
        return prompt
    
    def _format_structure_section(self, features: Dict, peptide: str) -> str:
        """Format structure features for the prompt"""
        if not features.get("available"):
            return f"""Structure Quality: {features.get('quality', 'unavailable')}
PDB File: Not generated (NeoaPred unavailable or failed)

Note: 3D structure not available. Provide analysis based on sequence biochemistry.
Use the following sequence-based reasoning:
- Anchor positions: P2={peptide[1] if len(peptide) >= 2 else '?'}, P{len(peptide)}={peptide[-1]}
- Central TCR-facing: positions P4-P6 = {peptide[3:6] if len(peptide) >= 6 else peptide}"""
        
        lines = [
            f"Structure Quality: {features['quality']}",
            f"PDB File: {features.get('pdb_path', 'N/A')}",
            f"Total Atoms: {features.get('atom_count', 0)}",
            f"Peptide Chain: {features.get('peptide_chain', 'N/A')}",
            f"Sequence in Structure: {features.get('sequence_in_structure', 'N/A')}",
            f"MHC Chains Present: {features.get('mhc_chains', [])}",
        ]
        
        if "residue_distances_to_mhc" in features:
            lines.append("\nResidue Distances to MHC (lower = more buried in pocket):")
            for pos_label, dist in features["residue_distances_to_mhc"].items():
                lines.append(f"  {pos_label}: {dist} Å")
        
        if "likely_anchors" in features:
            lines.append(f"\nLikely anchor residues (closest to MHC): {', '.join(features['likely_anchors'])}")
        
        if "tcr_facing" in features:
            lines.append(f"Likely TCR-facing residues (furthest from MHC): {', '.join(features['tcr_facing'])}")
        
        return "\n".join(lines)
    
    def _format_literature_section(self, context: Dict) -> str:
        """Format literature context for the prompt"""
        lines = [
            f"Evidence Level: {context['evidence_level']}",
            f"Evidence Score: {context['evidence_score']:.2f}",
            f"",
            f"IEDB (Immune Epitope Database):",
            f"  Total T cell assays: {context['iedb_assays']}",
            f"  Positive assays: {context.get('iedb_positive_assays', 0)}",
            f"  Negative assays: {context.get('iedb_negative_assays', 0)}",
            f"  Response rate: {context['iedb_response_rate']:.1%}",
        ]
        
        if context.get("iedb_diseases"):
            lines.append(f"  Disease contexts: {', '.join(context['iedb_diseases'])}")
        
        lines.append(f"\nPubMed:")
        lines.append(f"  Papers found: {context['pubmed_papers']}")
        
        if context.get("top_papers"):
            lines.append(f"\n  Top papers:")
            for i, paper in enumerate(context["top_papers"], 1):
                first_author = paper["authors"][0] if paper["authors"] else "Unknown"
                lines.append(f"\n  [{i}] {paper['title']}")
                lines.append(f"      {first_author} et al. ({paper['year']}) | {paper['journal']} | PMID: {paper['pmid']}")
                if paper["abstract_snippet"]:
                    lines.append(f"      Abstract excerpt: {paper['abstract_snippet']}...")
        
        return "\n".join(lines)
    
    def _parse_response(self, response_text: str, ensemble_result: EnsembleResult) -> ClaudeAnalysis:
        """Parse Claude's structured response with ## section headers"""
        
        sections = {
            "EXECUTIVE_SUMMARY": "",
            "PREDICTION_INTERPRETATION": "",
            "STRUCTURAL_ANALYSIS": "",
            "LITERATURE_CONTEXT": "",
            "MECHANISTIC_REASONING": "",
            "CONFIDENCE_ASSESSMENT": "",
            "FINAL_RECOMMENDATION": "BORDERLINE",
            "FINAL_CONFIDENCE": "0.5",
            "VACCINE_PRIORITY": "moderate",
            "KEY_STRENGTHS": [],
            "KEY_CONCERNS": [],
            "CAVEATS": [],
            "NEXT_STEPS": [],
        }
        
        # Split by ## headers
        pattern = r'##\s+(\w+)\s*\n(.*?)(?=##\s+\w+|\Z)'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for header, content in matches:
            header = header.strip().upper()
            content = content.strip()
            
            if header in sections:
                if header in ["KEY_STRENGTHS", "KEY_CONCERNS", "CAVEATS", "NEXT_STEPS"]:
                    # Parse bullet points
                    bullets = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith(('-', '•', '*')):
                            bullets.append(line.lstrip('-•* ').strip())
                        elif line and not line[0].isdigit():
                            bullets.append(line)
                    sections[header] = [b for b in bullets if b]
                else:
                    sections[header] = content
        
        # Parse recommendation (extract just the word)
        rec_text = sections["FINAL_RECOMMENDATION"]
        for rec in ["INCLUDE", "EXCLUDE", "BORDERLINE"]:
            if rec in rec_text.upper():
                sections["FINAL_RECOMMENDATION"] = rec
                break
        else:
            sections["FINAL_RECOMMENDATION"] = "BORDERLINE"
        
        # Parse confidence
        conf_text = sections["FINAL_CONFIDENCE"]
        conf_match = re.search(r'(\d+\.?\d*)', conf_text)
        if conf_match:
            try:
                conf_val = float(conf_match.group(1))
                if conf_val > 1.0:
                    conf_val = conf_val / 100.0  # Handle percentage format
                sections["FINAL_CONFIDENCE"] = max(0.0, min(1.0, conf_val))
            except ValueError:
                sections["FINAL_CONFIDENCE"] = 0.5
        else:
            sections["FINAL_CONFIDENCE"] = 0.5
        
        # Parse priority
        priority_text = sections["VACCINE_PRIORITY"].lower()
        for level in ["high", "moderate", "low", "not_recommended", "not recommended"]:
            if level in priority_text:
                sections["VACCINE_PRIORITY"] = level.replace(" ", "_")
                break
        
        return ClaudeAnalysis(
            peptide=ensemble_result.peptide,
            hla_allele=ensemble_result.hla_allele,
            executive_summary=sections["EXECUTIVE_SUMMARY"],
            prediction_interpretation=sections["PREDICTION_INTERPRETATION"],
            structural_analysis=sections["STRUCTURAL_ANALYSIS"],
            literature_context=sections["LITERATURE_CONTEXT"],
            mechanistic_reasoning=sections["MECHANISTIC_REASONING"],
            confidence_assessment=sections["CONFIDENCE_ASSESSMENT"],
            final_recommendation=sections["FINAL_RECOMMENDATION"],
            final_confidence=sections["FINAL_CONFIDENCE"],
            vaccine_priority=sections["VACCINE_PRIORITY"],
            key_strengths=sections["KEY_STRENGTHS"][:5],
            key_concerns=sections["KEY_CONCERNS"][:5],
            caveats=sections["CAVEATS"][:3],
            next_steps=sections["NEXT_STEPS"][:5],
            raw_response=response_text
        )


if __name__ == "__main__":
    from src.layer_2_predictors.ensemble import EnsemblePredictor
    
    print("=" * 100)
    print("CLAUDE EXPERT SYNTHESIS ENGINE - TEST")
    print("=" * 100)
    
    # Test with the strongest candidate
    ensemble = EnsemblePredictor()
    
    test_peptide = "NLVPMVATV"
    test_hla = "HLA-A*02:01"
    
    print(f"\nRunning ensemble for {test_peptide} + {test_hla}...")
    ensemble_result = ensemble.predict(test_peptide, test_hla)
    
    if ensemble_result:
        print(f"\n✓ Ensemble done: {ensemble_result.recommendation}")
        print(f"  Consensus: {ensemble_result.consensus_score:.2f}")
        
        # Synthesize with Claude
        print(f"\n{'='*100}")
        print("Calling Claude for expert synthesis...")
        print(f"{'='*100}")
        
        claude = ClaudeSynthesisEngine()
        analysis = claude.synthesize(ensemble_result)
        
        print(f"\n{'='*100}")
        print(f"CLAUDE EXPERT ANALYSIS — {analysis.peptide}")
        print(f"{'='*100}")
        
        print(f"\n📋 EXECUTIVE SUMMARY")
        print("─" * 100)
        print(analysis.executive_summary)
        
        print(f"\n🔬 PREDICTION INTERPRETATION")
        print("─" * 100)
        print(analysis.prediction_interpretation)
        
        print(f"\n🧬 STRUCTURAL ANALYSIS")
        print("─" * 100)
        print(analysis.structural_analysis)
        
        print(f"\n📚 LITERATURE CONTEXT")
        print("─" * 100)
        print(analysis.literature_context)
        
        print(f"\n⚙️  MECHANISTIC REASONING")
        print("─" * 100)
        print(analysis.mechanistic_reasoning)
        
        print(f"\n🎯 CONFIDENCE ASSESSMENT")
        print("─" * 100)
        print(analysis.confidence_assessment)
        
        print(f"\n{'='*100}")
        print(f"🏁 FINAL RECOMMENDATION: {analysis.final_recommendation}")
        print(f"   Confidence: {analysis.final_confidence:.2f}")
        print(f"   Vaccine Priority: {analysis.vaccine_priority}")
        print(f"{'='*100}")
        
        print(f"\n✅ KEY STRENGTHS")
        for i, s in enumerate(analysis.key_strengths, 1):
            print(f"   {i}. {s}")
        
        print(f"\n⚠️  KEY CONCERNS")
        for i, c in enumerate(analysis.key_concerns, 1):
            print(f"   {i}. {c}")
        
        print(f"\n📝 CAVEATS")
        for i, c in enumerate(analysis.caveats, 1):
            print(f"   {i}. {c}")
        
        print(f"\n🎬 NEXT STEPS")
        for i, n in enumerate(analysis.next_steps, 1):
            print(f"   {i}. {n}")