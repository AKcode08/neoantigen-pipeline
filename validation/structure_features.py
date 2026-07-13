"""
Structure Feature Extractor — Real Features from NeoaPred PDB Files

Replaces the previous structure scoring (which returned a constant 0.8 for
any peptide that folded successfully — discriminative power = 0).

This module reads the per-peptide relaxed PDB already produced by NeoaPred
and computes four interpretable structural features. Each is normalized to
[0,1] and combined with equal weights into a single structure score.

DESIGN PRINCIPLES (explicit, so it's clear what we're NOT claiming)
-------------------------------------------------------------------
1. Equal weights, no fitting. We do not tune against any benchmark.
   Any weights tuned on ITSNdb would memorize ITSNdb.
2. Single-structure only. We did not fold WT counterparts, so we use
   single-structure proxies for "foreignness," not true mutant-vs-WT.
3. We prefer features with mechanistic justification from the
   immunology literature over features that look smart.
4. The output is meant to be COMBINED with binding/presentation/immuno —
   not replace them. ITSNdb already filtered for binding, so this score
   in isolation is not expected to be strongly predictive.

THE FOUR FEATURES
-----------------
1. TCR contact exposure       — mean (peptide residue) → (MHC) distance at
                                positions P4-P6 (9-mers) or P4-P7 (10-mers).
                                Higher = TCR sees more of the peptide. Bounded.
2. Mutation TCR visibility    — distance to MHC at the mutated position,
                                attenuated 0.3x if it's an anchor position
                                (TCR can't see anchor mutations well).
3. Anchor compatibility       — does the actual P2 and PΩ residue match the
                                published preference for this HLA allele.
                                Pure sequence + lookup table. Well-established.
4. TCR-facing hydrophobicity  — Chowell et al. 2015 showed TCR-facing
                                hydrophobicity correlates with immunogenicity.
                                Kyte-Doolittle scale at central positions.

NORMALIZATION
-------------
Each feature normalized to [0,1] using fixed thresholds chosen from
literature / structural conventions, NOT fit to ITSNdb.

OUTPUT
------
StructureFeatures dataclass with all four features + a `combined_score`
(unweighted mean). Compatible drop-in for the existing structure layer.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import Counter
import math


# ---------------------------------------------------------------------------
# Constants — chosen from literature / structural conventions, NOT fit
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity. Used by ~everyone in immunoinformatics.
KYTE_DOOLITTLE = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}
KD_MIN, KD_MAX = -4.5, 4.5

# Anchor preferences from published HLA peptide-binding motifs.
# Sources: NetMHCpan training motifs, Rammensee's SYFPEITHI, etc.
# Format: {"HLA-X*YY:ZZ": {"P2": set(allowed_AAs), "PΩ": set(allowed_AAs)}}
# We use a coarse "preferred set" — present or not — rather than continuous PSSMs,
# to keep this tractable and not require allele-specific frequency files.
ANCHOR_PREFERENCES = {
    # The big ones for ITSNdb
    "HLA-A*02:01": {"P2": set("LMIVAT"), "POmega": set("LMIVA")},
    "HLA-A*01:01": {"P2": set("TSY"),    "POmega": set("Y")},
    "HLA-A*03:01": {"P2": set("LMIVT"),  "POmega": set("KRY")},
    "HLA-A*11:01": {"P2": set("LMIVT"),  "POmega": set("KR")},
    "HLA-A*24:02": {"P2": set("YFW"),    "POmega": set("FLIWM")},
    "HLA-A*25:01": {"P2": set("VLMIT"),  "POmega": set("WFL")},
    "HLA-A*68:01": {"P2": set("VTIAL"),  "POmega": set("KR")},
    "HLA-B*07:02": {"P2": set("P"),      "POmega": set("LFIM")},
    "HLA-B*08:01": {"P2": set("KR"),     "POmega": set("LFI")},
    "HLA-B*27:05": {"P2": set("R"),      "POmega": set("KRYFLM")},
    "HLA-B*35:01": {"P2": set("P"),      "POmega": set("YFM")},
    "HLA-B*35:03": {"P2": set("P"),      "POmega": set("YFM")},
    "HLA-B*44:03": {"P2": set("E"),      "POmega": set("YFW")},
    "HLA-B*57:01": {"P2": set("AT"),     "POmega": set("WFY")},
}

# 3-letter -> 1-letter amino acid code
AA_3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E',
    'GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V',
}

# Distance normalization: typical MHC groove gives 3-8 Å for buried residues,
# 8-15 Å for solvent/TCR-exposed central residues. These are conventions
# from looking at canonical pMHC structures (e.g. 1HHK, 2BNQ, 3MRG),
# NOT tuned on ITSNdb.
TCR_DIST_MIN = 3.0   # tightly buried in pocket
TCR_DIST_MAX = 12.0  # well-exposed to TCR side


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructureFeatures:
    """Per-peptide structural features and combined score."""
    peptide: str
    hla_allele: str

    # The four features, each on [0,1]
    tcr_contact_exposure: Optional[float] = None
    mutation_tcr_visibility: Optional[float] = None
    anchor_compatibility: Optional[float] = None
    tcr_hydrophobicity: Optional[float] = None

    # Equal-weight mean of available features (ignores None)
    combined_score: Optional[float] = None

    # Provenance / diagnostics
    feature_count: int = 0
    notes: str = ""

    # Optional raw measurements (debugging / Claude context)
    raw_central_mhc_dist_A: Optional[float] = None
    raw_mut_position_mhc_dist_A: Optional[float] = None
    sequence_in_structure: str = ""

    def __repr__(self):
        if self.combined_score is None:
            return f"StructureFeatures({self.peptide}: no features)"
        return (f"StructureFeatures({self.peptide}: "
                f"combined={self.combined_score:.2f}, "
                f"tcr={self.tcr_contact_exposure}, "
                f"mut={self.mutation_tcr_visibility}, "
                f"anc={self.anchor_compatibility}, "
                f"hyd={self.tcr_hydrophobicity})")


# ---------------------------------------------------------------------------
# PDB parsing — minimal, no Biopython dependency
# ---------------------------------------------------------------------------

def _parse_atoms(pdb_path: str) -> List[Dict]:
    """Return list of ATOM records. Same parsing the Claude engine uses."""
    atoms = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                try:
                    atoms.append({
                        'chain': line[21],
                        'res_num': int(line[22:26].strip()),
                        'res_name': line[17:20].strip(),
                        'atom_name': line[12:16].strip(),
                        'x': float(line[30:38]),
                        'y': float(line[38:46]),
                        'z': float(line[46:54]),
                    })
                except (ValueError, IndexError):
                    continue
    except IOError:
        return []
    return atoms


def _identify_peptide_chain(atoms: List[Dict]) -> Optional[str]:
    """
    Identify which chain is the peptide vs. the MHC.
    Peptide chain has the fewest unique residues (8-15) vs MHC's ~270+.
    """
    chains_to_residues: Dict[str, set] = {}
    for a in atoms:
        chains_to_residues.setdefault(a['chain'], set()).add(a['res_num'])
    if not chains_to_residues:
        return None
    # Pick chain with fewest residues, sanity-bounded
    candidate = min(chains_to_residues, key=lambda c: len(chains_to_residues[c]))
    if 6 <= len(chains_to_residues[candidate]) <= 20:
        return candidate
    return None


def _residue_to_mhc_distance(
    peptide_atoms: List[Dict],
    mhc_atoms: List[Dict],
    peptide_residue_num: int,
) -> Optional[float]:
    """
    Minimum distance from any atom of one peptide residue to any MHC atom.
    Proxy for "how buried in the MHC groove" — small = buried, large = exposed.
    """
    res_atoms = [a for a in peptide_atoms if a['res_num'] == peptide_residue_num]
    if not res_atoms or not mhc_atoms:
        return None
    # Minimum atom-pair distance
    min_d2 = float('inf')
    for ra in res_atoms:
        for ma in mhc_atoms:
            d2 = ((ra['x']-ma['x'])**2 +
                  (ra['y']-ma['y'])**2 +
                  (ra['z']-ma['z'])**2)
            if d2 < min_d2:
                min_d2 = d2
    return math.sqrt(min_d2) if min_d2 != float('inf') else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tcr_positions(length: int) -> List[int]:
    """
    Which 1-indexed peptide positions face the TCR (vs. anchor down into MHC).
    Convention from canonical pMHC-I structures: central positions, excluding
    P1, P2 (anchor-ish) and the C-terminal anchor.
    9-mer:  P4, P5, P6        (sometimes P7 too; we use 4-6 for stricter signal)
    10-mer: P4, P5, P6, P7
    11+mer: P4 through P(length-3)
    """
    if length == 9:
        return [4, 5, 6]
    if length == 10:
        return [4, 5, 6, 7]
    if length >= 11:
        return list(range(4, length - 2))
    # short peptides (8-mers): use middle only
    return [4, 5]


# ---------------------------------------------------------------------------
# The four features
# ---------------------------------------------------------------------------

def feature_tcr_contact_exposure(
    peptide_atoms: List[Dict],
    mhc_atoms: List[Dict],
    peptide_length: int,
    sorted_residue_nums: List[int],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Mean MHC-distance of TCR-facing residues, normalized to [0,1].
    Returns (normalized_score, raw_mean_distance_angstroms).
    """
    tcr_pos = _tcr_positions(peptide_length)
    distances: List[float] = []
    for p in tcr_pos:
        if p - 1 >= len(sorted_residue_nums):
            continue
        residue_num = sorted_residue_nums[p - 1]
        d = _residue_to_mhc_distance(peptide_atoms, mhc_atoms, residue_num)
        if d is not None:
            distances.append(d)
    if not distances:
        return None, None
    mean_d = sum(distances) / len(distances)
    normalized = _clip01((mean_d - TCR_DIST_MIN) / (TCR_DIST_MAX - TCR_DIST_MIN))
    return normalized, mean_d


def feature_mutation_tcr_visibility(
    peptide_atoms: List[Dict],
    mhc_atoms: List[Dict],
    mut_position: int,
    position_type: str,
    sorted_residue_nums: List[int],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Distance to MHC at the mutated position, attenuated for anchor positions.
    Returns (normalized_score, raw_distance_angstroms).
    Anchor mutations get 0.3x — TCR can't see what's pointing down into MHC.
    """
    if mut_position < 1 or mut_position > len(sorted_residue_nums):
        return None, None
    residue_num = sorted_residue_nums[mut_position - 1]
    d = _residue_to_mhc_distance(peptide_atoms, mhc_atoms, residue_num)
    if d is None:
        return None, None
    raw_norm = (d - TCR_DIST_MIN) / (TCR_DIST_MAX - TCR_DIST_MIN)
    # Attenuate for anchor positions
    attenuation = 0.3 if position_type.strip().lower() == "anchor" else 1.0
    return _clip01(raw_norm * attenuation), d


def feature_anchor_compatibility(peptide: str, hla_allele: str) -> Optional[float]:
    """
    Does the actual P2 and PΩ residue match published HLA anchor preferences?
    Returns score in {0.0, 0.5, 1.0}: both match / one matches / neither.
    Pure sequence + table lookup — no PDB needed.
    """
    if len(peptide) < 3:
        return None
    prefs = ANCHOR_PREFERENCES.get(hla_allele)
    if prefs is None:
        return None  # unknown allele -> abstain rather than guess
    p2 = peptide[1]                    # 1-indexed P2 = 0-indexed [1]
    p_omega = peptide[-1]
    matches = 0
    if p2 in prefs.get("P2", set()):
        matches += 1
    if p_omega in prefs.get("POmega", set()):
        matches += 1
    return matches / 2.0


def feature_tcr_hydrophobicity(peptide: str) -> Optional[float]:
    """
    Mean Kyte-Doolittle hydrophobicity at TCR-facing positions.
    Chowell et al. 2015 (PNAS) — TCR-facing hydrophobicity correlates with
    immunogenicity. Higher hydrophobicity at central positions -> higher score.
    Returns score in [0,1].
    """
    tcr_pos = _tcr_positions(len(peptide))
    values = []
    for p in tcr_pos:
        if 1 <= p <= len(peptide):
            aa = peptide[p - 1]
            if aa in KYTE_DOOLITTLE:
                values.append(KYTE_DOOLITTLE[aa])
    if not values:
        return None
    mean_kd = sum(values) / len(values)
    return _clip01((mean_kd - KD_MIN) / (KD_MAX - KD_MIN))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_structure_features(
    peptide: str,
    hla_allele: str,
    pdb_path: Optional[str],
    mut_position: int = -1,
    position_type: str = "",
) -> StructureFeatures:
    """
    Compute all four structural features from a relaxed PDB + sequence metadata.

    pdb_path can be None (e.g. if NeoaPred failed) — the sequence-based
    features (anchor compatibility, hydrophobicity) still compute.
    """
    result = StructureFeatures(peptide=peptide, hla_allele=hla_allele)
    notes: List[str] = []

    # --- Sequence-only features (always available) ---
    result.anchor_compatibility = feature_anchor_compatibility(peptide, hla_allele)
    if result.anchor_compatibility is None:
        notes.append(f"anchor: no preference table for {hla_allele}")

    result.tcr_hydrophobicity = feature_tcr_hydrophobicity(peptide)

    # --- Structure-dependent features ---
    if pdb_path and Path(pdb_path).exists():
        atoms = _parse_atoms(pdb_path)
        if atoms:
            pep_chain = _identify_peptide_chain(atoms)
            if pep_chain:
                pep_atoms = [a for a in atoms if a['chain'] == pep_chain]
                mhc_atoms = [a for a in atoms if a['chain'] != pep_chain]

                # Build sorted residue list + extract sequence from structure
                res_to_name = {}
                for a in pep_atoms:
                    res_to_name.setdefault(a['res_num'], a['res_name'])
                sorted_res = sorted(res_to_name.keys())
                result.sequence_in_structure = ''.join(
                    AA_3_TO_1.get(res_to_name[r], 'X') for r in sorted_res
                )

                tcr_score, raw_tcr_d = feature_tcr_contact_exposure(
                    pep_atoms, mhc_atoms, len(peptide), sorted_res
                )
                result.tcr_contact_exposure = tcr_score
                result.raw_central_mhc_dist_A = raw_tcr_d

                if mut_position > 0:
                    mut_score, raw_mut_d = feature_mutation_tcr_visibility(
                        pep_atoms, mhc_atoms, mut_position,
                        position_type, sorted_res
                    )
                    result.mutation_tcr_visibility = mut_score
                    result.raw_mut_position_mhc_dist_A = raw_mut_d
                else:
                    notes.append("mut_position not provided")
            else:
                notes.append("could not identify peptide chain in PDB")
        else:
            notes.append("no ATOM records parsed")
    else:
        notes.append("no PDB available — sequence-only features")

    # Equal-weight mean over available features
    available = [
        v for v in (
            result.tcr_contact_exposure,
            result.mutation_tcr_visibility,
            result.anchor_compatibility,
            result.tcr_hydrophobicity,
        ) if v is not None
    ]
    result.feature_count = len(available)
    if available:
        result.combined_score = sum(available) / len(available)
    result.notes = "; ".join(notes) if notes else ""
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Structure feature extractor")
    parser.add_argument("--pdb-dir", default="neoapred_structures",
                        help="Directory with *_pep_relaxed.pdb files")
    parser.add_argument("--peptide", help="Single peptide to test")
    parser.add_argument("--hla", help="HLA allele (HLA-A*02:01 form)")
    parser.add_argument("--mut-pos", type=int, default=-1)
    parser.add_argument("--pos-type", default="")
    args = parser.parse_args()

    if args.peptide:
        # Find matching PDB
        pdb_dir = Path(args.pdb_dir)
        hla_short = args.hla.replace("HLA-", "").replace("*", "").replace(":", "") if args.hla else ""
        # Try standard NeoaPred naming convention
        candidate = pdb_dir / f"{args.peptide}_{hla_short}_pep_relaxed.pdb"
        if not candidate.exists():
            # Fallback: any matching peptide
            matches = list(pdb_dir.glob(f"{args.peptide}*_pep_relaxed.pdb"))
            candidate = matches[0] if matches else None

        feats = extract_structure_features(
            peptide=args.peptide,
            hla_allele=args.hla,
            pdb_path=str(candidate) if candidate else None,
            mut_position=args.mut_pos,
            position_type=args.pos_type,
        )
        print(f"\n{feats}\n")
        for k, v in asdict(feats).items():
            print(f"  {k}: {v}")
    else:
        # Demonstrate sequence-only features (no PDB needed)
        demo = [
            ("NLVPMVATV", "HLA-A*02:01"),
            ("GILGFVFTL", "HLA-A*02:01"),
            ("KRASGSDFVQ", "HLA-A*02:01"),
            ("AAAAAAAA",  "HLA-A*02:01"),
        ]
        print(f"\n{'Peptide':<12} {'anchor':>8} {'hydro':>8}")
        print("-" * 32)
        for pep, hla in demo:
            feats = extract_structure_features(pep, hla, pdb_path=None)
            ac = f"{feats.anchor_compatibility:.2f}" if feats.anchor_compatibility is not None else "  —  "
            hy = f"{feats.tcr_hydrophobicity:.2f}" if feats.tcr_hydrophobicity is not None else "  —  "
            print(f"{pep:<12} {ac:>8} {hy:>8}")
        print("\n(Pass --peptide SEQ --hla HLA-A*02:01 --mut-pos N to test with PDB)")
