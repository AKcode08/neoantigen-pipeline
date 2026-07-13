#!/usr/bin/env python3
"""
NeoaPred Structure Prediction Wrapper
Calls Docker PepConf to generate peptide-MHC 3D structures
Parses PDB files to extract structural features
"""

import subprocess
import csv
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class StructurePrediction:
    """Structure prediction result from NeoaPred PepConf"""
    peptide: str
    hla_allele: str
    pdb_file_path: Optional[str]  # Path to relaxed PDB file
    structure_id: str
    residue_count: int
    has_structure: bool
    pdb_atoms: Optional[List[dict]]  # Parsed atom coordinates
    structure_quality: str  # high/medium/low based on PDB validity
    notes: str = ""
    
    def __repr__(self):
        return (
            f"StructurePrediction(peptide={self.peptide}, "
            f"hla={self.hla_allele}, "
            f"structure={'✅' if self.has_structure else '❌'}, "
            f"quality={self.structure_quality})"
        )


class NeoaPredWrapper:
    """Wrapper around Docker NeoaPred PepConf for structure prediction"""
    
    def __init__(self, docker_image: str = "panda1103/neoapred:1.0.0", 
                 output_dir: str = None):
        self.docker_image = docker_image
        self.temp_dir = None
        
        # Set output directory (where PDB files are saved)
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "neoapred_structures"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 NeoaPred structures will be saved to: {self.output_dir}")
        
    def predict(self, peptide: str, hla_allele: str, structure_id: str = None) -> StructurePrediction:
        """
        Predict 3D structure of peptide-MHC complex using NeoaPred PepConf
        
        Args:
            peptide: Amino acid sequence (e.g., "NLVPMVATV")
            hla_allele: HLA allele (e.g., "A0201")
            structure_id: Optional ID for this structure (default: peptide_hla)
            
        Returns:
            StructurePrediction object with PDB file path and metrics
        """
        if not structure_id:
            structure_id = f"{peptide}_{hla_allele}"
            
        # Create temp directory for this prediction
        temp_input = tempfile.mkdtemp(prefix="neoapred_input_")
        temp_output = tempfile.mkdtemp(prefix="neoapred_output_")
        
        try:
            # Create input CSV
            input_csv = os.path.join(temp_input, "input.csv")
            with open(input_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Allele", "Pep"])
                writer.writerow([structure_id, hla_allele, peptide])
            
            # Call Docker
            cmd = [
                "docker", "run",
                "-v", f"{temp_input}:/input",
                "-v", f"{temp_output}:/output",
                self.docker_image,
                "bash", "-c",
                (
                    "cd /var/software/NeoaPred && "
                    "source ~/.bashrc && "
                    "conda activate neoa && "
                    f"python run_NeoaPred.py --input_file /input/input.csv "
                    f"--output_dir /output --mode PepConf"
                )
            ]
            
            print(f"🔄 Running NeoaPred PepConf for {peptide}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return StructurePrediction(
                    peptide=peptide,
                    hla_allele=hla_allele,
                    structure_id=structure_id,
                    pdb_file_path=None,
                    residue_count=len(peptide),
                    has_structure=False,
                    pdb_atoms=None,
                    structure_quality="error",
                    notes=f"Docker error: {result.stderr[:200]}"
                )
            
            # Find output PDB file in temp directory
            temp_pdb_file = os.path.join(temp_output, "Structure", f"{structure_id}_pep_relaxed.pdb")
            
            if not os.path.exists(temp_pdb_file):
                return StructurePrediction(
                    peptide=peptide,
                    hla_allele=hla_allele,
                    structure_id=structure_id,
                    pdb_file_path=None,
                    residue_count=len(peptide),
                    has_structure=False,
                    pdb_atoms=None,
                    structure_quality="missing",
                    notes="PDB file not generated"
                )
            
            # Copy PDB file to project directory
            project_pdb_file = self.output_dir / f"{structure_id}_pep_relaxed.pdb"
            import shutil
            shutil.copy(temp_pdb_file, str(project_pdb_file))
            
            # Also copy the Structure directory contents
            temp_structure_dir = os.path.join(temp_output, "Structure")
            project_structure_dir = self.output_dir / structure_id
            project_structure_dir.mkdir(parents=True, exist_ok=True)
            
            for file in os.listdir(temp_structure_dir):
                src = os.path.join(temp_structure_dir, file)
                dst = project_structure_dir / file
                if os.path.isfile(src):
                    shutil.copy(src, str(dst))
            
            # Parse PDB file
            atoms = self._parse_pdb(str(project_pdb_file))
            quality = "high" if atoms and len(atoms) > 0 else "low"
            
            return StructurePrediction(
                peptide=peptide,
                hla_allele=hla_allele,
                structure_id=structure_id,
                pdb_file_path=str(project_pdb_file),
                residue_count=len(peptide),
                has_structure=True,
                pdb_atoms=atoms,
                structure_quality=quality,
                notes=f"✅ Structure saved to {project_pdb_file}"
            )
            
        except subprocess.TimeoutExpired:
            return StructurePrediction(
                peptide=peptide,
                hla_allele=hla_allele,
                structure_id=structure_id,
                pdb_file_path=None,
                residue_count=len(peptide),
                has_structure=False,
                pdb_atoms=None,
                structure_quality="timeout",
                notes="NeoaPred exceeded 5 minute timeout"
            )
        except Exception as e:
            return StructurePrediction(
                peptide=peptide,
                hla_allele=hla_allele,
                structure_id=structure_id,
                pdb_file_path=None,
                residue_count=len(peptide),
                has_structure=False,
                pdb_atoms=None,
                structure_quality="error",
                notes=f"Exception: {str(e)[:200]}"
            )
    
    def predict_batch(self, peptides: List[tuple]) -> List[StructurePrediction]:
        """
        Predict structures for multiple peptides
        
        Args:
            peptides: List of (peptide, hla_allele) tuples
            
        Returns:
            List of StructurePrediction objects
        """
        results = []
        for peptide, hla_allele in peptides:
            result = self.predict(peptide, hla_allele)
            results.append(result)
            print(f"  {result}")
        return results
    
    @staticmethod
    def _parse_pdb(pdb_file: str) -> List[dict]:
        """
        Parse PDB file and extract atom coordinates
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            List of dicts with atom info: {residue, atom_name, x, y, z, element}
        """
        atoms = []
        
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        # PDB format: columns are fixed-width
                        atom_dict = {
                            'atom_num': int(line[6:11].strip()),
                            'atom_name': line[12:16].strip(),
                            'residue': line[17:20].strip(),
                            'residue_num': int(line[22:26].strip()),
                            'x': float(line[30:38].strip()),
                            'y': float(line[38:46].strip()),
                            'z': float(line[46:54].strip()),
                            'element': line[76:78].strip()
                        }
                        atoms.append(atom_dict)
        except (IOError, ValueError) as e:
            print(f"⚠️  Error parsing PDB file: {e}")
            return []
        
        return atoms
    
    @staticmethod
    def get_pdb_summary(pdb_atoms: List[dict]) -> dict:
        """
        Get summary metrics from PDB atoms
        
        Args:
            pdb_atoms: List of atom dicts from _parse_pdb
            
        Returns:
            Dict with: atom_count, residue_count, x_range, y_range, z_range
        """
        if not pdb_atoms:
            return {}
        
        residues = set(a['residue_num'] for a in pdb_atoms)
        x_coords = [a['x'] for a in pdb_atoms]
        y_coords = [a['y'] for a in pdb_atoms]
        z_coords = [a['z'] for a in pdb_atoms]
        
        return {
            'atom_count': len(pdb_atoms),
            'residue_count': len(residues),
            'x_range': (min(x_coords), max(x_coords)),
            'y_range': (min(y_coords), max(y_coords)),
            'z_range': (min(z_coords), max(z_coords)),
            'structure_size': max(
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords),
                max(z_coords) - min(z_coords)
            )
        }


# Test
if __name__ == "__main__":
    wrapper = NeoaPredWrapper()
    
    # Test peptides
    test_peptides = [
        ("NLVPMVATV", "A0201"),
        ("GILGFVFTL", "A0201"),
        ("KRASGSDFVQ", "A0201"),
    ]
    
    print("=" * 80)
    print("NeoaPred Structure Prediction Wrapper - Test")
    print("=" * 80)
    
    results = wrapper.predict_batch(test_peptides)
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result.peptide} ({result.hla_allele})")
        print(f"  Status: {'✅ Success' if result.has_structure else '❌ Failed'}")
        print(f"  Quality: {result.structure_quality}")
        print(f"  PDB File: {result.pdb_file_path}")
        
        if result.pdb_atoms:
            summary = wrapper.get_pdb_summary(result.pdb_atoms)
            print(f"  Atoms: {summary.get('atom_count', 'N/A')}")
            print(f"  Structure Size: {summary.get('structure_size', 'N/A'):.1f} Å")