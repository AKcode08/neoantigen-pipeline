"""
Report Generator - Layer 6

Generates beautiful, professional HTML reports from the 5-layer ensemble
+ Claude expert analysis. Includes interactive 3D structure viewer.

Features:
- Executive summary card with recommendation
- 5-layer prediction breakdown with visualizations
- Interactive 3D structure viewer (NGL.js)
- Claude's expert analysis sections
- Literature citations with paper details
- Downloadable PDF-ready HTML
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layer_2_predictors.ensemble import EnsembleResult
from src.layer_5_synthesis.claude_engine import ClaudeAnalysis


@dataclass
class ReportPaths:
    """Paths to generated report files"""
    html_path: str
    csv_path: Optional[str] = None
    json_path: Optional[str] = None


class ReportGenerator:
    """
    Generate beautiful, professional reports for neoantigen analysis.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "reports"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self,
                ensemble_result: EnsembleResult,
                claude_analysis: ClaudeAnalysis,
                patient_id: Optional[str] = None,
                tumor_type: Optional[str] = None) -> ReportPaths:
        """
        Generate full report from ensemble + Claude analysis.
        
        Args:
            ensemble_result: 5-layer ensemble output
            claude_analysis: Claude expert synthesis
            patient_id: Optional patient identifier
            tumor_type: Optional tumor type context
        
        Returns:
            ReportPaths with paths to generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{ensemble_result.peptide}_{timestamp}"
        
        # Generate HTML
        html_path = self.output_dir / f"{report_name}.html"
        html_content = self._build_html(ensemble_result, claude_analysis, patient_id, tumor_type)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Generate JSON
        json_path = self.output_dir / f"{report_name}.json"
        json_data = self._build_json(ensemble_result, claude_analysis, patient_id, tumor_type)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return ReportPaths(
            html_path=str(html_path),
            json_path=str(json_path)
        )
    
    def _build_json(self,
                   result: EnsembleResult,
                   analysis: ClaudeAnalysis,
                   patient_id: Optional[str],
                   tumor_type: Optional[str]) -> dict:
        """Build JSON representation of report"""
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "patient_id": patient_id,
                "tumor_type": tumor_type,
            },
            "peptide": {
                "sequence": result.peptide,
                "length": len(result.peptide),
                "hla_allele": result.hla_allele,
            },
            "predictions": {
                "binding": {
                    "affinity_nm": result.binding_affinity_nm,
                    "level": result.binding_level,
                    "percentile": result.binding_percentile,
                },
                "presentation": {
                    "score": result.presentation_score,
                    "level": result.presentation_level,
                    "processing_score": result.processing_score,
                },
                "immunogenicity": {
                    "score": result.immunogenicity_score,
                    "level": result.immunogenicity_level,
                },
                "structure": {
                    "score": result.structure_score,
                    "level": result.structure_level,
                    "pdb_path": result.structure_pdb_path,
                    "atom_count": result.structure_atom_count,
                },
                "literature": {
                    "evidence_level": result.literature_evidence_level,
                    "evidence_score": result.literature_evidence_score,
                    "iedb_assays": result.iedb_assay_count,
                    "iedb_response_rate": result.iedb_response_rate,
                    "pubmed_papers": result.pubmed_paper_count,
                    "summary": result.literature_summary,
                },
            },
            "ensemble": {
                "consensus_score": result.consensus_score,
                "agreement_level": result.agreement_level,
                "recommendation": result.recommendation,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            },
            "claude_analysis": {
                "executive_summary": analysis.executive_summary,
                "prediction_interpretation": analysis.prediction_interpretation,
                "structural_analysis": analysis.structural_analysis,
                "literature_context": analysis.literature_context,
                "mechanistic_reasoning": analysis.mechanistic_reasoning,
                "confidence_assessment": analysis.confidence_assessment,
                "final_recommendation": analysis.final_recommendation,
                "final_confidence": analysis.final_confidence,
                "vaccine_priority": analysis.vaccine_priority,
                "key_strengths": analysis.key_strengths,
                "key_concerns": analysis.key_concerns,
                "caveats": analysis.caveats,
                "next_steps": analysis.next_steps,
            }
        }
    
    def _read_pdb_content(self, pdb_path: Optional[str]) -> str:
        """Read PDB file content for embedded 3D viewer"""
        if not pdb_path or not os.path.exists(pdb_path):
            return ""
        try:
            with open(pdb_path, 'r') as f:
                return f.read()
        except IOError:
            return ""
    
    def _build_html(self,
                   result: EnsembleResult,
                   analysis: ClaudeAnalysis,
                   patient_id: Optional[str],
                   tumor_type: Optional[str]) -> str:
        """Build the HTML report"""
        
        # Recommendation styling
        rec_colors = {
            "INCLUDE": ("#10b981", "#064e3b", "Recommended for Vaccine"),
            "EXCLUDE": ("#ef4444", "#7f1d1d", "Not Recommended"),
            "BORDERLINE": ("#f59e0b", "#78350f", "Needs Human Review"),
        }
        rec = analysis.final_recommendation
        rec_color, rec_bg, rec_label = rec_colors.get(rec, ("#6b7280", "#1f2937", "Unknown"))
        
        # Priority styling
        priority_colors = {
            "high": "#10b981",
            "moderate": "#f59e0b",
            "low": "#f97316",
            "not_recommended": "#ef4444",
        }
        priority_color = priority_colors.get(analysis.vaccine_priority, "#6b7280")
        
        # PDB content for 3D viewer
        pdb_content = self._read_pdb_content(result.structure_pdb_path)
        pdb_js_string = json.dumps(pdb_content) if pdb_content else "''"
        
        # Format bullet lists
        def format_bullets(items: List[str], icon: str = "•") -> str:
            if not items:
                return "<p class='empty-state'>None provided</p>"
            return "\n".join(f"<li><span class='bullet-icon'>{icon}</span><span>{item}</span></li>" for item in items)
        
        # Format papers
        papers_html = ""
        if result.literature_evidence and result.literature_evidence.pubmed_result:
            papers = result.literature_evidence.pubmed_result.papers[:5]
            for paper in papers:
                first_author = paper.authors[0] if paper.authors else "Unknown"
                more_authors = f" et al." if len(paper.authors) > 1 else ""
                abstract_snippet = paper.abstract[:300] + "..." if paper.abstract and len(paper.abstract) > 300 else paper.abstract or ""
                papers_html += f"""
                <div class="paper-card">
                    <div class="paper-title">{paper.title}</div>
                    <div class="paper-meta">
                        <span>{first_author}{more_authors}</span>
                        <span class="separator">·</span>
                        <span>{paper.journal}</span>
                        <span class="separator">·</span>
                        <span>{paper.year}</span>
                        <span class="separator">·</span>
                        <a href="https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/" target="_blank" class="pmid-link">PMID: {paper.pmid}</a>
                    </div>
                    {f'<div class="paper-abstract">{abstract_snippet}</div>' if abstract_snippet else ''}
                </div>
                """
        
        if not papers_html:
            papers_html = '<p class="empty-state">No PubMed papers found for this peptide-HLA combination.</p>'
        
        # IEDB disease tags
        iedb_diseases_html = ""
        if result.literature_evidence and result.literature_evidence.iedb_result and result.literature_evidence.iedb_result.diseases:
            diseases = result.literature_evidence.iedb_result.diseases[:8]
            iedb_diseases_html = '<div class="disease-tags">' + "".join(
                f'<span class="disease-tag">{d}</span>' for d in diseases if d
            ) + '</div>'
        
        # Structure section
        structure_section_html = ""
        if result.structure_pdb_path and pdb_content:
            structure_section_html = f"""
            <div class="structure-viewer-card">
                <div class="card-header">
                    <h3>3D Structure Visualization</h3>
                    <span class="badge">{result.structure_atom_count} atoms · {result.structure_level} quality</span>
                </div>
                <div id="ngl-viewer" class="ngl-viewer"></div>
                <div class="viewer-controls">
                    <button onclick="setRepresentation('cartoon')" class="viewer-btn">Cartoon</button>
                    <button onclick="setRepresentation('ball+stick')" class="viewer-btn">Ball+Stick</button>
                    <button onclick="setRepresentation('surface')" class="viewer-btn">Surface</button>
                    <button onclick="toggleSpin()" class="viewer-btn">Toggle Spin</button>
                    <button onclick="resetView()" class="viewer-btn">Reset View</button>
                </div>
                <div class="structure-meta">
                    <strong>PDB Source:</strong> NeoaPred PepConf · <strong>Atoms:</strong> {result.structure_atom_count} · <strong>Format:</strong> Relaxed PDB
                </div>
            </div>
            """
        else:
            structure_section_html = """
            <div class="structure-viewer-card">
                <p class="empty-state">3D structure not available for this peptide.</p>
            </div>
            """
        
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        patient_info = ""
        if patient_id or tumor_type:
            patient_info = f"""
            <div class="patient-info">
                {f'<div><span class="label">Patient ID:</span> {patient_id}</div>' if patient_id else ''}
                {f'<div><span class="label">Tumor Type:</span> {tumor_type}</div>' if tumor_type else ''}
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neoantigen Report: {result.peptide}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        :root {{
            --bg-primary: #fafaf9;
            --bg-card: #ffffff;
            --bg-subtle: #f5f5f4;
            --bg-dark: #1c1917;
            --text-primary: #1c1917;
            --text-secondary: #57534e;
            --text-tertiary: #a8a29e;
            --border: #e7e5e4;
            --border-strong: #d6d3d1;
            --accent: #0c4a6e;
            --rec-color: {rec_color};
            --rec-bg: {rec_bg};
            --priority-color: {priority_color};
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 40px 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        /* HEADER */
        .header {{
            margin-bottom: 48px;
            padding-bottom: 32px;
            border-bottom: 2px solid var(--border);
        }}
        
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            gap: 24px;
            margin-bottom: 16px;
        }}
        
        .header-meta {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-tertiary);
        }}
        
        .header-title {{
            font-family: 'Fraunces', serif;
            font-size: 48px;
            font-weight: 500;
            line-height: 1.05;
            letter-spacing: -0.02em;
            margin-top: 8px;
        }}
        
        .peptide-display {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
        }}
        
        .header-subtitle {{
            font-size: 18px;
            color: var(--text-secondary);
            margin-top: 12px;
        }}
        
        .patient-info {{
            display: flex;
            gap: 24px;
            margin-top: 16px;
            font-size: 14px;
            color: var(--text-secondary);
        }}
        
        .patient-info .label {{
            color: var(--text-tertiary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
        }}
        
        /* RECOMMENDATION HERO */
        .recommendation-hero {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 4px solid var(--rec-color);
            padding: 40px;
            margin-bottom: 32px;
            border-radius: 4px;
            position: relative;
        }}
        
        .recommendation-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--text-tertiary);
            margin-bottom: 8px;
        }}
        
        .recommendation-text {{
            font-family: 'Fraunces', serif;
            font-size: 56px;
            font-weight: 600;
            color: var(--rec-color);
            line-height: 1;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }}
        
        .recommendation-sublabel {{
            font-size: 16px;
            color: var(--text-secondary);
            margin-bottom: 24px;
        }}
        
        .recommendation-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 32px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
        }}
        
        .stat-block {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-tertiary);
            margin-bottom: 6px;
        }}
        
        .stat-value {{
            font-family: 'Fraunces', serif;
            font-size: 28px;
            font-weight: 500;
            color: var(--text-primary);
        }}
        
        .priority-pill {{
            display: inline-block;
            padding: 4px 12px;
            background: var(--priority-color);
            color: white;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
            border-radius: 3px;
        }}
        
        /* EXECUTIVE SUMMARY */
        .executive-card {{
            background: var(--bg-dark);
            color: #f5f5f4;
            padding: 40px;
            margin-bottom: 48px;
            border-radius: 4px;
        }}
        
        .executive-card .section-label {{
            color: #a8a29e;
        }}
        
        .executive-text {{
            font-family: 'Fraunces', serif;
            font-size: 22px;
            line-height: 1.5;
            font-weight: 400;
            color: #f5f5f4;
        }}
        
        /* SECTIONS */
        .section {{
            margin-bottom: 48px;
        }}
        
        .section-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--text-tertiary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .section-label::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: var(--border);
        }}
        
        .section-title {{
            font-family: 'Fraunces', serif;
            font-size: 32px;
            font-weight: 500;
            margin-bottom: 24px;
            letter-spacing: -0.01em;
        }}
        
        /* PREDICTION GRID */
        .prediction-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .prediction-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 24px;
            border-radius: 4px;
            position: relative;
            transition: border-color 0.2s;
        }}
        
        .prediction-card:hover {{
            border-color: var(--text-tertiary);
        }}
        
        .prediction-number {{
            position: absolute;
            top: 16px;
            right: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-tertiary);
        }}
        
        .prediction-card-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-tertiary);
            margin-bottom: 8px;
        }}
        
        .prediction-card-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }}
        
        .prediction-value {{
            font-family: 'Fraunces', serif;
            font-size: 36px;
            font-weight: 500;
            line-height: 1;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }}
        
        .prediction-unit {{
            font-size: 12px;
            color: var(--text-tertiary);
            margin-bottom: 12px;
        }}
        
        .prediction-meta {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        
        .level-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 8px;
        }}
        
        .level-strong, .level-high {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .level-medium, .level-moderate {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .level-weak, .level-low, .level-non-binder, .level-none, .level-unavailable, .level-unknown {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        /* 3D STRUCTURE VIEWER */
        .structure-viewer-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 24px;
            border-radius: 4px;
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}
        
        .card-header h3 {{
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 500;
        }}
        
        .badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
            background: var(--bg-subtle);
            padding: 4px 10px;
            border-radius: 3px;
        }}
        
        .ngl-viewer {{
            width: 100%;
            height: 500px;
            background: #0f172a;
            border-radius: 4px;
            margin-bottom: 12px;
        }}
        
        .viewer-controls {{
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }}
        
        .viewer-btn {{
            padding: 8px 16px;
            background: var(--bg-subtle);
            border: 1px solid var(--border);
            border-radius: 3px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            cursor: pointer;
            transition: all 0.2s;
            color: var(--text-primary);
        }}
        
        .viewer-btn:hover {{
            background: var(--text-primary);
            color: white;
            border-color: var(--text-primary);
        }}
        
        .structure-meta {{
            font-size: 12px;
            color: var(--text-secondary);
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }}
        
        /* CLAUDE ANALYSIS */
        .claude-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 32px;
            border-radius: 4px;
            margin-bottom: 16px;
        }}
        
        .claude-section h4 {{
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 500;
            margin-bottom: 16px;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}
        
        .claude-text {{
            font-size: 15px;
            line-height: 1.7;
            color: var(--text-primary);
        }}
        
        /* LISTS */
        .strength-concern-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        
        @media (max-width: 768px) {{
            .strength-concern-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .list-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 28px;
            border-radius: 4px;
        }}
        
        .list-card.strengths {{
            border-left: 3px solid #10b981;
        }}
        
        .list-card.concerns {{
            border-left: 3px solid #ef4444;
        }}
        
        .list-card.caveats {{
            border-left: 3px solid #f59e0b;
        }}
        
        .list-card.next-steps {{
            border-left: 3px solid #3b82f6;
        }}
        
        .list-card h4 {{
            font-family: 'Fraunces', serif;
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 16px;
        }}
        
        .list-card ul {{
            list-style: none;
        }}
        
        .list-card li {{
            display: flex;
            gap: 12px;
            padding: 8px 0;
            font-size: 14px;
            line-height: 1.6;
            border-bottom: 1px solid var(--border);
        }}
        
        .list-card li:last-child {{
            border-bottom: none;
        }}
        
        .bullet-icon {{
            color: var(--text-tertiary);
            flex-shrink: 0;
        }}
        
        /* LITERATURE */
        .literature-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .lit-stat {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 20px;
            border-radius: 4px;
        }}
        
        .lit-stat-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            text-transform: uppercase;
            color: var(--text-tertiary);
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }}
        
        .lit-stat-value {{
            font-family: 'Fraunces', serif;
            font-size: 28px;
            font-weight: 500;
        }}
        
        .disease-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 12px;
        }}
        
        .disease-tag {{
            background: var(--bg-subtle);
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 12px;
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }}
        
        .paper-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 20px;
            margin-bottom: 12px;
            border-radius: 4px;
        }}
        
        .paper-title {{
            font-family: 'Fraunces', serif;
            font-size: 17px;
            font-weight: 500;
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        
        .paper-meta {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }}
        
        .paper-meta .separator {{
            color: var(--text-tertiary);
            margin: 0 6px;
        }}
        
        .pmid-link {{
            color: var(--accent);
            text-decoration: none;
            border-bottom: 1px dotted;
        }}
        
        .pmid-link:hover {{
            border-bottom-style: solid;
        }}
        
        .paper-abstract {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }}
        
        .empty-state {{
            color: var(--text-tertiary);
            font-style: italic;
            font-size: 14px;
            padding: 20px;
            text-align: center;
        }}
        
        /* FOOTER */
        .footer {{
            margin-top: 64px;
            padding-top: 32px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-tertiary);
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .footer-stack {{
            margin-top: 8px;
            font-size: 11px;
        }}
        
        @media print {{
            body {{ background: white; }}
            .viewer-controls, .ngl-viewer {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header class="header">
            <div class="header-top">
                <div>
                    <div class="header-meta">Neoantigen Candidate Report · Generated {timestamp}</div>
                    <h1 class="header-title">
                        Candidate <span class="peptide-display">{result.peptide}</span>
                    </h1>
                    <div class="header-subtitle">
                        HLA Restriction: <strong>{result.hla_allele}</strong> · Length: {len(result.peptide)} aa
                    </div>
                    {patient_info}
                </div>
            </div>
        </header>
        
        <!-- RECOMMENDATION HERO -->
        <div class="recommendation-hero">
            <div class="recommendation-label">Final Recommendation</div>
            <div class="recommendation-text">{analysis.final_recommendation}</div>
            <div class="recommendation-sublabel">
                {rec_label} · <span class="priority-pill">Priority: {analysis.vaccine_priority.replace('_', ' ')}</span>
            </div>
            <div class="recommendation-stats">
                <div class="stat-block">
                    <div class="stat-label">Expert Confidence</div>
                    <div class="stat-value">{analysis.final_confidence:.0%}</div>
                </div>
                <div class="stat-block">
                    <div class="stat-label">Ensemble Consensus</div>
                    <div class="stat-value">{result.consensus_score:.2f}</div>
                </div>
                <div class="stat-block">
                    <div class="stat-label">Signal Agreement</div>
                    <div class="stat-value">{result.agreement_level.replace('_', ' ').title()}</div>
                </div>
            </div>
        </div>
        
        <!-- EXECUTIVE SUMMARY -->
        <div class="executive-card">
            <div class="section-label" style="color: #a8a29e;">Executive Summary</div>
            <div class="executive-text">{analysis.executive_summary}</div>
        </div>
        
        <!-- PREDICTION LAYERS -->
        <div class="section">
            <div class="section-label">Five-Layer Prediction Analysis</div>
            <h2 class="section-title">Computational Predictions</h2>
            
            <div class="prediction-grid">
                <div class="prediction-card">
                    <span class="prediction-number">01</span>
                    <div class="prediction-card-label">Layer 1 · MHCflurry</div>
                    <div class="prediction-card-title">MHC Binding</div>
                    <div class="prediction-value">{result.binding_affinity_nm:.1f}</div>
                    <div class="prediction-unit">nM affinity</div>
                    <div class="prediction-meta">Percentile: {result.binding_percentile:.3f}%</div>
                    <span class="level-badge level-{result.binding_level.replace(' ', '-').replace('non-binder', 'non-binder')}">{result.binding_level}</span>
                </div>
                
                <div class="prediction-card">
                    <span class="prediction-number">02</span>
                    <div class="prediction-card-label">Layer 2 · MHCflurry</div>
                    <div class="prediction-card-title">Antigen Presentation</div>
                    <div class="prediction-value">{result.presentation_score:.2f}</div>
                    <div class="prediction-unit">presentation score</div>
                    <div class="prediction-meta">Processing: {result.processing_score:.2f}</div>
                    <span class="level-badge level-{result.presentation_level}">{result.presentation_level}</span>
                </div>
                
                <div class="prediction-card">
                    <span class="prediction-number">03</span>
                    <div class="prediction-card-label">Layer 3 · BigMHC</div>
                    <div class="prediction-card-title">Immunogenicity</div>
                    <div class="prediction-value">{f'{result.immunogenicity_score:.2f}' if result.immunogenicity_score is not None else 'N/A'}</div>
                    <div class="prediction-unit">T cell response probability</div>
                    <div class="prediction-meta">Predictor: BigMHC IM</div>
                    <span class="level-badge level-{result.immunogenicity_level}">{result.immunogenicity_level}</span>
                </div>
                
                <div class="prediction-card">
                    <span class="prediction-number">04</span>
                    <div class="prediction-card-label">Layer 4 · NeoaPred</div>
                    <div class="prediction-card-title">3D Structure</div>
                    <div class="prediction-value">{result.structure_atom_count}</div>
                    <div class="prediction-unit">atoms in complex</div>
                    <div class="prediction-meta">Quality: {result.structure_level}</div>
                    <span class="level-badge level-{result.structure_level}">{result.structure_level}</span>
                </div>
                
                <div class="prediction-card">
                    <span class="prediction-number">05</span>
                    <div class="prediction-card-label">Layer 5 · IEDB + PubMed</div>
                    <div class="prediction-card-title">Literature Evidence</div>
                    <div class="prediction-value">{result.iedb_assay_count}</div>
                    <div class="prediction-unit">IEDB T cell assays</div>
                    <div class="prediction-meta">Response rate: {result.iedb_response_rate:.0%}</div>
                    <span class="level-badge level-{result.literature_evidence_level}">{result.literature_evidence_level}</span>
                </div>
            </div>
        </div>
        
        <!-- 3D STRUCTURE -->
        <div class="section">
            <div class="section-label">Structural Visualization</div>
            <h2 class="section-title">Peptide-MHC Complex</h2>
            {structure_section_html}
        </div>
        
        <!-- CLAUDE ANALYSIS -->
        <div class="section">
            <div class="section-label">Expert Analysis · Claude Opus 4</div>
            <h2 class="section-title">Senior Immunologist Interpretation</h2>
            
            <div class="claude-section">
                <h4>Prediction Interpretation</h4>
                <div class="claude-text">{analysis.prediction_interpretation}</div>
            </div>
            
            <div class="claude-section">
                <h4>Structural Analysis</h4>
                <div class="claude-text">{analysis.structural_analysis}</div>
            </div>
            
            <div class="claude-section">
                <h4>Literature Context</h4>
                <div class="claude-text">{analysis.literature_context}</div>
            </div>
            
            <div class="claude-section">
                <h4>Mechanistic Reasoning</h4>
                <div class="claude-text">{analysis.mechanistic_reasoning}</div>
            </div>
            
            <div class="claude-section">
                <h4>Confidence Assessment</h4>
                <div class="claude-text">{analysis.confidence_assessment}</div>
            </div>
        </div>
        
        <!-- STRENGTHS & CONCERNS -->
        <div class="section">
            <div class="section-label">Decision Matrix</div>
            <h2 class="section-title">Strengths · Concerns · Caveats · Next Steps</h2>
            
            <div class="strength-concern-grid">
                <div class="list-card strengths">
                    <h4>✓ Key Strengths</h4>
                    <ul>{format_bullets(analysis.key_strengths, "✓")}</ul>
                </div>
                <div class="list-card concerns">
                    <h4>⚠ Key Concerns</h4>
                    <ul>{format_bullets(analysis.key_concerns, "⚠")}</ul>
                </div>
                <div class="list-card caveats">
                    <h4>※ Caveats</h4>
                    <ul>{format_bullets(analysis.caveats, "※")}</ul>
                </div>
                <div class="list-card next-steps">
                    <h4>→ Next Steps</h4>
                    <ul>{format_bullets(analysis.next_steps, "→")}</ul>
                </div>
            </div>
        </div>
        
        <!-- LITERATURE EVIDENCE -->
        <div class="section">
            <div class="section-label">Layer 5 Detail · IEDB + PubMed</div>
            <h2 class="section-title">Published Evidence</h2>
            
            <div class="literature-stats">
                <div class="lit-stat">
                    <div class="lit-stat-label">IEDB Assays</div>
                    <div class="lit-stat-value">{result.iedb_assay_count}</div>
                </div>
                <div class="lit-stat">
                    <div class="lit-stat-label">Response Rate</div>
                    <div class="lit-stat-value">{result.iedb_response_rate:.0%}</div>
                </div>
                <div class="lit-stat">
                    <div class="lit-stat-label">PubMed Papers</div>
                    <div class="lit-stat-value">{result.pubmed_paper_count}</div>
                </div>
                <div class="lit-stat">
                    <div class="lit-stat-label">Evidence Level</div>
                    <div class="lit-stat-value" style="font-size: 18px;">{result.literature_evidence_level.title()}</div>
                </div>
            </div>
            
            {iedb_diseases_html}
            
            <h3 style="font-family: 'Fraunces', serif; font-size: 22px; font-weight: 500; margin: 32px 0 16px;">Relevant Publications</h3>
            {papers_html}
        </div>
        
        <!-- FOOTER -->
        <footer class="footer">
            Generated by Neoantigen Analysis Pipeline · {timestamp}
            <div class="footer-stack">
                MHCflurry · BigMHC · NeoaPred · IEDB Query API · NCBI E-utilities · Claude Opus 4
            </div>
        </footer>
    </div>
    
    <script>
        // 3D Structure Viewer
        const pdbContent = {pdb_js_string};
        let stage = null;
        let component = null;
        let isSpinning = false;
        
        if (pdbContent) {{
            document.addEventListener('DOMContentLoaded', function() {{
                stage = new NGL.Stage('ngl-viewer', {{
                    backgroundColor: '#0f172a'
                }});
                
                const stringBlob = new Blob([pdbContent], {{type: 'text/plain'}});
                
                stage.loadFile(stringBlob, {{ext: 'pdb'}}).then(function(comp) {{
                    component = comp;
                    comp.addRepresentation('cartoon', {{
                        sele: 'protein',
                        colorScheme: 'chainname',
                        smoothSheet: true
                    }});
                    comp.addRepresentation('ball+stick', {{
                        sele: 'not protein',
                        colorScheme: 'element'
                    }});
                    comp.autoView();
                }});
                
                window.addEventListener('resize', function() {{
                    if (stage) stage.handleResize();
                }});
            }});
        }}
        
        function setRepresentation(repType) {{
            if (!component) return;
            component.removeAllRepresentations();
            if (repType === 'cartoon') {{
                component.addRepresentation('cartoon', {{colorScheme: 'chainname'}});
                component.addRepresentation('ball+stick', {{sele: 'not protein'}});
            }} else if (repType === 'ball+stick') {{
                component.addRepresentation('ball+stick', {{colorScheme: 'element'}});
            }} else if (repType === 'surface') {{
                component.addRepresentation('surface', {{
                    sele: 'protein',
                    colorScheme: 'chainname',
                    opacity: 0.7
                }});
                component.addRepresentation('cartoon', {{sele: 'protein'}});
            }}
            component.autoView();
        }}
        
        function toggleSpin() {{
            if (!stage) return;
            isSpinning = !isSpinning;
            stage.setSpin(isSpinning);
        }}
        
        function resetView() {{
            if (component) component.autoView();
        }}
    </script>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    from src.layer_2_predictors.ensemble import EnsemblePredictor
    from src.layer_5_synthesis.claude_engine import ClaudeSynthesisEngine
    
    print("=" * 100)
    print("REPORT GENERATOR - END TO END TEST")
    print("=" * 100)
    
    # Step 1: Ensemble
    print("\n📊 Step 1: Running ensemble...")
    ensemble = EnsemblePredictor()
    test_peptide = "NLVPMVATV"
    test_hla = "HLA-A*02:01"
    
    ensemble_result = ensemble.predict(test_peptide, test_hla)
    print(f"   ✓ Ensemble: {ensemble_result.recommendation} (consensus={ensemble_result.consensus_score:.2f})")
    
    # Step 2: Claude
    print("\n🧠 Step 2: Claude expert analysis...")
    claude = ClaudeSynthesisEngine()
    analysis = claude.synthesize(ensemble_result)
    print(f"   ✓ Claude: {analysis.final_recommendation} (confidence={analysis.final_confidence:.2f})")
    
    # Step 3: Report
    print("\n📄 Step 3: Generating report...")
    generator = ReportGenerator()
    paths = generator.generate(
        ensemble_result=ensemble_result,
        claude_analysis=analysis,
        patient_id="DEMO_PATIENT_001",
        tumor_type="Demo"
    )
    
    print(f"\n{'='*100}")
    print("✅ REPORT GENERATED")
    print(f"{'='*100}")
    print(f"\n📄 HTML Report: {paths.html_path}")
    print(f"📊 JSON Data: {paths.json_path}")
    print(f"\nOpen the HTML file in a browser to view the report!")
    print(f"\n  open {paths.html_path}")