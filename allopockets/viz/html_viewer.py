"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Self-contained 3D HTML viewer generation using 3Dmol.js.
Visualizes protein cartoon, pocket alpha spheres, and ground-truth allosteric modulator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlloPockets 3D Debugger - {title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #1a1a24;
            color: #e0e0e0;
            overflow: hidden;
        }}
        #container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
        }}
        #panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(26, 26, 36, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid #33334d;
            border-radius: 8px;
            padding: 16px 20px;
            max-width: 380px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            z-index: 100;
        }}
        h2 {{
            margin: 0 0 10px 0;
            font-size: 18px;
            color: #00d2ff;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
            font-size: 13px;
        }}
        .metric-name {{
            color: #a0a0b0;
        }}
        .metric-val {{
            font-weight: 600;
            color: #ffffff;
        }}
        .legend {{
            margin-top: 14px;
            padding-top: 10px;
            border-top: 1px solid #33334d;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .color-box {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 8px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div id="panel">
        <h2>{title}</h2>
        <div class="metric-row">
            <span class="metric-name">PDB Structure:</span>
            <span class="metric-val">{pdb_id}</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Pocket:</span>
            <span class="metric-val">{pocket_id}</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Overlap (site in pocket):</span>
            <span class="metric-val">{site_in_pocket}</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Classification Label:</span>
            <span class="metric-val">{label_text}</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Alpha Spheres:</span>
            <span class="metric-val">{num_spheres}</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Pocket Volume:</span>
            <span class="metric-val">{volume} Å³</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Centroid Distance to Ligand:</span>
            <span class="metric-val">{distance} Å</span>
        </div>

        <div class="legend">
            <div class="legend-item"><span class="color-box" style="background:#b0b0b8;"></span> Protein Backbone (Cartoon)</div>
            <div class="legend-item"><span class="color-box" style="background:#ffcc00;"></span> Allosteric Modulator (Sticks)</div>
            <div class="legend-item"><span class="color-box" style="background:#ff7700;"></span> Annotated Site Residues</div>
            <div class="legend-item"><span class="color-box" style="background:#00e5ff;"></span> Pocket Alpha Spheres</div>
        </div>
    </div>

    <div id="container"></div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            let viewer = $3Dmol.createViewer("container", {{ backgroundColor: "#1a1a24" }});
            let pdbData = {pdb_data_json};
            let pocketData = {pocket_data_json};

            // 1. Add Protein Structure
            viewer.addModel(pdbData, "cif");
            viewer.setStyle({{hetflag: false}}, {{cartoon: {{color: '#b0b0b8', opacity: 0.85}}}});

            // 2. Modulator (Heteroatoms / Non-polymers)
            viewer.setStyle({{hetflag: true}}, {{stick: {{colorscheme: 'yellowCarbon', radius: 0.25}}}});

            // 3. Highlight Site Residues (if any)
            let siteResidues = {site_residues_json};
            for (let r of siteResidues) {{
                viewer.setStyle({{chain: r.chain, resi: r.resi}}, {{
                    cartoon: {{color: '#ff7700', opacity: 1.0}},
                    stick: {{colorscheme: 'orangeCarbon', radius: 0.15}}
                }});
            }}

            // 4. Add Pocket Alpha Spheres
            if (pocketData && pocketData.trim().length > 0) {{
                let pModel = viewer.addModel(pocketData, "cif");
                pModel.setStyle({{}}, {{sphere: {{color: '#00e5ff', radius: 0.8, opacity: 0.7}}}});
            }}

            viewer.zoomTo();
            viewer.render();
        }});
    </script>
</body>
</html>
"""


def generate_3d_pocket_html(
    pdb_id: str,
    pocket_id: str,
    pdb_cif_content: str,
    pocket_cif_content: str,
    metrics: Dict,
    site_residues: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate a self-contained interactive 3D HTML document visualizing the pocket.
    """
    label_num = metrics.get("label", 0)
    label_text = "Positive (Allosteric)" if label_num == 1 else "Negative (Non-allosteric)"

    html_content = HTML_TEMPLATE.format(
        title=f"{pdb_id.upper()} - {pocket_id}",
        pdb_id=pdb_id.upper(),
        pocket_id=pocket_id,
        site_in_pocket=f"{metrics.get('site_in_pocket', 0.0)*100:.1f}%",
        label_text=label_text,
        num_spheres=metrics.get("num_spheres", "N/A"),
        volume=f"{metrics.get('volume', 0.0):.1f}",
        distance=(
            f"{metrics.get('modulator_distance', 0.0):.2f}"
            if metrics.get("modulator_distance") is not None
            else "N/A"
        ),
        pdb_data_json=json.dumps(pdb_cif_content),
        pocket_data_json=json.dumps(pocket_cif_content),
        site_residues_json=json.dumps(site_residues or []),
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_content, encoding="utf-8")

    return html_content
