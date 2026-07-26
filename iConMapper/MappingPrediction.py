"""
MappingPrediction.py
────────────────────
Inference script for the DSGPM-TP coarse-graining model.

Given a SMILES string, this script:
  1. Builds an atomistic graph from the SMILES
  2. Runs the trained DSGPM-TP model to produce per-atom embeddings
     and per-atom CG bead type predictions
  3. Clusters the embeddings into N CG beads using graph cuts
  4. Enforces connectivity so every bead is a connected subgraph
  5. Writes the CG mapping to a JSON file (same format as the training dataset)
  6. Renders a 2D visualization of the mapping as PNG or SVG

Usage (CLI):
    python MappingPrediction.py \
        --smiles "CCO" \
        --num    2 \
        --output ./aa2cg \
        --style  1 \
        --labels \
        --svg \
        --cg

Usage (Python):
    from MappingPrediction import DSGPM_TPtoCG
    result = DSGPM_TPtoCG(smiles="CCO", out_dir="./aa2cg", num_cg_bead=2,
                          svg=True, write_cg=True)

Visualization styles:
    style=1  — colored atom beads with colored bonds (bead-centric)
    style=2  — black molecule with semi-transparent bounding ellipses per bead group

Notes:
    - The model checkpoint must be at model/best_epoch.pth
    - CG bead types are predicted per atom then majority-voted per bead group
      when --use_regular_mapping_from_prediction is active (default True)
    - If --num is not provided, defaults to n_heavy_atoms // 3
    - `svg` is threaded explicitly through DSGPM_TPtoCG -> gen_vis -> draw_graph_style{1,2}
      (it is NOT a module-level global). Use --svg on the CLI, or pass svg=True in Python.
    - Use --cg on the CLI (or write_cg=True in Python) to also write an
      all-atom PDB (RDKit AllChem.EmbedMolecule + MMFF/UFF-optimized
      conformer) and a coarse-grained PDB built from that same conformer.
"""

# ── standard library ──────────────────────────────────────────────────────────
import os
import io
import re
import sys
import json
import copy
import math
import random
import argparse
from collections import Counter
from warnings import simplefilter

# ── third-party: numerics ─────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

# ── third-party: cheminformatics ──────────────────────────────────────────────
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdDepictor, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

# ── third-party: visualization ────────────────────────────────────────────────
import seaborn as sns
import tqdm
from PIL import Image, ImageDraw, ImageFont
from skimage.io import imsave

# ── third-party: ML / graph ───────────────────────────────────────────────────
from sklearn.exceptions import UndefinedMetricWarning
from torch_geometric.data import DataListLoader

# ── project ───────────────────────────────────────────────────────────────────
from dataset.ham import ATOMS
from dataset.ham_per_file import HAMPerFile
from model.networks import DSGPM_TP
from model.graph_cuts import graph_cuts
from utils.post_processing import enforce_connectivity

# ── warning filters ───────────────────────────────────────────────────────────
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)
simplefilter(action='ignore', category=Warning)

# ── globals ───────────────────────────────────────────────────────────────────
# NOTE: `svg` used to live here as a module-level global and was read by
# gen_vis() while draw_graph_style1/2 took their own *local* `svg` parameter
# with an independent default. That split-brain setup caused gen_vis() to
# pick a file-writing branch (imsave vs. plain text write) that could
# disagree with what draw_graph_style1/2 actually returned (ndarray vs. str).
# `svg` is now threaded explicitly as a parameter everywhere instead.
debug = False

CACTUS       = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
commad_file  = './command.log'

# CG type index ↔ name  (inverted at the bottom so dict is readable as written)
_CG_TYPE_NAME_TO_IDX = {
    # amino-acid side chains
    "C2E":  0,  "C3E":  1,  "A2V":  2,  "A1L":  3,  "A1I":  4,
    "A5M":  5,  "P5N":  6,  "QaD":  7,  "P4Q":  8,  "QaE":  9,
    "P1C": 10,  "P1S": 11,  "P1T": 12,  "A2P": 13,  "A3K": 14,
    "QdK": 15,  "A3R": 16,  "QdR": 17,  "A4H": 18,  "P1H": 19,
    "P2H": 20,  "A1F": 21,  "A2F": 22,  "A1Y": 23,  "A2Y": 24,
    "P1Y": 25,  "A1W": 26,  "P1W": 27,  "A2W": 28,
    # RNA beads
    "RS1": 29,  "RS2": 30,  "RA1": 31,  "RG1": 32,  "RA2": 33,
    "RG2": 34,  "RA3": 35,  "RG4": 36,  "RC2": 37,  "RA4": 38,
    "RG3": 39,  "RU2": 40,  "RC1": 41,  "RU1": 42,  "RC3": 43,
    "RU3": 44,  "PHO": 45,
    # metabolite beads
    "M01": 46,  "M02": 47,  "M03": 48,  "M04": 49,  "M05": 50,
    "M06": 51,  "M07": 52,  "M08": 53,  "MCI": 54,  "MSO": 55,
    "MSS": 56,  "MCL": 57,  "MCF": 58,  "MBR": 59,
}
CG_TYPE_DICT = {idx: name for name, idx in _CG_TYPE_NAME_TO_IDX.items()}


# ── helper: SMILES → name ─────────────────────────────────────────────────────

def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def smiles_to_formula(smiles):
    """Returns the molecular formula (e.g. C6H12O6) from a SMILES string."""
    try:
        m = Chem.MolFromSmiles(smiles)
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None


# ── helper: majority-vote bead type per group ─────────────────────────────────

def adjust_list(lst):
    counts = Counter(lst)
    max_count = counts.most_common(1)[0][1]
    most_common_elements = [
        element for element, count in counts.items() if count == max_count
    ]
    chosen_element = random.choice(most_common_elements)
    return [chosen_element] * len(lst)


# ── visualization: style 1 — colored beads + colored bonds ───────────────────

def draw_graph_style1(graph, hard_assign, cg_types, show_labels=False, svg=False):
    smiles   = graph.graph['smiles']
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    rdCoordGen.AddCoords(molecule)

    hard_assign = np.array(hard_assign)
    palette     = np.array(sns.color_palette("Set2", hard_assign.max() + 1))

    atom_index        = list(range(len(graph.nodes)))
    undirected_edges  = np.array([
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()
    ])

    non_cut_mask = hard_assign[undirected_edges[:, 0]] == hard_assign[undirected_edges[:, 1]]
    non_cut_edges_indices = np.nonzero(non_cut_mask)[0]
    cut_edges_indices     = np.nonzero(~non_cut_mask)[0]

    atom_colors          = list(map(tuple, palette[hard_assign]))
    atom_id_to_color_dict = dict(zip(atom_index, atom_colors))

    # ── bead label annotations ────────────────────────────────────────────────
    annotations = []
    conf = molecule.GetConformer()

    if show_labels:
        cg_group_to_atoms = {}
        for l in range(molecule.GetNumAtoms()):
            cg_group_to_atoms.setdefault(int(hard_assign[l]), []).append(l)

        for cg_group_id, indices in cg_group_to_atoms.items():
            label = f"{cg_types[indices[0]]}"
            avg_x = sum(conf.GetAtomPosition(idx).x for idx in indices) / len(indices)
            avg_y = sum(conf.GetAtomPosition(idx).y for idx in indices) / len(indices)
            annotations.append((label, Point2D(avg_x, avg_y), cg_group_id))

    # ── drawer setup ──────────────────────────────────────────────────────────
    drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600) if svg \
             else rdMolDraw2D.MolDraw2DCairo(1200, 600)

    options = drawer.drawOptions()
    options.addAtomIndices  = False
    options.clearBackground = False

    atom_radius_size = 0.43
    bond_width_scale = 1.0

    drawer.DrawMolecule(
        molecule,
        highlightAtoms      = atom_index,
        highlightBonds      = [],
        highlightAtomColors = atom_id_to_color_dict,
        highlightAtomRadii  = dict(zip(atom_index, [atom_radius_size] * len(atom_index))),
    )
    drawer.FinishDrawing()

    # pixel scale for bond width
    p0_px = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(0).x,
                                          conf.GetAtomPosition(0).y))
    p1_px = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(0).x + 1.0,
                                          conf.GetAtomPosition(0).y))
    pixels_per_unit   = abs(p1_px.x - p0_px.x)
    bond_width        = max(1, int(atom_radius_size * pixels_per_unit * 2 * bond_width_scale))
    internal_bond_width = bond_width
    cut_bond_width      = bond_width

    # ── compute bond line segments ────────────────────────────────────────────
    internal_bond_lines, cut_edge_lines = [], []

    for idx in non_cut_edges_indices:
        u, v = undirected_edges[idx]
        pu = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(u)).x,
                                           conf.GetAtomPosition(int(u)).y))
        pv = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(v)).x,
                                           conf.GetAtomPosition(int(v)).y))
        internal_bond_lines.append((pu.x, pu.y, pv.x, pv.y, palette[hard_assign[u]]))

    for idx in cut_edges_indices:
        u, v = undirected_edges[idx]
        pu = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(u)).x,
                                           conf.GetAtomPosition(int(u)).y))
        pv = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(v)).x,
                                           conf.GetAtomPosition(int(v)).y))
        mid_x = (pu.x + pv.x) / 2.0
        mid_y = (pu.y + pv.y) / 2.0
        cut_edge_lines.append((pu.x, pu.y, mid_x, mid_y, palette[hard_assign[u]]))
        cut_edge_lines.append((mid_x, mid_y, pv.x, pv.y, palette[hard_assign[v]]))

    # ── render ────────────────────────────────────────────────────────────────
    if svg:
        txt = drawer.GetDrawingText()
        custom_bonds_svg = ""
        for x1, y1, x2, y2, color in internal_bond_lines + cut_edge_lines:
            r, g, b = [int(c * 255) for c in color]
            custom_bonds_svg += (
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="rgb({r},{g},{b})" stroke-width="{bond_width}" '
                f'stroke-opacity="1.0" stroke-linecap="butt" />\n'
            )
        txt = txt.replace('<path', custom_bonds_svg + '<path', 1)

        if show_labels:
            svg_insert = ""
            for label, pos, cg_group_id in annotations:
                p = drawer.GetDrawCoords(pos)
                svg_insert += (
                    f'<text x="{p.x}" y="{p.y}" font-size="22" font-family="Arial" '
                    f'text-anchor="middle" dominant-baseline="middle" '
                    f'fill="black">{label}</text>\n'
                )
            txt = txt.replace('</svg>', svg_insert + '</svg>')
        img = txt.replace('svg:', '')

    else:
        txt     = drawer.GetDrawingText()
        pil_img = Image.open(io.BytesIO(txt)).convert("RGBA")
        final_img = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))

        highlight_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        highlight_draw  = ImageDraw.Draw(highlight_layer, "RGBA")

        for x1, y1, x2, y2, color in internal_bond_lines + cut_edge_lines:
            r, g, b = [int(c * 255) for c in color]
            highlight_draw.line([(x1, y1), (x2, y2)],
                                fill=(r, g, b, 255), width=bond_width)

        final_img = Image.alpha_composite(final_img, highlight_layer)
        final_img = Image.alpha_composite(final_img, pil_img)

        if show_labels:
            draw = ImageDraw.Draw(final_img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 25)
            except IOError:
                font = ImageFont.load_default()

            for label, pos, cg_group_id in annotations:
                p = drawer.GetDrawCoords(pos)
                try:
                    w, h = draw.textsize(label, font=font)
                except AttributeError:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                r, g, b = [int(c * 255) for c in palette[cg_group_id]]
                draw.text(
                    (p.x - w / 2, p.y - h / 2), label,
                    fill="black", font=font,
                    stroke_width=4, stroke_fill=(r, g, b),
                )

        img = np.asarray(final_img.convert("RGB"))

    return img


# ── visualization: style 2 — black molecule + translucent ellipses ────────────

def draw_graph_style2(graph, hard_assign, cg_types, show_labels=False, svg=False):
    """
    Draws a 2D molecule with structural groups highlighted by semi-transparent
    oriented ellipses. The molecule itself is drawn in entirely black.
    """
    smiles   = graph.graph['smiles']
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, "Failed to parse SMILES string."
    rdCoordGen.AddCoords(molecule)

    hard_assign = np.array(hard_assign)

    palette = np.array(sns.color_palette("Set2", hard_assign.max() + 1))

    drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600) if svg \
             else rdMolDraw2D.MolDraw2DCairo(1200, 600)

    options = drawer.drawOptions()
    options.addAtomIndices  = False
    options.clearBackground = False
    options.useBWAtomPalette()

    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()

    # ── per-group ellipse geometry ────────────────────────────────────────────
    conf = molecule.GetConformer()
    cg_group_to_atoms = {}
    for l in range(molecule.GetNumAtoms()):
        cg_group_to_atoms.setdefault(int(hard_assign[l]), []).append(l)

    half_bond_px = 28 # Fallback
    if molecule.GetNumBonds() > 0:
        b = molecule.GetBondWithIdx(0)
        p1 = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(b.GetBeginAtomIdx()).x, conf.GetAtomPosition(b.GetBeginAtomIdx()).y))
        p2 = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(b.GetEndAtomIdx()).x, conf.GetAtomPosition(b.GetEndAtomIdx()).y))
        bond_len = math.hypot(p1.x - p2.x, p1.y - p2.y)
        if bond_len > 0:
            half_bond_px = bond_len / 2.0

    circle_data   = []
    label_centers = {}

    for gid, indices in cg_group_to_atoms.items():
        pixel_coords = []
        for i in indices:
            ap = conf.GetAtomPosition(i)
            px = drawer.GetDrawCoords(Point2D(ap.x, ap.y))
            pixel_coords.append(px)

        if len(pixel_coords) == 1:
            cx, cy = pixel_coords[0].x, pixel_coords[0].y
            rx = half_bond_px * 1.5
            ry = rx * (2.0 / 3.0)
            theta = 0.0
        else:
            max_d_sq = -1
            p1_best, p2_best = pixel_coords[0], pixel_coords[1]
            for i in range(len(pixel_coords)):
                for j in range(i + 1, len(pixel_coords)):
                    p1, p2 = pixel_coords[i], pixel_coords[j]
                    d_sq = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
                    if d_sq > max_d_sq:
                        max_d_sq = d_sq
                        p1_best, p2_best = p1, p2

            dist = math.sqrt(max_d_sq)
            cx = (p1_best.x + p2_best.x) / 2.0
            cy = (p1_best.y + p2_best.y) / 2.0

            dx = p2_best.x - p1_best.x
            dy = p2_best.y - p1_best.y
            theta = math.degrees(math.atan2(dy, dx))

            rx = (dist / 2.0) + half_bond_px
            ry = rx * (2.0 / 3.0)

        circle_data.append((gid, cx, cy, rx, ry, theta))
        label_centers[gid] = (cx, cy)

    # ── render ────────────────────────────────────────────────────────────────
    if svg:
        txt        = drawer.GetDrawingText()
        svg_ellipses = ""
        for gid, cx, cy, rx, ry, theta in circle_data:
            rc, gc, bc = palette[gid]
            r_c, g_c, b_c = int(rc * 255), int(gc * 255), int(bc * 255)
            rx_small = rx * 0.8
            ry_small = ry * 0.8

            # Reverted to your original SVG opacity (0.25) and stroke mapping
            svg_ellipses += (
                f'<g transform="translate({cx:.1f}, {cy:.1f}) rotate({theta:.1f})">'
                f'<ellipse cx="0" cy="0" rx="{rx_small:.1f}" ry="{ry_small:.1f}" '
                f'fill="rgba({r_c},{g_c},{b_c},0.25)" '
                f'stroke="rgba({r_c},{g_c},{b_c},0.6)" stroke-width="1"/>'
                f'</g>\n'
            )
        txt = txt.replace('<path', svg_ellipses + '<path', 1)

        if show_labels:
            svg_labels = ""
            for gid, indices in cg_group_to_atoms.items():
                cx, cy = label_centers[gid]
                label   = f"{cg_types[indices[0]]}"
                svg_labels += (
                    f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="22" '
                    f'font-family="Arial" text-anchor="middle" '
                    f'dominant-baseline="middle" fill="black">{label}</text>\n'
                )
            txt = txt.replace('</svg>', svg_labels + '</svg>')
        img = txt.replace('svg:', '')

    else:
        txt     = drawer.GetDrawingText()
        pil_img = Image.open(io.BytesIO(txt)).convert("RGBA")
        circles_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))

        for gid, cx, cy, rx, ry, theta in circle_data:
            rc, gc, bc  = palette[gid]
            r_int, g_int, b_int = int(rc * 255), int(gc * 255), int(bc * 255)

            # Reverted to your original PIL opacity (65) and grey stroke
            fill_color   = (r_int, g_int, b_int, 240)
            stroke_color = 'grey'

            # Matched transparent background prevents muddy fringes during rotation
            transparent_bg = (r_int, g_int, b_int, 0)

            rx_small     = rx * 0.8
            ry_small     = ry * 0.8

            box_size = int(math.ceil(max(rx_small, ry_small) * 2 + 4))
            local_layer = Image.new("RGBA", (box_size, box_size), transparent_bg)
            local_draw  = ImageDraw.Draw(local_layer, "RGBA")

            ctr = box_size / 2.0
            local_draw.ellipse(
                [ctr - rx_small, ctr - ry_small, ctr + rx_small, ctr + ry_small],
                fill=fill_color, outline=stroke_color, width=1,
            )

            rotated_layer = local_layer.rotate(-theta, resample=Image.BICUBIC, expand=True)

            paste_x = int(cx - rotated_layer.width / 2.0)
            paste_y = int(cy - rotated_layer.height / 2.0)

            temp_canvas = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            temp_canvas.paste(rotated_layer, (paste_x, paste_y), rotated_layer)

            circles_layer = Image.alpha_composite(circles_layer, temp_canvas)

        pil_img = Image.alpha_composite(circles_layer, pil_img)

        if show_labels:
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 25)
            except IOError:
                font = ImageFont.load_default()

            for gid, indices in cg_group_to_atoms.items():
                cx, cy = label_centers[gid]
                label  = f"{cg_types[indices[0]]}"
                try:
                    w, h = draw.textsize(label, font=font)
                except AttributeError:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((cx - w / 2, cy - h / 2), label,
                          fill="black", font=font)

        final_bg  = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        final_img = Image.alpha_composite(final_bg, pil_img)
        img       = np.asarray(final_img.convert("RGB"))

    return img


# ── visualization: dispatch + save ───────────────────────────────────────────

def gen_vis(dataloader, output_file, style=1, show_labels=False, svg=False):
    """
    Parameters
    ----------
    svg : bool
        When True, ask the draw_graph_* function for an SVG string and write
        it to disk as text. When False, ask for an RGB ndarray and write it
        with skimage.io.imsave. This flag is passed straight through to the
        draw_graph_* call below so the two decisions (what gets returned,
        how it gets saved) can never disagree.
    """
    vis_path = output_file

    for i, data in enumerate(dataloader):
        data      = data[0]
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        graph_nx  = data.graph
        cg_types  = data.json.get('cgnode_types', [])

        # skip the initial _aa.json which has no predictions yet
        if len(cg_types) == 0:
            continue

        gt_hard_assigns = data.y.cpu().numpy()
        smiles_str      = graph_nx.graph['smiles']
        mol_name        = data.json.get('compound name', 'unknown_molecule')

        if style == 1:
            draw_graph = draw_graph_style1
        elif style == 2:
            draw_graph = draw_graph_style2
        else:
            raise ValueError("Invalid style. Choose 1 or 2.")

        if not debug:
            gt_img = draw_graph(graph_nx, gt_hard_assigns, cg_types,
                                show_labels=show_labels, svg=svg)
            print("Success:")
            print(f"    smiles: {smiles_str}")
            print(f"    name:   {mol_name}")

            if svg:
                fpath = os.path.join(vis_path, f"{mol_name}.svg")
                with open(fpath, "wt") as svg_file:
                    svg_file.write(gt_img)
            else:
                fpath = os.path.join(vis_path, f"{mol_name}.png")
                imsave(fpath, gt_img)


# ── inference: single molecule ────────────────────────────────────────────────

def eval(test_dataloader, model, output_dir, name,
         num_cg_beads=None, use_regular_mapping_from_prediction=True,
         cluster_random_seed=None):
    model.eval()

    tbar = tqdm.tqdm(enumerate(test_dataloader),
                     total=len(test_dataloader), dynamic_ncols=True)

    for i, data in tbar:
        data      = data[0]
        json_data = data.json
        json_data['cgnodes'] = []

        num_nodes  = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        data       = data.to(torch.device(0))

        edge_index_cpu = data.edge_index.cpu().numpy()

        fg_embed, node_cg_type_pred = model(data)
        softmax_output       = F.softmax(node_cg_type_pred, dim=1)
        predicted_cg_types_id = torch.argmax(softmax_output.cpu(), dim=1)
        predicted_cg_types    = [
            CG_TYPE_DICT[cgtype.item()] for cgtype in predicted_cg_types_id
        ]

        if num_cg_beads is None:
            iter_num_cg_beads = range(2, num_nodes)
        else:
            iter_num_cg_beads = num_cg_beads

        hard_assign, _ = graph_cuts(
            fg_embed, data.edge_index, num_cg_beads,
            random_state=cluster_random_seed,
        )
        hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)

        actual_num_cg = max(hard_assign) + 1
        if actual_num_cg != num_cg_beads:
            print(f'warning: actual vs. expected: {actual_num_cg} vs. {num_cg_beads}')

        result_json = copy.deepcopy(json_data)
        for atom_idx, cg_idx in enumerate(hard_assign):
            result_json['nodes'][atom_idx]['cg_id']   = int(cg_idx)
            result_json['nodes'][atom_idx]['cg_type'] = predicted_cg_types[atom_idx]
        result_json['cgnode_types'] = predicted_cg_types

        for cg_idx in range(num_cg_beads):
            atom_indices = np.nonzero(hard_assign == cg_idx)[0].tolist()
            result_json['cgnodes'].append([int(x) for x in atom_indices])

        # majority-vote bead type per group
        if use_regular_mapping_from_prediction:
            cg_groups = []
            for b in range(num_cg_beads):
                cg_groups.append([
                    predicted_cg_types[idx]
                    for idx, cg in enumerate(hard_assign) if b == cg
                ])
            new_cg_groups = [adjust_list(sublist) for sublist in cg_groups]
            new_predicted_cg_types = [new_cg_groups[b][0] for b in hard_assign]

            for atom_idx, cg_idx in enumerate(hard_assign):
                result_json['nodes'][atom_idx]['cg_type'] = new_predicted_cg_types[atom_idx]
            result_json['cgnode_types'] = new_predicted_cg_types

        fpath = os.path.join(output_dir, f'{name}_cg_{actual_num_cg}.json')
        if os.path.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'w') as f:
            json.dump(result_json, f, indent=4)

        return result_json


# ── output: all-atom + coarse-grained PDB ─────────────────────────────────────

def _bead_element_for_pdb(cg_type):
    """PDB atom names should start with an element symbol so viewers (VMD,
    PyMOL, etc.) render something sensible. We use a generic carbon; the
    real bead identity lives in the residue name (cg_type)."""
    return "C"


def _embed_3d_conformer(smiles, embed_seed=42, mmff_iters=500):
    """
    Build a 3D atomistic conformer for `smiles` via RDKit ETKDG embedding
    + MMFF (falling back to UFF) geometry optimization.

    Returns
    -------
    mol_h   : RDKit Mol with explicit Hs and one embedded/optimized conformer
    n_heavy : int, number of heavy atoms (== Chem.MolFromSmiles(smiles).GetNumAtoms())

    Notes
    -----
    RDKit's AddHs appends new H atoms after the existing heavy atoms, so
    heavy-atom indices 0..n_heavy-1 in `mol_h` line up exactly with the
    `nodes`/`cgnodes` atom indices written by DSGPM_TPtoCG.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    n_heavy = mol.GetNumAtoms()

    mol_h = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(mol_h, randomSeed=embed_seed, useRandomCoords=True)
    if cid < 0:
        # one retry with a different seed before giving up
        cid = AllChem.EmbedMolecule(mol_h, randomSeed=embed_seed + 1, useRandomCoords=True)
    if cid < 0:
        raise RuntimeError(f"3D embedding failed for SMILES: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=mmff_iters)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=mmff_iters)

    return mol_h, n_heavy


def write_aa_pdb(mol_h, out_path, compound_name="MOL"):
    """
    Write the all-atom (explicit-H) RDKit conformer produced by
    `_embed_3d_conformer` out to a PDB file, exactly as embedded/optimized
    (i.e. the RDKit AllChem.EmbedMolecule + MMFF/UFF-optimized structure).
    """
    Chem.MolToPDBFile(mol_h, out_path)

    # RDKit's writer doesn't add a HEADER/REMARK, so prepend one for
    # traceability, matching the style of the CG PDB below.
    with open(out_path, "r") as f:
        body = f.read()
    header = (
        f"HEADER    ALL-ATOM CONFORMER FOR {compound_name}\n"
        f"REMARK    generated by MappingPrediction.py (RDKit "
        f"AllChem.EmbedMolecule + MMFF/UFF optimization)\n"
    )
    with open(out_path, "w") as f:
        f.write(header + body)

    return out_path


def generate_cg_pdb(result_json, out_path, embed_seed=42, mmff_iters=500,
                    aa_out_path=None):
    """
    Build a 3D atomistic conformer for the molecule described in
    `result_json`, collapse it onto CG beads (mass-weighted center of mass
    per bead), and write the result as a PDB file (one HETATM per bead,
    plus CONECT records for bonds that cross bead boundaries).

    Optionally also writes the underlying all-atom conformer (the RDKit
    AllChem.EmbedMolecule + MMFF/UFF-optimized structure the beads were
    derived from) to `aa_out_path`, so the two files are guaranteed to be
    geometrically consistent (same embedding, same seed).

    Parameters
    ----------
    result_json : dict
        Mapping dict produced by DSGPM_TPtoCG/eval() — needs 'smiles',
        'nodes', 'cgnodes' (atom-index groups per bead), 'cgnode_types'
        (bead type per atom), and 'edges'.
    out_path : str
        Path to write the CG .pdb file to.
    embed_seed : int
        RNG seed for the 3D embedding (ETKDG).
    mmff_iters : int
        Max iterations for the MMFF/UFF geometry optimization.
    aa_out_path : str or None
        If given, also write the all-atom conformer used to build the CG
        beads to this path.

    Returns
    -------
    (cg_path, aa_path) : tuple(str, str or None)

    Notes
    -----
    `cgnodes` atom indices must refer to the heavy-atom ordering of
    Chem.MolFromSmiles(smiles) with no explicit Hs — this matches what
    DSGPM_TPtoCG writes into `nodes`/`cgnodes`. See `_embed_3d_conformer`
    for why heavy-atom indices line up between `mol_h` and `cgnodes`.
    """
    smiles = result_json['smiles']
    mol_h, n_heavy = _embed_3d_conformer(smiles, embed_seed=embed_seed,
                                         mmff_iters=mmff_iters)

    compound_name = result_json.get('compound name', 'MOL')
    aa_path = None
    if aa_out_path is not None:
        aa_path = write_aa_pdb(mol_h, aa_out_path, compound_name=compound_name)

    conf = mol_h.GetConformer()
    mol_noh = Chem.MolFromSmiles(smiles)  # for heavy-atom masses only
    heavy_coords  = np.array([list(conf.GetAtomPosition(i)) for i in range(n_heavy)])
    heavy_masses  = np.array([mol_noh.GetAtomWithIdx(i).GetMass() for i in range(n_heavy)])

    cgnodes  = result_json['cgnodes']       # list[list[int]] — atom indices per bead
    cg_types = result_json['cgnode_types']  # bead type PER ATOM (already majority-voted uniform within a bead)

    bead_type_per_bead = [cg_types[atom_indices[0]] for atom_indices in cgnodes]

    bead_coords = []
    for atom_indices in cgnodes:
        idx = np.array(atom_indices)
        w   = heavy_masses[idx]
        com = (heavy_coords[idx] * w[:, None]).sum(axis=0) / w.sum()
        bead_coords.append(com)

    # bead-bead bonds = atomistic bonds that cross a bead boundary
    atom_to_bead = {}
    for bead_id, atom_indices in enumerate(cgnodes):
        for a in atom_indices:
            atom_to_bead[a] = bead_id

    bead_bonds = set()
    for e in result_json['edges']:
        b1 = atom_to_bead[e['source']]
        b2 = atom_to_bead[e['target']]
        if b1 != b2:
            bead_bonds.add(tuple(sorted((b1, b2))))

    # ── write CG PDB ──────────────────────────────────────────────────────
    lines = [
        f"HEADER    COARSE-GRAINED MAPPING FOR {compound_name}",
        "REMARK    generated by MappingPrediction.py (DSGPM-TP)",
        f"REMARK    bead position = mass-weighted center of mass of "
        f"atomistic conformer (embed_seed={embed_seed})",
    ]

    for bead_id, (coord, btype) in enumerate(zip(bead_coords, bead_type_per_bead)):
        serial   = bead_id + 1
        atomname = btype[:4].ljust(4)
        resname  = btype[:3].ljust(3)
        x, y, z  = coord
        lines.append(
            f"HETATM{serial:5d} {atomname:<4s} {resname:<3s} A{serial:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          "
            f"{_bead_element_for_pdb(btype):>2s}"
        )

    for b1, b2 in sorted(bead_bonds):
        lines.append(f"CONECT{b1 + 1:5d}{b2 + 1:5d}")

    lines.append("END")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path, aa_path


# ── public API ────────────────────────────────────────────────────────────────

def DSGPM_TPtoCG(smiles, out_dir, num_cg_bead, name=None,
                 cluster_random_seed=None, style=1, show_labels=False, svg=False,
                 write_cg=False, pdb_embed_seed=42):
    """Takes a SMILES string and returns a CG mapping in JSON format + PNG/SVG.

    Parameters
    ----------
    smiles             : SMILES string of the molecule
    out_dir            : directory to write JSON and image outputs
    num_cg_bead        : number of CG beads to predict
    name               : molecule name; defaults to molecular formula
    cluster_random_seed: random seed for graph-cut clustering
    style               : visualization style (1 = colored beads, 2 = ellipses)
    show_labels        : overlay bead type labels on the output image
    svg                 : if True, write the visualization as .svg (text);
                          if False (default), write it as .png (raster image)
    write_cg            : if True, also write two PDB files: the all-atom
                          RDKit-embedded/MMFF-optimized conformer
                          (`{name}_aa.pdb`) and the coarse-grained mapping
                          built from it (`{name}_cg_{N}.pdb`)
    pdb_embed_seed      : RNG seed used for the RDKit 3D embedding
    """
    m     = Chem.MolFromSmiles(smiles)
    edges = []
    for j in range(m.GetNumBonds()):
        b     = m.GetBonds()[j]
        begin = b.GetBeginAtomIdx()
        end   = b.GetEndAtomIdx()
        bond  = m.GetBondWithIdx(j).GetBondTypeAsDouble()
        edges.append({"source": begin, "target": end, "bondtype": bond})

    nodes = []
    for l in range(m.GetNumAtoms()):
        element = m.GetAtomWithIdx(l).GetSymbol()
        nodes.append({"id": l, "element": element, "charge": 0,
                      "cg_id": 0, "cg_type": "C3E"})

    AllChem.ComputeGasteigerCharges(m)
    for l in range(m.GetNumAtoms()):
        nodes[l]["charge"] = round(
            float(m.GetAtomWithIdx(l).GetProp('_GasteigerCharge')), 3
        )

    if name is None:
        try:
            name = smiles_to_formula(smiles)
        except Exception:
            print("\nMolecular formula could not be computed.")
            name = smiles

    cg_dict = {
        "compound name": name,
        "smiles":        smiles,
        "cgnodes":       [],
        "cgnode_types":  [],
        "nodes":         nodes,
        "edges":         edges,
        "note":          "generated by MappingPrediction.py",
    }

    os.makedirs(out_dir, exist_ok=True)
    predict_out_dir = os.path.join(out_dir, f"{name}")
    os.makedirs(predict_out_dir, exist_ok=True)

    # clear any stale JSON from a previous run
    for file in os.listdir(predict_out_dir):
        if file.endswith('.json'):
            os.remove(os.path.join(predict_out_dir, file))

    oname = re.sub('[^A-Za-z0-9]+', '', name) + '_aa.json'
    ofile = os.path.join(predict_out_dir, oname)
    with open(ofile, 'w') as f:
        f.write(json.dumps(cg_dict, sort_keys=False, indent=4))

    # ── load data ─────────────────────────────────────────────────────────────
    test_set = HAMPerFile(
        data_root=predict_out_dir,
        cycle_feat=True, degree_feat=True,
        charge_feat=False, aromatic_feat=True,
        automorphism=False,
    )
    test_dataloader = DataListLoader(test_set, batch_size=1,
                                     num_workers=4, pin_memory=True)

    # ── load model ────────────────────────────────────────────────────────────
    model = DSGPM_TP(
        input_dim      = len(ATOMS),
        hidden_dim     = 128,
        embedding_dim  = 128,
        use_cycle_feat = True,
        use_degree_feat= True,
        use_charge_feat= False,
        use_aromatic_feat=True,
    ).cuda()
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'model/best_epoch.pth'))
    model.load_state_dict(ckpt)

    # ── predict ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        predict_json = eval(
            test_dataloader, model,
            output_dir=predict_out_dir, name=name,
            num_cg_beads=num_cg_bead,
            cluster_random_seed=cluster_random_seed,
        )

    # ── visualize ─────────────────────────────────────────────────────────────
    test_set = HAMPerFile(
        data_root=predict_out_dir,
        cycle_feat=True, degree_feat=True,
        charge_feat=False, aromatic_feat=False,
        automorphism=False,
    )
    test_dataloader = DataListLoader(test_set, batch_size=1,
                                     num_workers=0, pin_memory=True)
    gen_vis(test_dataloader, output_file=predict_out_dir,
            style=style, show_labels=show_labels, svg=svg)

    # ── all-atom + CG PDB ─────────────────────────────────────────────────────
    if write_cg:
        cg_pdb_fpath = os.path.join(
            predict_out_dir, f"{name}_cg_{len(predict_json['cgnodes'])}.pdb"
        )
        aa_pdb_fpath = os.path.join(predict_out_dir, f"{name}_aa.pdb")
        try:
            cg_path, aa_path = generate_cg_pdb(
                predict_json, cg_pdb_fpath,
                embed_seed=pdb_embed_seed, aa_out_path=aa_pdb_fpath,
            )
            print(f"    all-atom PDB written: {aa_path}")
            print(f"    CG PDB written:       {cg_path}")
        except Exception as ex:
            print(f"    warning: PDB generation failed: {ex}")

    print('    CG bead number: ', num_cg_bead)
    print('DSGPM_TP prediction complete.')
    return predict_json


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="iConMapper — CG mapping via DSGPM-TP"
    )
    parser.add_argument('--name',   type=str,  default=None,
                        help='Molecule name (default: molecular formula)')
    parser.add_argument('--smiles', type=str,  required=True,
                        help='SMILES string of the molecule')
    parser.add_argument('--output', type=str,  default='./aa2cg',
                        help='Output directory (default: ./aa2cg)')
    parser.add_argument('--num',    type=int,  default=None,
                        help='Number of CG beads (default: n_heavy_atoms // 3)')
    parser.add_argument('--labels', action='store_true',
                        help='Overlay bead type labels on the output image')
    parser.add_argument('--style',  type=int,  default=1,
                        help='Visualization style: 1=colored beads, 2=ellipses (default: 1)')
    parser.add_argument('--svg',    action='store_true',
                        help='Write the visualization as .svg instead of .png')
    parser.add_argument('--cg',     action='store_true',
                        help='Also write an all-atom .pdb (RDKit-embedded, '
                             'MMFF/UFF-optimized conformer) and a coarse-grained '
                             '.pdb built from it (one bead per CG group, at the '
                             'atomistic center of mass)')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    num = args.num
    if not num:
        mol = Chem.MolFromSmiles(args.smiles)
        num = int(mol.GetNumHeavyAtoms() / 3)

    DSGPM_TPtoCG(
        smiles      = args.smiles,
        out_dir     = args.output,
        num_cg_bead = num,
        name        = args.name,
        style       = args.style,
        show_labels = args.labels,
        svg         = args.svg,
        write_cg    = args.cg,
    )


if __name__ == "__main__":
    main()