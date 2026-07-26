import sys
import random
import os
import re
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Geometry import Point2D
import requests
import tqdm
import json
import copy
from model.networks import DSGPM_TP
from model.graph_cuts import graph_cuts
from utils.post_processing import enforce_connectivity
import torch.nn.functional as F
from dataset.ham import ATOMS
from sklearn.exceptions import UndefinedMetricWarning
from collections import Counter
import torch
import seaborn as sns
import numpy as np
import io
from dataset.ham_per_file import HAMPerFile
from torch_geometric.data import DataListLoader
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.io import imsave
from warnings import simplefilter
from PIL import ImageDraw, ImageFont


simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)
simplefilter(action='ignore', category=Warning)

import argparse

svg = False
debug = False

def draw_graph_style1(graph, hard_assign, cg_types, show_labels=False, svg=False):
    smiles = graph.graph['smiles']
    molecule = Chem.MolFromSmiles(smiles)

    assert molecule is not None
    # rdDepictor.Compute2DCoords(molecule)
    rdCoordGen.AddCoords(molecule)

    hard_assign = np.array(hard_assign)
    
    # Using 'tab20' so neighboring IDs don't get similar colors (fixes the color blending issue!)
    palette = np.array(sns.color_palette("Set2", hard_assign.max() + 1))

    atom_index = list(range(len(graph.nodes)))
    undirected_edges = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()])
    
    non_cut_edges_indices = np.nonzero(hard_assign[undirected_edges[:, 0]] == hard_assign[undirected_edges[:, 1]])[0]
    cut_edges_indices = np.nonzero(hard_assign[undirected_edges[:, 0]] != hard_assign[undirected_edges[:, 1]])[0]

    atom_colors = list(map(tuple, palette[hard_assign]))
    atom_id_to_color_dict = dict(zip(atom_index, atom_colors))

    # --- 1. CALCULATE MOLECULE COORDINATES & LABELS ---
    annotations = []
    conf = molecule.GetConformer()
    
    if show_labels:
        cg_group_to_atoms = {}
        for l in range(molecule.GetNumAtoms()):
            cg_group_id = int(hard_assign[l]) 
            if cg_group_id not in cg_group_to_atoms:
                cg_group_to_atoms[cg_group_id] = []
            cg_group_to_atoms[cg_group_id].append(l)
            
        for cg_group_id, indices in cg_group_to_atoms.items():
            label = f"{cg_types[indices[0]]}"
            avg_x = sum(conf.GetAtomPosition(idx).x for idx in indices) / len(indices)
            avg_y = sum(conf.GetAtomPosition(idx).y for idx in indices) / len(indices)
            annotations.append((label, Point2D(avg_x, avg_y), cg_group_id))

    # --- 2. SETUP DRAWER AND DRAW ONLY ATOM HIGHLIGHTS ---
    if svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600)
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(1200, 600)

    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.clearBackground = False 

    # --- TUNE YOUR SIZES HERE ---
    atom_radius_size = 0.43      # Size of the beads (molecule coordinate units)
    bond_width_scale = 1.0       # 1.0 = bonds exactly fill the bead diameter; tune 0.8–1.2
    # ----------------------------

    drawer.DrawMolecule(
        molecule,
        highlightAtoms=atom_index,
        highlightBonds=[],
        highlightAtomColors=atom_id_to_color_dict,
        highlightAtomRadii=dict(zip(atom_index, [atom_radius_size] * len(atom_index)))
    )
    drawer.FinishDrawing()

    # Compute pixel scale AFTER FinishDrawing so GetDrawCoords is valid
    p0_px = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(0).x,       conf.GetAtomPosition(0).y))
    p1_px = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(0).x + 1.0, conf.GetAtomPosition(0).y))
    pixels_per_unit = abs(p1_px.x - p0_px.x)

    bond_width = max(1, int(atom_radius_size * pixels_per_unit * 2 * bond_width_scale))
    internal_bond_width = bond_width
    cut_bond_width      = bond_width

    # --- 3. CALCULATE ALL CUSTOM BONDS (INTERNAL & CUT) ---
    internal_bond_lines = []
    for idx in non_cut_edges_indices:
        u = undirected_edges[idx][0]
        v = undirected_edges[idx][1]
        
        pu = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(u)).x, conf.GetAtomPosition(int(u)).y))
        pv = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(v)).x, conf.GetAtomPosition(int(v)).y))
        
        c = palette[hard_assign[u]] # Same group, single color
        internal_bond_lines.append((pu.x, pu.y, pv.x, pv.y, c))

    cut_edge_lines = []
    for idx in cut_edges_indices:
        u = undirected_edges[idx][0]
        v = undirected_edges[idx][1]
        
        pu = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(u)).x, conf.GetAtomPosition(int(u)).y))
        pv = drawer.GetDrawCoords(Point2D(conf.GetAtomPosition(int(v)).x, conf.GetAtomPosition(int(v)).y))
        
        mid_x = (pu.x + pv.x) / 2.0
        mid_y = (pu.y + pv.y) / 2.0
        
        cu = palette[hard_assign[u]]
        cv = palette[hard_assign[v]]
        
        cut_edge_lines.append((pu.x, pu.y, mid_x, mid_y, cu))
        cut_edge_lines.append((mid_x, mid_y, pv.x, pv.y, cv))

    # --- 4. OVERLAY EVERYTHING ---
    if svg:
        txt = drawer.GetDrawingText()
        custom_bonds_svg = ""
        
        # 1. Draw Internal Bonds
        for x1, y1, x2, y2, color in internal_bond_lines:
            r, g, b = [int(c * 255) for c in color]
            custom_bonds_svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="rgb({r},{g},{b})" stroke-width="{internal_bond_width}" stroke-opacity="1.0" stroke-linecap="butt" />\n'
            
        # 2. Draw Cut Bonds
        for x1, y1, x2, y2, color in cut_edge_lines:
            r, g, b = [int(c * 255) for c in color]
            custom_bonds_svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="rgb({r},{g},{b})" stroke-width="{cut_bond_width}" stroke-opacity="1.0" stroke-linecap="butt" />\n'
        
        txt = txt.replace('<path', custom_bonds_svg + '<path', 1)

        if show_labels:
            svg_insert = ""
            for label, pos, cg_group_id in annotations:
                p = drawer.GetDrawCoords(pos)
                svg_insert += f'<text x="{p.x}" y="{p.y}" font-size="22" font-family="Arial" text-anchor="middle" dominant-baseline="middle" fill="black">{label}</text>\n'
            txt = txt.replace('</svg>', svg_insert + '</svg>')
            
        img = txt.replace('svg:','')
        
    else:
        txt = drawer.GetDrawingText()
        pil_img = Image.open(io.BytesIO(txt)).convert("RGBA")
        
        final_img = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        
        highlight_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        highlight_draw = ImageDraw.Draw(highlight_layer, "RGBA")
        
        # 1. Draw Internal Bonds
        for x1, y1, x2, y2, color in internal_bond_lines:
            r, g, b = [int(c * 255) for c in color]
            highlight_draw.line([(x1, y1), (x2, y2)], fill=(r, g, b, 255), width=internal_bond_width)

        # 2. Draw Cut Bonds
        for x1, y1, x2, y2, color in cut_edge_lines:
            r, g, b = [int(c * 255) for c in color]
            highlight_draw.line([(x1, y1), (x2, y2)], fill=(r, g, b, 255), width=cut_bond_width)
            
        final_img = Image.alpha_composite(final_img, highlight_layer)
        final_img = Image.alpha_composite(final_img, pil_img)
        
        if show_labels:
            draw = ImageDraw.Draw(final_img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 25)
            except IOError:
                font = ImageFont.load_default()

            for label, pos, cg_group_id in annotations:
                p = drawer.GetDrawCoords(pos)
                
                try:
                    w, h = draw.textsize(label, font=font)
                except AttributeError:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                
                r, g, b = [int(c * 255) for c in palette[cg_group_id]]
                
                draw.text(
                    (p.x - w/2, p.y - h/2), 
                    label, 
                    fill="black", 
                    font=font,
                    stroke_width=4,
                    stroke_fill=(r, g, b)
                )

        img = np.asarray(final_img.convert("RGB"))

    return img


def draw_graph_style2(graph, hard_assign, cg_types, show_labels=False, svg=False):
    """
    Draws a 2D molecule with structural groups highlighted by semi-transparent circles.
    The molecule itself is drawn in entirely black.
    """
    smiles = graph.graph['smiles']
    molecule = Chem.MolFromSmiles(smiles)

    assert molecule is not None, "Failed to parse SMILES string."
    # rdDepictor.Compute2DCoords(molecule)
    rdCoordGen.AddCoords(molecule)


    # Cast to numpy array to ensure .max() works
    hard_assign = np.array(hard_assign)
    palette = np.array(sns.hls_palette(hard_assign.max() + 1))

    # --- SET UP DRAWER ---
    if svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600)
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(1200, 600)

    options = drawer.drawOptions()
    options.addAtomIndices = False
    
    # CRITICAL 1: Disables opaque white background
    options.clearBackground = False 
    
    # CRITICAL 2: Forces all atoms and bonds to be drawn in black
    options.useBWAtomPalette()

    # Draw molecule first so bounds are calculated for pixel mapping
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()

    # --- COMPUTE PER-GROUP CIRCLE CENTERS AND RADII ---
    conf = molecule.GetConformer()
    cg_group_to_atoms = {}
    for l in range(molecule.GetNumAtoms()):
        gid = int(hard_assign[l])
        cg_group_to_atoms.setdefault(gid, []).append(l)

    circle_data = [] 
    label_centers = {} 

    for gid, indices in cg_group_to_atoms.items():
        # 1. Map all atoms in the group to pixel space first
        pixel_coords = []
        for i in indices:
            ap = conf.GetAtomPosition(i)
            px = drawer.GetDrawCoords(Point2D(ap.x, ap.y))
            pixel_coords.append(px)

        # 2. Find the bounding box of the group in pixel space
        min_x = min(p.x for p in pixel_coords)
        max_x = max(p.x for p in pixel_coords)
        min_y = min(p.y for p in pixel_coords)
        max_y = max(p.y for p in pixel_coords)

        # 3. Calculate the visual center (middle of the bounding box)
        center_px_x = (min_x + max_x) / 2.0
        center_px_y = (min_y + max_y) / 2.0

        # 4. Calculate radius (max distance from this new center to any atom)
        max_dist = 0.0
        for p in pixel_coords:
            dist = ((p.x - center_px_x) ** 2 + (p.y - center_px_y) ** 2) ** 0.5
            max_dist = max(max_dist, dist)
            
        radius = max_dist + 28 # padding in pixels
        circle_data.append((gid, center_px_x, center_px_y, radius))
        label_centers[gid] = (center_px_x, center_px_y)

# --- OVERLAY CIRCLES ---
    if svg:
        txt = drawer.GetDrawingText()
        svg_circles = ""
        for gid, cx, cy, r in circle_data:
            rc, gc, bc = palette[gid]
            r_c, g_c, b_c = int(rc*255), int(gc*255), int(bc*255)
            
            # Change this multiplier (e.g., 0.5 to 0.8) to adjust the size
            r_small = r * 0.8 
            
            # Using standard rgba() for better viewer transparency compatibility
            svg_circles += (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_small:.1f}" '
                f'fill="rgba({r_c},{g_c},{b_c},0.25)" stroke="rgba({r_c},{g_c},{b_c},0.6)" '
                f'stroke-width="1"/>\n'
            )
            
        txt = txt.replace('<path', svg_circles + '<path', 1)

        if show_labels:
            svg_labels = ""
            for gid, indices in cg_group_to_atoms.items():
                cx, cy = label_centers[gid]
                label = f"{cg_types[indices[0]]}"
                svg_labels += (
                    f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="22" font-family="Arial" '
                    f'text-anchor="middle" dominant-baseline="middle" fill="black">{label}</text>\n'
                )
            txt = txt.replace('</svg>', svg_labels + '</svg>')

        img = txt.replace('svg:', '')

    else:
        txt = drawer.GetDrawingText()
        pil_img = Image.open(io.BytesIO(txt)).convert("RGBA")

        # Base transparent layer for ALL circles combined
        circles_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))

        for gid, cx, cy, r in circle_data:
            rc, gc, bc = palette[gid]
            fill_color = (int(rc*255), int(gc*255), int(bc*255), 65)    # ~25% opacity
            # stroke_color = (int(rc*255), int(gc*255), int(bc*255), 150) # ~60% opacity
            stroke_color = 'grey'

            # CRITICAL FIX: Draw EACH circle on its own temporary layer to force PIL to alpha-blend them
            temp_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_layer, "RGBA")

            # Change this multiplier (e.g., 0.5 to 0.8) to adjust the size
            r_small = r * 0.8  
            temp_draw.ellipse(
                [cx - r_small, cy - r_small, cx + r_small, cy + r_small],
                fill=fill_color,
                outline=stroke_color,
                width=1
            )

            # Blend this individual circle onto the main circles layer
            circles_layer = Image.alpha_composite(circles_layer, temp_layer)

        # Composite the molecule ON TOP of the blended circles
        pil_img = Image.alpha_composite(circles_layer, pil_img)

        if show_labels:
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 25)
            except IOError:
                font = ImageFont.load_default()

            for gid, indices in cg_group_to_atoms.items():
                cx, cy = label_centers[gid]
                label = f"{cg_types[indices[0]]}"
                
                try:
                    w, h = draw.textsize(label, font=font)
                except AttributeError:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    
                draw.text((cx - w/2, cy - h/2), label, fill="black", font=font)

        # Final white background
        final_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        final_img = Image.alpha_composite(final_bg, pil_img)
        
        img = np.asarray(final_img.convert("RGB"))

    return img

    return img


def gen_vis(dataloader, output_file, style=1, show_labels=False):
    vis_path = output_file

    for i, data in enumerate(dataloader):
        # skip saved smiles
        data = data[0]
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        graph_nx = data.graph
        cg_types = data.json.get('cgnode_types', [])

        # Skip the initial _aa.json file which has no predictions
        if len(cg_types) == 0:
            continue

        gt_hard_assigns = data.y.cpu().numpy()

        # Extract the SMILES string and the compound name
        smiles_str = graph_nx.graph['smiles']
        mol_name = data.json.get('compound name', 'unknown_molecule')

        if style == 1:
            draw_graph = draw_graph_style1
        elif style == 2:
            draw_graph = draw_graph_style2
        else:
            raise ValueError("Invalid style. Choose from 1 and 2.")
        
        if not debug:
            gt_img = draw_graph(graph_nx, gt_hard_assigns, cg_types, show_labels=show_labels)
            
            # Updated print formatting
            print("Success:")
            print(f"    smiles: {smiles_str}")
            print(f"    name: {mol_name}")

            # Save the image using the molecule name
            if svg:
                fpath = os.path.join(vis_path, f"{mol_name}.svg")
                svg_file = open(fpath, "wt")
                svg_file.write(gt_img)
                svg_file.close()
            else:
                fpath = os.path.join(vis_path, f"{mol_name}.png")
                imsave(fpath, gt_img)



def adjust_list(lst):
    counts = Counter(lst)
    max_count = counts.most_common(1)[0][1]
    most_common_elements = [element for element, count in counts.items() if count == max_count]
    chosen_element = random.choice(most_common_elements)
    return [chosen_element] * len(lst)


CG_TYPE_DICT = {
    "C2E": 0,  "C3E": 1,  "A2V": 2,  "A1L": 3,  "A1I": 4,  "A5M": 5,  "P5N": 6,  "QaD": 7,  "P4Q": 8,  "QaE": 9, 
    "P1C": 10, "P1S": 11, "P1T": 12, "A2P": 13, "A3K": 14, "QdK": 15, "A3R": 16, "QdR": 17, "A4H": 18, "P1H": 19,
    "P2H": 20, "A1F": 21, "A2F": 22, "A1Y": 23, "A2Y": 24, "P1Y": 25, "A1W": 26, "P1W": 27, "A2W": 28,

    "RS1": 29, "RS2": 30, "RA1": 31, "RG1": 32, "RA2": 33, "RG2": 34, "RA3": 35, "RG4": 36, "RC2": 37, "RA4": 38,
    "RG3": 39, "RU2": 40, "RC1": 41, "RU1": 42, "RC3": 43, "RU3": 44, "PHO": 45,

    "M01": 46, "M02": 47, "M03": 48, "M04": 49, "M05": 50, "M06": 51, "M07": 52, "M08": 53, "MCI": 54, "MSO": 55,
    "MSS": 56, "MCL": 57, "MCF": 58, "MBR": 59
}


CG_TYPE_DICT = {value: key for key, value in CG_TYPE_DICT.items()}

CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
commad_file = './command.log'

def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def smiles_to_formula(smiles):
    """Returns the empirical formula instead of the IUPAC name."""
    try:
        m = Chem.MolFromSmiles(smiles)
        return rdMolDescriptors.CalcMolFormula(m)
    except:
        return None
    
def eval(test_dataloader, model, output_dir, name, num_cg_beads=None, use_regular_mapping_from_prediction=True, cluster_random_seed=None):
    model.eval()

    tbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), dynamic_ncols=True)
    for i, data in tbar:
        data = data[0]
        json_data = data.json
        json_data['cgnodes'] = []
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        data = data.to(torch.device(0))
        edge_index_cpu = data.edge_index.cpu().numpy()
        fg_embed, node_cg_type_pred = model(data)
        softmax_output = F.softmax(node_cg_type_pred, dim=1)
        predicted_cg_types_id = torch.argmax(softmax_output.cpu(), dim=1)
        predicted_cg_types = [CG_TYPE_DICT[cgtype.item()] for cgtype in predicted_cg_types_id]

        # dense_adj = torch.sparse.LongTensor(data.edge_index, data.no_bond_edge_attr, (num_nodes, num_nodes)).to_dense()

        if num_cg_beads is None:
            iter_num_cg_beads = range(2, num_nodes)
        else:
            iter_num_cg_beads = num_cg_beads


        hard_assign, _ = graph_cuts(fg_embed, data.edge_index, num_cg_beads, random_state=cluster_random_seed)
        # print(hard_assign)
        hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
        # print(hard_assign)
        actual_num_cg = max(hard_assign) + 1
        if actual_num_cg != num_cg_beads:
            print('warning: actual vs. expected: {} vs. {}'.format(actual_num_cg, num_cg_beads))

        result_json = copy.deepcopy(json_data)
        for atom_idx, cg_idx in enumerate(hard_assign):
            result_json['nodes'][atom_idx]['cg_id'] = int(cg_idx)
            result_json['nodes'][atom_idx]['cg_type'] = predicted_cg_types[atom_idx]
        result_json['cgnode_types'] = predicted_cg_types

        for cg_idx in range(num_cg_beads):
            atom_indices = np.nonzero(hard_assign == cg_idx)[0].tolist()
            atom_indices = [int(x) for x in atom_indices]
            result_json['cgnodes'].append(atom_indices)

        if use_regular_mapping_from_prediction:
            cg_groups = []
            for i in range(num_cg_beads):
                cg_groups_tmp = []
                for id, cg in enumerate(hard_assign):
                    if i == cg:
                        cg_groups_tmp.append(predicted_cg_types[id])
                cg_groups.append(cg_groups_tmp)

            new_cg_groups = [adjust_list(sublist) for sublist in cg_groups]


            new_predicted_cg_types = [new_cg_groups[i][0] for i in hard_assign]
            for atom_idx, cg_idx in enumerate(hard_assign):
                result_json['nodes'][atom_idx]['cg_type'] = new_predicted_cg_types[atom_idx]
            result_json['cgnode_types'] = new_predicted_cg_types

        fpath = os.path.join(output_dir, name + '_cg_{}.json'.format(actual_num_cg))

        if os.path.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'w') as f:
            json.dump(result_json, f, indent=4)

        return result_json


def DSGPM_TPtoCG(smiles, out_dir, num_cg_bead, name=None, cluster_random_seed=None, style=1, show_labels=False):
    '''Takes in a PDB file or a SMILES string and returns one bead mapping
       in JSON format and .png based on the DGSPM model prediction.

       file_dir : path to the output
       smile : SMILES string
       num_cg_bead: the number of cg bead you want to predict
    '''

    m = Chem.MolFromSmiles(smiles)
    edges = []
    for j in range(m.GetNumBonds()):
        begin = m.GetBonds()[j].GetBeginAtomIdx()
        end = m.GetBonds()[j].GetEndAtomIdx()
        bond = m.GetBondWithIdx(j).GetBondTypeAsDouble()
        value = {"source": begin, "target": end, "bondtype": bond}
        edges.append(value)

    # Create one bead mappings
    nodes = []
    cgnodes = []
    cgnode_types = []

    for l in range(m.GetNumAtoms()):
        element = m.GetAtomWithIdx(l).GetSymbol()
        val = {"id": l, "element": element, "charge": 0, "cg_id": 0, "cg_type": "C3E"}
        nodes.append(val)

    AllChem.ComputeGasteigerCharges(m)
    t_charge = 0
    for l in range(m.GetNumAtoms()):
        nodes[l]["charge"] = round(float(m.GetAtomWithIdx(l).GetProp('_GasteigerCharge')), 3)
        t_charge += nodes[l]["charge"]

    if name is None:
        try:
            # name = smiles_to_iupac(smiles)
            name = smiles_to_formula(smiles)
        except:
            print("\nThe IUPAC name of commpound is not found! ")
            name = smiles

    # Create a nested dictionary to be given in json format
    cg_dict = {"compound name": name, "smiles": smiles, "cgnodes": cgnodes,
               "cgnode_types": cgnode_types, "nodes": nodes, "edges": edges, "note": "generated by Drep Zhong script"}

    # Writing to json file

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    predict_out_dir = os.path.join(out_dir, f"{name}")
    if not os.path.exists(predict_out_dir):
        os.mkdir(predict_out_dir)
    else:
        for file in os.listdir(predict_out_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(predict_out_dir, file))

    oname = re.sub('[^A-Za-z0-9]+', '', name) + '_aa.json'
    ofile = os.path.join(predict_out_dir, oname)
    with open(ofile, 'w') as f:
        f.write(json.dumps(cg_dict, sort_keys=False, indent=4))

    test_set = HAMPerFile(data_root=predict_out_dir, cycle_feat=True, degree_feat=True,
                          charge_feat=False, aromatic_feat=False,
                          automorphism=False)

    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=4,
                                     pin_memory=True)

    model = DSGPM_TP(input_dim=len(ATOMS), hidden_dim=128,
                  embedding_dim=128,
                  use_cycle_feat=True,
                  use_degree_feat=True,
                  use_charge_feat=False,
                  use_aromatic_feat=False,
                  ).cuda()
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'model/best_epoch.pth'))
    model.load_state_dict(ckpt)

    if not os.path.exists(predict_out_dir):
        os.mkdir(predict_out_dir)
    with torch.no_grad():
        predict_json = eval(test_dataloader, model, output_dir=predict_out_dir, name=name, num_cg_beads=num_cg_bead,
                            cluster_random_seed=cluster_random_seed)

    test_set = HAMPerFile(data_root=predict_out_dir, cycle_feat=True, degree_feat=True,
                          charge_feat=False, aromatic_feat=False,
                          automorphism=False)
    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)
    gen_vis(test_dataloader, output_file=predict_out_dir, style=style, show_labels=show_labels)

    print('DSGPM_TP prediction complete.')
    return predict_json


def main():
    parser = argparse.ArgumentParser(description="========= DSGPM-TP model for CG Mapping =========")
    parser.add_argument('--name', type=str, default=None, help='Name of the molecule')
    parser.add_argument('--mol_form', type=str, default='sml', help='Molecular format (default: sml, smiles)')
    parser.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    parser.add_argument('--pdb', type=str, default=None, help='Path to the PDB file')
    parser.add_argument('--output', type=str, default='./aa2cg', help='Path to the JSON output directory')
    parser.add_argument('--num', type=int, default=4, help='Number of CG beads (default: 4)')
    # NEW ARGUMENT: Only show labels if this flag is used
    parser.add_argument('--show_labels', action='store_true', help='Show the CG index and type text labels on the output image')
    parser.add_argument('--style', type=int, default=1, help='Style of the visualization, chose from 1 and 2 (default: 1)')

    args = parser.parse_args()

    if args.pdb:
        mol = AllChem.MolFromPDBFile(args.pdb)
        if mol is None:
            print(f"Error: RDKit could not parse the PDB file '{args.pdb}'. Check if it has valid formatting.")
            sys.exit(1)
        smiles = Chem.MolToSmiles(mol)
    elif args.smiles:
        smiles = args.smiles
    else:
        print("Error: You must provide either a --smiles string or a --pdb file.")
        sys.exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Pass the argument directly (False by default, True if --show_labels is used)
    DSGPM_TPtoCG(smiles=smiles, out_dir=args.output, num_cg_bead=args.num, name=args.name, style=args.style, show_labels=args.show_labels)

if __name__ == "__main__":
    main()
