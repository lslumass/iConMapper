import os
import glob
import json
import torch
import numpy as np
import networkx as nx
import random
import torch.nn.functional as F

from collections import OrderedDict
from torch.utils.data import Dataset
from networkx.algorithms.cycles import cycle_basis
from torch_geometric.data import Data
from tqdm import tqdm
from utils.automorphism_group import node_equal, edge_equal
from rdkit import Chem

MASK_ATOM_INDEX = 0

ATOMS = OrderedDict([
    ('B', 10.81), ('C', 12.011), ('N', 14.007), ('O', 15.999),
    ('F', 18.998403163), ('Si', 28.085), ('P', 30.973761998), ('S', 32.06),
    ('Cl', 35.45), ('K', 39.0983), ('Fe', 55.845), ('Se', 78.971),
    ('Br', 79.904), ('Ru', 101.07), ('Sn', 118.71), ('I', 126.90447)
])

BOND_TYPE_DICT = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3,
                  '-': 0, '/': 0, '\\': 0, ':': 1, '=': 2, '#': 3}

BOND_STR_TO_FLOAT = {'-': 1.0, '/': 1.0, '\\': 1.0, ':': 1.5, '=': 2.0, '#': 3.0}


"""
rename CG types to be more descriptive and consistent ones:
A3F -> A2F
A3W -> A1W
A4W -> A1W
DS1 -> RS1
DS2 -> RS2
DA1 -> RA1
DA2 -> RA2
DA3 -> RA3
DA4 -> RA4
DG1 -> RG1
DG2 -> RG2
DG3 -> RG3
DG4 -> RG4
DC1 -> RC1
DC2 -> RC2
DC3 -> RC3
DT1 -> RU1
DT2 -> RU2
DT3 -> RU3
"""
CG_TYPE_DICT = {
    "C2E": 0,  "C3E": 1,  "A2V": 2,  "A1L": 3,  "A1I": 4,  "A5M": 5,  "P5N": 6,  "QaD": 7,  "P4Q": 8,  "QaE": 9, 
    "P1C": 10, "P1S": 11, "P1T": 12, "A2P": 13, "A3K": 14, "QdK": 15, "A3R": 16, "QdR": 17, "A4H": 18, "P1H": 19,
    "P2H": 20, "A1F": 21, "A2F": 22, "A1Y": 23, "A2Y": 24, "P1Y": 25, "A1W": 26, "P1W": 27, "A2W": 28,

    "RS1": 29, "RS2": 30, "RA1": 31, "RG1": 32, "RA2": 33, "RG2": 34, "RA3": 35, "RG4": 36, "RC2": 37, "RA4": 38,
    "RG3": 39, "RU2": 40, "RC1": 41, "RU1": 42, "RC3": 43, "RU3": 44, "PHO": 45,

    "M01": 46, "M02": 47, "M03": 48, "M04": 49, "M05": 50, "M06": 51, "M07": 52, "M08": 53, "MCI": 54, "MSO": 55,
    "MSS": 56, "MCL": 57, "MCF": 58, "MBR": 59
}


def _smiles_to_key(smiles):
    """Convert SMILES to filesystem/dict-safe key."""
    return smiles.replace('/', '|').replace('\\', '~')


class HAM(Dataset):
    def __init__(self, data_root, dataset_type='train', for_vis=False,
                 cycle_feat=False, degree_feat=False, charge_feat=False,
                 aromatic_feat=False, cross_validation=False, automorphism=True,
                 transform=None):
        assert dataset_type in {'train', 'test'}
        self.dataset_type = dataset_type
        self.transform = transform
        self.for_vis = for_vis
        self.cycle_feat = cycle_feat
        self.degree_feat = degree_feat
        self.charge_feat = charge_feat
        self.aromatic_feat = aromatic_feat

        if not cross_validation:
            jsons_root = os.path.join(data_root, dataset_type, '*.json')
        else:
            jsons_root = os.path.join(data_root, '*.json')
        self.json_file_path_lst = glob.glob(jsons_root)

        self.smiles_cluster_idx_dict = {}
        self.smiles_json_fpath_lst = OrderedDict()

        for json_fpath in self.json_file_path_lst:
            with open(json_fpath) as f:
                json_data = json.load(f)
            if 'smiles' not in json_data:
                smiles = os.path.splitext(os.path.basename(json_fpath))[0]
            else:
                smiles = json_data['smiles']
            smiles_key = _smiles_to_key(smiles)

            if smiles_key not in self.smiles_cluster_idx_dict:
                self.smiles_cluster_idx_dict[smiles_key] = []
            cluster_idx = self.compute_cluster_idx(json_data)
            self.smiles_cluster_idx_dict[smiles_key].append(cluster_idx.unsqueeze(0))

            if smiles_key not in self.smiles_json_fpath_lst:
                self.smiles_json_fpath_lst[smiles_key] = []
            self.smiles_json_fpath_lst[smiles_key].append(json_fpath)

        self.smiles_lst = list(self.smiles_json_fpath_lst.keys())

        if automorphism:
            for smile_key in tqdm(self.smiles_lst, desc='computing automorphisms'):
                json_fpath = self.smiles_json_fpath_lst[smile_key][0]
                with open(json_fpath) as f:
                    json_data = json.load(f)
                graph = nx.Graph()
                for node in json_data['nodes']:
                    graph.add_node(node['id'], element=node['element'], cg=node['cg_id'])
                for edge in json_data['edges']:
                    bond_type = edge['bondtype']
                    if isinstance(bond_type, str):
                        bond_type = BOND_STR_TO_FLOAT[bond_type]
                    graph.add_edge(edge['source'], edge['target'], bond_type=bond_type)

                gm = nx.isomorphism.GraphMatcher(graph, graph,
                                                 node_match=node_equal,
                                                 edge_match=edge_equal)
                mapping_lst = []
                for node_mapping in gm.isomorphisms_iter():
                    key_value_lst = torch.tensor(list(node_mapping.items())).transpose(1, 0)
                    for original_mapping in self.smiles_cluster_idx_dict[smile_key]:
                        new_mapping = original_mapping.clone()
                        new_mapping[:, key_value_lst[0]] = new_mapping[:, key_value_lst[1]]
                        mapping_lst.append(new_mapping)
                self.smiles_cluster_idx_dict[smile_key] = mapping_lst

    def __getitem__(self, index):
        smiles_key = self.smiles_lst[index]
        json_fpaths = self.smiles_json_fpath_lst[smiles_key]
        json_fpath = json_fpaths[random.randrange(len(json_fpaths))]

        with open(json_fpath) as f:
            json_data = json.load(f)

        data = Data()

        # Get original SMILES for RDKit (not the sanitized key)
        if 'smiles' not in json_data:
            smiles = os.path.splitext(os.path.basename(json_fpath))[0]
        else:
            smiles = json_data['smiles']

        # Build NetworkX graph
        graph = nx.Graph(smiles=smiles)
        for node in json_data['nodes']:
            graph.add_node(node['id'], element=node['element'], cg=node['cg_id'])
        for edge in json_data['edges']:
            bond_type = edge['bondtype']
            if isinstance(bond_type, str):
                bond_type = BOND_STR_TO_FLOAT[bond_type]
            graph.add_edge(edge['source'], edge['target'], bond_type=bond_type)

        # ========== atom types ==========
        fg_beads: list = json_data['nodes']
        fg_beads.sort(key=lambda x: x['id'])
        atom_types = torch.LongTensor(
            [list(ATOMS.keys()).index(bead['element']) for bead in fg_beads]).reshape(-1, 1)
        data.x = atom_types

        # ========== atom charges ==========
        if 'charge' in fg_beads[0]:
            data.atom_charges = torch.tensor(
                [[bead['charge']] for bead in fg_beads], dtype=torch.float)
        else:
            data.atom_charges = torch.zeros(len(fg_beads), 1)

        # ========== atom aromatic ==========
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() == len(fg_beads):
            atom_aromatics = []
            for atom in mol.GetAtoms():
                atom_aromatics.append(int(atom.GetIsAromatic()))
            data.atom_aromatics = torch.tensor(atom_aromatics, dtype=torch.float).reshape(-1, 1)
        else:
            data.atom_aromatics = torch.zeros(len(fg_beads), 1)

        # ========== CG types ==========
        if 'cg_type' in fg_beads[0]:
            atom_CG_types = torch.LongTensor(
                [list(CG_TYPE_DICT.keys()).index(bead['cg_type']) for bead in fg_beads]).reshape(-1, 1)
            data.atom_CG_types = atom_CG_types
        else:
            data.atom_CG_types = torch.zeros(len(fg_beads), 1).long()

        # ========== extended features ==========
        extended_feats = []
        if self.degree_feat:
            degrees = np.array(graph.degree)[:, 1]
            extended_feats.append(torch.tensor(degrees, dtype=torch.float).unsqueeze(-1) / 4)
        if self.cycle_feat:
            cycle_indicator = torch.zeros(len(fg_beads), 1)
            for cycle in cycle_basis(graph):
                cycle_indicator[torch.tensor(cycle)] = 1
            extended_feats.append(cycle_indicator)
        if self.charge_feat:
            extended_feats.append(data.atom_charges)
        if self.aromatic_feat:
            extended_feats.append(data.atom_aromatics)

        if extended_feats:
            data.extended_feat = torch.cat(extended_feats, dim=1)

        # ========== edges ==========
        edges = []
        bond_types = []
        for x in json_data['edges']:
            edges.append([x['source'], x['target']])
            edges.append([x['target'], x['source']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])
        data.edge_index = torch.tensor(edges, dtype=torch.long).t()
        data.edge_attr = F.one_hot(
            torch.tensor(bond_types, dtype=torch.long), num_classes=4).float()

        # ========== ground truth ==========
        multi_anno = self.smiles_cluster_idx_dict[smiles_key]
        multi_anno = torch.cat(multi_anno, dim=0)  # [num_annotations, num_nodes]

        if self.dataset_type == 'train':
            rand_idx = random.randrange(len(multi_anno))
            data.y = multi_anno[rand_idx]  # [num_nodes] 1D
        else:
            data.y = multi_anno  # [num_annotations, num_nodes] 2D

        if self.for_vis or self.dataset_type == 'test':
            data.graph = graph

        if self.dataset_type == 'train':
            # ========== triplet and positive pair sampling ==========
            row_idx, column_idx = zip(*[(x['source'], x['target']) for x in json_data['edges']])
            i = torch.LongTensor([row_idx, column_idx])
            v = torch.ones(len(row_idx))
            fg_adj = torch.sparse_coo_tensor(
                i, v, torch.Size([len(atom_types), len(atom_types)]))
            fg_adj = (fg_adj + fg_adj.t()).to_dense()
            np_fg_adj = fg_adj.numpy()

            fg_id_cg_id_dict = {int(x['id']): int(x['cg_id']) for x in json_data['nodes']}

            anchors, positives, negatives = [], [], []
            pos_pairs = []

            def find_positive_vertex(fg_id, cur_cg_id):
                neighbors = np.where(np_fg_adj[fg_id] > 0)[0]
                np.random.shuffle(neighbors)
                for n in neighbors:
                    if fg_id_cg_id_dict[n] == cur_cg_id:
                        return n
                return None

            for edge in json_data['edges']:
                u, v = edge['source'], edge['target']
                u_cg, v_cg = fg_id_cg_id_dict[u], fg_id_cg_id_dict[v]

                if u_cg != v_cg:
                    if np.random.random() < 0.5:
                        u, v = v, u
                        u_cg, v_cg = v_cg, u_cg
                    pos = find_positive_vertex(u, u_cg)
                    anchor, neg = u, v
                    if pos is None:
                        pos = find_positive_vertex(v, v_cg)
                        anchor, neg = v, u
                    if pos is not None:
                        anchors.append(anchor)
                        positives.append(pos)
                        negatives.append(neg)
                else:
                    pos_pairs.append([u, v])

            # Guarantee shape [3, K] even when K=0
            if anchors:
                data.triplet_index = torch.tensor([anchors, positives, negatives], dtype=torch.long)
            else:
                data.triplet_index = torch.zeros(3, 0, dtype=torch.long)

            # Guarantee shape [2, M] even when M=0
            if pos_pairs:
                data.pos_pair_index = torch.tensor(pos_pairs, dtype=torch.long).t()
            else:
                data.pos_pair_index = torch.zeros(2, 0, dtype=torch.long)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.smiles_cluster_idx_dict)

    @staticmethod
    def compute_cluster_idx(json_data):
        node_cluster_index = -1 * torch.ones((len(json_data['nodes']),)).long()
        for node in json_data['nodes']:
            node_cluster_index[node['id']] = node['cg_id']
        assert torch.all(node_cluster_index >= 0)
        return node_cluster_index