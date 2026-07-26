import random
import torch
from dataset.ham import MASK_ATOM_INDEX


class MaskAtomType(object):
    def __init__(self, mask_ratio):
        assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0, 1]"
        self.mask_ratio = mask_ratio

    def __call__(self, data):
        num_atom = data.x.size(0)
        num_masked_atoms = min(int(num_atom * self.mask_ratio + 1), num_atom)

        data.masked_atom_index = torch.tensor(
            random.sample(range(num_atom), num_masked_atoms))
        data.masked_atom_type = data.x[data.masked_atom_index].squeeze(1)

        # Clone to avoid corrupting original dataset across epochs
        data.x = data.x.clone() + 1  # shift for padding index
        data.x[data.masked_atom_index] = MASK_ATOM_INDEX

        return data