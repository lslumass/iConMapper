"""
LEGACY MODULE - Not used by current PyG DataLoader/DataListLoader pipeline.
Kept for backward compatibility with older evaluation scripts.
"""
import torch


def pad_tensor(input_tensor, pad, dim):
    pad_size = list(input_tensor.shape)
    pad_size[dim] = pad - input_tensor.size(dim)
    if pad_size[dim] <= 0:
        return input_tensor
    return torch.cat([input_tensor, torch.zeros(*pad_size).type(input_tensor.type())], dim=dim)


def pad_tensor_two_dims(input_tensor, pad: tuple):
    assert len(pad) == 2
    holder = torch.zeros(pad)
    holder[:input_tensor.shape[0], :input_tensor.shape[1]] = input_tensor
    return holder


def pad_tensor_lst_three_dims(input_tensor_lst, pad: tuple):
    assert len(pad) == 3
    holder = torch.zeros(pad)
    for idx, tensor in enumerate(input_tensor_lst):
        holder[idx, :tensor.shape[0], :tensor.shape[1]] = tensor
    return holder


def fg_cg_data_pad_collate_train(batch):
    batch_len = len(batch[0])
    assert batch_len == 5
    num_fg_atoms = [data[0].shape[0] for data in batch]
    num_cg_beads = [data[2].shape[1] for data in batch]
    num_triplet_sample = [data[3].shape[0] for data in batch]
    num_pos_pair_sample = [data[4].shape[0] for data in batch]

    max_fg = max(num_fg_atoms)
    max_cg = max(num_cg_beads)
    max_trip = max(num_triplet_sample)
    max_pos = max(num_pos_pair_sample)

    padded_batch = []
    for data in batch:
        atom_types, fg_adj, mapping_op, triplet_idx, pos_pair = data
        padded_batch.append((
            pad_tensor(atom_types, max_fg, dim=0),
            pad_tensor_two_dims(fg_adj, (max_fg, max_fg)),
            pad_tensor_two_dims(mapping_op, (max_fg, max_cg)),
            pad_tensor(triplet_idx, max_trip, dim=0),
            pad_tensor(pos_pair, max_pos, dim=0)
        ))

    ret = tuple(torch.stack([d[i] for d in padded_batch], dim=0) for i in range(batch_len))
    ret += (torch.LongTensor(num_fg_atoms).reshape(-1, 1),
            torch.LongTensor(num_cg_beads).reshape(-1, 1),
            torch.LongTensor(num_triplet_sample).reshape(-1, 1),
            torch.LongTensor(num_pos_pair_sample).reshape(-1, 1))
    return ret


def fg_cg_data_pad_collate_test(batch):
    batch_len = len(batch[0])
    assert batch_len == 3
    num_fg_atoms = [data[0].shape[0] for data in batch]
    num_cg_beads = []
    for data in batch:
        for mapping_op in data[2]:
            num_cg_beads.append(mapping_op.shape[1])
    num_annotations = [len(data[2]) for data in batch]

    max_fg = max(num_fg_atoms)
    max_cg = max(num_cg_beads)
    max_anno = max(num_annotations)

    padded_batch = []
    for data in batch:
        atom_types, fg_adj, mapping_ops = data
        padded_batch.append((
            pad_tensor(atom_types, max_fg, dim=0),
            pad_tensor_two_dims(fg_adj, (max_fg, max_fg)),
            pad_tensor_lst_three_dims(mapping_ops, (max_anno, max_fg, max_cg))
        ))

    ret = tuple(torch.stack([d[i] for d in padded_batch], dim=0) for i in range(batch_len))
    ret += (torch.LongTensor(num_fg_atoms).reshape(-1, 1),
            torch.LongTensor(num_cg_beads).reshape(-1, 1),
            torch.LongTensor(num_annotations).reshape(-1, 1))
    return ret