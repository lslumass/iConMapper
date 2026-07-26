import copy
import socket
from datetime import datetime
import os
import torch


def get_run_name(title):
    """A unique name for each run. No colons (Windows-safe)."""
    return datetime.now().strftime('%b%d-%H-%M-%S') + '_' + socket.gethostname() + '_' + title


def average_model_parameters(parent_folder, train_model, pth_name='best_epoch.pth',
                             save_path='average_best_epoch.pth'):
    """Average model parameters across all fold checkpoints without mutating inputs."""
    model_folders = [f for f in os.listdir(parent_folder)
                     if os.path.isdir(os.path.join(parent_folder, f))]

    if len(model_folders) == 0:
        print(f"Warning: No model subfolders found in {parent_folder}")
        return

    model_parameters = []
    for folder in model_folders:
        model_path = os.path.join(parent_folder, folder, pth_name)
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping.")
            continue
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model_parameters.append(copy.deepcopy(state_dict))

    if len(model_parameters) == 0:
        print("Error: No valid model checkpoints found.")
        return

    # Average while preserving original dtypes
    average_state_dict = {}
    for key in model_parameters[0].keys():
        original_dtype = model_parameters[0][key].dtype
        stacked = sum(p[key].float() for p in model_parameters) / len(model_parameters)
        average_state_dict[key] = stacked.to(original_dtype)

    average_model = copy.deepcopy(train_model)
    average_model.load_state_dict(average_state_dict)
    torch.save(average_model.state_dict(), save_path)
    print(f'Average model parameters saved to {save_path}')