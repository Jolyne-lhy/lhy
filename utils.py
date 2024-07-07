import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def get_default_device():
    """
    Pick GPU if available, else CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """
    Move tensor(s) to chosen device
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def plot_history_cnn(history, zone_name):
    """
    Plot training history
    """
    trn_losses = [x['trn_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(trn_losses, '-x', label="trn_loss")
    plt.plot(val_losses, '-x', label="val_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    # plt.savefig(f'./output_cnn/{zone_name}.png')
    plt.show()
    
def check_and_create_directory(directory_path):
    """
    Check if directory exists
    """
    if not os.path.exists(directory_path):
        # If not exists, create new directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' not exists，already created.")
    else:
        print(f"Directory '{directory_path}' already created.")