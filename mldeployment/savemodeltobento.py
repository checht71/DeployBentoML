"""This module saves a Keras model to BentoML."""

from pathlib import Path
import bentoml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def load_model_and_save_it_to_bento(model_file: Path) -> None:
    # PyTorch models inherit from torch.nn.Module
    """Loads a keras model from disk and saves it to BentoML."""
    # For loading the entire model 
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    model = torch.load('full_model_final_86', map_location=map_location)
    bento_model = bentoml.pytorch.save_model("torch_model_86", model)
    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(Path("model"))
