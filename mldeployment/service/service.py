"""This module defines a BentoML service that uses a Pytorch model to classify
roadway images. 0. no deformation, 1. crack, 2. pothole, 3. crack & pothole
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from torchvision.transforms import v2
import torch

BENTO_MODEL_TAG = "your tag here"

bento_model = bentoml.pytorch.get(BENTO_MODEL_TAG)
classifier_runner = bento_model.to_runner()
torch_service = bentoml.Service("torch_service", runners=[classifier_runner])
model = bentoml.pytorch.load_model(bento_model)

@torch_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    input_data = torch.tensor(input_data).to(torch.float32)
    return model(input_data)
