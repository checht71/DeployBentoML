"""This module defines a BentoML service that uses a Pytorch model to classify
roadway images. 0. no deformation, 1. crack, 2. pothole, 3. crack & pothole
"""
#from bentoml.adapters import ImageInput
import numpy as np
import bentoml
from PIL.Image import Image as PILImage
from torchvision.transforms import v2
import torch
import json

SERVICE_NAME = "torch_service_image"
BENTO_MODEL_TAG = "torch_model_86:26bscybtekyarrdv"


bento_model = bentoml.pytorch.get(BENTO_MODEL_TAG)
classifier_runner = bento_model.to_runner()
torch_service = bentoml.Service(SERVICE_NAME, runners=[classifier_runner])
model = bentoml.pytorch.load_model(bento_model)


#@torch_service.api(input=ImageInput(), output=str)

@bentoml.service
def infer(self, image: PILImage) -> str:
    trans_image = np.array(transforms(image))
    input_data = np.expand_dims(trans_image, axis=0)
    input_data = torch.tensor(input_data).to(torch.float32)
    output_tensor = torch.argmax(model(input_data))
    output_dimension_0_size = output_tensor.size(0)
    output_list = output_tensor.tolist()
    json_data = json.dumps(output_list)
    return json_data

print("service created")