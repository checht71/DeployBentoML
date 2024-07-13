from typing import Tuple
import json
from PIL import Image
from torchvision.transforms import v2
import numpy as np
import requests
from prepdata import prepare_data
import torch

BATCH_SIZE = 2

SERVICE_URL = "http://localhost:3000/classify"


transforms = v2.Compose([
    v2.Resize(size=(224, 224)),
    #v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #v2.RandomHorizontalFlip(p=0.5),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])


def load_single_image():
    image = Image.open("YOUR IMAGE HERE")
    trans_image= np.array(transforms(image))
    inputs = np.expand_dims(trans_image, axis=0)

    #inputs = trans_image.transpose(0, 3, 1, 2)
    return inputs


def make_request_to_bento_service(
    service_url: str, input_array
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text


def main():
    #input_data, expected_output = sample_random_mnist_data_point()
    input_data = load_single_image()
    #print(np.shape(input_data))
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)
    print(f"Prediction: {prediction}")
    #print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()
