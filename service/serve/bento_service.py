import pandas as pd
import torch
import torch.nn.functional as F
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import ImageInput
from bentoml.adapters import DataframeInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from typing import List
import imageio
import numpy as np
from bentoml import BentoService, api, artifacts
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.adapters import ImageInput
from typing import List
import numpy as np

SURFACE_CLASSES = ['negatieve', 'positive']
@env(infer_pip_packages=True, pip_packages=['torch','pillow','numpy', 'imageio'])
@artifacts([PytorchModelArtifact('pytorch_model')])
class SurfaceClassification(BentoService):
    @api(input=ImageInput(), batch=True)
    def predict(
        self, image_arrays: List[imageio.core.util.Array]
    ) -> List[str]:
        return [1]

