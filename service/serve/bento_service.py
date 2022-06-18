import torch
import torch.nn.functional as F
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
SURFACE_CLASSES = ['negatieve', 'positive']
print("dsadsa")
@env(infer_pip_packages=True, pip_packages=['torch','pillow','numpy'])
@artifacts([PytorchModelArtifact('pytorch_model')])
class SurfaceClassification(BentoService):
    @api(input=ImageInput(), batch=True)
    def predict(self, imgs):  # imgs = [b, w, h, c], list
        print(imgs)
        imgs = torch.tensor(imgs).permute(0, 3, 1, 2)
        imgs = F.interpolate(imgs, size=224) / 255.0
        outputs = self.artifacts.pytorch_model(imgs)
        return torch.argmax(outputs, dim=1).tolist()
