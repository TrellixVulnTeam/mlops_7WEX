from bento_service import SurfaceClassification
from mlflow_model import load_model
surface_classifier_service = SurfaceClassification()

model = load_model(model_name='Animals', version=5)
surface_classifier_service.pack('pytorch_model', model)
saved_path = surface_classifier_service.save()
