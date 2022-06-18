from bento_service import SurfaceClassification
from mlflow_model import load_model
# Create a classification service instance
surface_classifier_service = SurfaceClassification()
model = load_model(model_name='surface', version=1)
# Pack the newly trained model artifact
surface_classifier_service.pack('pytorch_model', model)
# Save the prediction service to disk for model serving
saved_path = surface_classifier_service.save()
