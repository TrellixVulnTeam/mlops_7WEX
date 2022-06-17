import mlflow
from models.model import model_select
from arguments import Arguments
from torchinfo import summary
# 로깅하고자 하는 대상들이 저장될 위치를 지정
mlflow.set_tracking_uri("http://210.123.42.43:5000")
# client = mlflow.tracking.MlflowClient("http://210.123.42.43:5000")

exp = mlflow.get_experiment_by_name("Test_Blog")
if exp is None:
    exp_id = mlflow.create_experiment("Test_Blog")
else:
    exp_id = exp.experiment_id

print(exp_id)

mlflow_run = mlflow.start_run(run_name = 'test', experiment_id=exp_id)

mlflow.log_param('Learning Rate', 0.01)
mlflow.log_param('Batch Size', 64)
mlflow.log_param('Optimizer', 'SGD')
mlflow.log_param('Train Ratio', 0.8)
args = Arguments().parser().parse_args()
args_dict = vars(args)
for k, v in args_dict.items():
    mlflow.log_param(k, v)

weights_path = '/home/jongjin/workspace/DeepLearning-Model/classification/models/resnet18.pth'
model = model_select(backbone='resnet18', num_classes=3, pretrained=True)
# model_summary = str(summary(model, input_size=(1, 3, 224, 224)))
# with open("model_architecture.txt", 'w') as f:
    # f.write(model_summary)
# mlflow.log_artifact("model_architecture.txt")



mlflow.log_metric('loss', 0.0891)
mlflow.log_metric('Accruacy', 99.9)

 #아래는 모델을 로깅하는 방식으로 첫번째 인자는 model 또는 torch.jit.script 로 변환된 모델
 #두번 째 인자는 모델을 저장할 위치
mlflow.pytorch.log_model(model, "model")
