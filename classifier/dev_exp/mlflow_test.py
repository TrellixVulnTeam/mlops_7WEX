import mlflow
from models import ResNet
from arguments import Arguments
# 로깅하고자 하는 대상들이 저장될 위치를 지정
mlflow.set_tracking_uri("http://210.123.42.41:5000")
client = mlflow.tracking.MlflowClient("http://210.123.42.41:5000")

exp = mlflow.get_experiment_by_name("Classification")
if exp is None:
    exp_id = mlflow.create_experiment("Classification")
else:
    exp_id = exp.experiment_id

print(exp_id)
# for exp in client.list_experiments():
    # if exp.name == 'Classification':
        # experiment = mlflow.get_experiment(exp.experiment_id)
    # print(exp)
    # print(exp.name)

mlflow_run = mlflow.start_run(run_name = 'test', experiment_id=exp_id)

# # mlflow.log_param('initial_lr', 0.01)
args = Arguments().parser().parse_args()
args_dict = vars(args)
for k, v in args_dict.items():
    mlflow.log_param(k, v)

weights_path = '/home/jongjin/workspace/DeepLearning-Model/classification/model/resnet18.pth'
model = ResNet(backbone='RESNET18', num_classes=2, weights_path=weights_path)

mlflow.log_metric('loss', 0.1) # 'loss' 는 key 이고 loss는 값

 # #아래는 모델을 로깅하는 방식으로 첫번째 인자는 model 또는 torch.jit.script 로 변환된 모델
 # #두번 째 인자는 모델을 저장할 위치
mlflow.pytorch.log_model(model, "model")
