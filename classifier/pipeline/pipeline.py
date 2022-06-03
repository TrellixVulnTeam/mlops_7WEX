import kfp
from kfp import dsl
from kfp import onprem
import requests

def preprocess_op(pvc_name, volume_name, volume_mount_path):
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='kaejong/mlops-mask-detection:v0.1',
        arguments=['--input-path', f'{volume_mount_path}/mask',
                   '--output-path', f'{volume_mount_path}/data',
                   '--img-width', 640,
                   '--img-height', 480],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

# def train_op(pvc_name, volume_name, volume_mount_path, device):
    # return dsl.ContainerOp(
        # name='Train Model',
        # image='kaejong/bentoml-surface-train:v0.1',
        # arguments=['--data-path', volume_mount_path,
                   # '--img-size', 224,
                   # '--model-name', 'surface-ConvNeXt-T',
                   # '--device', device]
    # ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)

# def test_op(pvc_name, volume_name, volume_mount_path, model_path, device):
    # return dsl.ContainerOp(
        # name='Test Model',
        # image='kaejong/bentoml-surface-test:v0.1',
        # arguments=['--data-path', volume_mount_path,
                   # '--img-size', 224,
                   # '--model-path', model_path,
                   # '--device', device]
    # ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)

@dsl.pipeline(
    name='Mask Detection using YoloV5',
    description=''
)
def surface_pipeline(mode_hyp_train_test: str,
                     preprocess_yes_no: str,
                     model_path: str,
                     device: str):
    pvc_name = "workspace-yolov5"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jovyan'
    with dsl.Condition(preprocess_yes_no == 'yes'):
        _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path)
    # with dsl.Condition(mode_hyp_train_test == 'train'):
        # _train_op = train_op(pvc_name, volume_name, volume_mount_path, device).after(_preprocess_op)
    # with dsl.Condition(mode_hyp_train_test == 'test'):
        # _test_op = test_op(pvc_name, volume_name, volume_mount_path, model_path, device).after(_train_op)
if __name__ == '__main__':
    USERNAME='user@example.com'
    PASSWORD='12341234'
    NAMESPACE='kubeflow-user-example-com'
    HOST="http://210.123.42.43:8080"
    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    print(session.cookies.get_dict())
    session_cookie = session.cookies.get_dict()["authservice_session"]

    print(session_cookie)

    client = kfp.Client(
        host=f"{HOST}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=NAMESPACE
    )

    pipeline_name = "mask_detection"
    pipeline_package_path = "yolov5.zip"
    version = "v0.01"

    experiment_name = "Yolov5"
    run_name = "kubeflow study {}".format(version)

    kfp.compiler.Compiler().compile(surface_pipeline, pipeline_package_path)

    pipeline_id = client.get_pipeline_id(pipeline_name)
    print(pipeline_id)

    if pipeline_id:
        client.upload_pipeline_version(
            pipeline_package_path=pipeline_package_path,
            pipeline_version_name=version,
            pipeline_name=pipeline_name
        )
    else:
        client.upload_pipeline(
            pipeline_package_path=pipeline_package_path,
            pipeline_name=pipeline_name
        )

    experiment = client.create_experiment(name=experiment_name, namespace=NAMESPACE)
    # run = client.run_pipeline(experiment.id, run_name, pipeline_package_path, params={})

