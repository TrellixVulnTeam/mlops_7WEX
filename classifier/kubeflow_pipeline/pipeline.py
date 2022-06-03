import kfp
import kfp.components as comp
from kfp import dsl
from kfp import onprem
import requests
from kubernetes.client.models import V1EnvVar
import os


@dsl.pipeline(
    name="MNIST using Arcface",
    description="CT pipeline"
)
def mnist_pipeline():
    ENV_MANAGE_URL = V1EnvVar(name='MANAGE_URL', value='http://210.123.42.43:8080/send')
    pvc_name = "workspace-mnist"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jovyan'

    data_0 = dsl.ContainerOp(
        name="load & preprocess data pipeline",
        image="kaejong/mnist-pre-data:v0.2",
        arguments=['--train_data_path', os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--test_data_path', os.path.join(volume_mount_path, 'data/mnist/test'),
                   '--train_data_file',  'train_mnist',
                   '--test_data_file',  'test_mnist',
                   '--faiss_train_data_path',  os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--faiss_test_data_path', os.path.join(volume_mount_path, 'data/mnist/test'),
                   '--faiss_train_data_file',  'train_faiss',
                   '--faiss_test_data_file',  'test_faiss',
                   '--image_width',  28,
                   '--image_height', 28,
                   '--image_channel', 1,
                   '--npy_interval', 5000],
    ).set_display_name('collect & preprocess data')\
    .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))
    data_1 = dsl.ContainerOp(
        name="validate data pipeline",
        image="kaejong/mnist-val-data:v0.2",
        arguments=['--train_data_path', os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--test_data_path', os.path.join(volume_mount_path, 'data/mnist/test'),
                   '--train_data_file',  'train_mnist',
                   '--test_data_file',  'test_mnist',
                   '--faiss_train_data_path',  os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--faiss_test_data_path', os.path.join(volume_mount_path, 'data/mnist/test'),
                   '--faiss_train_data_file',  'train_faiss',
                   '--faiss_test_data_file',  'test_faiss',
                   '--image_width',  28,
                   '--image_height', 28,
                   '--image_channel', 1,
                   '--npy_interval', 5000],
    ).set_display_name('validate data').after(data_0)\
    .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

    train_model = dsl.ContainerOp(
        name="train embedding model",
        image="kaejong/mnist-train-model:v0.7",
        arguments=['--train_data_path', os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--test_data_path', os.path.join(volume_mount_path, 'data/mnist/test'),
                   '--image_width',  28,
                   '--image_height', 28,
                   '--image_channel', 1,
                   '--d_embedding', 128,
                   '--scale_size', 32,
                   '--class_nums', 10,
                   '--optimizer', 'SGD',
                   '--lr', 0.01,
                   '--weight_decay', 5e-4,
                   '--model_dir', os.path.join(volume_mount_path, 'model'),
                   '--model_file', 'model.pt',
                   '--metric_file', 'metric.pt',
                   '--ckpt_dir', os.path.join(volume_mount_path, 'model/ckpt'),
                   '--model_ckpt', 'model.ckpt',
                   '--metric_ckpt', 'metric_ckpt',
                   '--epoch', 1,
                   '--batch_size', 64,
                   '--log_iter', 100,
                   '--save_iter', 2000,
                   '--eval_iter', 1000],
    ).set_display_name('train model').after(data_1)\
    .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))\

    embedding = dsl.ContainerOp(
        name="embedding data using embedding model",
        image="kaejong/mnist-embedding:v0.3",
        arguments=['--npy_interval', 5000,
                   '--model_dir', os.path.join(volume_mount_path, 'model'),
                   '--model_file', 'model.pt',
                   '--d_embedding', 128,
                   '--faiss_train_data_path',  os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--faiss_test_data_path', os.path.join(volume_mount_path, 'data/mnist/test'),
                   '--faiss_train_data_file',  'train_faiss',
                   '--faiss_test_data_file',  'test_faiss',
                   '--batch_size', 10],
    ).set_display_name('embedding').after(train_model)\
    .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))\

    train_faiss = dsl.ContainerOp(
        name="train faiss",
        image="kaejong/mnist-train-faiss:v0.2",
        arguments=['--class_nums', 10,
                   '--model_dir', os.path.join(volume_mount_path, 'model'),
                   '--faiss_model_file', 'faiss_index.bin',
                   '--faiss_label_file', 'faiss_label.json',
                   '--d_embedding', 128,
                   '--faiss_train_data_path',  os.path.join(volume_mount_path, 'data/mnist/train'),
                   '--faiss_test_data_path', os.path.join(volume_mount_path, 'data/mnist/test')],
    ).set_display_name('train faiss').after(embedding)\
    .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))\

    analysis = dsl.ContainerOp(
        name="analysis total",
        image="kaejong/mnist-analysis:v0.2",
        arguments=['--class_nums', 10,
                   '--model_dir', os.path.join(volume_mount_path, 'model'),
                   '--model_file', 'model.pt',
                   '--faiss_model_file', 'faiss_index.bin',
                   '--faiss_label_file', 'faiss_label.json',
                   '--image_width', 28,
                   '--image_height', 28,
                   '--image_channel', 1,
                   '--batch_size', 16,
                   # '--result_path', os.path.join(volume_mount_path, 'result'),
                   '--test_data_path', os.path.join(volume_mount_path, 'data/mnist/test')],
        file_outputs={
            "confusion_matrix": "/confusion_matrix.csv",
            "mlpipeline-ui-metadata": "/mlpipeline-ui-metadata.json",
            "accuracy": "/accuracy.json",
            "mlpipeline_metrics": "/mlpipeline-metrics.json"
        }
    ).set_display_name('analysis').after(train_faiss)\\
    .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))\
    # ).add_env_variable(ENV_MANAGE_URL).set_display_name('analysis').after(train_faiss)\

    # baseline = 0.8
    # with dsl.Condition(analysis.outputs["accuracy"] > baseline) as check_deploy:
        # deploy = dsl.ContainerOp(
            # name="deploy mar",
            # image="kaejong/mnist-deploy:v0.6",
            # arguments=['--model_name', "embedding",
                       # '--model_dir', os.path.join(volume_mount_path, 'model'),
                       # '--model_file', 'model.pt',
                       # '--faiss_model_file', 'faiss_index.bin',
                       # '--faiss_label_file', 'faiss_label.json',
                       # '--requirements',  "requirements.txt",
                       # '--handler_class', "MyHandler.py",
                       # '--handler', "handler.py",
                       # '--export_path', os.path.join(volume_mount_path, 'deploy-model/model-store'),
                       # '--config_path', os.path.join(volume_mount_path, 'deploy-model/config'),
                       # '--max_num_models', 3,
                       # '--pred_port', 8082,
                      # '--manage_port', 8083,
                       # '--metric_port', 8084,
                       # '--deploy_name', 'torchserve',
                       # '--svc_name', 'torchserve',
                       # '--namespace', 'default'],
        # ).set_display_name('deploy').after(analysis)\
        # .apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))\
        # ).add_env_variable(ENV_MANAGE_URL).set_display_name('deploy').after(analysis)\


if __name__=="__main__":
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

    pipeline_name = "Mnist"
    pipeline_package_path = "pipeline.zip"
    version = "v0.02"

    experiment_name = "For Develop"
    run_name = "kubeflow study {}".format(version)

    kfp.compiler.Compiler().compile(mnist_pipeline, pipeline_package_path)

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
    run = client.run_pipeline(experiment.id, run_name, pipeline_package_path)


