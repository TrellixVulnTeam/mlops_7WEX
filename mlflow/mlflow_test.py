import mlflow
mlflow.set_tracking_uri("http://210.123.42.41:5000")

client = mlflow.tracking.MlflowClient("http://210.123.42.41:5000")
experiment = client.get_experiment_by_name("classification")

exp = mlflow.get_experiment_by_name("Classificationa")
if exp is None:
    exp_id = mlflow.create_experiment("Classificationa")
else:
    exp_id = exp.experiment_id

mlflow_run = mlflow.start_run(run_name = 'Animals', experiment_id=exp_id)
mlflow.log_param('Learning Rate', 0.1)
mlflow.log_metric('Accuracy', 100)
