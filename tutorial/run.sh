SAVE_DIR=/mnt/mlflow

mlflow server \
<<<<<<< HEAD
--default-artifact-root s3://mlflow/ \
--backend-store-uri sqlite:///$SAVE_DIR/mlflow_test.db \
=======
--default-artifact-root $SAVE_DIR/artifacts \
--backend-store-uri sqlite:///$SAVE_DIR/mlflow.db \
>>>>>>> 16f4ac22ce38985890d6350fe198f8e8f2a9ea4e
--host 0.0.0.0
