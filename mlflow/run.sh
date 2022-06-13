SAVE_DIR=/mnt/mlflow

mlflow server \
--backend-store-uri sqlite:///$SAVE_DIR/mlflow_test.db \
--default-artifact-root $SAVE_DIR/artifacts \
--host 0.0.0.0
