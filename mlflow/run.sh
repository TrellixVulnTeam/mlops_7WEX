SAVE_DIR=/mnt/mlflow

mlflow server \
--default-artifact-root $SAVE_DIR/artifacts \
--backend-store-uri sqlite:///$SAVE_DIR/mlflow.db \
--host 0.0.0.0
