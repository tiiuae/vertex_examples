import xgboost as xgb
from tensorboardX import SummaryWriter
from tensorflow.python.lib.io import file_io

class SaveBestModel(xgb.callback.TrainingCallback):
    """
    This callback can be used to output each model trained during 
    XGB CV.
    """
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def TensorBoardCallback():
    """
    Writes training statistics straight to GCP Storage Bucket.
    This way we can use the Vertex AI Experiments tab to monitor our experiments
    and training jobs.
    """
    logdir = file_io.FileIO(
        "gs://vertexai_demos1/experiments/titanic/xgboost/tensorboard", "w")
    writer = SummaryWriter("tb_logs")

    def callback(env):
        for k, v in env.evaluation_result_list:
            writer.add_scalar(k, v, env.iteration)
    return callback
