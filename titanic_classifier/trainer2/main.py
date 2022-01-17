import argparse
import logging
import os
import sys
import traceback
from datetime import date

from loguru import logger
from six.moves import cPickle as pickle
from tensorflow.io import gfile

basepath = os.path.dirname(__file__)
trainer_path = os.path.abspath(os.path.join(basepath, ".."))
sys.path.append(trainer_path)

from trainer2.model import Model

# This code does just the training of the model
# There is a seperate Vertex AI batch prediction job that handles
# Predicing on the test dataset
# Read environmental variables

training_data_uri = os.environ["AIP_TRAINING_DATA_URI"]
validation_data_uri = os.environ["AIP_VALIDATION_DATA_URI"]
test_data_uri = os.environ["AIP_TEST_DATA_URI"]
output_uri = os.environ["AIP_MODEL_DIR"]

def train_model(config):
    # Our model here is actually a class object
    # The Object holds all of the necessary functionalities to train a
    # XGBoost model eg. loading the training data is handled there
    try:
        model = Model()
        model.train(config)
        logger.info("Training job completed succesfully!")
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        # Printing this causes the exception to be in the training job logs, as well.
        logger.info("Exception during task: " +
                    str(e) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == "__main__":
    # Path for the local dummy data.
    # Normally input data is read from a cloud storage bucket
    # and output/results are also written to a bucket
    dummy_data_output_path = os.path.abspath(
        os.path.join(basepath, "..", "outputs")
    )
    dummy_data_input_path = os.path.abspath(
        os.path.join(basepath, "..", "inputs")
    )
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input_path",
        type=str,
        default=training_data_uri,
        help="Data input (bucket) location.",
    )
    parser.add_argument(
        "--trainer_output_path",
        type=str,
        default=output_uri,
        help="Trainer code Output/result (bucket) location.",
    )
    options, args = parser.parse_known_args()
    # Set logging level to ERROR by default, change to DEBUG for more robust logging
    logging.basicConfig(format="%(levelname)s:%(message)s",
                        level=logging.ERROR)
    train_model(options)
