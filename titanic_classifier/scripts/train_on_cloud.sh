#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This script performs cloud training for a PyTorch model.

echo "Submitting Custom Job to Vertex AI to train XGBoost model"

# BUCKET_NAME: Change to your bucket name
BUCKET_NAME="vertex_ai_demos1" # <-- CHANGE TO YOUR BUCKET NAME
PACKAGE_PATH="${BUCKET_NAME}/vertexai_pipelines/titanic_classifier/trainer/trainer.tar.gz"
# The Tensorflow image provided by Vertex AI Training.
IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tensorflow.2.7:latest"

# JOB_NAME: the name of your job running on Vertex AI.
JOB_PREFIX="xgboost-titanic-classifier-pkg-ar-"
JOB_NAME=${JOB_PREFIX}-$(date +%Y%m%d%H%M%S)-custom-job

# REGION: select a region from https://cloud.google.com/vertex-ai/docs/general/locations#available_regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION="europe-west4"

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR="gs://${BUCKET_NAME}/${JOB_PREFIX}/model/${JOB_NAME}"

# CONFIG_YAML: Vertex AI training instance config
CONFIG_YAML="/home/jupyter/titanic_classifier/scripts/vertex_conf.yml"

# validate bucket name
if [ "${BUCKET_NAME}" != "vertex_ai_demos1" ]
then
  echo "[ERROR] INVALID VALUE: Please update the variable BUCKET_NAME with valid Cloud Storage bucket name. Exiting the script..."
  exit 1
fi

# Submit Custom Job to Vertex AI
gcloud ai custom-jobs create \
    --display-name=${JOB_NAME} \
    --region=${REGION} \
    --python-package-uris=${PACKAGE_PATH} \
    --config=${CONFIG_YAML}

echo "After the job is completed successfully, model files will be saved at $JOB_DIR/"

# uncomment following lines to monitor the job progress by streaming logs

# Stream the logs from the job
gcloud ai custom-jobs stream-logs $(gcloud ai custom-jobs list --region=$REGION --filter="displayName:"$JOB_NAME --format="get(name)")

# # Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/