import os
import shutil
import sys
import traceback
from collections import Counter
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
from tensorflow.io import gfile
from xgboost import XGBClassifier

# This is not mandatory needed to do this so that
# import worked correctly locally
basepath = os.path.dirname(__file__)
trainer_path = os.path.abspath(os.path.join(basepath, ".."))
sys.path.append(trainer_path)

from trainer2 import cv_fit
from trainer2 import data_utils
from trainer2.callbacks import TensorBoardCallback


class Model:
    def __init__(self):
        self.training_feature_cols = None
        self.config = None
        self.booster = None
        self.training_features = None

    def train(self, config):
        self.config = config
        logger.info("Load train dataset.")
        # If you have multiple files you can refer to them
        # with asterisk f.ex. /train*.csv
        train = data_utils.load_dataset(config.data_input_path)
        train_len = len(train)
        test = data_utils.load_dataset(config.data_input_path)
        logger.info("Detect outliers from dataset.")
        Outliers_to_drop = data_utils.detect_outliers(
            train, 2, ["Age", "SibSp", "Parch", "Fare"])
        logger.info("Drop outliers")
        train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
        # Kaggle doesn't provide ground truth for test dataset
        # We'll face problems if we have a column mismatch
        test['Survived'] = 2
        # Concatenate the dataset so we have a matching amount of categorical features after feature engineering
        logger.info("Concatenate train and test datasets.")
        dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
        logger.info("Handle missing values.")
        dataset = data_utils.fill_missing(dataset)
        logger.info("Feature Engineering.")
        dataset = data_utils.feature_engineering(dataset)
        logger.info("Seperate train and test datasets again.")
        # original len_train=891 but 10 rows get lost in transformation
        train = dataset[:881]
        logger.info("Separate train features and target")
        # Drop rows where label = 2 if something went awry during feature engineering
        train.drop(train[train['Survived'] == 2].index, inplace=True)
        Y_train = pd.to_numeric(
            train["Survived"].astype(int), downcast="integer")
        logger.info(np.unique(Y_train))
        train.drop("Survived", axis=1, inplace=True)
        test = dataset[881:]
        # Drop dummy target col
        test.drop("Survived", axis=1, inplace=True)
        # For experiment monitoring purposes it's nice to output what features
        # were in use during training
        feature_cols = [f for f in train.columns]
        # Store training feature columns used for scoring
        self.training_features = feature_cols
        logger.info(
            f"Dataset columns(features) used for training: {feature_cols}")
        dtrain = xgb.DMatrix(train, label=Y_train)
        logger.info(
            "Start Cross-validation training for XGBoost Binary Classifier.")
        grid = pd.DataFrame({"learning_rate": [0.001, 0.01, 0.05],
                             "n_estimators": [2000, 2500, 3000],
                             "reg_alpha": [0.00001, 0.00006, 0.001]})
        grid[['train-auc-mean', 'train-auc-std',
              'test-auc-mean', 'test-auc-std']] = grid.apply(lambda x: cv_fit.fit_model(x=grid,
                                                             dmatrix=dtrain),
                                                             axis=1,
                                                             result_type='expand')
        logger.info(f"Best num_boost_round:{len(grid['test-auc-mean'])}")
        best_score_row = grid['test-auc-mean'].idxmax()
        best_params = grid.iloc[best_score_row]
        logger.info(f"Best CV score: {best_params['test-auc-mean']}")
        best_params.drop(['train-auc-mean', 'train-auc-std',
                          'test-auc-mean', 'test-auc-std'], inplace=True)
        best_params = best_params.to_dict()
        params = {"objective": "binary:logistic",
                  "eval_metric": "auc",
                  "learning_rate": 0.01,
                  "n_estimators": 100,
                  "max_depth": 4,
                  "min_child_weight": 0,
                  "gamma": 0,
                  "subsample": 0.7,
                  "colsample_bytree": 0.7,
                  "scale_pos_weight": 1,
                  "seed": 27,
                  "reg_alpha": 0.0,
                  "early_stopping_rounds": 15
                  }
        params.update(**best_params)
        params['n_estimators'] = int(params['n_estimators'])
        logger.info(
            f"Training XGBoost classifier with parameters {params}")
        # You'll want to use the sklearn API to train the model
        # This way you can access the feature importances more easily
        bst = XGBClassifier(**params, use_label_encoder=False)
        tb_callback = TensorBoardCallback()
        # 10 rows will be missing so we need to filter
        Y_train = Y_train.iloc[train.index.tolist()]
        logger.info(Y_train.shape)
        logger.info(train.shape)
        bst.fit(train, Y_train, callbacks=[tb_callback])
        # Load data into DMatrix object
        dtrain = xgb.DMatrix(train, label=Y_train)
        bst2 = xgb.train(params=params, dtrain=dtrain)
        logger.info("Predict on the Test dataset.")
        y_pred = bst.predict(train)
        y_pred = [round(value) for value in y_pred]
        self.booster = bst
        logger.info("Evaluate model against Train dataset.")
        logger.info("Kaggle doesn't provide ground truth for test dataset.")
        tn, fp, fn, tp = confusion_matrix(Y_train, y_pred).ravel()
        class_dist = Counter(Y_train)
        logger.info(f"Model Precision {precision_score(Y_train, y_pred)}")
        logger.info(f"Model Recall: {recall_score(Y_train, y_pred)}")
        logger.info(f"Model Accuracy: {accuracy_score(Y_train, y_pred)}")
        logger.info(f"Model F1 score: {f1_score(Y_train, y_pred)}")
        logger.info(f"Model Log Loss: {log_loss(Y_train, y_pred)}")
        logger.info(f"Model ROC Curve: {roc_curve(Y_train, y_pred)}")
        logger.info(
            f"Model tn, fp, fn, tp: {confusion_matrix(Y_train, y_pred).ravel()}"
        )
        logger.info(
            f"Model Area Under the Curve: {roc_auc_score(Y_train, y_pred)}")
        logger.info(
            f"Test dataset Class Distribution: {class_dist}")
        logger.info(
            "XGBoost BinaryClassifier model training completed succesfully!")
        model_file_name = "model.joblib"
        stats_df = {
            "Train_Dataset_size": len(train),
            "Test_Dataset_size": len(test),
            "Train_Class_0_Distribution": class_dist[0],
            "Test_Class_1_Distribution": class_dist[1],
            "Precision": precision_score(Y_train, y_pred),
            "Recall": recall_score(Y_train, y_pred),
            "Accuracy": accuracy_score(Y_train, y_pred),
            "F1_Score": f1_score(Y_train, y_pred),
            "Binary_log_loss": log_loss(Y_train, y_pred),
            "ROC_AUC_Score": roc_auc_score(Y_train, y_pred),
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp
        }
        logger.info("Extract model feature importances.")
        imps = self.booster.feature_importances_
        feats = self.booster.get_booster().feature_names
        feature_imp = sorted(zip(feats, imps))
        logger.info("Store feature importances to dataframe.")
        feature_imp = pd.DataFrame(
            feature_imp, columns=['Feature', 'Value'])
        feature_imp['Date'] = datetime.today().strftime("%Y-%m-%d")
        logger.info("Store model evaluation statistics to dataframe.")
        stats_df = pd.DataFrame(
            stats_df, index=[datetime.today().strftime("%Y-%m-%d")])
        stats_df.reset_index(inplace=True)
        logger.info(
            "Store training job outputs to local disk before uploading.")
        # Export the classifier to a file
        model_filename = 'model.bst'
        bst2.save_model(model_filename)
        joblib.dump(bst.get_booster(), model_file_name)
        stats_df.to_csv("model_training_stats.csv")
        feature_imp.to_csv("model_feature_importances.csv")
        test_data_filename = f"Test_dataset_{date.today()}.npz"
        np.savez_compressed(test_data_filename,
                            X_test=train[self.training_features],
                            Y_test=Y_train,
                            names=self.training_features)
        training_job_output = [#test_data_filename,
                               #model_file_name,
                               model_filename,
                               #"model_training_stats.csv",
                               #"model_feature_importances.csv"
                                ]
        logger.info(f"Upload model to {config.trainer_output_path}")
        try:
            if config.trainer_output_path.startswith("gs://"):
                logger.info("Checking if output Storage bucket exists.")
                bucket_status = gfile.exists(config.trainer_output_path)
                if not bucket_status:
                    logger.warning("Creating new output bucket.")
                    gfile.mkdir(config.trainer_output_path)
                logger.info(
                    "Start uploading training job results to output Storage bucket.")
                # Upload training job outputs
                for file_name in training_job_output:
                    if not gfile.exists(config.trainer_output_path + file_name):
                        gfile.copy(
                            "./" + file_name, config.trainer_output_path + file_name
                        )
                    else:
                        continue
                # Upload training job tensorboard logs
                for file_name in gfile.glob("./tb_logs/*"):
                    if not gfile.exists(config.trainer_output_path + file_name):
                        gfile.copy(
                            file_name, config.trainer_output_path + "tensorboard/titanic_classifier", overwrite=True
                        )
                    else:
                        continue
            # Else the output points to a local directory
            # So we just move the files
            else:
                for file_name in training_job_output:
                    shutil.move(file_name, config.trainer_output_path + "/" +
                                file_name)
        except Exception as e:
            # Write out an error file. This will be returned as the failureReason in the
            # DescribeTrainingJob result.
            trc = traceback.format_exc()
            # Printing this causes the exception to be in the training job logs, as well.
            logger.info("Exception during task: " +
                        str(e) + "\n" + trc, file=sys.stderr)
            # A non-zero exit code causes the training job to be marked as Failed.
            sys.exit(255)
