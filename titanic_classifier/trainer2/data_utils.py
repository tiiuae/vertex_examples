import warnings
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger
from tensorflow.io import gfile
from tensorflow.python.lib.io import file_io
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def reduce_mem_usage(df, verbose=True):
    """
    If your dataset is large f.ex. it has a lot of continuous variables, 
    it is worth doing this step when possible. It considerably reduces the 
    memory usage of the dataset. The tradeoff being that it takes some time 
    to go through the data and do conversions.
    """
    numerics = ["int8", "int16", "int32",
                "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def read_csv_file(filename):
    """
    Read a CSV straight to pandas dataframe.
    Tensorflow File_io method is leveraged here. With this you can use
    the GCP Storage bucket as filesystem and you don't need to download the 
    training dataset to the machine.
    """
    with file_io.FileIO(filename, "r") as f:
        return pd.read_csv(f)


def load_dataset(filename_pattern):
    """
    Takes a filename pattern f.ex. Train*.csv and loads each CSV to a 
    individual dataframe before concatenation
    """

    filenames = gfile.glob(filename_pattern)
    try:
        dataframes = [read_csv_file(filename) for filename in filenames]
        dataset = pd.concat(dataframes)
        dataset = reduce_mem_usage(dataset)
    except ValueError as e:
        logger.info(" Loading single csv file as dataset.")
        dataset = read_csv_file(filename_pattern)
        dataset = reduce_mem_usage(dataset)
    logger.info("File null counts")
    logger.info(dataset.isnull().sum())
    return dataset


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step)
                              | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def fill_age(dataset):
    """
    A seperate function for handling missing values in "Age" column.
    For code readability.
    """
    # Fill Age with the median age of similar rows according to
    # Pclass, Parch and SibSp
    # Index of NaN age rows
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

    for i in index_NaN_age:
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (
            dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred):
            dataset['Age'].iloc[i] = age_pred
        else:
            dataset['Age'].iloc[i] = age_med
    return dataset


def fill_missing(dataset):
    """
    Takes a dataframe as input and fills missing values appropriately.
    Returns dataframe with no empty values.
    """
    # Drop rows where target label is not present, should be 0
    dataset.dropna(subset=['Survived'], inplace=True)
    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(np.nan)
    # Fill Fare missing values with the median value
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    # Apply log to Fare to reduce skewness distribution
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    # Fill Embarked nan values of dataset set with 'S' most frequent value
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    # Convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
    # Fill Age
    dataset = fill_age(dataset)
    return dataset


def handle_ticket_col(dataset):
    """
    Treat Ticket by extracting the ticket prefix. 
    When there is no prefix it returns X.
    """
    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace(
                "/", "").strip().split(' ')[0])  # Take prefix
        else:
            Ticket.append("X")
    dataset["Ticket"] = Ticket
    return dataset


def feature_engineering(dataset):
    """
    Takes a dataframe as input and does feature engineering.
    """
    # Get Title from Name
    dataset_title = [i.split(",")[1].split(".")[0].strip()
                     for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    # Convert to categorical values Title
    dataset["Title"] = dataset["Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"] = dataset["Title"].map(
        {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
    dataset["Title"] = dataset["Title"].astype(int)
    # Drop Name variable
    dataset.drop(labels=["Name"], axis=1, inplace=True)
    # Create a family size descriptor from SibSp and Parch
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
    # Create new feature of family size
    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
    # Convert to indicator values Title and Embarked
    dataset = pd.get_dummies(dataset, columns=["Title"])
    dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
    # Replace the Cabin number by the type of cabin 'X' if not available
    dataset["Cabin"] = pd.Series(
        [i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
    # One-hot-encode
    dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
    # Extract from Ticket col
    dataset = handle_ticket_col(dataset)
    # One-hot-encode
    dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
    # Create categorical values for Pclass
    dataset["Pclass"] = dataset["Pclass"].astype("category")
    dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
    # Drop useless variables
    dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
    return dataset
