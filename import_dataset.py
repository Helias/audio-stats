from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def imp_dataset(dataset_path: str, drop_feature: List[str], conditions: str = None) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  df = pd.read_csv(dataset_path)

  if conditions:
    df = df.query(conditions)

  # fix issue "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')."
  # source: https://github.com/pycaret/pycaret/issues/290#issuecomment-660607984
  df = df.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

  df = shuffle(df)

  # replacing real -> 0 and spoof -> 1
  df["label"] = df["label"].replace("bonafide", 0)
  df["label"] = df["label"].replace("spoof", 1)

  features = df.keys()
  features = features.drop(drop_feature)

  # splitting dataset 66 / 34 %
  df1 = df.iloc[:int(len(df) * 0.66)]    # 66% of data
  df2 = df.iloc[int(len(df) * 0.66) + 1:]    # 34% of data

  # training set
  x_train: np.ndarray = df1.loc[:, features].values
  y_train: np.ndarray = df1.loc[:, ['label']].values

  # test set
  x_test: np.ndarray = df2.loc[:, features].values
  y_test: np.ndarray = df2.loc[:, ['label']].values

  return x_train, y_train, x_test, y_test
