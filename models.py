import xgboost as xgb
from sklearn import svm
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_trained_models(x_train, y_train):
  return [
      {
          "name": "CART",
          "model": DecisionTreeClassifier().fit(x_train, y_train)
      },

      {
          "name": "SVM",
          "model": svm.NuSVC().fit(x_train, y_train.ravel())
      },
      {
          "name": "LR",
          "model": LogisticRegression().fit(x_train, y_train.ravel())
      },
      {
          "name": "KNN",
          "model": KNeighborsClassifier(11).fit(x_train, y_train.ravel())
      },
      {
          "name": "GMM",
          "model": GaussianMixture(n_components=2, random_state=0).fit(x_train, y_train.ravel())
      },
      {
          "name": "LDA",
          "model": LinearDiscriminantAnalysis().fit(x_train, y_train.ravel())
      },
      {
          "name": "SVC1",
          "model": SVC(gamma=2, C=1).fit(x_train, y_train.ravel())
      },
      {
          "name": "SVC2",
          "model": SVC(kernel="linear", C=0.025).fit(x_train, y_train.ravel())
      },
      {
          "name": "GPC",
          "model": GaussianProcessClassifier(1.0 * RBF(1.0)).fit(x_train, y_train.ravel())
      },
      {
          "name": "RFC",
          "model": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(x_train, y_train.ravel())
      },
      {
          "name": "MLP",
          "model": MLPClassifier(alpha=1, max_iter=1000).fit(x_train, y_train.ravel())
      },
      {
          "name": "ADC",
          "model": AdaBoostClassifier().fit(x_train, y_train.ravel())
      },
      {
          "name": "GNB",
          "model": GaussianNB().fit(x_train, y_train.ravel())
      },
      {
          "name": "QDA",
          "model": QuadraticDiscriminantAnalysis().fit(x_train, y_train.ravel())
      },
      {
          "name": "NB",
          "model": BernoulliNB().fit(x_train, y_train.ravel())
      },

    # { "name":"XGB", "model": xgb.XGBRegressor().fit(x_train, y_train) },
  ]
