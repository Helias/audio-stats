import pickle

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

from import_dataset import imp_dataset
from models import get_trained_models


def predict_and_score(model, x_test, y_test):
  p_test = model.predict(x_test)
  return accuracy_score(y_test, p_test), confusion_matrix(y_test, p_test), p_test


# speakers = [
#     'LA_0012', 'LA_0013', 'LA_0047', 'LA_0023', 'LA_0038', 'LA_0027', 'LA_0033', 'LA_0022', 'LA_0007', 'LA_0003', 'LA_0018', 'LA_0041', 'LA_0009', 'LA_0011', 'LA_0004', 'LA_0024',
#     'LA_0035', 'LA_0048', 'LA_0029', 'LA_0034', 'LA_0037', 'LA_0036', 'LA_0044', 'LA_0028', 'LA_0042', 'LA_0017', 'LA_0030', 'LA_0039', 'LA_0006', 'LA_0019', 'LA_0016', 'LA_0015',
#     'LA_0032', 'LA_0005', 'LA_0031', 'LA_0025', 'LA_0014', 'LA_0045', 'LA_0008', 'LA_0043', 'LA_0001', 'LA_0002', 'LA_0020', 'LA_0040', 'LA_0021', 'LA_0010', 'LA_0026', 'LA_0046'
# ]

speakers = ['LA_0012', 'LA_0013', 'LA_0047', 'LA_0023', 'LA_0038']

drop_features = ["AUDIO_FILE_NAME", "label", "SPEAKER_ID", "Unused", "SYSTEM_ID", "label", "duration", "size"]
query_conditions = '(SYSTEM_ID == "-" | SYSTEM_ID ==  "A07")'
speaker_condition = ' | '.join(['SPEAKER_ID == "' + speaker + '"' for speaker in speakers])
query_conditions = '(' + speaker_condition + ')' + " & " + query_conditions

# print('loading dataset')
x_train, y_train, x_test, y_test = imp_dataset("ASVspoof_data.csv", drop_features, query_conditions)

# print('training models')
trained_models = get_trained_models(x_train, y_train)

for t_m in trained_models:
  acc, conf_matrix, y_pred = predict_and_score(t_m["model"], x_test, y_test)

  # print(conf_matrix)
  print(t_m["name"] + "\t ACC: \t\t", str(acc)[:5])
