from typing import Union
import pathos as pa
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from import_dataset import imp_dataset
from models import get_trained_models


def predict_and_score(model, x_test, y_test):
  p_test = model.predict(x_test)
  return accuracy_score(y_test, p_test), confusion_matrix(y_test, p_test), p_test


# speakers for A07+
# speakers = [
#     'LA_0012', 'LA_0013', 'LA_0047', 'LA_0023', 'LA_0038', 'LA_0027', 'LA_0033', 'LA_0022', 'LA_0007', 'LA_0003', 'LA_0018', 'LA_0041', 'LA_0009', 'LA_0011', 'LA_0004', 'LA_0024',
#     'LA_0035', 'LA_0048', 'LA_0029', 'LA_0034', 'LA_0037', 'LA_0036', 'LA_0044', 'LA_0028', 'LA_0042', 'LA_0017', 'LA_0030', 'LA_0039', 'LA_0006', 'LA_0019', 'LA_0016', 'LA_0015',
#     'LA_0032', 'LA_0005', 'LA_0031', 'LA_0025', 'LA_0014', 'LA_0045', 'LA_0008', 'LA_0043', 'LA_0001', 'LA_0002', 'LA_0020', 'LA_0040', 'LA_0021', 'LA_0010', 'LA_0026', 'LA_0046'
# ]

# speakers for A07+
# speakers = ['LA_0012', 'LA_0013', 'LA_0047', 'LA_0023', 'LA_0038']
speakers = ['LA_0069', 'LA_0070', 'LA_0071', 'LA_0072', 'LA_0073', 'LA_0074', 'LA_0075']

# SYSTEM_IDS = ['A07', 'A10', 'A11', 'A13', 'A14']
SYSTEM_IDS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']
# SYSTEM_IDS = ['A08', 'A09', 'A12', 'A15', 'A16', 'A17', 'A18', 'A19']


def parallelize(system_id: str) -> dict:
    drop_features = ["AUDIO_FILE_NAME", "label", "SPEAKER_ID", "Unused", "SYSTEM_ID", "label", "duration", "size"]
    query_conditions = '(SYSTEM_ID == "-" | SYSTEM_ID ==  "' + system_id + '")'
    speaker_condition = ' | '.join(['SPEAKER_ID == "' + speaker + '"' for speaker in speakers])
    query_conditions = '(' + speaker_condition + ')' + " & " + query_conditions

    # print('loading dataset')
    x_train, y_train, x_test, y_test = imp_dataset("ASVspoof_data.csv", drop_features, query_conditions)

    # print('training models')
    trained_models = get_trained_models(x_train, y_train)

    res = {}
    for t_m in trained_models:
      res[t_m["name"]] = 0

    REPEAT = 10
    for _ in range(REPEAT):
      for t_m in trained_models:
        acc, conf_matrix, y_pred = predict_and_score(t_m["model"], x_test, y_test)

        res[t_m["name"]] += acc
        # print(conf_matrix)
        # print(system_id + " \t" + t_m["name"] + "\t ACC: \t\t", str(acc)[:5])

    for t_m in trained_models:
      res[t_m["name"]] /= REPEAT
    
    return res 

# ncpu = int(pa.helpers.cpu_count() / 2)
# with pa.multiprocessing.ProcessingPool(ncpu) as p:
#     results = list(tqdm(p.imap(parallelize, SYSTEM_IDS), total=len(SYSTEM_IDS)))

resx = {}
for system_id in SYSTEM_IDS:
    resx[system_id] = parallelize(system_id)

# print(resx)

for model in resx[SYSTEM_IDS[0]]:
  print(model, end='')
  # print(model + '\t' + str(resx[system_id][model])[:5])

  for system_id in resx:
    print('\t' + str(resx[system_id][model])[:5], end='')
    # print(system_id)
  print()
