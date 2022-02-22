from typing import Union
import pathos as pa
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from import_dataset import imp_dataset
from models import get_trained_models


def predict_and_score(model, x_test, y_test):
  p_test = model.predict(x_test)
  return accuracy_score(y_test, p_test), confusion_matrix(y_test, p_test), p_test

SPEAKERS_A01_A06 = ['LA_0069', 'LA_0070', 'LA_0071', 'LA_0072', 'LA_0073', 'LA_0074', 'LA_0075']
SPEAKERS_A07_A19 = ['LA_0012', 'LA_0013', 'LA_0047', 'LA_0023', 'LA_0038']

SYSTEM_IDS_A01_A06 = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']
SYSTEM_IDS_A07_A19 = ['A08', 'A09', 'A12', 'A15', 'A16', 'A17', 'A18', 'A19']
SYSTEM_IDS_BIT_RATE_MAIN_FEATURE = ['A07', 'A10', 'A11', 'A13', 'A14']

SPEAKER_SYSTEM_IDS = [
  { "speakers": SPEAKERS_A01_A06, "system_ids": SYSTEM_IDS_A01_A06 },
  { "speakers": SPEAKERS_A07_A19, "system_ids": SYSTEM_IDS_A07_A19 },
  { "speakers": SPEAKERS_A07_A19, "system_ids": SYSTEM_IDS_BIT_RATE_MAIN_FEATURE },
]

def parallelize(speakers_system_ids: list) -> Union[str, dict]:
    [system_id, speakers] = speakers_system_ids

    drop_features = ["AUDIO_FILE_NAME", "label", "SPEAKER_ID", "Unused", "SYSTEM_ID", "label", "duration", "size", "spectral_bandwidth"] #, "bit_rate"]
    query_conditions = '(SYSTEM_ID == "-" | SYSTEM_ID ==  "' + system_id + '")'
    speaker_condition = ' | '.join(['SPEAKER_ID == "' + speaker + '"' for speaker in speakers])
    query_conditions = '(' + speaker_condition + ')' + " & " + query_conditions

    # print('loading dataset')
    # ASVspoof_all_data.csv
    # ASVspoof_loud_norm_data.csv
    # ASVspoof_resample_bit_rate_data.csv
    x_train, y_train, x_test, y_test = imp_dataset("ASVspoof_loud_norm_data.csv", drop_features, query_conditions)

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
    
    return system_id, res 


formatted_markdown = ["|Model"]
formatted_markdown.append("|---" + ''.join(["|---" for _ in (SYSTEM_IDS_A01_A06+SYSTEM_IDS_A07_A19+SYSTEM_IDS_BIT_RATE_MAIN_FEATURE)]))

for model_name in ["|**CART**", "|**SVM**", "|**LR**", "|**KNN**", "|**GMM**", "|**LDA**", "|**SVC1**", "|**SVC2**", "|**GPC**", "|**RFC**", "|**MLP**", "|**ADC**", "|**GNB**", "|**QDA**", "|**NB**"]:
  formatted_markdown.append(model_name)

for speaker_system_id in SPEAKER_SYSTEM_IDS:
  speakers = speaker_system_id["speakers"]
  SYSTEM_IDS = speaker_system_id["system_ids"]

  formatted_markdown[0] += '|' + '|'.join(SYSTEM_IDS)

  system_id_and_speakers = [[system_id, speakers] for system_id in SYSTEM_IDS]

  resx = {}
  ncpu = int(pa.helpers.cpu_count() / 2)
  with pa.multiprocessing.ProcessingPool(ncpu) as p:
      results = list(tqdm(p.imap(parallelize, system_id_and_speakers), total=len(SYSTEM_IDS)))

  for r in results:
    system_id = r[0]
    model_results = r[1]

    resx[system_id] = model_results

  # for system_id in SYSTEM_IDS:
  #     resx[system_id] = parallelize(system_id)

  # print(resx)

  idx = 2
  for model in resx[SYSTEM_IDS[0]]:
    # print("|" + model, end='')
    # print(model + '\t' + str(resx[system_id][model])[:5])

    for system_id in resx:
      val = str(resx[system_id][model])[:5]
      # print('\t |' + val, end='')

      formatted_markdown[idx] += "|" + val

    idx += 1

for idx in range(len(formatted_markdown)):
  formatted_markdown[idx] += "|"

for markdown_line in formatted_markdown:
  print(markdown_line)
