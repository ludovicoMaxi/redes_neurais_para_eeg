import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) 

import random
import numpy as np
import warnings
from statistics import mode 
import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, cohen_kappa_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from enums import *
from models import ModelAi
from memory_profiler import profile

##################################################################
#############   Definição de Funções e constantes    #############
##################################################################
warnings.simplefilter(action='ignore', category=FutureWarning)
SEED = 42

def metrics(y_true, y_pred):
    
    f1 = 0
    precision = 0
    recall = 0
    if( np.max(y_pred) <= 1 and np.max(y_true) <= 1 ):
        f1 = round(f1_score(y_true, y_pred), 3)
        precision = round(precision_score(y_true, y_pred), 3)
        recall = round(recall_score(y_true, y_pred), 3)

    acc = round(accuracy_score(y_true, y_pred), 3)
    f1_macro = round(f1_score(y_true, y_pred, average='macro'), 3)
    f1_weighted = round(f1_score(y_true, y_pred, average='weighted'), 3)
    precision_macro = round(precision_score(y_true, y_pred, average='macro'), 3)
    recall_macro = round(recall_score(y_true, y_pred, average='macro'), 3)
    specificity_macro = compute_macro_specificity(y_true, y_pred)
    
    print(f'Accuracy: {acc}')
    print(f'F1 score: {f1}')
    print(f'F1 score (macro): {f1_macro}')
    print(f'F1 score (weighted): {f1_weighted}')
    print(f'Precision: {precision}')
    print(f'Precision (macro): {precision_macro}')
    print(f'Recall: {recall}')
    print(f'Recall (macro): {recall_macro}')
    print(f'specificity (macro): {specificity_macro}')

    
    return acc, f1, f1_macro, f1_weighted, precision, precision_macro, recall, recall_macro, specificity_macro


def compute_macro_specificity(y_true, y_pred):

    # Obtenha as classes únicas
    classes = np.unique(y_true)
    specificities = {}

    # Iterar sobre cada classe
    for cls in classes:
        # Crie um vetor binário: 1 para a classe atual, 0 para todas as outras classes
        y_true_binary = np.where(y_true == cls, 1, 0)
        y_pred_binary = np.where(y_pred == cls, 1, 0)

        # Calcular a matriz de confusão para a classe atual
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        # Calcular a especificidade
        specificity = tn / (tn + fp)
        specificities[cls] = specificity

    macro_specificity = np.mean(list(specificities.values()))

    return macro_specificity
    


def printMeanAndStandardDeviationMetric(metricName, values):
    print(f'{metricName}: {round(np.mean(values), 3)} +/- {round(np.std(values), 3)}')

##################################################################
############ FIM   Definição de Funções e constantes  ############
##################################################################

##################################################################
################       Setup Processamento         ###############
##################################################################
@profile
def process_training_ai(file_path='./data/patients.xlsx', ai_type=AiType.DEATH_PROGNOSTIC_CLINICAL_FEATURE, number_splits=5, seed=SEED, resample_data_same_size=False, group_fold=False):
    print("##################################################################")
    print("############     iniciando treinamento da IA          ############")
    print("##################################################################")
    print(f"file_path={file_path}")
    print(f"ai_type={ai_type} ({ai_type.description})")
    print(f"number_splits={number_splits}")
    print(f"seed={seed}")
    print("##################################################################")
    directory_path = os.path.dirname(file_path)
    (df_info_patients, df_info_patients_copy) = ai_type.model_class.read_file_with_datas_for_training(file_path=file_path)

    data = ai_type.model_class.read_datas_eeg_for_training(df_info_patients, df_info_patients_copy, resample_data_same_size=resample_data_same_size, dir_path_files=directory_path)

    index = np.array(list(data.keys()))
    acc = []
    f1 = []
    f1_macro = []
    f1_weighted = [] 
    precision = [] 
    precision_macro = []
    recall = []
    recall_macro = []
    specificity_macro = []


    groups = None
    if (group_fold == False):
        skfolds = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=42)
    else:
        #skfolds = GroupKFold(n_splits=number_splits)
        skfolds = StratifiedGroupKFold(n_splits=number_splits, shuffle=True, random_state=42)
        groups = df_info_patients['Group']
        

    fold = 1
    #criar outra estrutura de dados que somente passa o id da pessoa nao amostra
    for train_index, test_index  in skfolds.split(list(data.keys()), df_info_patients_copy[ai_type.model_class.get_analysis_type().assumption_label], groups=groups):
        print(f'-------------------- Fold {fold} --------------------')
        keras.backend.clear_session()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
        # If using TensorFlow, this will make GPU ops as deterministic as possible,
        # but it will affect the overall performance, so be mindful of that.
        #tf.config.experimental.enable_op_determinism()
        #tf.config.experimental.enable_op_determinism = False

        train_data = {key: data[key] for key in index[train_index]}
        test_data = {key: data[key] for key in index[test_index]}
        
        ######################
        ai_type.model_class.initialize_csp()
        datas_for_training = ai_type.model_class.get_data_for_training(train_data)
        datas_for_test = ai_type.model_class.get_data_for_test(test_data)
        model = ai_type.model_class.get_model(datas_for_training)
        with tf.device('/GPU:0'): #para os outros casos sempre usar GPU devido a performance 
            model.fit(ai_type.model_class.data_generator(datas_for_training), steps_per_epoch=len(datas_for_training[0]), epochs=30, verbose=1)
        
        y_test_real = []
        y_predict = []
        datas_for_test = ai_type.model_class.data_test_before_generator(datas_for_test)
        for data_test in datas_for_test:          
            *_, y_test = data_test
            if( ai_type.model_class.get_analysis_type() in ModelAi.BINARY_ANALYSIS_TYPE ):
                predict = model.predict(ai_type.model_class.data_generator(data_test), steps=(len(y_test)))

                y_test_real.append(y_test[0])
                predict[predict > 0.5] = 1
                predict[predict <= 0.5] = 0
                
                predict = np.array(predict).flatten()
                y_predict.append(mode(predict))

            else:
                predict = model.predict(ai_type.model_class.data_generator(data_test), steps=(len(y_test)))

                predict_ = []
                y_test_real.append(np.array(y_test[0]).argmax())
                for j in predict:
                    predict_.append(np.array(j).argmax())

                y_predict.append(mode(np.array(predict_)))


        print(f'\n PERFORMANCE')
        print(classification_report(y_test_real, y_predict))
        print(confusion_matrix(y_test_real, y_predict))
        print(f'Performance of model CNN + LSTM + features ({ai_type.description})')
        acc_fold, f1_fold, f1_macro_fold, f1_weighted_fold, precision_fold, precision_macro_fold, recall_fold, recall_macro_fold, specificity_macro_fold = metrics(y_test_real, y_predict)
        
        acc.append(acc_fold)
        f1.append(f1_fold)
        f1_macro.append(f1_macro_fold)
        f1_weighted.append(f1_weighted_fold) 
        precision.append(precision_fold) 
        precision_macro.append(precision_macro_fold)
        recall.append(recall_fold)
        recall_macro.append(recall_macro_fold)
        specificity_macro.append(specificity_macro_fold)
        
        fold += 1


    print(f'Performance of model ({ai_type.description})')
    printMeanAndStandardDeviationMetric('Accuracy', acc)
    printMeanAndStandardDeviationMetric('F1 score', f1)
    printMeanAndStandardDeviationMetric('F1 score (macro)', f1_macro)
    printMeanAndStandardDeviationMetric('F1 score (weighted)', f1_weighted)
    printMeanAndStandardDeviationMetric('Precision', precision)
    printMeanAndStandardDeviationMetric('Precision (macro)', precision_macro)
    printMeanAndStandardDeviationMetric('Recall', recall)
    printMeanAndStandardDeviationMetric('Recall (macro)', recall_macro)
    printMeanAndStandardDeviationMetric('Specificity (macro)', specificity_macro)

##################################################################
################  FIM  Setup Processamento         ###############
##################################################################