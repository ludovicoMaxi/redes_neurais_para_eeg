import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import numpy.typing as npt
from category_encoders import OneHotEncoder
import tensorflow as tf
from statistics import mode 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, cohen_kappa_score
from tensorflow import keras
from sklearn.utils import shuffle
import itertools
from statistics import geometric_mean
import scipy
import warnings
import matplotlib.pyplot as plt
from EegRecord import EegRecord
from Epoch import Epoch, BandFrequence
import mne
#from enum import StrEnum
from strenum import StrEnum
from enum import Enum
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
from sklearn.utils import shuffle
import time
from varname import nameof
from abc import ABC, abstractmethod


##################################################################
#############   Definição de Funções e constantes    #############
##################################################################
warnings.simplefilter(action='ignore', category=FutureWarning)
SEED = 42
SFREQ = 400


class ModelAi(ABC):
    @abstractmethod
    def get_model(self, features):
        """
        Returns de model
        """

    @abstractmethod
    def data_genereator(self, *args, **kwargs):
        """
        Returns de model
        """

class CnnLstmFeatures(ModelAi):

    def get_model(self, features):
        """
        Returns de model
        """
    

    def data_genereator(self, *args, **kwargs):
        """
        Returns de model
        """


class CnnLstm(ModelAi):

    def get_model(self, features):
        """
        Returns de model
        """
    

    def data_genereator(self, *args, **kwargs):
        """
        Returns de model
        """


class FeatureType(StrEnum):
    CLINICAL = 'Clinico'
    STATISTIC = 'Estatistico'
    STATISTIC_AND_CLINICAL = 'Estatistico e Clinico'
    PCP = 'Porcentagem de contribuicao de pontencia'
    PCP_AND_CLINICAL = 'Porcentagem de contribuicao de pontencia e Clinico'
    FM = 'Frequencia Mediana'
    FM_AND_CLINICAL = 'Frequencia Mediana e Clinico'
    PCP_FM = 'Porcentagem de contribuicao de pontencia e Frequencia Mediana'

class AiType(Enum):
    DEATH_PROGNOSTIC = ('Cenario para prognostico de Morte', 'label')
    ETIOLOGY = ('Cenario para classificação da etiologia', 'Etiology')

    def __new__(cls, description, assumption_label):
        obj = object.__new__(cls)
        obj._description_ = description
        obj.assumption_label = assumption_label
        return obj

class Patient:
    def __init__(self, id_patient, eeg_data, eeg_record, clinical_features = []):
        self.id_patient = id_patient
        self.clinical_features = clinical_features 
        self.eeg_data = eeg_data 
        self.eeg_record = eeg_record       


parameters_names = np.array(['filters.1', 'kernel_size.1', 'padding.1', 'pool_size.1', 'dropout.1',  # layer 1 Cov1d
                            'filters.2', 'kernel_size.2', 'padding.2', 'pool_size.2', 'dropout.2',  # layer 2 Cov1d
                             'filters.3', 'kernel_size.3', 'padding.3', 'pool_size.3', 'dropout.3',  # layer 3 Cov1d
                             'filters.4', 'kernel_size.4', 'padding.4', 'pool_size.4', 'dropout.4',  # layer 4 Cov1d
                             'lstm_units.5', 'dropout.5',  # layer 1 LSTM
                             'lstm_units.6', 'dropout.6',  # layer 2 LSTM
                             'lstm_units.7', 'dropout.7',  # layer 3 LSTM
                             'dense_units.8', 'dropout.8',  # layer 1 Dense
                             'dense_units.9', 'dropout.9',  # layer 2 Dense
                             'activation_cnn', 'activation_dense',
                             'epochs', 'optimizer', 'learning_rate', 'momentum', 'scaler'])

CHANNELS_NAME = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3',
        'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']


FEATURES = ['FP1_mean',  'FP1_std', 'FP1_max', 'FP1_min', 'FP1_var',
        'FP2_mean',   'FP2_std', 'FP2_max', 'FP2_min', 'FP2_var',
        'F7_mean',   'F7_std', 'F7_max', 'F7_min', 'F7_var',
        'F3_mean',   'F3_std', 'F3_max', 'F3_min', 'F3_var',
        'FZ_mean',  'FZ_std', 'FZ_max', 'FZ_min', 'FZ_var',
        'F4_mean',   'F4_std', 'F4_max', 'F4_min', 'F4_var',
        'F8_mean',   'F8_std', 'F8_max', 'F8_min', 'F8_var',  
        'T3_mean',   'T3_std', 'T3_max', 'T3_min', 'T3_var',
        'C3_mean',  'C3_std', 'C3_max', 'C3_min', 'C3_var',
        'CZ_mean',   'CZ_std', 'CZ_max', 'CZ_min', 'CZ_var', 
        'C4_mean',   'C4_std', 'C4_max', 'C4_min', 'C4_var', 
        'T4_mean',   'T4_std', 'T4_max', 'T4_min', 'T4_var',
        'T5_mean',  'T5_std', 'T5_max', 'T5_min', 'T5_var',
        'P3_mean',   'P3_std', 'P3_max', 'P3_min', 'P3_var',
        'PZ_mean',   'PZ_std', 'PZ_max', 'PZ_min', 'PZ_var',  
        'P4_mean',   'P4_std', 'P4_max', 'P4_min', 'P4_var',
        'T6_mean',  'T6_std', 'T6_max', 'T6_min', 'T6_var',
        'O1_mean',   'O1_std', 'O1_max', 'O1_min', 'O1_var', 
        'OZ_mean',   'OZ_std', 'OZ_max', 'OZ_min', 'OZ_var', 
        'O2_mean',   'O2_std', 'O2_max', 'O2_min', 'O2_var']


def extract_statistic_features(df):
    import pandas as pd
    import numpy as np

    result = pd.DataFrame(columns=FEATURES)
    
    
    sample_statistic_features = {'mean':np.round(df[0, :, :].mean(0), 3), 'std':np.round(df[0, :, :].std(0), 3), 
                            'max':np.round(df[0, :, :].max(0), 3), 'min':np.round(df[0, :, :].min(0), 3), 
                            'var':np.round(df[0, :, :].var(0), 3)}


    r = pd.DataFrame(sample_statistic_features).to_numpy().reshape(-1)

    result = pd.concat([result, pd.DataFrame(r.reshape(-1, len(r)), columns=FEATURES)], ignore_index = True)

    return result


def transform_train_data(df_dict, type=FeatureType.STATISTIC, assumption_label='label'):

    patients_data_signal = []
    patients_data_features = []
    patients_labels = []
    
    if ( type in (FeatureType.STATISTIC, FeatureType.STATISTIC_AND_CLINICAL, FeatureType.CLINICAL) ):

        for key in df_dict.keys():
            
            for epoch in df_dict[key].eeg_data:
                df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
                
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                
                if (type == FeatureType.STATISTIC): 
                    statistical_features = extract_statistic_features(epoch_signal)
                    patients_data_features.append(statistical_features.values[0])

                elif (type == FeatureType.STATISTIC_AND_CLINICAL): 
                    statistical_features = extract_statistic_features(epoch_signal)
                    patients_data_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], statistical_features.values[0])))

                elif (type == FeatureType.CLINICAL): 
                    patients_data_features.append(df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0])            
                
                #print(f'{key} - size: {np.shape(epoch_signal)}')
                patients_data_signal.append(epoch_signal)
                patients_labels.append(df_clinical_features[assumption_label].values[0])

    elif ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM) ):

        for key in df_dict.keys():
            df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
                        
            df_dict[key].eeg_record.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal()
            df_dict[key].eeg_record.compute_pcp()
            df_dict[key].eeg_record.compute_fm()

            if ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL) ): 
                for epoch in  df_dict[key].eeg_record.epoch_list[0:9]:
                    pcp_epoch_band = [] 
                    [pcp_epoch_band.append(band.pcp) for band in epoch.band_frequence_signal_list]                   
                    features = np.ndarray.flatten(np.array(pcp_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    if (type == FeatureType.PCP):
                        patients_data_features.append(features)
                    else:
                        patients_data_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))

            elif ( type in (FeatureType.FM, FeatureType.FM_AND_CLINICAL) ):
                for epoch in  df_dict[key].eeg_record.epoch_list[0:9]:
                    fm_epoch_band = [] 
                    [fm_epoch_band.append(band.fm) for band in epoch.band_frequence_signal_list]                   
                    features = np.ndarray.flatten(np.array(fm_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    if (type == FeatureType.FM):
                        patients_data_features.append(features)
                    else:
                        patients_data_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))    
            elif (type == FeatureType.PCP_FM):        
                patients_data_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], statistical_features.values[0])))


            for epoch in df_dict[key].eeg_data:                
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                patients_data_signal.append(epoch_signal)
                patients_labels.append(df_clinical_features[assumption_label].values[0])
   
   
    #print(f'{key} - size: {np.shape(patients_data_signal)}')
    print(f'{key} - size: {np.shape(patients_labels)}')  
    print(f'{key} - size: {np.shape(patients_data_features)}') 
    return patients_data_signal, patients_data_features, patients_labels


def transform_test_data(df_dict, type=FeatureType.STATISTIC, assumption_label='label'):

    patients_data_signal = []
    patients_data_features = []
    patients_labels = []
    
    if ( type in (FeatureType.STATISTIC, FeatureType.STATISTIC_AND_CLINICAL, FeatureType.CLINICAL) ):

        for key in df_dict.keys():

            X_signal = []
            X_features = []
            y = []            
            for epoch in df_dict[key].eeg_data:
                df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
                
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                
                if (type == FeatureType.STATISTIC): 
                    statistical_features = extract_statistic_features(epoch_signal)
                    X_features.append(statistical_features.values[0])

                elif (type == FeatureType.STATISTIC_AND_CLINICAL): 
                    statistical_features = extract_statistic_features(epoch_signal)
                    X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], statistical_features.values[0])))

                elif (type == FeatureType.CLINICAL): 
                    X_features.append(df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0])            
                
                X_signal.append(epoch_signal)
                y.append(df_clinical_features[assumption_label].values[0])
        
            patients_data_signal.append(X_signal)
            patients_data_features.append(X_features)
            patients_labels.append(y)

    elif ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM) ):

        for key in df_dict.keys():
            df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
                        
            df_dict[key].eeg_record.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal()
            df_dict[key].eeg_record.compute_pcp()
            df_dict[key].eeg_record.compute_fm()

            if ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL) ): 
                X_features = [] 
                for epoch in  df_dict[key].eeg_record.epoch_list[0:9]:
                    pcp_epoch_band = [] 
                    [pcp_epoch_band.append(band.pcp) for band in epoch.band_frequence_signal_list]                   
                    features = np.ndarray.flatten(np.array(pcp_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    if (type == FeatureType.PCP):
                        X_features.append(features)
                    else:
                        X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))

            elif ( type in (FeatureType.FM, FeatureType.FM_AND_CLINICAL) ):
                X_features = [] 
                for epoch in  df_dict[key].eeg_record.epoch_list[0:9]:
                    fm_epoch_band = [] 
                    [fm_epoch_band.append(band.pcp) for band in epoch.band_frequence_signal_list]                   
                    features = np.ndarray.flatten(np.array(fm_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    if (type == FeatureType.FM):
                        X_features.append(features)
                    else:
                        X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))
    
            elif (type == FeatureType.PCP_FM):        
                X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], statistical_features.values[0])))

            X_signal = []
            y = []
            for epoch in df_dict[key].eeg_data:                
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                X_signal.append(epoch_signal)
                y.append(df_clinical_features[assumption_label].values[0])
        
            patients_data_signal.append(X_signal)
            patients_data_features.append(X_features)
            patients_labels.append(y)

    print(f'{key} - size: {np.shape(patients_data_features)}')
    #print(f'{key} - size: {np.shape(patients_data_signal)}')
    print(f'{key} - size: {np.shape(patients_labels)}')        
   
    return patients_data_signal, patients_data_features, patients_labels


def resample(sample, sfreq: int):
    '''
    Realiza o resampling das amostras considerando o parâmetro sfreq.

    Considera uma amostra com o seguinte shape: (amostra, caminho, frequencia, target).

    Parâmetros:
        sample: amostra.
        sfreq: frequência de amostragem desejada.
    '''
    epochs = sample[0]
    freq = sample[2]

    info = mne.create_info(ch_names=CHANNELS_NAME, sfreq=freq, ch_types='eeg')
    sample = np.copy(sample)
    sample[2] = sfreq
    sample[0] = np.array([mne.io.RawArray(epoch, info=info, verbose='CRITICAL').resample(sfreq, verbose='CRITICAL')._data for epoch in epochs])
    
    return sample


def metrics(y_true, y_pred):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
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
    
    print(f'Accuracy: {acc}')
    print(f'F1 score: {f1}')
    print(f'F1 score (macro): {f1_macro}')
    print(f'F1 score (weighted): {f1_weighted}')
    print(f'Precision: {precision}')
    print(f'Precision (macro): {precision_macro}')
    print(f'Recall: {recall}')
    print(f'Recall (macro): {recall_macro}')
    
    return acc, f1, f1_macro, f1_weighted, precision, precision_macro, recall, recall_macro


def data_with_features_generator(X_signal, X_features, y):
    while True:
        for i in range(0, len(X_signal)):
            shape = (-1,1)  if (len(np.shape(y)) == 1) else (1, np.shape(y)[1])
            yield ((tf.convert_to_tensor(X_signal[i]), tf.convert_to_tensor(np.array(X_features[i]).reshape(1, -1))), np.array(y[i]).reshape(shape))


def read_file_with_datas_for_training(path_file='./data/patients.xlsx', ai_type=AiType.DEATH_PROGNOSTIC):

    df_info_patients = pd.read_excel(path_file, header=0, index_col=0)
    df_info_patients.head()
    df_info_patients_copy = df_info_patients.copy()

    if( ai_type == AiType.DEATH_PROGNOSTIC ):
         #prognostico morte
        CLINICAL_FEATURES = ['Gender', 'Etiology', 'Outcome', 'Age']
        df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
        df_info_patients_copy.head()
        df_info_patients_copy[ai_type.assumption_label] = np.where(df_info_patients_copy['Outcome'] == 'Alive', 1, 0)
        df_info_patients_copy = df_info_patients_copy.drop('Outcome', axis=1)
        df_info_patients_copy.head()

        # Scaler features
        encoder_cat = OneHotEncoder(cols=['Gender', 'Etiology'], use_cat_names=True)

    elif( ai_type == AiType.ETIOLOGY ):
        #etiologia
        CLINICAL_FEATURES = ['Gender', 'Etiology', 'Age'] #etiologia 
        df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
        df_info_patients_copy.head()
        df_info_patients_copy[ai_type.assumption_label].value_counts()
        df_info_patients_copy = df_info_patients_copy.replace({'Post anoxic encephalopathy': 'Other', 'Neoplasia': 'Other',
                                            'Hydroelectrolytic Disorders': 'Other', 'Firearm Injury': 'Other',
                                            'Chronic subdural hematoma': 'Other', 'Hydrocephalus': 'Other',
                                            'Hypoxic encephalopathy': 'Other', 'Neurocysticercosis': 'Other'}, regex=True) 
        df_info_patients_copy[ai_type.assumption_label].value_counts()
        # Scaler features
        encoder_cat = OneHotEncoder(cols=['Gender'], use_cat_names=True)
    

    df_info_patients_copy = encoder_cat.fit_transform(df_info_patients_copy)
    df_info_patients_copy.head()

    return (df_info_patients, df_info_patients_copy)


def read_datas_eeg_for_training(df_info_patients, df_info_patients_copy, feature_type=FeatureType.STATISTIC, resample_data_same_size=False):
    #a vida é curta para viver, mas tem tanto tempo para aprender, quanto tempo para se jogar, quanto tempo ate você perceber, que os seus sonhos só dependem de você
    data = {}
    frequence_band_list = [BandFrequence('delta', 1, 3.5),
                        BandFrequence('teta', 3.5, 7.5),
                        BandFrequence('alfa', 7.5, 12.5),
                        BandFrequence('beta', 12.5, 30)]
                        #BandFrequence('gama', 30, 80)]
                        #BandFrequence('supergama', 80, 100),
                        #BandFrequence('ruido', 58, 62)]

    for id_patient, outcome in zip(df_info_patients.index, df_info_patients.Outcome):
        #print(f'./data/eeg_{outcome.lower()}/{id_patient}.mat')
        df_eeg = scipy.io.loadmat(f'./data/eeg_{outcome.lower()}/{id_patient}.mat')
        

        if( feature_type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM) ):
            eeg_record = EegRecord(id_patient, CHANNELS_NAME, df_eeg['fs'][0][0], df_eeg['XN'], customize=False)
            fs = df_eeg['fs'][0][0]
            for i in range(len(df_eeg['epochsRange'][0]) - 1):
                eeg_record.epoch_list.append(Epoch(df_eeg['epochsTime'][0][i], 2, fs, df_eeg['epochsRange'][0][i], frequence_band_list))
        else:
            eeg_record = None


        if( resample_data_same_size ):
            sample: npt.NDArray[np.object_] = np.zeros(shape=(5), dtype=np.object_)
            sample[0] = df_eeg['epochsRange'][0][0:9]
            sample[2] = fs
            eeg_data = resample(sample, SFREQ)[0]
            eeg_data_trunc = eeg_data[:, :, 0: 801]
        else:
            #sem igualar a quantidade de atributos/amostras
            eeg_data_trunc =  df_eeg['epochsRange'][0][0:9]

        patient = Patient(id_patient = id_patient, clinical_features = df_info_patients_copy.loc[f'{id_patient}'].copy(), eeg_data = eeg_data_trunc, eeg_record = eeg_record) 
        
        data[f'{id_patient}'] = patient
    
    return data


def printMeanAndStandardDeviationMetric(metricName, values):
    print(f'{metricName}: {round(np.mean(values), 3)} +/- {round(np.std(values), 3)}')

##################################################################
############ FIM   Definição de Funções e constantes  ############
##################################################################

##################################################################
################       Setup Processamento         ###############
##################################################################
def process_training_ai(path_file='./data/patients.xlsx', feature_type=FeatureType.STATISTIC, ai_type=AiType.DEATH_PROGNOSTIC, number_splits=5, seed=SEED):
    print("##################################################################")
    print("############     iniciando treinamento da IA          ############")
    print("##################################################################")
    print(f"path_file={path_file}")
    print(f"feature_type={feature_type}")
    print(f"number_splits={number_splits}")
    print(f"seed={seed}")
    print("##################################################################")

    (df_info_patients, df_info_patients_copy) = read_file_with_datas_for_training(path_file=path_file, ai_type=ai_type)
    data = read_datas_eeg_for_training(df_info_patients, df_info_patients_copy, feature_type=feature_type)

    index = np.array(list(data.keys()))
    acc = []
    f1 = []
    f1_macro = []
    f1_weighted = [] 
    precision = [] 
    precision_macro = []
    recall = []
    recall_macro = []

    skfolds = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=42)

    fold = 1
    for train_index, test_index  in skfolds.split(list(data.keys()), df_info_patients_copy[ai_type.assumption_label]):
        print(f'-------------------- Fold {fold} --------------------')
        keras.backend.clear_session()
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        train_data = {key: data[key] for key in index[train_index]}
        X_train_signal, X_train_features, y_train = transform_train_data(train_data, type=feature_type, assumption_label=ai_type.assumption_label)
        print("######################")
        X_train_signal, X_train_features, y_train = shuffle(X_train_signal, X_train_features, y_train, random_state=42)
        
        test_data = {key: data[key] for key in index[test_index]}
        X_test_signal, X_test_features, y_test = transform_test_data(test_data, type=feature_type, assumption_label=ai_type.assumption_label)
        
        X_train_features_array = np.array(X_train_features)
        X_test_features_array = np.array(X_test_features)

        n_features = np.shape(X_train_features)[1]
        
        input_B = tf.keras.layers.Input(shape=[n_features], name='categorical_input')
        
        #print(input_B.summary())
        input_A = tf.keras.layers.Input(shape=[None, 20], name='timeseries_input')

        hidden1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(input_A)
        hidden2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(hidden1)
        max_pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(hidden2)
        droopout_layer1 = tf.keras.layers.Dropout(0.2)(max_pooling_layer)
        hidden3 = tf.keras.layers.Conv1D(filters=128, kernel_size=13, activation='relu')(droopout_layer1)
        hidden4 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu')(hidden3)
        hidden5 = tf.keras.layers.LSTM(32, return_sequences=True)(hidden4)
        gp = tf.keras.layers.GlobalAveragePooling1D()(hidden5)
        
        concat = None
        activation_type = None
        loss = None
        model_metrics = None
        if( ai_type == AiType.DEATH_PROGNOSTIC ):
            flatten_layer = tf.keras.layers.Flatten()(gp)
            concat = tf.keras.layers.concatenate([flatten_layer, input_B])
            activation_type = 'sigmoid'
            loss = tf.keras.losses.binary_crossentropy
            model_metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
        else:
            concat = tf.keras.layers.concatenate([gp, input_B])
            activation_type = 'softmax'
            loss = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
            model_metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]

            # Scaler label
            encoder = OneHotEncoder(handle_unknown='ignore')
            #transforma a escrite em numero para sair no modelo

            y_train = encoder.fit_transform(np.array(y_train)).values.tolist()
                
            y_test_enconder = []
            for i in range(0, len(y_test)):
                y_test_enconder.append(encoder.transform(np.array(y_test[i])).values.tolist())



        hidden6 = tf.keras.layers.Dense(64, activation='relu')(concat)
        droopout_layer2 = tf.keras.layers.Dropout(0.2)(hidden6)
        dimension = 1  if (len(np.shape(y_train)) == 1) else np.shape(y_train)[1]
        output = tf.keras.layers.Dense(dimension, activation=activation_type, name='output')(droopout_layer2)

        model = tf.keras.Model(inputs=[input_A, input_B], outputs=[output])
        model.compile( loss=loss, metrics=model_metrics, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4) )
        
        scale_features = StandardScaler()
        X_train_features = scale_features.fit_transform(X_train_features_array.reshape(-1, X_train_features_array.shape[-1])).reshape(X_train_features_array.shape).tolist()
        X_test_features = scale_features.transform(X_test_features_array.reshape(-1, X_test_features_array.shape[-1])).reshape(X_test_features_array.shape).tolist()
        
        with tf.device('/GPU:0'):    
            model.fit(data_with_features_generator(X_train_signal, X_train_features, y_train), steps_per_epoch=len(X_train_signal), epochs=30, verbose=1)
        
        y_test_real = []
        y_predict = []
        for i in range(len(X_test_signal)):            
            
            if( ai_type == AiType.DEATH_PROGNOSTIC ):
                predict = model.predict(data_with_features_generator(X_test_signal[i], X_test_features[i], y_test[i]), steps=(len(X_test_signal[i]))) 

                y_test_real.append(y_test[i][0])
                predict[predict > 0.5] = 1
                predict[predict < 0.5] = 0
                
                predict = np.array(predict).flatten()
                y_predict.append(mode(predict))

            else:
                predict = model.predict(data_with_features_generator(X_test_signal[i], X_test_features[i], y_test_enconder[i]), steps=(len(X_test_signal[i])))

                predict_ = []
                y_test_real.append(np.array(y_test_enconder[i][0]).argmax())
                for j in predict:
                    predict_.append(np.array(j).argmax())

                y_predict.append(mode(np.array(predict_)))


        print(f'\n PERFORMANCE')
        print(classification_report(y_test_real, y_predict))
        print(confusion_matrix(y_test_real, y_predict))
        print(f'Performance of model CNN + LSTM + features ({feature_type})')
        acc_fold, f1_fold, f1_macro_fold, f1_weighted_fold, precision_fold, precision_macro_fold, recall_fold, recall_macro_fold = metrics(y_test_real, y_predict)
        
        acc.append(acc_fold)
        f1.append(f1_fold)
        f1_macro.append(f1_macro_fold)
        f1_weighted.append(f1_weighted_fold) 
        precision.append(precision_fold) 
        precision_macro.append(precision_macro_fold)
        recall.append(recall_fold)
        recall_macro.append(recall_macro_fold)
        
        fold += 1


    print(f'Performance of model ({ai_type}) CNN + LSTM + features ({feature_type})')
    printMeanAndStandardDeviationMetric('Accuracy', acc)
    printMeanAndStandardDeviationMetric('F1 score', f1)
    printMeanAndStandardDeviationMetric('F1 score (macro)', f1_macro)
    printMeanAndStandardDeviationMetric('F1 score (weighted)', f1_weighted)
    printMeanAndStandardDeviationMetric('Precision', precision)
    printMeanAndStandardDeviationMetric('Precision (macro)', precision_macro)
    printMeanAndStandardDeviationMetric('Recall', recall)
    printMeanAndStandardDeviationMetric('Recall (macro)', recall_macro)

##################################################################
################  FIM  Setup Processamento         ###############
##################################################################
start_time = time.time()
process_training_ai(feature_type=FeatureType.PCP_AND_CLINICAL, ai_type=AiType.DEATH_PROGNOSTIC)
print("--- %s seconds ---" % (time.time() - start_time))

# Check if a GPU is available
tf.config.experimental.list_physical_devices()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available.")
    for gpu in gpus:
        print("GPU:", gpu)
else:
    print("No GPU found.")