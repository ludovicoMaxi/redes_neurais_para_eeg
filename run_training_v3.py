import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import scipy
import numpy as np
import numpy.typing as npt
import pickle
import mne
import itertools
from statistics import geometric_mean
import warnings
import matplotlib.pyplot as plt
from statistics import mode 
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, cohen_kappa_score, recall_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone 
from category_encoders import OneHotEncoder
from EegRecord import EegRecord, Channel
from Epoch import Epoch, BandFrequence
from strenum import StrEnum
from enum import Enum
import time
from varname import nameof
from abc import ABC, abstractmethod
from typing import List

##################################################################
#############   Definição de Funções e constantes    #############
##################################################################
warnings.simplefilter(action='ignore', category=FutureWarning)
SEED = 42
SFREQ = 400

CHANNELS_BASE = [
    Channel('FP1', 'EEG'),
    Channel('FP2', 'EEG'),
    Channel('F7', 'EEG'),
    Channel('F3', 'EEG'),
    Channel('FZ', 'EEG'),
    Channel('F4', 'EEG'),
    Channel('F8', 'EEG'),
    Channel('T3', 'EEG'),
    Channel('C3', 'EEG'),
    Channel('CZ', 'EEG'),
    Channel('C4', 'EEG'),
    Channel('T4', 'EEG'),
    Channel('T5', 'EEG'),
    Channel('P3', 'EEG'),
    Channel('PZ', 'EEG'),
    Channel('P4', 'EEG'),
    Channel('T6', 'EEG'),
    Channel('O1', 'EEG'),
    Channel('OZ', 'EEG'),
    Channel('O2', 'EEG')
]

class AnalysisType(Enum):
    DEATH_PROGNOSTIC = ('Prognostico de Morte para coma', 'label')
    ETIOLOGY = ('Etiologia do Coma', 'Etiology')
    MUSIC_LIKE = ('Apreciação Estimulação Musical', 'label')

    def __new__(cls, description, assumption_label):
        obj = object.__new__(cls)
        obj.description = description
        obj.assumption_label = assumption_label
        return obj

    
class Channels(Enum):
    ALL = (['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'],
           [   0,     1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19])
    FRONTAL = (['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8'],
               [   0,     1,    2,    3,    4,    5,    6])
    CENTRAL = (['C3', 'CZ', 'C4'],
               [  8,    9,    10])
    PARIENTAL = (['P3', 'PZ', 'P4'],
                 [ 13,   14,   15])
    OCCIPITAL = (['O1', 'OZ', 'O2'],
                 [ 17,   18,   19])
    TEMPORAL = (['T3', 'T4', 'T5', 'T6'],
                [  7,   11,   12,   16])
    T3_T4_Pz_O2_Oz = (['T3', 'T4', 'PZ', 'OZ', 'O2'],
                      [  7,    11,   14,   18,  19])

    def __new__(cls, channels_name, positions):
        obj = object.__new__(cls)
        obj.channels_name = channels_name
        obj.positions = positions
        return obj

class BandFrequenceEnum(Enum):   
    DELTA = ('Banda Delta de 1 a 3.5 Hz', 0)
    TETA = ('Banda Teta de 3.5 a 7.5', 1)
    ALFA = ('Banda Alfa de 7.5 a 12.5', 2)
    BETA = ('Banda Beta de 12.5 a 30', 3)
    GAMA = ('Banda Gama de 30 a 80', 4)
    SUPERGAMA = ('Banda Super gama de 80 a 100', 5)
    RUIDO = ('Banda Ruido de 58 a 62', 6)

    def __new__(cls, description, position):
        obj = object.__new__(cls)
        obj.description = description
        obj.position = position
        return obj

class FeatureType(Enum):
    CLINICAL = (Channels.ALL, 'Clinico')
    STATISTIC = (Channels.ALL, 'Estatistico')
    STATISTIC_AND_CLINICAL = (Channels.ALL, 'Estatistico e Clinico')
    PCP = (Channels.ALL, 'Porcentagem de contribuicao de pontencia')
    PCP_AND_CLINICAL = (Channels.ALL, 'Porcentagem de contribuicao de pontencia e Clinico')
    FM = (Channels.ALL, 'Frequencia Mediana')
    FM_AND_CLINICAL = (Channels.ALL, 'Frequencia Mediana e Clinico')
    PCP_FM = (Channels.ALL, 'Porcentagem de contribuicao de pontencia e Frequencia Mediana')
    COHERENCE = (Channels.ALL, 'Coerência lado esquerdo com o lado direito')
    COHERENCE_AND_CLINICAL = (Channels.ALL, 'Coerência lado esquerdo com o lado direito e Clinico')
    CLINICAL_FRONTAL = (Channels.FRONTAL, 'Clinico somente com os canais Frontais')
    CLINICAL_CENTRAL = (Channels.CENTRAL, 'Clinico somente com os canais Centrais')
    CLINICAL_PARIENTAL = (Channels.PARIENTAL, 'Clinico somente com os canais Parientais')
    CLINICAL_OCCIPITAL = (Channels.OCCIPITAL, 'Clinico somente com os canais Occipitais')
    CLINICAL_TEMPORAL = (Channels.TEMPORAL, 'Clinico somente com os canais Temporais')
    CLINICAL_T3_T4_Pz_O2_Oz = (Channels.T3_T4_Pz_O2_Oz, 'Clinico somente com os canais T3 T4 Pz O2 Oz')

    def __new__(cls, channels, description):
        obj = object.__new__(cls)
        obj.channels = channels
        obj.description = description
        return obj


class ModelAi(ABC):
    @abstractmethod
    def get_analysis_type(self):
        """
        Return analysis type
        """

    @abstractmethod
    def get_feature_type(self):
        """
        Return feature type
        """

    @abstractmethod
    def get_band_frequences(self):
        """
        Return analysis type
        """

    @abstractmethod
    def get_model(self, datas_for_training):
        """
        Return model
        """

    @abstractmethod
    def get_data_for_training(self, train_data):
        """
        Return data for training
        """

    @abstractmethod
    def get_data_for_test(self, test_data):
        """
        Return data for testing
        """

    @abstractmethod
    def data_test_before_generator(self, *args, **kwargs):
        """
        Return generator data
        """

    @abstractmethod
    def data_generator(self, *args, **kwargs):
        """
        Return generator data
        """

    def read_mat_file_with_until_30_hz_for_pcp_fm(self, file_path, df_info_patients, resample_data_same_size=False, channels=Channels.ALL):
        frequence_band_list_30_hz = [BandFrequence('delta', 1, 3.5),
                            BandFrequence('teta', 3.5, 7.5),
                            BandFrequence('alfa', 7.5, 12.5),
                            BandFrequence('beta', 12.5, 30)]
                            #BandFrequence('gama', 30, 80)]
                            #BandFrequence('supergama', 80, 100),
                            #BandFrequence('ruido', 58, 62)]

        file_name_with_extension = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        df_eeg = scipy.io.loadmat(file_path)

        fs = df_eeg['fs'][0][0]

        eeg_data_trunc =  df_eeg['epochsRange'][0]
        shape_before = np.shape(eeg_data_trunc[0])
        #sem igualar a quantidade de atributos/amostras
        for i in range(len(eeg_data_trunc)):
            eeg_data_trunc[i] = eeg_data_trunc[i][channels.positions, :]

        
        shape_channels = np.shape(eeg_data_trunc[0])
        #igualar a quantidade de atributos/amostras
        if( resample_data_same_size ):
            sample: npt.NDArray[np.object_] = np.zeros(shape=(5), dtype=np.object_)
            sample[0] = eeg_data_trunc
            sample[2] = fs
            eeg_data = resample(sample, SFREQ, channels=channels)[0]
            eeg_data = eeg_data[:, :, 0: 801]
            eeg_data_trunc = np.empty(np.shape(eeg_data)[0], dtype=object)
            eeg_data_trunc[:] = list(eeg_data)

        # TODO criar regrar para quantidade de canais para PCP e FM
        if( self.get_feature_type() in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, 
                                        FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM,
                                        FeatureType.COHERENCE, FeatureType.COHERENCE_AND_CLINICAL) ):
            fs_epoch = SFREQ if resample_data_same_size else fs
            channels_proccess =  [channel for channel in CHANNELS_BASE if channel.name in channels.channels_name]
            eeg_record = EegRecord(file_name, channels_proccess, fs_epoch, df_eeg['XN'], customize=False)
            
            for i in range(len(eeg_data_trunc)):
                eeg_record.epoch_list.append(Epoch(df_eeg['epochsTime'][0][i][0], 2, fs_epoch, eeg_data_trunc[i], frequence_band_list_30_hz))
        
        else:
            eeg_record = None

        amount_epochs = len(eeg_data_trunc)
        eeg_data_trunc = eeg_data_trunc[0:(amount_epochs-1)]
        patient = Patient(id_patient = file_name, clinical_features = df_info_patients.loc[f'{file_name}'].copy(), eeg_data = eeg_data_trunc, eeg_record = eeg_record) 
        print(f'{file_name} - {fs} - shape before {shape_before} - shape canais {shape_channels} - shape after {np.shape(eeg_data_trunc[0])}')   

        return patient
    

    def read_pkl_file(self, file_path, df_info_patients, resample_data_same_size=False, channels=Channels.ALL):

        file_name_with_extension = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        eeg_record = pickle.load(open(file_path, 'rb'))
        
        shape_before = np.shape(eeg_record.epoch_list[0].xn)
        if(channels != Channels.ALL):
            eeg_record.split_channel_list(channels.channels_name)
            #eeg_record.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal()
            #eeg_record.compute_pcp()
            #eeg_record.compute_fm()
            #eeg_record.compute_coherence()
  
        
        eeg_data_trunc = []
        for index in range(len(eeg_record.epoch_list) - 1):
            eeg_data_trunc.append(eeg_record.epoch_list[index].xn)

        
        shape_channels = np.shape(eeg_data_trunc[0])
        #igualar a quantidade de atributos/amostras
        if( resample_data_same_size ):
            sample: npt.NDArray[np.object_] = np.zeros(shape=(5), dtype=np.object_)
            sample[0] = eeg_data_trunc
            sample[2] = eeg_record.fa
            eeg_data = resample(sample, SFREQ, channels=channels)[0]
            eeg_data = eeg_data[:, :, 0: 801]
            eeg_data_trunc = np.empty(np.shape(eeg_data)[0], dtype=object)
            eeg_data_trunc[:] = list(eeg_data)


        patient = Patient(id_patient = file_name, clinical_features = df_info_patients.loc[f'{file_name}'].copy(), eeg_data = eeg_data_trunc, eeg_record = eeg_record) 
        print(f'{file_name} - {eeg_record.fa} - shape before {shape_before} - shape canais {shape_channels} - shape after {np.shape(eeg_data_trunc[0])}')   

        return patient


    def read_datas_eeg_for_training(self, df_info_patients, df_info_patients_copy, resample_data_same_size=False, channels=Channels.ALL, dir_path_files='./data'):
        #a vida é curta para viver, mas tem tanto tempo para aprender, quanto tempo para se jogar, quanto tempo ate você perceber, que os seus sonhos só dependem de você
        #FBCSP --- braindecode --- para processar o sinal 
        data = {}
        
        for id_patient in df_info_patients.index:    
            #print(f'./data/eeg_{outcome.lower()}/{id_patient}.mat')
            patient = None
            if (self.get_analysis_type() in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.ETIOLOGY)):
                outcome = df_info_patients.at[id_patient, 'Outcome']
                file_path = dir_path_files + f'/eeg_{outcome.lower()}/{id_patient}.mat'
                patient = self.read_mat_file_with_until_30_hz_for_pcp_fm(file_path, df_info_patients_copy, resample_data_same_size, channels)

            else:
                file_path = dir_path_files + f'/data/{id_patient}.pkl'
                patient = self.read_pkl_file(file_path, df_info_patients_copy, resample_data_same_size, channels)

        
            data[f'{id_patient}'] = patient
        
        return data
    

    def extract_statistic_features(self, df):

        result = pd.DataFrame(columns=FEATURES)
        
        
        sample_statistic_features = {'mean':np.round(df[0, :, :].mean(0), 3), 'std':np.round(df[0, :, :].std(0), 3), 
                                'max':np.round(df[0, :, :].max(0), 3), 'min':np.round(df[0, :, :].min(0), 3), 
                                'var':np.round(df[0, :, :].var(0), 3)}


        r = pd.DataFrame(sample_statistic_features).to_numpy().reshape(-1)

        result = pd.concat([result, pd.DataFrame(r.reshape(-1, len(r)), columns=FEATURES)], ignore_index = True)

        return result

    def read_file_with_datas_for_training(self, file_path='./data/patients.xlsx'):

        df_info_patients = pd.read_excel(file_path, header=0, index_col=0)
        df_info_patients.head()
        df_info_patients_copy = df_info_patients.copy()

        if( self.get_analysis_type() == AnalysisType.DEATH_PROGNOSTIC ):
            #prognostico morte
            CLINICAL_FEATURES = ['Gender', 'Etiology', 'Outcome', 'Age']
            df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
            df_info_patients_copy.head()
            df_info_patients_copy[self.get_analysis_type().assumption_label] = np.where(df_info_patients_copy['Outcome'] == 'Alive', 1, 0)
            df_info_patients_copy = df_info_patients_copy.drop('Outcome', axis=1)
            df_info_patients_copy.head()

            # Scaler features
            encoder_cat = OneHotEncoder(cols=['Gender', 'Etiology'], use_cat_names=True)

        elif( self.get_analysis_type() == AnalysisType.ETIOLOGY ):
            #etiologia
            CLINICAL_FEATURES = ['Gender', 'Etiology', 'Age'] #etiologia 
            df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
            df_info_patients_copy.head()
            df_info_patients_copy[self.get_analysis_type().assumption_label].value_counts()
            df_info_patients_copy = df_info_patients_copy.replace({'Post anoxic encephalopathy': 'Other', 'Neoplasia': 'Other',
                                                'Hydroelectrolytic Disorders': 'Other', 'Firearm Injury': 'Other',
                                                'Chronic subdural hematoma': 'Other', 'Hydrocephalus': 'Other',
                                                'Hypoxic encephalopathy': 'Other', 'Neurocysticercosis': 'Other'}, regex=True) 
            df_info_patients_copy[self.get_analysis_type().assumption_label].value_counts()
            # Scaler features
            encoder_cat = OneHotEncoder(cols=['Gender'], use_cat_names=True)

        elif( self.get_analysis_type() == AnalysisType.MUSIC_LIKE ):
            #prognostico morte
            CLINICAL_FEATURES = ['Gender', 'Age', 'Weight',  'Height', 'Profession', 'Handedness', 'Like']
            df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
            df_info_patients_copy.head()
            df_info_patients_copy[self.get_analysis_type().assumption_label] = np.where(df_info_patients_copy['Like'] == 'Like', 1, 0)
            df_info_patients_copy = df_info_patients_copy.drop('Like', axis=1)
            df_info_patients_copy.head()

            # Scaler features
            encoder_cat = OneHotEncoder(cols=['Gender', 'Profession', 'Handedness'], use_cat_names=True)
        

        df_info_patients_copy = encoder_cat.fit_transform(df_info_patients_copy)
        df_info_patients_copy.head()

        return (df_info_patients, df_info_patients_copy)
    

    def transform_data_statistic_and_clinical(self, df_dict, type, assumption_label):
        
        patients_data_signal = []
        patients_data_features = []
        patients_labels = []

        for key in df_dict.keys():

            X_signal = []
            X_features = []
            y = []            
            for epoch in df_dict[key].eeg_data:
                df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
                
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                
                if (type == FeatureType.STATISTIC): 
                    statistical_features = self.extract_statistic_features(epoch_signal)
                    X_features.append(statistical_features.values[0])

                elif (type == FeatureType.STATISTIC_AND_CLINICAL): 
                    statistical_features = self.extract_statistic_features(epoch_signal)
                    X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], statistical_features.values[0])))

                elif ('CLINICAL' in type.name):
                    X_features.append(df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0])            
                
                X_signal.append(epoch_signal)
                y.append(df_clinical_features[assumption_label].values[0])
        
            patients_data_signal.append(X_signal)
            patients_data_features.append(X_features)
            patients_labels.append(y)

            #print(f'{key} - size: {np.shape(patients_data_features)}')
            #print(f'{key} - size: {np.shape(patients_data_signal)}')
            #print(f'{key} - size: {np.shape(patients_labels)}')  
    
        return patients_data_signal, patients_data_features, patients_labels
    

    def transform_data_pcp_fm_coherence(self, df_dict, type, assumption_label):

        patients_data_signal = []
        patients_data_features = []
        patients_labels = []

        for key in df_dict.keys():
            df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()

            if ( not hasattr(df_dict[key].eeg_record.epoch_list[0],'band_frequence_signal_list') ):  
                df_dict[key].eeg_record.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal()
                df_dict[key].eeg_record.compute_pcp()
                df_dict[key].eeg_record.compute_fm()
                df_dict[key].eeg_record.compute_coherence()

            amount_epochs = len(df_dict[key].eeg_record.epoch_list)
            if ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL) ): 
                X_features = [] 
                for epoch in  df_dict[key].eeg_record.epoch_list[0: (amount_epochs-1)]:
                    pcp_epoch_band = [] 
                    if (self.get_band_frequences() == None or  len(self.get_band_frequences()) == 0):
                        [pcp_epoch_band.append(band.pcp) for band in epoch.band_frequence_signal_list]
                    else:
                        [pcp_epoch_band.append(epoch.band_frequence_signal_list[band.position].pcp) for band in self.get_band_frequences()]

                    features = np.ndarray.flatten(np.array(pcp_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    if (type == FeatureType.PCP):
                        X_features.append(features)
                    else:
                        X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))

            elif ( type in (FeatureType.FM, FeatureType.FM_AND_CLINICAL) ):
                X_features = [] 
                for epoch in  df_dict[key].eeg_record.epoch_list[0: (amount_epochs-1)]:
                    fm_epoch_band = [] 
                    [fm_epoch_band.append(band.fm) for band in epoch.band_frequence_signal_list]                   
                    features = np.ndarray.flatten(np.array(fm_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    if (type == FeatureType.FM):
                        X_features.append(features)
                    else:
                        X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))
    
            elif (type == FeatureType.PCP_FM):        
                X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], statistical_features.values[0])))

            elif (type in (FeatureType.COHERENCE, FeatureType.COHERENCE_AND_CLINICAL)):
                
                X_features = [] 
                for epoch in  df_dict[key].eeg_record.epoch_list[0: (amount_epochs-1)]:
                    coherence_epoch_band = epoch._cor_pairs_of_electrodes
                    features = np.ndarray.flatten(np.array(coherence_epoch_band))
                    features[np.isnan(features)] = 0  #removendo not a number do vetor
                    
                    if (type == FeatureType.COHERENCE):
                        X_features.append(features)
                    else:
                        X_features.append(np.concatenate((df_clinical_features.loc[:, df_clinical_features.columns != assumption_label].values[0], features)))


            X_signal = []
            y = []
            for epoch in df_dict[key].eeg_data:                
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                X_signal.append(epoch_signal)
                y.append(df_clinical_features[assumption_label].values[0])
        
            patients_data_signal.append(X_signal)
            patients_data_features.append(X_features)
            patients_labels.append(y)

            #print(f'{key} - size: {np.shape(patients_data_features)}')
            #print(f'{key} - size: {np.shape(patients_data_signal)}')
            #print(f'{key} - size: {np.shape(patients_labels)}')  
        
        return patients_data_signal, patients_data_features, patients_labels
    

    def transform_train_data(self, df_dict, type=FeatureType.CLINICAL, assumption_label='label'):
        
        (patients_data_signal, patients_data_features, patients_labels) = self.transform_test_data(df_dict, type, assumption_label)

        #reduzindo uma dimensão para dados de treino
        patients_data_signal = np.reshape(patients_data_signal, (-1,) + np.shape(patients_data_signal)[2:])
        patients_data_features = np.reshape(patients_data_features, (-1,) + np.shape(patients_data_features)[2:])
        patients_labels = np.reshape(patients_labels, (-1,) + np.shape(patients_labels)[2:])

        return patients_data_signal, patients_data_features, patients_labels


    def transform_test_data(self, df_dict, type=FeatureType.CLINICAL, assumption_label='label'):
        
        if ( type in (FeatureType.STATISTIC, FeatureType.STATISTIC_AND_CLINICAL, FeatureType.CLINICAL, FeatureType.CLINICAL_FRONTAL,
                      FeatureType.CLINICAL_CENTRAL, FeatureType.CLINICAL_OCCIPITAL, FeatureType.CLINICAL_PARIENTAL, 
                      FeatureType.CLINICAL_TEMPORAL, FeatureType.CLINICAL_T3_T4_Pz_O2_Oz ) ):
            
            return self.transform_data_statistic_and_clinical(df_dict, type, assumption_label)


        elif ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, 
                        FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM,
                        FeatureType.COHERENCE, FeatureType.COHERENCE_AND_CLINICAL) ):
            
            return self.transform_data_pcp_fm_coherence(df_dict, type, assumption_label)


class CnnLstmFeatures(ModelAi):

    analysis_type: AnalysisType
    feature_type: FeatureType
    scale_features = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore')

    def __init__(self, analysis_type: AnalysisType, feature_type: FeatureType):
        self.analysis_type = analysis_type
        self.feature_type = feature_type

    def get_analysis_type(self):
        return self.analysis_type
    
    def get_feature_type(self):
        return self.feature_type
    
    def get_band_frequences(self):
        return None

    def get_model(self, datas_for_training):
        X_train_signal = datas_for_training[0]
        X_train_features = datas_for_training[1]
        y_train = datas_for_training[2]
        n_features = np.shape(X_train_features)[1]
        input_B = tf.keras.layers.Input(shape=[n_features], name='categorical_input')
        
        #print(input_B.summary())
        amount_electrodes = np.shape(X_train_signal[0])[2]
        input_A = tf.keras.layers.Input(shape=[None, amount_electrodes], name='timeseries_input')

        hidden1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(input_A)
        hidden2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(hidden1)
        max_pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(hidden2)
        droopout_layer1 = tf.keras.layers.Dropout(0.2)(max_pooling_layer)
        hidden3 = tf.keras.layers.Conv1D(filters=128, kernel_size=13, activation='relu')(droopout_layer1)
        hidden4 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu')(hidden3)
        hidden5 = tf.keras.layers.LSTM(32, return_sequences=True)(hidden4)
        gp = tf.keras.layers.GlobalAveragePooling1D()(hidden5)
        
        #### Atualização do LSTM , usar o transforme https://keras.io/api/keras_nlp/modeling_layers/transformer_encoder/
        ### avaliando os hiperParametros (Refina-los) pesquisar transformers para EEG
        concat = None
        activation_type = None
        loss = None
        model_metrics = None
        if( self.analysis_type in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.MUSIC_LIKE) ):
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


        hidden6 = tf.keras.layers.Dense(64, activation='relu')(concat)
        droopout_layer2 = tf.keras.layers.Dropout(0.2)(hidden6)
        dimension = 1  if (len(np.shape(y_train)) == 1) else np.shape(y_train)[1]
        output = tf.keras.layers.Dense(dimension, activation=activation_type, name='output')(droopout_layer2)

        model = tf.keras.Model(inputs=[input_A, input_B], outputs=[output])
        model.compile( loss=loss, metrics=model_metrics, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4) )

        return model

    def get_data_for_training(self, train_data):
        X_train_signal, X_train_features, y_train = self.transform_train_data(train_data, type=self.feature_type, assumption_label=self.analysis_type.assumption_label)
        print("######################")
        X_train_signal, X_train_features, y_train = shuffle(X_train_signal, X_train_features, y_train, random_state=42)
        X_train_features_array = np.array(X_train_features)
        X_train_features = self.scale_features.fit_transform(X_train_features_array.reshape(-1, X_train_features_array.shape[-1])).reshape(X_train_features_array.shape).tolist()        

        if AnalysisType.ETIOLOGY == self.analysis_type:
            y_train = self.encoder.fit_transform(np.array(y_train)).values.tolist()

        return X_train_signal, X_train_features, y_train
        

    def get_data_for_test(self, test_data):
        X_test_signal, X_test_features, y_test = self.transform_test_data(test_data, type=self.feature_type, assumption_label=self.analysis_type.assumption_label)       
        X_test_features_array = np.array(X_test_features)
        X_test_features = self.scale_features.transform(X_test_features_array.reshape(-1, X_test_features_array.shape[-1])).reshape(X_test_features_array.shape).tolist()

        return X_test_signal, X_test_features, y_test
    

    def data_test_before_generator(self, datas):
        X_signal = datas[0]
        X_features = datas[1]
        y = datas[2]
            
        if AnalysisType.ETIOLOGY == self.analysis_type:
            y_test_enconder = []
            for i in range(0, len(y)):
                y_test_enconder.append(self.encoder.transform(np.array(y[i])).values.tolist())

            y = y_test_enconder

        for i in range(0, len(X_signal)):
            yield (X_signal[i], X_features[i], y[i])            

    def data_generator(self, datas):
        X_signal = datas[0]
        X_features = datas[1]
        y = datas[2]
        while True:
            for i in range(0, len(X_signal)):
                shape = (-1,1)  if (len(np.shape(y)) == 1) else (1, np.shape(y)[1])
                yield ((tf.convert_to_tensor(X_signal[i]), tf.convert_to_tensor(np.array(X_features[i]).reshape(1, -1))), np.array(y[i]).reshape(shape))


class CnnLstm(ModelAi):

    analysis_type: AnalysisType
    feature_type: FeatureType
    encoder = OneHotEncoder(handle_unknown='ignore')

    def __init__(self, analysis_type: AnalysisType, feature_type: FeatureType):
        self.analysis_type = analysis_type
        self.feature_type = feature_type

    def get_analysis_type(self):
        return self.analysis_type
    
    def get_feature_type(self):
        return self.feature_type
    
    def get_band_frequences(self):
        return None

    def get_model(self, datas_for_training):
        X_train_signal = datas_for_training[0]
        y_train = datas_for_training[1]
        amount_electrodes = np.shape(X_train_signal[0])[2]

        layers = [
            tf.keras.layers.Input(shape=[None, amount_electrodes]),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=13, activation='relu'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu'),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ]
        
        if( self.analysis_type in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.MUSIC_LIKE) ):
            layers.append(tf.keras.layers.Flatten())

        layers.append(tf.keras.layers.Dense(64, activation='relu'))
        layers.append(tf.keras.layers.Dropout(0.2))

        loss = None
        model_metrics = None
        if( self.analysis_type in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.MUSIC_LIKE) ):
            layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

            loss = tf.keras.losses.binary_crossentropy
            model_metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]

        else:
            dimension = np.shape(y_train)[1]
            layers.append(tf.keras.layers.Dense(dimension, activation='softmax'))

            loss = tf.keras.losses.CategoricalCrossentropy()
            model_metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]


        model = tf.keras.models.Sequential(layers)

        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=model_metrics
        )

        return model
    
    def get_data_for_training(self, train_data):
        X_train_signal, _, y_train = self.transform_train_data(train_data, type=FeatureType.CLINICAL, assumption_label=self.analysis_type.assumption_label)        
        X_train_signal, y_train = shuffle(X_train_signal, y_train, random_state=42)

        if AnalysisType.ETIOLOGY == self.analysis_type:
            y_train = self.encoder.fit_transform(np.array(y_train)).values.tolist()

        return X_train_signal, y_train

    def get_data_for_test(self, test_data):
        # o underline ignora o valor da variavel
        X_test, _, y_test = self.transform_test_data(test_data, type=FeatureType.CLINICAL, assumption_label=self.analysis_type.assumption_label)
        return X_test, y_test
    
    def data_test_before_generator(self, datas):
        X_signal = datas[0]
        y = datas[1]

        if AnalysisType.ETIOLOGY == self.analysis_type:
            y_test_enconder = []
            for i in range(0, len(y)):
                y_test_enconder.append(self.encoder.transform(np.array(y[i])).values.tolist())

            y = y_test_enconder

        for i in range(0, len(X_signal)):
            yield (X_signal[i], y[i])  


    def data_generator(self, datas):
        X_signal = datas[0]
        y = datas[1]
        while True:
            for i in range(0, len(X_signal)):
                shape = (-1,1)  if (len(np.shape(y)) == 1) else (1, np.shape(y)[1])
                yield (tf.convert_to_tensor(X_signal[i]), np.array(y[i]).reshape(shape))
                

class Mlp(ModelAi):

    analysis_type: AnalysisType
    encoder = OneHotEncoder(handle_unknown='ignore')
    scale_features = StandardScaler()
    feature_type: FeatureType
    band_frequences: List[BandFrequenceEnum]


    def __init__(self, analysis_type: AnalysisType, feature_type: FeatureType, band_frequences: List[BandFrequenceEnum]):
        self.analysis_type = analysis_type
        self.feature_type = feature_type
        self.band_frequences = band_frequences

    def get_analysis_type(self):
        return self.analysis_type
    
    def get_feature_type(self):
        return self.feature_type
    
    def get_band_frequences(self):
        return self.band_frequences

    def get_model(self, datas_for_training):
        X_train_features = datas_for_training[0]
        y_train = datas_for_training[1]
        amount_features = len(X_train_features[0])

        layers = [
            tf.keras.layers.Input(shape=[amount_features]),
            tf.keras.layers.Dense(amount_features, activation='relu'),
        ]

        loss = None
        model_metrics = None
        if( self.analysis_type in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.MUSIC_LIKE) ):
            layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

            loss = tf.keras.losses.binary_crossentropy
            model_metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]

        else:
            dimension = np.shape(y_train)[1]
            layers.append(tf.keras.layers.Dense(dimension, activation='softmax'))

            loss = tf.keras.losses.CategoricalCrossentropy()
            model_metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]


        model = tf.keras.models.Sequential(layers)

        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=model_metrics
        )

        return model
    
    def get_data_for_training(self, train_data):
        _, X_train_features, y_train = self.transform_train_data(train_data, type=self.get_feature_type(), assumption_label=self.analysis_type.assumption_label)        
        X_train_features, y_train = shuffle(X_train_features, y_train, random_state=42)

        if AnalysisType.ETIOLOGY == self.analysis_type:
            y_train = self.encoder.fit_transform(np.array(y_train)).values.tolist()

        X_train_features_array = np.array(X_train_features)
        X_train_features = self.scale_features.fit_transform(X_train_features_array.reshape(-1, X_train_features_array.shape[-1])).reshape(X_train_features_array.shape).tolist()

        return X_train_features, y_train

    def get_data_for_test(self, test_data):
        # o underline ignora o valor da variavel
        _, X_test_features, y_test = self.transform_test_data(test_data, type=self.get_feature_type(), assumption_label=self.analysis_type.assumption_label)

        X_test_features_array = np.array(X_test_features)
        X_test_features = self.scale_features.transform(X_test_features_array.reshape(-1, X_test_features_array.shape[-1])).reshape(X_test_features_array.shape).tolist()

        return X_test_features, y_test
    
    def data_test_before_generator(self, datas):
        X_features = datas[0]
        y = datas[1]

        if AnalysisType.ETIOLOGY == self.analysis_type:
            y_test_enconder = []
            for i in range(0, len(y)):
                y_test_enconder.append(self.encoder.transform(np.array(y[i])).values.tolist())

            y = y_test_enconder

        for i in range(0, len(X_features)):
            yield (X_features[i], y[i])  
    

    def data_generator(self, datas):
        X_features = datas[0]
        y = datas[1]
        while True:
            for i in range(0, len(X_features)):
                shape = (-1,1)  if (len(np.shape(y)) == 1) else (1, np.shape(y)[1])
                yield (tf.convert_to_tensor(np.array(X_features[i]).reshape(1, -1)), np.array(y[i]).reshape(shape))

class AiType(Enum):
    DEATH_PROGNOSTIC_CLINICAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas clinicas)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL))
    DEATH_PROGNOSTIC_STATISTIC_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas estatisticas)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.STATISTIC))
    DEATH_PROGNOSTIC_STATISTIC_AND_CLINICAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas estatisticas e clinicas)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.STATISTIC_AND_CLINICAL))
    DEATH_PROGNOSTIC_PCP_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas de PCP)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.PCP))
    DEATH_PROGNOSTIC_PCP_AND_CLINICAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas de PCP e clinicas)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.PCP_AND_CLINICAL))
    DEATH_PROGNOSTIC_COHERENCE_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas de Coerência)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.COHERENCE))
    DEATH_PROGNOSTIC_COHERENCE_AND_CLINICAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas de Coerência e clinicas)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.COHERENCE_AND_CLINICAL))
    DEATH_PROGNOSTIC_FM_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas de FM)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.FM))
    DEATH_PROGNOSTIC_FM_AND_CLINICAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas de FM e clinicas)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.FM_AND_CLINICAL))
    DEATH_PROGNOSTIC_SIGNAL = ('Cenario para prognostico de Morte (CNN LSTM Somente Pelo sinal Puro)', CnnLstm(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL))
    DEATH_PROGNOSTIC_MLP_CLINICAL = ('Cenario para prognostico de Morte (MLP Caracteristicas Clinicas)', Mlp(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL, None))
    DEATH_PROGNOSTIC_CLINICAL_FRONTAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas clinicas Canais Frontais)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL_FRONTAL))
    DEATH_PROGNOSTIC_CLINICAL_PARIENTAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas clinicas Canais Parientais)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL_PARIENTAL))
    DEATH_PROGNOSTIC_CLINICAL_CENTRAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas clinicas Canais Centrais)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL_CENTRAL))
    DEATH_PROGNOSTIC_CLINICAL_OCCIPITAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas clinicas Canais Occipitais)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL_OCCIPITAL))
    DEATH_PROGNOSTIC_CLINICAL_TEMPORAL_FEATURE = ('Cenario para prognostico de Morte (CNN + LSTM + caracteristicas clinicas Canais Temporais)', CnnLstmFeatures(AnalysisType.DEATH_PROGNOSTIC, FeatureType.CLINICAL_TEMPORAL))
    ETIOLOGY_CLINICAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas clinicas)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.CLINICAL))
    ETIOLOGY_STATISTIC_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas estatisticas)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.STATISTIC))
    ETIOLOGY_STATISTIC_AND_CLINICAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas estatisticas e clinicas)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.STATISTIC_AND_CLINICAL))
    ETIOLOGY_PCP_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas de PCP)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.PCP))
    ETIOLOGY_PCP_AND_CLINICAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas de PCP e clinicas)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.PCP_AND_CLINICAL))
    ETIOLOGY_COHERENCE_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas de Coerencia)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.COHERENCE))
    ETIOLOGY_COHERENCE_AND_CLINICAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas de Coerência e Clinicas)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.COHERENCE_AND_CLINICAL))
    ETIOLOGY_FM_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas de FM)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.FM))
    ETIOLOGY_FM_AND_CLINICAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas de FM e clinicas)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.FM_AND_CLINICAL))
    ETIOLOGY_SIGNAL = ('Cenario para classificação da etiologia (CNN LSTM Somente Pelo sinal Puro)', CnnLstm(AnalysisType.ETIOLOGY, FeatureType.CLINICAL))
    ETIOLOGY_MLP_CLINICAL = ('Cenario para classificação da etiologia (MLP Caracteristicas Clinicas)', Mlp(AnalysisType.ETIOLOGY, FeatureType.CLINICAL, None))
    ETIOLOGY_CLINICAL_FRONTAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas clinicas Canais Frontais)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.CLINICAL_FRONTAL))
    ETIOLOGY_CLINICAL_PARIENTAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas clinicas Canais Parientais)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.CLINICAL_PARIENTAL))
    ETIOLOGY_CLINICAL_CENTRAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas clinicas Canais Centrais)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.CLINICAL_CENTRAL))
    ETIOLOGY_CLINICAL_OCCIPITAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas clinicas Canais Occipitais)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.CLINICAL_OCCIPITAL))
    ETIOLOGY_CLINICAL_TEMPORAL_FEATURE = ('Cenario para classificação da etiologia (CNN + LSTM + caracteristicas clinicas Canais Temporais)', CnnLstmFeatures(AnalysisType.ETIOLOGY, FeatureType.CLINICAL_TEMPORAL))
    MUSIC_LIKE_CLINICAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL))
    MUSIC_LIKE_STATISTIC_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas estatisticas)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.STATISTIC))
    MUSIC_LIKE_STATISTIC_AND_CLINICAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas estatisticas e clinicas)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.STATISTIC_AND_CLINICAL))
    MUSIC_LIKE_PCP_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas de PCP)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.PCP))
    MUSIC_LIKE_PCP_AND_CLINICAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas de PCP e clinicas)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.PCP_AND_CLINICAL))
    MUSIC_LIKE_FM_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas de FM)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.FM))
    MUSIC_LIKE_FM_AND_CLINICAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas de FM e clinicas)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.FM_AND_CLINICAL))
    MUSIC_LIKE_COHERENCE_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas de Coerência)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.COHERENCE))
    MUSIC_LIKE_COHERENCE_AND_CLINICAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas de Coerência e clinicas)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.COHERENCE_AND_CLINICAL))
    MUSIC_LIKE_SIGNAL = ('Cenario para Gosto Musical (CNN LSTM Somente Pelo sinal Puro)', CnnLstm(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL))
    MUSIC_LIKE_MLP_CLINICAL = ('Cenario para Gosto Musical (MLP Caracteristicas Clinicas)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL, None))
    MUSIC_LIKE_MLP_PCP = ('Cenario para Gosto Musical (MLP Caracteristicas PCP)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, None))
    MUSIC_LIKE_MLP_PCP_ALFA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Alfa)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.ALFA]))
    MUSIC_LIKE_MLP_PCP_BETA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Beta)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.BETA]))
    MUSIC_LIKE_MLP_PCP_GAMA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Gama)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.GAMA]))
    MUSIC_LIKE_MLP_PCP_SUPER_GAMA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Super Gama)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.SUPERGAMA]))
    MUSIC_LIKE_MLP_PCP_DELTA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Delta)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.DELTA]))
    MUSIC_LIKE_MLP_PCP_TETA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Teta)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.TETA]))
    MUSIC_LIKE_MLP_PCP_DELTA_TETA_SUPER_GAMA = ('Cenario para Gosto Musical (MLP Caracteristicas PCP Banda Delta, Teta, Super Gama)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.PCP, [BandFrequenceEnum.DELTA, BandFrequenceEnum.TETA, BandFrequenceEnum.SUPERGAMA]))
    MUSIC_LIKE_MLP_STATISTIC = ('Cenario para Gosto Musical (MLP Caracteristicas estatisticas)', Mlp(AnalysisType.MUSIC_LIKE, FeatureType.STATISTIC, None))
    MUSIC_LIKE_CLINICAL_FRONTAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas Canais Frontais)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_FRONTAL))
    MUSIC_LIKE_CLINICAL_PARIENTAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas Canais Parientais)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_PARIENTAL))
    MUSIC_LIKE_CLINICAL_CENTRAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas Canais Centrais)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_CENTRAL))
    MUSIC_LIKE_CLINICAL_OCCIPITAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas Canais Occipitais)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_OCCIPITAL))
    MUSIC_LIKE_CLINICAL_TEMPORAL_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas Canais Temporais)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_TEMPORAL))
    MUSIC_LIKE_CLINICAL_T3_T4_Pz_O2_Oz_FEATURE = ('Cenario para Gosto Musical (CNN + LSTM + caracteristicas clinicas Canais T3 T4 Pz O2 Oz)', CnnLstmFeatures(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_T3_T4_Pz_O2_Oz))
    MUSIC_LIKE_SIGNAL_T3_T4_Pz_O2_Oz = ('Cenario para Gosto Musical (CNN LSTM Somente Pelo sinal Puro T3 T4 Pz O2 Oz)', CnnLstm(AnalysisType.MUSIC_LIKE, FeatureType.CLINICAL_T3_T4_Pz_O2_Oz))

    def __new__(cls, description, model_class):
        obj = object.__new__(cls)
        obj.description = description
        obj.model_class = model_class
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


def resample(sample, sfreq: int, channels=Channels.ALL):
    '''
    Realiza o resampling das amostras considerando o parâmetro sfreq.

    Considera uma amostra com o seguinte shape: (amostra, caminho, frequencia, target).

    Parâmetros:
        sample: amostra.
        sfreq: frequência de amostragem desejada.
    '''
    epochs = sample[0]
    freq = sample[2]

    info = mne.create_info(ch_names=channels.channels_name, sfreq=freq, ch_types='eeg')
    sample = np.copy(sample)
    sample[2] = sfreq
    sample[0] = np.array([mne.io.RawArray(epoch, info=info, verbose='CRITICAL').resample(sfreq, verbose='CRITICAL')._data for epoch in epochs])
    
    return sample


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
    
    channels = Channels.ALL
    if (ai_type.model_class.get_feature_type() != None):
        channels = ai_type.model_class.get_feature_type().channels

    data = ai_type.model_class.read_datas_eeg_for_training(df_info_patients, df_info_patients_copy, channels=channels, resample_data_same_size=resample_data_same_size, dir_path_files=directory_path)

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
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        train_data = {key: data[key] for key in index[train_index]}
        test_data = {key: data[key] for key in index[test_index]}
        
        ######################
        datas_for_training = ai_type.model_class.get_data_for_training(train_data)
        datas_for_test = ai_type.model_class.get_data_for_test(test_data)
        model = ai_type.model_class.get_model(datas_for_training)
        with tf.device('/GPU:0'):    
            model.fit(ai_type.model_class.data_generator(datas_for_training), steps_per_epoch=len(datas_for_training[0]), epochs=30, verbose=1)
        
        y_test_real = []
        y_predict = []
        datas_for_test = ai_type.model_class.data_test_before_generator(datas_for_test)
        for data_test in datas_for_test:          
            *_, y_test = data_test
            if( ai_type.model_class.get_analysis_type() in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.MUSIC_LIKE) ):
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
start_time = time.time()
#./data/patients_without_others.xlsx  ou ./data/patients.xlsx  ou ./data/patients_test.xlsx 
#./data_music/patients.xlsx ./data_music/patients_half.xlsx ./data_music/patients_another_half.xlsx  patients_same_half
process_training_ai(file_path='./data/patients.xlsx', ai_type=AiType.DEATH_PROGNOSTIC_FM_AND_CLINICAL_FEATURE, resample_data_same_size=False, group_fold=True)
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