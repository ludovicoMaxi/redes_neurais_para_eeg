import os
import numpy as np
import numpy.typing as npt
import scipy
import mne
import pickle
from abc import ABC, abstractmethod
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP
from EegRecord import EegRecord, Channel
from Epoch import Epoch, BandFrequence as BandFrequenceEeg
from enums import *
from .patient import Patient
from memory_profiler import profile
import pywt


SFREQ = 400

class ModelAi(ABC):

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
    
    BINARY_ANALYSIS_TYPE = (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.MUSIC_LIKE, AnalysisType.RIGHT_ARM)

    encoder = OneHotEncoder(handle_unknown='ignore')

    amount_wavelet_bands = 8
    list_csp = None
    label_encoder_csp = LabelEncoder()

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

    def read_mat_file_with_until_30_hz_for_pcp_fm(self, file_path, df_info_patients, resample_data_same_size=False):
        
        frequence_band_list_30_hz = [
                            BandFrequenceEeg('delta', 1, 3.5),
                            BandFrequenceEeg('teta', 3.5, 7.5),
                            BandFrequenceEeg('alfa', 7.5, 12.5),
                            BandFrequenceEeg('beta', 12.5, 30)
                            ]
                            #BandFrequenceEeg('gama', 30, 80)]
                            #BandFrequenceEeg('supergama', 80, 100),
                            #BandFrequenceEeg('ruido', 58, 62)]

        file_name_with_extension = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        df_eeg = scipy.io.loadmat(file_path)

        fs = df_eeg['fs'][0][0]

        eeg_data_trunc =  df_eeg['epochsRange'][0]
        shape_before = np.shape(eeg_data_trunc[0])
        channels = self.get_feature_type().channels
        #sem igualar a quantidade de atributos/amostras
        for i in range(len(eeg_data_trunc)):
            eeg_data_trunc[i] = eeg_data_trunc[i][channels.positions, :]

        
        shape_channels = np.shape(eeg_data_trunc[0])
        #igualar a quantidade de atributos/amostras
        if( resample_data_same_size ):
            sample: npt.NDArray[np.object_] = np.zeros(shape=(5), dtype=np.object_)
            sample[0] = eeg_data_trunc
            sample[2] = fs
            eeg_data = self.resample(sample, SFREQ, channels=channels)[0]
            eeg_data = eeg_data[:, :, 0: 801]
            eeg_data_trunc = np.empty(np.shape(eeg_data)[0], dtype=object)
            eeg_data_trunc = eeg_data

        fs_epoch = SFREQ if resample_data_same_size else fs
        # TODO criar regrar para quantidade de canais para PCP e FM
        if( self.get_feature_type() in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, 
                                        FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM,
                                        FeatureType.COHERENCE, FeatureType.COHERENCE_AND_CLINICAL) ):
            
            channels_proccess =  [channel for channel in self.CHANNELS_BASE if channel.name in channels.channels_name]
            eeg_record = EegRecord(file_name, channels_proccess, fs_epoch, df_eeg['XN'], customize=False)
            
            for i in range(len(eeg_data_trunc)):
                eeg_record.epoch_list.append(Epoch(df_eeg['epochsTime'][0][i][0], 2, fs_epoch, eeg_data_trunc[i], frequence_band_list_30_hz))
        
        else:
            eeg_record = None

        amount_epochs = len(eeg_data_trunc)
        eeg_data_trunc = eeg_data_trunc[0:(amount_epochs-1)]
        patient = Patient(id_patient = file_name, clinical_features = df_info_patients.loc[f'{file_name}'].copy(), eeg_data = eeg_data_trunc, eeg_record = eeg_record, fs = fs_epoch) 
        print(f'{file_name} - {fs} - shape before {shape_before} - shape canais {shape_channels} - shape after {np.shape(eeg_data_trunc[0])}')   

        return patient
    

    def read_pkl_file(self, file_path, df_info_patients, resample_data_same_size=False):

        file_name_with_extension = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        eeg_record = pickle.load(open(file_path, 'rb'))
        channels = self.get_feature_type().channels
        
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
        fs_epoch = SFREQ if resample_data_same_size else eeg_record.fs
        if( resample_data_same_size ):
            sample: npt.NDArray[np.object_] = np.zeros(shape=(5), dtype=np.object_)
            sample[0] = eeg_data_trunc
            sample[2] = eeg_record.fs
            eeg_data = self.resample(sample, SFREQ, channels=channels)[0]
            eeg_data = eeg_data[:, :, 0: 801]
            eeg_data_trunc = np.empty(np.shape(eeg_data)[0], dtype=object)
            eeg_data_trunc = eeg_data


        patient = Patient(id_patient = file_name, clinical_features = df_info_patients.loc[f'{file_name}'].copy(), eeg_data = eeg_data_trunc, eeg_record = eeg_record, fs = fs_epoch) 
        print(f'{file_name} - {eeg_record.fs} - shape before {shape_before} - shape canais {shape_channels} - shape after {np.shape(eeg_data_trunc[0])}')   

        return patient

    def separe_epoch_mat(self, file_mat, position, file_name_index, window_size, df_info_patients):
        clinical_features = df_info_patients.loc[f'{file_name_index}'].copy()
        fs = file_mat['nfo']['fs'][0][0][0][0]
        channels = self.get_feature_type().channels

        frequence_band = [
                    BandFrequenceEeg('delta', 1, 3.5),
                    BandFrequenceEeg('teta', 3.5, 7.5),
                    BandFrequenceEeg('alfa', 7.5, 12.5),
                    BandFrequenceEeg('beta', 12.5, 30),
                    BandFrequenceEeg('gama', 30, 80)]
                    #BandFrequenceEeg('supergama', 80, 100)]
        
        start = position - window_size
        end = position + window_size
        
        # Verificar limites da matriz
        if start >= 0 and end <= file_mat['cnt'].shape[0]:
            eeg_data_trunc = np.copy(file_mat['cnt'][start:end])
            eeg_data_trunc = [np.transpose(eeg_data_trunc)]

            positions_summary = [0, 4, 13, 15, 17, 19, 21, 49, 51, 53, 55, 57, 86, 88, 90, 92, 94, 111, 112, 113]
            eeg_summary = np.array(eeg_data_trunc)[:, positions_summary, :][0]
            channels_proccess =  [channel for channel in self.CHANNELS_BASE if channel.name in channels.channels_name]
            eeg_record = EegRecord(file_name_index, channels_proccess, fs, eeg_summary, customize=False)
            eeg_record.epoch_list.append(Epoch(eeg_summary, 1, fs, eeg_summary, frequence_band))


        else:
            raise ValueError("Erro: posição fora dos limites da matriz.")

        patient = Patient(id_patient = file_name_index, eeg_data = eeg_data_trunc, eeg_record=eeg_record, fs = fs, clinical_features=clinical_features) 
        print(f'{file_name_index} - {fs} - shape after {np.shape(eeg_data_trunc[0])}')   

        return patient
    

    def separe_epoch_gdf(self, file_gdf, position, init_second, end_second, file_name_index, df_info_patients):
        clinical_features = df_info_patients.loc[f'{file_name_index}'].copy()
        fs = file_gdf.info['sfreq']
        channel_names = file_gdf.info.ch_names[0:22]

        frequence_band = [
                    BandFrequenceEeg('delta', 1, 3.5),
                    BandFrequenceEeg('teta', 3.5, 7.5),
                    BandFrequenceEeg('alfa', 7.5, 12.5),
                    BandFrequenceEeg('beta', 12.5, 30),
                    BandFrequenceEeg('gama', 30, 80)]
                    #BandFrequenceEeg('supergama', 80, 100)]
        
        start = position + int(init_second*fs)
        end = position + int(end_second*fs)
        
        # Verificar limites da matriz
        if start >= 0 and end <= file_gdf.get_data().shape[1]:

            # Copia apenas os dados necessários, evitando manter uma referência ao array original
            eeg_data_trunc = np.copy(file_gdf.get_data()[0:22, start:end])
            eeg_record = EegRecord(file_name_index, channel_names, fs, eeg_data_trunc, customize=False)
            eeg_record.epoch_list.append(Epoch(start, (end_second - init_second), fs, eeg_data_trunc, frequence_band))


        else:
            raise ValueError("Erro: posição fora dos limites da matriz.")

        patient = Patient(id_patient = file_name_index, eeg_data = [eeg_data_trunc], eeg_record=eeg_record, fs = fs, clinical_features=clinical_features) 
        #print(f'{file_name_index} - {fs} - shape after {np.shape(eeg_data_trunc)}')   

        return patient

        ''' 
            anotacoes = raw.annotations
            eventos, descricao_eventos = mne.events_from_annotations(raw)
            
            A matriz de eventos tem a forma (n_eventos, 3), onde:

            * A primeira coluna representa o tempo (em amostras) em que o evento ocorreu.
            * A segunda coluna é usada como um código para diferenciar entre eventos consecutivos (geralmente 0).
            * A terceira coluna é o ID do evento.


            276 0x0114 Idling EEG (eyes open)
            277 0x0115 Idling EEG (eyes closed)
            768 0x0300 Start of a trial
            769 0x0301 Cue onset left (class 1)
            770 0x0302 Cue onset right (class 2)
            771 0x0303 Cue onset foot (class 3)
            772 0x0304 Cue onset tongue (class 4)
            783 0x030F Cue unknown
            1023 0x03FF Rejected trial
            1072 0x0430 Eye movements
            32766 0x7FFE Start of a new run
        '''


    @profile
    def read_datas_eeg_for_training(self, df_info_patients, df_info_patients_copy, resample_data_same_size=False, dir_path_files='./data'):
        #a vida é curta para viver, mas tem tanto tempo para aprender, quanto tempo para se jogar, quanto tempo ate você perceber, que os seus sonhos só dependem de você
        #FBCSP --- braindecode --- para processar o sinal 
        data = {}
        for id_patient in df_info_patients.index:    
            #print(f'./data/eeg_{outcome.lower()}/{id_patient}.mat')
            patient = None
            if (self.get_analysis_type() in (AnalysisType.DEATH_PROGNOSTIC, AnalysisType.ETIOLOGY)):
                outcome = df_info_patients.at[id_patient, 'Outcome']
                file_path = dir_path_files + f'/eeg_{outcome.lower()}/{id_patient}.mat'
                patient = self.read_mat_file_with_until_30_hz_for_pcp_fm(file_path, df_info_patients_copy, resample_data_same_size)

            elif (self.get_analysis_type() == AnalysisType.RIGHT_ARM):
                id_patient_without_sufix = id_patient.rsplit('_', 1)[0]
                file_path = dir_path_files + f'/{id_patient_without_sufix}.mat'
                
                if('id_patient_reference' not in locals()):
                    id_patient_reference = id_patient_without_sufix
                    file_mat = scipy.io.loadmat(file_path)
                
                if id_patient_reference != id_patient_without_sufix :
                    id_patient_reference = id_patient_without_sufix
                    file_mat = scipy.io.loadmat(file_path)

                position = df_info_patients.at[id_patient, 'pos']
                patient = self.separe_epoch_mat(file_mat=file_mat, position=position, file_name_index=id_patient, window_size=100, df_info_patients=df_info_patients_copy)

            elif (self.get_analysis_type() == AnalysisType.DATA_MOTOR_IMAGINARY):
                id_patient_without_sufix = id_patient.rsplit('_', 1)[0]
                file_path = dir_path_files + f'/{id_patient_without_sufix}.gdf'
                
                if('id_patient_reference' not in locals()):
                    id_patient_reference = id_patient_without_sufix
                    file_gdf = self.prepare_gdf_file(file_path)
                
                if id_patient_reference != id_patient_without_sufix :
                    id_patient_reference = id_patient_without_sufix
                    file_gdf = self.prepare_gdf_file(file_path)

                position = df_info_patients.at[id_patient, 'pos']
                patient = self.separe_epoch_gdf(file_gdf=file_gdf, position=position, init_second=1, end_second=4, file_name_index=id_patient, df_info_patients=df_info_patients_copy)

            else:
                file_path = dir_path_files + f'/data/{id_patient}.pkl'
                patient = self.read_pkl_file(file_path, df_info_patients_copy, resample_data_same_size)

        
            data[f'{id_patient}'] = patient
        
        return data
    
    
    def initialize_csp(self):

        type = self.get_feature_type()
        
        if(type in (FeatureType.COMMON_SPATIAL_PATTERNS_4, FeatureType.COMMON_SPATIAL_PATTERNS_10,
                    FeatureType.COMMON_SPATIAL_PATTERNS_20, FeatureType.COMMON_SPATIAL_PATTERNS_AND_CLINICAL,
                    FeatureType.WAVELET_DECOMPOSITION)):
            
            amount_components = None
            match type:
                case FeatureType.COMMON_SPATIAL_PATTERNS_4:
                    amount_components = 4
                case FeatureType.COMMON_SPATIAL_PATTERNS_10:
                    amount_components = 10
                case FeatureType.COMMON_SPATIAL_PATTERNS_20:
                    amount_components = 20
                case FeatureType.COMMON_SPATIAL_PATTERNS_100:
                    amount_components = 100
                case FeatureType.WAVELET_DECOMPOSITION:
                    amount_components = 4

            # Instanciando o CSP
            self.list_csp = [CSP(n_components=amount_components, reg=None, log=True, norm_trace=False) for _ in range(self.amount_wavelet_bands)]


    @staticmethod
    def prepare_gdf_file(file_path_gdf):

        #file_gdf = mne.io.read_raw_edf(file_path_gdf, preload=True)
        file_gdf = mne.io.read_raw_gdf(file_path_gdf, preload=True)

        #Separacao dos dados em outras IA
        file_gdf.filter(7., 35., fir_design='firwin')

        #Remove the EOG channels and pick only desired EEG channels

        file_gdf.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        picks = mne.pick_types(file_gdf.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

        """ # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10})

        events, _ = mne.events_from_annotations(file_gdf)
        tmin, tmax = 1., 4.
        epochs = mne.Epochs(file_gdf, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
        """
        return file_gdf

    @staticmethod
    def wavelet_packet_decomposition(X): 
        coeffs = pywt.WaveletPacket(X,'db4',mode='symmetric',maxlevel=5)
        return coeffs
           
             
    def extract_wavelet_feature_bands(signal, amount_bands=8):
    
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        amount_epochs = np.shape(signal)[0]
        amount_channesl = np.shape(signal)[1]

        C = ModelAi.wavelet_packet_decomposition(signal[0,0,:]) 
        elements_generate_wavelet = len(C['aaaaa'].data)

        bands = np.empty((amount_bands, amount_epochs, amount_channesl, elements_generate_wavelet)) # 8 freq band coefficients are chosen from the range 4-32Hz
        
        for epoch_index in range(amount_epochs):
            for channel_index in range(amount_channesl):
                pos = []
                C = ModelAi.wavelet_packet_decomposition(signal[epoch_index,channel_index,:]) 
                pos = np.append(pos,[node.path for node in C.get_level(5, 'natural')])
                for b in range(1, amount_bands + 1):
                    bands[b-1,epoch_index,channel_index,:] = C[pos[b]].data
            
        return bands    

    def extract_statistic_features(self, df):

        result = pd.DataFrame(columns=self.FEATURES)
        
        
        sample_statistic_features = {'mean':np.round(df[0, :, :].mean(0), 3), 'std':np.round(df[0, :, :].std(0), 3), 
                                'max':np.round(df[0, :, :].max(0), 3), 'min':np.round(df[0, :, :].min(0), 3), 
                                'var':np.round(df[0, :, :].var(0), 3)}


        r = pd.DataFrame(sample_statistic_features).to_numpy().reshape(-1)

        if (len(self.FEATURES) == len(r)):
            result = pd.concat([result, pd.DataFrame(r.reshape(-1, len(r)), columns=self.FEATURES)], ignore_index = True)
        else:
            result = pd.DataFrame(r).transpose()

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

        elif( self.get_analysis_type() == AnalysisType.RIGHT_ARM ):
           #prognostico morte
            CLINICAL_FEATURES = ['Outcome']
            df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
            df_info_patients_copy.head()
            df_info_patients_copy[self.get_analysis_type().assumption_label] = np.where(df_info_patients_copy['Outcome'] == 1, 1, 0)
            df_info_patients_copy = df_info_patients_copy.drop('Outcome', axis=1)
            df_info_patients_copy.head()

        elif( self.get_analysis_type() == AnalysisType.DATA_MOTOR_IMAGINARY ):
            CLINICAL_FEATURES = ['Outcome', 'Description']
            df_info_patients_copy = df_info_patients_copy[CLINICAL_FEATURES]    
            df_info_patients_copy.head()
            df_info_patients_copy[self.get_analysis_type().assumption_label].value_counts()
        

        if(  self.get_analysis_type() not in (AnalysisType.RIGHT_ARM, AnalysisType.DATA_MOTOR_IMAGINARY) ):
            df_info_patients_copy = encoder_cat.fit_transform(df_info_patients_copy)

        df_info_patients_copy.head()

        return (df_info_patients, df_info_patients_copy)
    

    def transform_data_statistic_and_clinical(self, df_dict):
        assumption_label = self.get_analysis_type().assumption_label
        type = self.get_feature_type()
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
    

    def transform_data_pcp_fm_coherence(self, df_dict ):

        assumption_label = self.get_analysis_type().assumption_label
        type = self.get_feature_type()
        patients_data_signal = []
        patients_data_features = []
        patients_labels = []

        for key in df_dict.keys():
            df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()

            if ( not hasattr(df_dict[key].eeg_record.epoch_list[0],'band_frequence_signal_list') ):  
                df_dict[key].eeg_record.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal()
                df_dict[key].eeg_record.compute_pcp()
                df_dict[key].eeg_record.compute_fm()
                #df_dict[key].eeg_record.compute_coherence()

            amount_epochs = len(df_dict[key].eeg_record.epoch_list)
            amount_epochs_iterable = amount_epochs if amount_epochs < 5 else (amount_epochs - 1)
            
            if ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL) ): 
                X_features = [] 
                for epoch in  df_dict[key].eeg_record.epoch_list[0: amount_epochs_iterable]:
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
                for epoch in  df_dict[key].eeg_record.epoch_list[0: amount_epochs_iterable]:
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
                for epoch in  df_dict[key].eeg_record.epoch_list[0: amount_epochs_iterable]:
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
    
    
    def extract_common_spatial_pattern(self, datas, labels, standardize=True):

        datas_mean = datas.mean(0)
        datas_var = np.sqrt(datas.var(0))

        """if standardize:
            X_train -= datas_mean
            X_train /= datas_var
            X_test -= datas_mean
            X_test /= datas_var

        """
                
        if(self.get_analysis_type() in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY)):
            labels_encoded = self.label_encoder_csp.fit_transform(labels)
        else:
            labels_encoded = labels

        # Ajustando o CSP aos dados
        #csp.fit(datas, labels) por causa do fit_tranform ja faz os dois juntos, por isso ficou somente em um lugar
        print('###########################################')
        print(labels)
        print('###########################################')
        # Transformando os dados usando o CSP para extrair as características
        #datas_csp = csp.transform(datas)

        amount_bands = 1
        if(self.get_feature_type() == FeatureType.WAVELET_DECOMPOSITION):
            amount_bands = self.amount_wavelet_bands

        if hasattr(self.list_csp[0], 'filters_'):
            datas_csp = np.concatenate(tuple( self.list_csp[band_index].transform( datas[band_index::amount_bands] ) for band_index in range(amount_bands) ), axis=-1)
        else:
            datas_csp = np.concatenate(tuple( self.list_csp[band_index].fit_transform( datas[band_index::amount_bands],  labels_encoded[band_index::amount_bands]) for band_index in range(amount_bands) ), axis=-1)

        return datas_csp, labels[0::amount_bands]
                    

    def transform_data_common_spatial_pattern(self, df_dict):
        
        patients_data_signal = []
        patients_data_features = []
        patients_labels = []

        assumption_label = self.get_analysis_type().assumption_label

        features = []
        y = []
        for key in df_dict.keys():
            df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
            
                        
            if(self.get_feature_type() == FeatureType.WAVELET_DECOMPOSITION):
                wavelet_feature_band = ModelAi.extract_wavelet_feature_bands(df_dict[key].eeg_data, self.amount_wavelet_bands)
                #reduzindo a dimiensao (x,y,a,b) para (x*y,a,b)
                wavelet_feature_band = np.reshape(wavelet_feature_band, (-1,) + np.shape(wavelet_feature_band)[2:])
                data = np.stack(wavelet_feature_band, axis=0)

            else:
                data = np.stack(df_dict[key].eeg_data, axis=0)


            y = y + [df_clinical_features[assumption_label].values[0]] * len(data)

            #channels = self.get_feature_type().channels
            #info = mne.create_info(ch_names=channels.channels_name, sfreq=df_dict[key].fs, ch_types='eeg')
            #epochs = mne.EpochsArray(data, info)   
            #X = epochs.get_data()  # Obtendo os dados segmentados em épocas
            features.append(data)
            
            X_signal = []
            for epoch in df_dict[key].eeg_data:
                df_clinical_features = pd.DataFrame(df_dict[key].clinical_features).transpose()
                epoch_signal = epoch.reshape(1, epoch.shape[1], epoch.shape[0])
                X_signal.append(epoch_signal)

            
            patients_data_signal.append(X_signal)
                
        
        features = np.concatenate(features, axis=0)
        patients_data_features, y = self.extract_common_spatial_pattern(features, y)

        amount_epochs = len(patients_data_signal[0])
        patients_labels = np.reshape(y, (len(y)//amount_epochs, amount_epochs) + np.shape(y)[1:])
        patients_data_features = np.reshape(patients_data_features, (len(patients_data_features)//amount_epochs, amount_epochs) + np.shape(patients_data_features)[1:])
        return patients_data_signal, patients_data_features, patients_labels
    

    def transform_train_data(self, df_dict):
        (patients_data_signal, patients_data_features, patients_labels) = self.transform_test_data(df_dict)

        #reduzindo uma dimensão para dados de treino
        #patients_data_signal = np.reshape(patients_data_signal, (-1,) + np.shape(patients_data_signal)[2:])
        flattened_matrices = []
        for group in patients_data_signal:
            for matrix in group:
                flattened_matrices.append(matrix)

        patients_data_signal = flattened_matrices
        patients_data_features = np.reshape(patients_data_features, (-1,) + np.shape(patients_data_features)[2:])
        patients_labels = np.reshape(patients_labels, (-1,) + np.shape(patients_labels)[2:])

        return patients_data_signal, patients_data_features, patients_labels


    def transform_test_data(self, df_dict):
        
        type = self.get_feature_type()
        if ( type in (FeatureType.STATISTIC, FeatureType.STATISTIC_AND_CLINICAL, FeatureType.CLINICAL, FeatureType.CLINICAL_FRONTAL,
                      FeatureType.CLINICAL_CENTRAL, FeatureType.CLINICAL_OCCIPITAL, FeatureType.CLINICAL_PARIENTAL, 
                      FeatureType.CLINICAL_TEMPORAL, FeatureType.CLINICAL_T3_T4_Pz_O2_Oz ) ):
            
            return self.transform_data_statistic_and_clinical(df_dict)


        elif ( type in (FeatureType.PCP, FeatureType.PCP_AND_CLINICAL, 
                        FeatureType.FM, FeatureType.FM_AND_CLINICAL, FeatureType.PCP_FM,
                        FeatureType.COHERENCE, FeatureType.COHERENCE_AND_CLINICAL) ):
            
            return self.transform_data_pcp_fm_coherence(df_dict)

        elif ( type in (FeatureType.COMMON_SPATIAL_PATTERNS_4, FeatureType.COMMON_SPATIAL_PATTERNS_10,
                        FeatureType.COMMON_SPATIAL_PATTERNS_20, FeatureType.COMMON_SPATIAL_PATTERNS_AND_CLINICAL) ):
            return self.transform_data_common_spatial_pattern(df_dict)
        
        elif ( type == FeatureType.WAVELET_DECOMPOSITION):
            return self.transform_data_common_spatial_pattern(df_dict)

        

    @staticmethod
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