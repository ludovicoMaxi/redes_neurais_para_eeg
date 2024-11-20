from .model_ai import ModelAi
from enums import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import keras
import tensorflow as tf

class CnnLstmFeatures(ModelAi):

    analysis_type: AnalysisType
    feature_type: FeatureType
    scale_features = StandardScaler()

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
        input_B = keras.layers.Input(shape=[n_features], name='categorical_input')
        
        #print(input_B.summary())
        amount_electrodes = np.shape(X_train_signal[0])[2]
        input_A = keras.layers.Input(shape=[None, amount_electrodes], name='timeseries_input')

        hidden1 = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(input_A)
        hidden2 = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(hidden1)
        max_pooling_layer = keras.layers.MaxPooling1D(pool_size=2)(hidden2)
        droopout_layer1 = keras.layers.Dropout(0.2)(max_pooling_layer)
        hidden3 = keras.layers.Conv1D(filters=128, kernel_size=13, activation='relu')(droopout_layer1)
        hidden4 = keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu')(hidden3)
        hidden5 = keras.layers.LSTM(32, return_sequences=True)(hidden4)
        gp = keras.layers.GlobalAveragePooling1D()(hidden5)
        
        #### Atualização do LSTM , usar o transforme https://keras.io/api/keras_nlp/modeling_layers/transformer_encoder/
        ### avaliando os hiperParametros (Refina-los) pesquisar transformers para EEG
        concat = None
        activation_type = None
        loss = None
        model_metrics = None
        if( self.analysis_type in self.BINARY_ANALYSIS_TYPE ):
            flatten_layer = keras.layers.Flatten()(gp)
            concat = keras.layers.concatenate([flatten_layer, input_B])
            activation_type = 'sigmoid'
            loss = keras.losses.binary_crossentropy
            model_metrics = [keras.metrics.BinaryAccuracy(name='accuracy')]
        else:
            concat = keras.layers.concatenate([gp, input_B])
            activation_type = 'softmax'
            loss = keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
            model_metrics=[keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]


        hidden6 = keras.layers.Dense(64, activation='relu')(concat)
        droopout_layer2 = keras.layers.Dropout(0.2)(hidden6)
        dimension = 1  if (len(np.shape(y_train)) == 1) else np.shape(y_train)[1]
        output = keras.layers.Dense(dimension, activation=activation_type, name='output')(droopout_layer2)

        model = keras.Model(inputs=[input_A, input_B], outputs=[output])
        model.compile( loss=loss, metrics=model_metrics, optimizer=keras.optimizers.Adam(learning_rate=1e-4) )

        return model

    def get_data_for_training(self, train_data):
        X_train_signal, X_train_features, y_train = self.transform_train_data(train_data)
        print("######################")
        X_train_signal, X_train_features, y_train = shuffle(X_train_signal, X_train_features, y_train, random_state=42)
        X_train_features_array = np.array(X_train_features)
        X_train_features = self.scale_features.fit_transform(X_train_features_array.reshape(-1, X_train_features_array.shape[-1])).reshape(X_train_features_array.shape).tolist()        

        if self.analysis_type in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY):
            y_train = self.encoder.fit_transform(np.array(y_train)).values.tolist()

        return X_train_signal, X_train_features, y_train
        

    def get_data_for_test(self, test_data):
        X_test_signal, X_test_features, y_test = self.transform_test_data(test_data)       
        X_test_features_array = np.array(X_test_features)
        X_test_features = self.scale_features.transform(X_test_features_array.reshape(-1, X_test_features_array.shape[-1])).reshape(X_test_features_array.shape).tolist()

        return X_test_signal, X_test_features, y_test
    

    def data_test_before_generator(self, datas):
        X_signal = datas[0]
        X_features = datas[1]
        y = datas[2]
            
        if self.analysis_type in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY):
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

