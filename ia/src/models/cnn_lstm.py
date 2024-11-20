from .model_ai import ModelAi
from enums import *
import numpy as np
import keras
from sklearn.utils import shuffle
import tensorflow as tf

class CnnLstm(ModelAi):

    analysis_type: AnalysisType
    feature_type: FeatureType

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
            keras.layers.Input(shape=[None, amount_electrodes]),
            keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.2),
            keras.layers.Conv1D(filters=128, kernel_size=13, activation='relu'),
            keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu'),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.GlobalAveragePooling1D(),
        ]
        
        
        if( self.analysis_type in  self.BINARY_ANALYSIS_TYPE):
            layers.append(keras.layers.Flatten())

        layers.append(keras.layers.Dense(64, activation='relu'))
        layers.append(keras.layers.Dropout(0.2))

        loss = None
        model_metrics = None
        if( self.analysis_type in self.BINARY_ANALYSIS_TYPE):
            layers.append(keras.layers.Dense(1, activation='sigmoid'))

            loss = keras.losses.binary_crossentropy
            model_metrics = [keras.metrics.BinaryAccuracy(name='accuracy')]

        else:
            dimension = np.shape(y_train)[1]
            layers.append(keras.layers.Dense(dimension, activation='softmax'))

            loss = keras.losses.CategoricalCrossentropy()
            model_metrics = [keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]


        model = keras.models.Sequential(layers)

        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            metrics=model_metrics
        )

        return model
    
    def get_data_for_training(self, train_data):
        X_train_signal, _, y_train = self.transform_train_data(train_data)        
        X_train_signal, y_train = shuffle(X_train_signal, y_train, random_state=42)

        if self.analysis_type in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY):
            y_train = self.encoder.fit_transform(np.array(y_train)).values.tolist()

        return X_train_signal, y_train

    def get_data_for_test(self, test_data):
        # o underline ignora o valor da variavel
        X_test, _, y_test = self.transform_test_data(test_data)
        return X_test, y_test
    
    def data_test_before_generator(self, datas):
        X_signal = datas[0]
        y = datas[1]

        if self.analysis_type in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY):
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
                
