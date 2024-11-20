from .model_ai import ModelAi
from enums import *
import numpy as np
import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List
from sklearn.utils import shuffle

class Mlp(ModelAi):

    analysis_type: AnalysisType
    scale_features = StandardScaler()
    feature_type: FeatureType
    band_frequences: List[BandFrequence]


    def __init__(self, analysis_type: AnalysisType, feature_type: FeatureType, band_frequences: List[BandFrequence]):
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
            keras.layers.Input(shape=[amount_features]),
            keras.layers.Dense(amount_features, activation='relu'),
        ]

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
        _, X_train_features, y_train = self.transform_train_data(train_data)        
        X_train_features, y_train = shuffle(X_train_features, y_train, random_state=42)

        if self.analysis_type in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY):
            y_train = self.encoder.fit_transform(np.array(y_train)).values.tolist()

        X_train_features_array = np.array(X_train_features)
        X_train_features = self.scale_features.fit_transform(X_train_features_array.reshape(-1, X_train_features_array.shape[-1])).reshape(X_train_features_array.shape).tolist()

        return X_train_features, y_train

    def get_data_for_test(self, test_data):
        # o underline ignora o valor da variavel
        _, X_test_features, y_test = self.transform_test_data(test_data)

        X_test_features_array = np.array(X_test_features)
        X_test_features = self.scale_features.transform(X_test_features_array.reshape(-1, X_test_features_array.shape[-1])).reshape(X_test_features_array.shape).tolist()

        return X_test_features, y_test
    
    def data_test_before_generator(self, datas):
        X_features = datas[0]
        y = datas[1]

        if self.analysis_type in (AnalysisType.ETIOLOGY, AnalysisType.DATA_MOTOR_IMAGINARY):
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
