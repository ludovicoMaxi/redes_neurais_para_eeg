import keras
import numpy as np
from enums import *
from keras_nlp.layers import TransformerEncoder
from .cnn_lstm_features import CnnLstmFeatures

class CnnTranformerEncodeFeatures(CnnLstmFeatures):
    
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

        # Definir o Transformer Encoder
        transformer_encoder = TransformerEncoder(num_heads=4, intermediate_dim=64, dropout=0.1)
        # Adicionando o TransformerEncoder no lugar do LSTM
        hidden5 = transformer_encoder(hidden4)

        gp = keras.layers.GlobalAveragePooling1D()(hidden5)
        
        #### Atualização do LSTM , usar o transforme https://keras.io/api/keras_nlp/modeling_layers/transformer_encoder/
        ### avaliando os hiperParametros (Refina-los) pesquisar transformers para EEG
        concat = None
        activation_type = None
        loss = None
        model_metrics = None
        if( self.analysis_type in self.BINARY_ANALYSIS_TYPE):
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
