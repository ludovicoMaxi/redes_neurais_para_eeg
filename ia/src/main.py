import os
import tensorflow as tf
#When use TransformerEncode necessario usar o CPU devido ao seguinte erro: GPU MaxPool gradient ops do not yet have a deterministic XLA implementation.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#tf.config.set_visible_devices([], 'GPU')

import time
from enums import *
from process import process_training_ai

start_time = time.time()
#/media/maxi/dados_maxi/workspace/Projeto Coma/data/patients_without_others.xlsx  ou ./data/patients.xlsx  ou ./data/patients_test.xlsx 
#../data_music/data/10_epocas/patients.xlsx ./data_music/patients_half.xlsx ./data_music/patients_another_half.xlsx  patients_same_half
#/media/maxi/dados_maxi/workspace/Projeto Coma/data_rigth_arm/eeg_data.xlsx
#/media/maxi/dados_maxi/workspace/Projeto Coma/data_motor_imaginary/eeg_data.xlsx
#/media/maxi/dados_maxi/workspace/Projeto Coma/data_motor_imaginary/eeg_data_test.xlsx
process_training_ai(file_path='/media/maxi/dados_maxi/workspace/Projeto Coma/data_music/data/10_epocas/patients.xlsx', ai_type=AiType.MUSIC_LIKE_WAVELET_DECOMPOSITION_FEATURE, resample_data_same_size=True, group_fold=True)
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