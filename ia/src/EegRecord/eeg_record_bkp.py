#python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose tk

import logging as LOG
import numpy as np
import ntpath
import datetime
import time
import pickle
from collections import namedtuple
from Epoch import Epoch, BandFrequence
from scipy import io
from tkinter import *
import sys
from pathlib import Path
import multiprocessing
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

#leitura do CSV remover posteriormente
import csv
import os

Channel = namedtuple('Channel', ['name', 'type'])
Variable = namedtuple('Variable', ['name', 'description'])
PrefixTranslate = namedtuple('PrefixTranslate', ['prefix', 'translation'])

LOG.basicConfig(level=LOG.DEBUG, stream=sys.stdout, format="%(asctime)s\t%(levelname)s\t%(message)s")

class EegRecord:
    """Representa um registro/exame de EEG coletado.
    """

    default_variables_description = []
    default_variables_description.append(
        Variable('name', '(string): Nome do registro'))
    default_variables_description.append(Variable(
        'channel_list', '(vetor de namedtuple("Channel", ["name", "type"]) ): canais utilizados\n' +
        '\tContem L elementos (L = quantidade de canais EEG do registro [comumente L=21])\n' +
        '\tname pode ser (FP1,O1,T2, etc),\n' +
        '\ttype pode ser (EEG, EKG, FOTO).\n' +
        '\tPossui  correspondencia direta linha a linha com "xn"'))
    default_variables_description.append(Variable(
        'fs', '(escalar): frequencia de amostragem do exame'))
    default_variables_description.append(Variable(
        'xn', '(ndarray numpy de escalares): cada linha desta matriz representa os valores de coleta de cada canal\n' +
        '\tMatriz de LxN\n' +
        '\tOnde: \n' +
        '\t\tL = quantidade de canais EEG do registro  [comumente L=21]\n' +
        '\t\tN = numero de amostras (Os valores aqui estao em uV e representa a amplitude daquela amostra)'))
    default_variables_description.append(Variable(
        'variables_description', '(vetor de namedtuple("Variable", ["name", "description"])): descricao das variaveis de saida'))
    default_variables_description.append(
        Variable('amount_channels', '(escalar):  numero de canais sendo igual a L, numero de linhas de xn'))
    default_variables_description.append(
        Variable('N', '(escalar): numero de amostras  (valor depende do tempo de coleta do exame)'))
    default_variables_description.append(
        Variable('fs', '(escalar): frequencia de amostragem do exame em Hz'))
    default_variables_description.append(Variable(
        'T', '(escalar): tempo de amostragrem que eh igual 1/fs, periodo de amostragem (baseado na taxa de amostragem) em segundos'))
    default_variables_description.append(Variable('t', '(vetor de escalares): vetor com a linha do tempo de acordo com o numero de amostras e frequencia de amostragem.' +
                                                  '\nTamanho N vetor tempo para plotagem (em segundos) tamanho N'))
    default_variables_description.append(Variable(
        'freq_max', '(escalar): Frequencia de Corte [Hz], salva o filtro passa baixa que deve ser utilizado (100 Normal, 30 Coma)'))
    default_variables_description.append(Variable(
        'created_at', '(datetime): data de criacao do registro no qual o PLG foi convertido'))
    default_variables_description.append(Variable(
        'remaining_xn', '(ndarray de escalares): contem os registros de canais do EEG convertido, mas que nao fazem parte da lista descrita em channel_list'))
    default_variables_description.append(Variable(
        'remaining_channels', '(vetor de namedtuple("Channel", ["name", "type"]) ): canais que estavam presente no exame .PLG mas que nao fazem parte da lista descrita em channel_list'))
    default_variables_description.append(
        Variable('epochs_values', 'Valores das epocas'))
    default_variables_description.append(Variable(
        'epochs_time', '(vetor de escalares): com a representacao dos instantes de tempo associados a cada epoca'))
    default_variables_description.append(Variable('epochs_values_zeros_in_noise_channel',
                                                  'Igual a epochsValues, mas com zeros nos canais que foram considerados ruidosos, tanto no calculo como pelo medico'))
    default_variables_description.append(Variable(
        'PCP', ': valores da contribuicao de potencia de cada ritmo cerebral, para aquela Epoca'))
    default_variables_description.append(
        Variable('power_signal', ': potencia total de cada epoca'))
    default_variables_description.append(
        Variable('FM', ': Frequencia Mediana de cada ritmo cerebral'))
    default_variables_description.append(Variable(
        'COR_pairs_of_electrodes', ': Essas matrizes contem os valores de coerencia calculados'))
    default_variables_description.append(
        Variable('CORRELATION', ': CORRELACAO'))
    default_variables_description.append(Variable(
        'F_cor', ' Vetor de frequencias (em Hz) [tamanho Y], gerado pela funcao do Matlab "mscohere"'))
    default_variables_description.append(
        Variable('frequences', '(vetor de namedtuple("BandFrequence", ["name", "band"]) ): contem as faixas de frequencia utilizadas'))
    default_variables_description.append(Variable('VPC', ''))
    default_variables_description.append(Variable('PSNE', ''))
    default_variables_description.append(Variable('PSNG', ''))

    def __init__(self, name, channel_list, fs, xn, freq_max=100, variables_description=default_variables_description, customize=True):
        """Construtor
            Args:
                name (string): Nome do registro
                channel_list (vetor de namedtuple('Channel', ['name', 'type']) ): canais utilizados
                    Contém L elementos (L = quantidade de canais EEG do registro [comumente L=21])
                    name pode ser (FP1,O1,T2, etc),
                    type pode ser (EEG, EKG, FOTO)
                    Possui  correspondencia direta linha a linha com "xn"
                fs (escalar): frequencia de amostragem do exame em Hz
                xn (ndarray numpy de escalares): cada linha desta matriz representa os valores de coleta de cada canal
                    Matriz de LxN
                    Onde:
                        L = quantidade de canais EEG do registro  [comumente L=21]
                        N = numero de amostras (Os valores aqui estao em uV e representa a amplitude daquela amostra)
                variables_description(vetor de namedtuple('Channel', ['name', 'description'])): descricao das variaveis de saida
                created_at (datetime): data de criacao do registro no qual o PLG foi convertido
                remaining_channel_list (vetor de namedtuple("Channel", ["name", "type"]) ): canais que estavam presente no exame .PLG mas que nao fazem parte da lista descrita em channel_list
                remaining_xn (ndarray de escalares): contem os registros de canais do EEG convertido, mas que nao fazem parte da lista descrita em channel_list
        """
        self.validate_record(channel_list, xn)

        self.__name = name
        self.__channel_list = channel_list
        self.__fs = fs
        self.__xn = xn
        self.__variables_description = variables_description
        self.__created_at = datetime.datetime.now()
        self.__remaining_channel_list = []
        self.__remaining_xn = []
        self.__epoch_list = []
        self.__freq_max = freq_max
        self.__frequence_band_list = [BandFrequence('delta', 1, 3.5),
                                     BandFrequence('teta', 3.5, 7.5),
                                     BandFrequence('alfa', 7.5, 12.5),
                                     BandFrequence('beta', 12.5, 30),
                                     BandFrequence('gama', 30, 80),
                                     BandFrequence('supergama', 80, 100),
                                     BandFrequence('ruido', 58, 62)]

        if customize:
            self.customize_record()

    def __repr__(self):
        return "EegRecord={{name: {}, fs: {}}}".format(self.__name, self.__fs)

    @staticmethod
    def validate_record(channel_list, xn):
        if len(channel_list) != len(xn):
            raise ValueError(
                "Tamanho da lista de canais diferente do tamanho/linhas de Xn.")

        amount_sample = len(xn[0])

        for i, channel_sample in enumerate(xn):
            if len(channel_sample) != amount_sample:
                raise ValueError(
                    "O numero de amostra dos canais estao divergentes. {}".format(channel_list[i]))

    @property
    def variables_description(self):
        """
            variables_description (vetor de namedtuple('Channel', ['name', 'description'])): descricao das variaveis de saida
        """
        return self.__variables_description
    
    @property
    def created_at(self):
        """
            created_at (date):  data de criação
        """
        return self.__created_at
    
    @property
    def amount_channels(self):
        """
            amount_channels (escalar):  numero de canais sendo igual a L
        """
        return len(self.__channel_list)
    
    @property
    def channel_list(self):
        """
            channel_list (vetor de namedtuple("Channel", ["name", "type"]) ): canais utilizados
        """
        return len(self.__channel_list)

    @property
    def N(self):
        """
            N (escalar): numero de amostras (valor depende do tempo de coleta do exame)
        """
        if len(self.__xn) == 0:
            return 0
        else:
            return len(self.__xn[0])

    @property
    def fs(self):
        """
            fs (escalar): frequencia de amostragem do exame em Hz
        """
        return self.__fs

    @property
    def T(self):
        """
            T (escalar): tempo de amostragrem que eh igual 1/fs
        """
        return 1/self.__fs
    
    @property
    def freq_max(self):
        """
            freq_max (freq_max): frequencia maxima adotada para processar o EEG
        """
        return self.__freq_max

    @property
    def xn(self):
        """
            xn (ndarray numpy de escalares): cada linha desta matriz representa os valores de coleta de cada canal
                    Matriz de LxN
                    Onde:
                        L = quantidade de canais EEG do registro  [comumente L=21]
                        N = numero de amostras (Os valores aqui estao em uV e representa a amplitude daquela amostra)
        """
        return self.__xn
    
    @property
    def t(self):
        """
            t (vetor de escalares): (vetor de escalares): vetor com a linha do tempo de acordo com o numero de amostras e frequencia de amostragem. Tamanho N vetor tempo para plotagem (em segundos) tamanho N
        """
        return [i*self.T for i in range(self.N)]

    @property
    def total_time(self):
        """
            total_time (escalar): ultimo valor de t
        """
        return self.N*self.T
    
    @property
    def remaining_channel_list(self):
        """
            remaining_channel_list (vetor de namedtuple("Channel", ["name", "type"]) ): canais que estavam presente no exame .PLG mas que nao fazem parte da lista descrita em channel_list
        """
        return self.__remaining_channel_list
    
    @property
    def remaining_xn(self):
        """
            remaining_xn  (ndarray de escalares): contem os registros de canais do EEG convertido, mas que nao fazem parte da lista descrita em channel_list
        """
        return self.__remaining_xn
    
    @property
    def epoch_list(self):
        """
            __epoch_list  (list de epocas):  lista de epocas do exame, contem os dados do processamento
        """
        return self.__epoch_list

    @staticmethod
    def read_file_plg(path_file, customize=True):
        """Metodo responsavel em ler um arquivo PLG e retornar os dados do Exame.

            Args:
                path_file (string): diretorio no qual o arquivo se encontra
        """
        with open(path_file, "rb") as file_plg:

            header = file_plg.read(1024)
            LOG.debug("Type header is: {}".format(type(header)))
            LOG.debug(header)

            amount_channels = header[3]
            fs = header[10]*256 + header[9]
            T = 1 / fs
            LOG.debug('amount_channels: {}'.format(amount_channels))
            LOG.debug('fs: {}'.format(fs))
            LOG.debug('T: {}'.format(T))

            channel_list = []
            for i in range(amount_channels):
                header_channel = file_plg.read(512)

                channel_name = header_channel[11:16].decode('latin1')
                channel_type = header_channel[41:66].decode('latin1')
                # \x00 na tabela ascii representa um caracter nulo por isso ele eh removido da string
                # channel_type = header_channel[41:66].decode('UTF-8', errors='ignore').replace('\x00', '')
                #channel_type = header_channel[41:66].decode(
                #    'UTF-8').replace('\x9a', '')
                channel_list.append(Channel(channel_name, channel_type))

            LOG.debug('channel_list: {}'.format(channel_list))

            xn = np.fromfile(file_plg, dtype=np.int16)
            #LOG.debug(xn)
            
        N = int(len(xn) / amount_channels)
        xn = np.reshape(xn, (amount_channels, N), order='F')
  
        #Transforma em microvolts
        xn = (xn/10)*(-1)
        #Matriz gerada a patir de um comando necessario para que os valores da
        #matriz dados fiquem em microvolts.
        
        name = ntpath.basename(path_file).split('.')
        name = name[-2] if len(name) > 1 else name[0]
        return EegRecord(name, channel_list, fs, xn, customize=customize)

    def translate_channel_name_by_prefix_comparator(self, prefix_translate_list=[]):
        """translate_channel_name_by_prefix_comparator
            Traduz os nomes dos canais conforme mapeamento passado como parametro

            Args:
                prefix_translate_list  (vetor de namedtuple('PrefixTranslate', ['prefix', 'translation']) ): prefixos a serem traduzidos
        """
        for i in range(len(self.__channel_list)):
            for prefix_translate in prefix_translate_list:
                if self.__channel_list[i].name.startswith(prefix_translate.prefix):
                    self.__channel_list[i] = self.__channel_list[i]._replace(
                        name=prefix_translate.translation)
                    break

    def sort_channels_by_name_list(self, name_list=[]):
        """sort_channels_by_name_list
            Ordena os canais pelo nome de acordo com a lista/ordem passada como parametro

            Args:
                name_list (vetor string) = vetor com a ordem de nomes dos canais que se deseja
        """

        sort_channel_list = []
        sort_xn = []
        for i in range(len(name_list)):
            for j in range(len(self.__channel_list)):
                if name_list[i] == self.__channel_list[j].name:
                    sort_channel_list.append(self.__channel_list[j])
                    sort_xn.append(self.__xn[j])
                    break

        for i in range(len(self.__channel_list)):
            if self.__channel_list[i].name not in name_list:
                sort_channel_list.append(self.__channel_list[i])
                sort_xn.append(self.__xn[i])

        self.__channel_list = sort_channel_list
        self.__xn = sort_xn

    def split_channel_list(self, keep_channel_list=[]):
        """split_channels
            Divide os canais de acordo com a lista passada como parametro, onde os que estiverem na lista
            continuaram em xn e channel_list e os que nao estiverem passaram para as variaveis
            remaining_channel_list e remaining_xn

            Args:
                keep_channel_list (vetor string) = vetor com os canais que devem ser mantidos
        """

        new_channel_list = []
        new_xn = []
        channel_positions_to_keep = []
        for i in range(len(self.__channel_list)):
            if self.__channel_list[i].name in (keep_channel_list):
                new_channel_list.append(self.__channel_list[i])
                new_xn.append(self.__xn[i])
                channel_positions_to_keep.append(i)
            else:
                self.__remaining_channel_list.append(self.__channel_list[i])
                self.__remaining_xn.append(self.__xn[i])

        #ANALISAR# verificar nome de variaveis
        self.__channel_list = new_channel_list
        self.__xn = np.array(new_xn)

        if len(self.__epoch_list) > 0 :
            for epoch in self.__epoch_list:
                epoch.xn = [epoch.xn[i] for i in channel_positions_to_keep]


    def customize_record(self):

        prefix_translate_list = []
        prefix_translate_list.append(PrefixTranslate('F7', 'F7'))
        prefix_translate_list.append(PrefixTranslate('T3', 'T3'))
        prefix_translate_list.append(PrefixTranslate('T5', 'T5'))
        prefix_translate_list.append(PrefixTranslate('Fp1', 'FP1'))
        prefix_translate_list.append(PrefixTranslate('F3', 'F3'))
        prefix_translate_list.append(PrefixTranslate('C3', 'C3'))
        prefix_translate_list.append(PrefixTranslate('P3', 'P3'))
        prefix_translate_list.append(PrefixTranslate('O1', 'O1'))
        prefix_translate_list.append(PrefixTranslate('F8', 'F8'))
        prefix_translate_list.append(PrefixTranslate('T4', 'T4'))
        prefix_translate_list.append(PrefixTranslate('T6', 'T6'))
        prefix_translate_list.append(PrefixTranslate('Fp2', 'FP2'))
        prefix_translate_list.append(PrefixTranslate('F4', 'F4'))
        prefix_translate_list.append(PrefixTranslate('C4', 'C4'))
        prefix_translate_list.append(PrefixTranslate('P4', 'P4'))
        prefix_translate_list.append(PrefixTranslate('O2', 'O2'))
        prefix_translate_list.append(PrefixTranslate('Fz', 'FZ'))
        prefix_translate_list.append(PrefixTranslate('Cz', 'CZ'))
        prefix_translate_list.append(PrefixTranslate('Pz', 'PZ'))
        prefix_translate_list.append(PrefixTranslate('Oz', 'OZ'))
        prefix_translate_list.append(PrefixTranslate('Card', 'ECG'))
        prefix_translate_list.append(PrefixTranslate('FOTO', 'FOTO'))

        self.translate_channel_name_by_prefix_comparator(prefix_translate_list)

        order_name_list = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3',
                           'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2', 'FOTO']
        self.sort_channels_by_name_list(order_name_list)

        #Ajustando o valor do ECG
        for i, channel in enumerate(self.__channel_list):
            if channel.type == 'EKG':
                self.__xn[i] = (-1 * np.array(self.__xn[i]))

        keep_channel_list = [order_name_list[i]
                             for i in range(len(order_name_list)-1)]

        self.split_channel_list(keep_channel_list)

        self.update_frequence_band_list_and_freq_max_according_with_freqMax_and_fa(
            self.__freq_max)

    def epoch_separator_list(self, start_time_epoch_list, window_length):
        """epoch_separator

            Args:
                start_time_epoch_list (lista de escalares): lista do tempo inicial de cada janela em milissegundos
                window_length (escalar): tamanho da janela em milissegundos
        """

        for start_time_epoch in start_time_epoch_list:
            self.epoch_separator(start_time_epoch, window_length)
        

    def epoch_separator(self, start_time_epoch, windown_length):
        """epoch_separator

            Args:
                start_time_epoch (escalar): lista do tempo inicial da janela em milissegundos
                window_length (escalar): tamanho da janela em milissegundos
        """
        start_time = int((start_time_epoch/1000) * self.__fs)
        end_time = start_time + int(windown_length * self.__fs / 1000)

        if start_time >= self.N:
            raise ValueError(
                'O tempo inicial da epoca eh maior que o tempo do registro')
        elif end_time > self.N:
            raise ValueError(
                'O tempo final da epoca eh maior que o tempo do registro')
        
        #Devido a indexação do python iniciar em zero é decrementado um valor
        #ANALISAR# temporario assim para facilitar os testes
        start_time = start_time - 1
        start_time = 0 if start_time <= 0 else start_time  

        xn_epoch = self.__xn[:, start_time:end_time]
        self.__epoch_list.append(
            Epoch(start_time_epoch, windown_length, self.__fs, xn_epoch, self.__frequence_band_list)
        )


    def validate_frequence_band_list(self):
        """validate_frequence_band_list
            Valida se __frequence_band_list possui todos os init maior que o end
        """
        for frequence_band in self.__frequence_band_list:
            if frequence_band.init >= frequence_band.end:
                raise ValueError(
                    'Inicio da frequencia maior que o seu final. {}'.format(frequence_band))

    def update_frequence_band_list_and_freq_max_according_with_freqMax_and_fa(self, freq_max):
        """update_frequence_band_list_and_freq_max_according_with_freqMax_and_fa
            programa que retorna as novas bandas/faixas de acordo com a frequencia maxima do exame, assim como  o nome dessas 
            bandas e a nova frequencia maxima.
            Elimina as bandas/faixas de frequencias que nao se enquadram completamente dentro da freqMax
            Assim se a freqMax esta no meio uma faixa de frequencia o novo valor de freqMax será igual ao valor inicial
            dessa faixa de frequencia. freqMax não pode ser maior do que metade da frequencia de amostragem,
            e tambem não pode ser maior do que 100, caso for é setado em 100;
            As faixas de frequencias que iram permanecer serão as que tiverem o valor final da faixa menor do que freqMax novo

            Args:
                freq_max (escalar) = Filtro passa baixa
        """
        if freq_max > (self.__fs / 2):
            freq_max = int(self.__fs / 2)

        #Limite maximo de freq = 100
        if freq_max > 100:
            freq_max = 100

        for frequence_band in self.__frequence_band_list:
            if (freq_max > frequence_band.init) and (freq_max < frequence_band.end):
                freq_max = frequence_band.init

        new_frequence_band_list = []
        for frequence_band in self.__frequence_band_list:
            if freq_max >= frequence_band.end:
                new_frequence_band_list.append(frequence_band)

        self.__frequence_band_list = new_frequence_band_list
        self.__freq_max = freq_max

    def get_list_name_noise_channel_in_epoch_list(self, error_threshold=0.7):
        """get_list_name_noise_channel_in_epoch_list
            Este programa objetiva validar o sinal EEG conforme o nivel de ruido verificado nos eletrodos.
            Dessa forma o programa calcula, baseado na densidade espectral de potencia, 
            o nível do sinal em porcentagem na faixa 1-40Hz e o nível de ruído na
            faixa 58-62Hz. Para um sinal ser completamente aprovado, os eletrodos devem  
            seguir a condicao que o valor máximo de potencia da faixa
            do ruído seja metade ou menos (errorThreshold) do valor máximo de potencia da faixa do
            sinal.

            Args:
                error_threshold (escalar) = limiar de erro para zerar o canal

            Parametros de saida:
                not_valid_channel_list (vetor de (vetor de namedtuple("Channel", ["name", "type"]) )) = vetor que contem
             	os canais do sinal que não foram validos de acordo com
              	os parametros, ou seja, as posições dos canais que nao foram considerados validos

        """
        tau_max = int(1.67*self.__fs)

        amount_channel_noise_dict = {}
        for epoch in self.__epoch_list:
            list_position_channel_noise = epoch.validate_channels(
                self.__freq_max, tau_max)

            for position_channel_noise in list_position_channel_noise:
                if position_channel_noise in amount_channel_noise_dict:
                    amount_channel_noise_dict[position_channel_noise] = amount_channel_noise_dict[position_channel_noise] + 1
                else:
                    amount_channel_noise_dict[position_channel_noise] = 1

        list_position_noise_channel_epochs = []
        amount_epoch = len(self.__epoch_list)
        for position_channel_noise in amount_channel_noise_dict:
            percentage_noise_in_epochs = amount_channel_noise_dict[
                position_channel_noise] / amount_epoch
            if percentage_noise_in_epochs > (1 - error_threshold):
                list_position_noise_channel_epochs.append(
                    position_channel_noise)

        list_noise_channel_epochs = []
        for position_channel in list_position_noise_channel_epochs:
            list_noise_channel_epochs.append(
                self.__channel_list[position_channel].name)

        return list_noise_channel_epochs

    def compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal(self):
        """compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal
            Este programa objetiva calcular a:
                - Media Zero
                - Potencia do Sinal
                - Rx
                - Tx
                - XM
                - XF
                - fTest
                - XM filtrado
                - XM Normalizado

            Args:

        """
        # Create pool of workers
        #num_cpu = multiprocessing.cpu_count() - 1
        #LOG.info(num_cpu)
        #multiprocessing.set_start_method('spawn')
        #pool = multiprocessing.Pool(processes=num_cpu)

        tau_max = int(1.67*self.__fs)
     
        #threads = [Thread(target=Epoch._process_compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal,
        #                  args=(epoch, tau_max, self.__freq_max)) for epoch in self.__epoch_list]
        #[t.start() for t in threads]
        #[t.join() for t in threads]
        #LOG.info(f'sera. ({self.__epoch_list} segundos)')
        #with ProcessPoolExecutor(num_cpu) as executor:
        #    executor.map(Epoch._process_compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal,
        #                        self.__epoch_list, [tau_max]*len(self.__epoch_list), [self.__freq_max]*len(self.__epoch_list))
 
        for epoch in self.__epoch_list:
            epoch.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal(tau_max, self.__freq_max)
        # Wait until workers complete execution
        #pool.close()
        # shutdown the pool, cancels scheduled tasks, returns when running tasks complete
        #executor.shutdown(wait=True, cancel_futures=True)
        LOG.info(f'caraca. ({self.__epoch_list} segundos)')

    def compute_pcp(self):
        """compute_pcp
            Este programa objetiva calcular a:
                - Media PCP
            Args:

        """
        for epoch in self.__epoch_list:
            epoch.compute_pcp()

    def compute_fm(self):
        """compute_FM
            Este programa objetiva calcular a:
                - Media fm
            Args:

        """
        for epoch in self.__epoch_list:
            epoch.compute_fm()

    def compute_coherence(self):
        """compute_coherence
            Este programa objetiva calcular a:
                - Media coherence
            Args:

        """
        for epoch in self.__epoch_list:
            epoch.compute_coherence_side_left_with_right(self.__channel_list)

        """mean_epoch_list = []
        for i, epoch in enumerate(self.__epoch_list):
            signal_epoch = np.matrix(epoch.__xn)
            mean_epoch_list.append(signal_epoch.mean(1))

            deviation_mean_epoch = signal_epoch - mean_epoch_list[i]
            variance_epoch = np.power(deviation_mean_epoch, 2).sum(
                1) / (signal_epoch[0].size - 1)
            #standard_deviation_epoch = signal_epoch.std(1,ddof=1)
            standard_deviation_epoch = variance_epoch**(1/2)

            #ANALISAR# calculo da potencia total deveria ser com o sinal puro? No matlab esta se utilizando o desvio da media
            #epoch._power_signal = np.trapz(np.power(signal_epoch,2),dx=self._T).tolist()
            epoch._power_signal = np.trapz(
                np.power(deviation_mean_epoch, 2), dx=self.T).tolist()
            #rms
            #epoch._power_signal = (np.power(signal_epoch,2)/len(signal_epoch)).sum(1)**(1/2)

            #formula da autocorrelação esta correta uma vez Rxx = somatorio( f(x)*f(x+taul) )
            #http://eceweb1.rutgers.edu/~gajic/solmanual/slides/chapter9_CORR.pdf

            #outras referencias diz que eh: somatorio( [f(x)- media(fx)] * [f(x+taul)-media(fx)]) dividido pela varianca de fx
            #https://dsp.stackexchange.com/questions/15658/autocorrelation-function-of-a-discrete-signal

            #A transformada de fourrier inversa da Densidade espectral de potencia é igual a autocorrelação do sinal
            #A transformada de fourrier do sinal vezes a fft do sinal asterixo é igual a densidade espectral de potencia

            #resultado da FFT divergente do esperado
            """

    def validate_variables_for_brain_power_variation(self):

        if not hasattr(self, '__epoch_list'):
            raise ValueError(
                "__epoch_list ainda nao calculada")

        for i, epoch in enumerate(self.__epoch_list):

            if not hasattr(epoch, '_band_xm_filtered_normalized_list'):
                raise ValueError(
                    ("band_xm_filtered_normalized_list ainda nao calculada! Epoca[" + str(i) + "]"))

            for j, band in enumerate(epoch._band_xm_filtered_normalized_list):
                if not band.pcp:
                    raise ValueError(
                        ("PCP ainda nao calculada! Epoca[" + str(i) + "] Band[" + str(j) + "]"))

    def brain_power_variation(self):

        self.validate_variables_for_brain_power_variation()

        #Separando todas as epocas por banda de frequencia
        BPC_dict = {}
        for epoch in self.__epoch_list:
            for band in epoch._band_xm_filtered_normalized_list:
                if band.band_frequence.name in BPC_dict:
                    BPC_dict[band.band_frequence.name].append(band.pcp)
                else:
                    BPC_dict[band.band_frequence.name] = [band.pcp]

        #ANALISAR# O calculo esta sendo realizado com a mediano, nao deveria ser a media, uma vez que eh o calculo do desvio padrão
        for band_name in BPC_dict:
            median = np.median(BPC_dict[band_name], 0)
            numerator = np.sum((np.array(BPC_dict[band_name]) - median)**2, 0)
            denominator = len(BPC_dict[band_name]) - 1
            dp_mdn = np.sqrt(numerator/denominator)

            BPC_dict[band_name] = dp_mdn.tolist()

        return BPC_dict


    @staticmethod
    def get_time_in_milliseconds(string_mm_ss):
        """get_time_in_miliseconds
            Função que recebe como entrada uma string no formato:
                - mm:ss
            retornando o tempo representativo em milliseconds
            Args:
                - string_mm_ss: string que representa o tempo no formato mm:ss

        """
        time_minute = int(string_mm_ss.split(':')[0])
        time_second = float(string_mm_ss.split(':')[1])

        time_milliseconds = (time_minute*60 + time_second)*1000
        return time_milliseconds

    @staticmethod
    def proccess_records(abs_path_csv_file):
        """proccess_records
            Este programa objetiva calcular a:
                - Media PCP
            Args:

        """
        start_time = time.time()        

        LOG.info('Iniciando a leitura do arquivo: ' + abs_path_csv_file)
        file_name_path = Path(abs_path_csv_file)
        script_dir = file_name_path.parent

        with open(abs_path_csv_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=';')

            name_column_file = "Nome Arquivo PLG"
            list_eeg_records = []
            for idx_line, row in enumerate(csv_reader):
                if idx_line == 0:
                    LOG.info(f'Column names are {", ".join(row)}')

                file_name_plg = row[name_column_file].rstrip("'")
                file_name_plg = file_name_plg + ".PLG"
                start_time_plg_process = time.time()
                LOG.info(
                    f'(Linha={idx_line+2}) Iniciando processamento do PLG: ' + file_name_plg)

                abs_path_plg_file = os.path.join(
                    script_dir, file_name_plg)

                start_time_step = time.time()
                LOG.info('Realizando Leitura PLG...')
                eeg_record = EegRecord.read_file_plg(abs_path_plg_file)
                LOG.info(type(eeg_record.__xn))
    
                filter_pass_low = row['Filtro Passa Baixa']
                if (not filter_pass_low) == False:
                    eeg_record.__freq_max = int(filter_pass_low)

                LOG.info(
                    f'Fim Leitura PLG. ({time.time() - start_time_step} segundos)')

                start_time_step = time.time()
                LOG.info('Realizando Separação de Epocas...')
                epoch_size = int(row["Duracao Epocas (segundos)"])*1000

                epochs_amount = row["Qtd Epocas"]
                start_time_epoch_list = []
                if ( epochs_amount.upper().startswith('SEQUENCIAL=') ):
                    epochs_amount = int(epochs_amount.split('=')[1].strip())
                    
                    column_epoch_x = ("Ep1")
                    time_epoch = row[column_epoch_x]
                    start_time = EegRecord.get_time_in_milliseconds(time_epoch)
                    start_time_epoch_list = [start_time + epoch_size * i for i in range(epochs_amount)]

                else:
                    epochs_amount = int(epochs_amount)

                    for i in range(epochs_amount):
                        column_epoch_x = ("Ep" + str(i+1))
                        time_epoch = row[column_epoch_x]
                        start_time_epoch_list.append(EegRecord.get_time_in_milliseconds(time_epoch))
                    
                
                eeg_record.epoch_separator_list(start_time_epoch_list, epoch_size)
                LOG.info(f'Fim Separação de Epocas. ({time.time() - start_time_step} segundos)')

                start_time_step = time.time()
                LOG.info('Inicio Validação de canais ruidosos...')
                list_channel_name_noise = eeg_record.get_list_name_noise_channel_in_epoch_list()
                LOG.info(f'Canais ruidoso: {list_channel_name_noise}')
                LOG.info(f'Fim Validação de canais ruidosos. ({time.time() - start_time_step} segundos)')

                channel_noise_doctor = row["Medico Canais Ruidosos"]

                LOG.info(f'Medico Canais Ruidosos: {channel_noise_doctor}')

                if (not channel_noise_doctor) == False:
                    channel_noise_doctor_list = channel_noise_doctor.split(',')
                    list_channel_name_noise.extend(channel_noise_doctor_list)
                    list_channel_name_noise = list(
                        dict.fromkeys(list_channel_name_noise))

                LOG.info(f'Canais ruidoso: {list_channel_name_noise}')

                #zerando canais ruidosos
                for channel_name in list_channel_name_noise:
                    position_channel = Epoch.find_position_channel_name(
                        eeg_record.__channel_list, channel_name)
                    for epoch in eeg_record.__epoch_list:
                        epoch.set_zeros_in_channel(position_channel)

                compute_quantifiers = row["Quantificadores"]
                compute_quantifiers_list = compute_quantifiers.upper().split(",")
                is_for_compute_pcp = 'PCP' in compute_quantifiers_list
                is_for_compute_fm = 'FM' in compute_quantifiers_list
                is_for_compute_coherence = 'COERENCIA' in compute_quantifiers_list

                is_for_compute_todos = ('TODOS' in compute_quantifiers_list) or (
                    not compute_quantifiers)
                if is_for_compute_todos:
                    is_for_compute_pcp = True
                    is_for_compute_fm = True
                    is_for_compute_coherence = True

                if is_for_compute_pcp or is_for_compute_fm:
                    start_time_step = time.time()
                    LOG.info('Inicio Pre-Calculos...')
                    eeg_record.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal()
                    LOG.info(
                        f'Fim Pre-Calculos. ({time.time() - start_time_step} segundos)')

                if is_for_compute_pcp:
                    start_time_step = time.time()
                    LOG.info('Inicio PCP...')
                    eeg_record.compute_pcp()
                    LOG.info(
                        f'Fim PCP. ({time.time() - start_time_step} segundos)')

                if is_for_compute_fm:
                    start_time_step = time.time()
                    LOG.info('Inicio FM...')
                    eeg_record.compute_fm()
                    LOG.info(
                        f'Fim FM. ({time.time() - start_time_step} segundos)')

                if is_for_compute_coherence:
                    start_time_step = time.time()
                    LOG.info('Inicio coherence...')
                    eeg_record.compute_coherence()
                    LOG.info(
                        f'Fim coherence. ({time.time() - start_time_step} segundos)')

                list_eeg_records.append(eeg_record)

                LOG.info(f'(Linha={idx_line+2}) Fim processamento do PLG: ' + file_name_plg +
                         f'\nTempo decorrido={time.time() - start_time_plg_process} segundos')

                name_file_output = os.path.join(
                    script_dir, row["Nome Saida"] + ".pkl")
                filePkl = open(name_file_output, 'wb')
                pickle.dump(eeg_record, filePkl)
                filePkl.close()

                pcp_matlab = []
                fm_matlab = []
                coherence = []
                frequences_coherence = []
                for epoch in eeg_record.epoch_list:
                    pcp_epoch = []
                    fm_epoch = []
                    if hasattr(epoch, '_band_xm_filtered_normalized_list'):
                        for band_frequence_signal in epoch.band_frequence_signal_list:
                
                            if is_for_compute_pcp:
                                pcp_epoch.append(band_frequence_signal.pcp)
                            if is_for_compute_fm:
                                fm_epoch.append(band_frequence_signal.fm)

                    if is_for_compute_pcp:
                        pcp_matlab.append(np.array(pcp_epoch))
                    if is_for_compute_fm:
                        fm_matlab.append(np.array(fm_epoch))
                    if is_for_compute_coherence:
                        coherence.append(epoch._cor_pairs_of_electrodes)
                        frequences_coherence.append(
                            epoch._frequences_coherence)

                abs_path_mat_file = os.path.join(
                    script_dir, row["Nome Saida"] + "p.mat")
                io.savemat(abs_path_mat_file, dict(pcpPython=np.array(
                    pcp_matlab), fmPython=np.array(fm_matlab),
                    coherencePython=np.array(coherence), frequenceCoherence=np.array(frequences_coherence)))
                # io.savemat(abs_path_mat_file, dict(pcpPython=np.array(
                #     pcp_matlab), fmPython=np.array(fm_matlab), coherencePython=np.array(coherence)))
                
                print(
                    f'\tArquivo de entrada: {row[name_column_file]}.plg, limiar de erro utilizado: {row["Limiar de erro"]}, arquivo gerado {row["Nome Saida"]}.pkl.')

            print(f'Processed {idx_line} lines.')

        LOG.info(
            f'Finalizando processamento do arquivo. Tempo decorrido={time.time() - start_time} segundos')
