import numpy as np
import math
import time
from collections import namedtuple
from scipy import signal
import logging as LOG

LOG.basicConfig(level=LOG.DEBUG,
                format="%(asctime)s\t%(levelname)s\t%(message)s")

BandFrequence = namedtuple('BandFrequence', ['name', 'init', 'end'])
BandFrequenceSignal = namedtuple(
    'BandFrequenceSignal', ['band_frequence', 'xn', 'f', 'pcp', 'fm'])
DirectFourierTransform = namedtuple(
    'DirectFourierTransform', ['f', 'XF', 'XM', 'XFF'])


class Epoch:
    """Representa uma epoca separada do exame.
    """

    def __init__(self, start_at, window_length, fs, xn, frequence_band_list):
        """Construtor

            Args:
                start_at (escalar): inicio da epoca em milissegundos
                window_length (escalar): tamanho da janela em milissegundos
                fs (escalar): frequencia de amostragem do exame em Hz
                xn (matrix numpy de escalares): cada linha desta matriz representa os valores de coleta de cada canal na epoca delimitada
                    Matriz de LxTA
                    Onde:
                        L = quantidade de canais EEG do registro  [comumente L=21]
                        TA = tamanho de amostras (Os valores aqui estao em uV e representa a amplitude daquela amostra)
        """
        self.__start_at = start_at
        self.__window_length = window_length
        self.__fs = fs
        self.__xn = xn
        #ANALISAR# validar tamanho da janela com fs e xn
        self.__frequence_band_list = frequence_band_list

    @property
    def T(self):
        """
            T (escalar): tempo de amostragrem que eh igual 1/fs
        """
        return 1/self.__fs
    
    @property
    def window_length(self):
        """
            window_length (escalar): tamanho da janela em milissegundos
        """
        return self.__window_length
    
    @property
    def fs(self):
        """
            fs (escalar): frequencia de amostragem do exame em Hz
        """
        return self.__fs
    
    @property
    def start_at(self):
        """
            start_at (escalar): inicio da epoca em milissegundos
        """
        return self.__start_at
    
    @property
    def xn(self):
        """
            xn (matrix numpy de escalares): cada linha desta matriz representa os valores de coleta de cada canal na epoca delimitada
                    Matriz de LxTA
                    Onde:
                        L = quantidade de canais EEG do registro  [comumente L=21]
                        TA = tamanho de amostras (Os valores aqui estao em uV e representa a amplitude daquela amostra)
        """
        return self.__xn
    
    @xn.setter
    def xn(self, xn):
        self.__xn = xn
    

    @property
    def band_frequence_signal_list(self):
        """
            band_frequence_signal (vetor de BandFrequenceSignal): lista de bandas de frequencia do sinal ('xn', 'f', 'pcp', 'fm')
        """
        return self._band_xm_filtered_normalized_list
    
    @staticmethod
    def auto_correlation(xn, fs, tau_max):
        """auto_correlation

            Args:
                xn (matrix numpy de escalares): cada linha desta matriz representa os valores de coleta de cada canal na epoca delimitada
                        Matriz de LxTA
                        Onde:
                            L = quantidade de canais EEG do registro  [comumente L=21]
                            TA = tamanho de amostras (Os valores aqui estao em uV e representa a amplitude daquela amostra)
                fs (escalar): frequencia de amostragem do exame em Hz
                tau_max (escalar): representa o comprimeto do vetor Rx (quantidade de amostras).
                    Obs: taumax deve ser menor que o comprimento de x (valor de N)
                    (Em 2015-2017 para sinais EEG recomenda-se usar igual a 100) ou fix(1.67*fs)

            Return:
                rx (matrix numpy de escalares):
        """ 
        xn = xn.tolist()
        amount_lines = len(xn)
        amount_columns = len(xn[0])

        if tau_max > amount_columns:
            raise ValueError(
                "Taumax deve ser menor ou igual ao comprimento do vetor de dados!")

        rx = [[0]*tau_max]*amount_lines
        for i in range(amount_lines):
            rx_channel = [0]*tau_max
            rx_channel[0] = (np.power(xn[i], 2).sum()/amount_columns)

            for tau in range(1, tau_max):

                for j in range(amount_columns-tau):
                    rx_channel[tau] += xn[i][j]*xn[i][j+tau]

                #rx_channel[tau] = rx_channel[tau]/(amount_columns - tau)*(1/fs)
                #ANALISAR# Normalização aqui deveria ser o codigo de cima, no matlab isso esta errado o codigo abaixo foi extraido da versão matlab
                rx_channel[tau] = rx_channel[tau]/(amount_columns - tau)*(fs)

            rx_channel = rx_channel / np.amax(rx_channel)
            rx[i] = rx_channel

        return rx

    @staticmethod
    def dft_matriz_v3(rx, fs):
        """dft_matriz_v3
        """

        rx = np.matrix(rx)
        shape_rx = rx.shape
        amount_lines = shape_rx[0]
        amount_columns = shape_rx[1]

        t = np.array([i/fs for i in range(amount_columns)])
        f = [i*fs/amount_columns for i in range(int(amount_columns/2) + 1)]

        XF = []
        XM = []
        XFF = []
        for r in range(amount_lines):
            XF.append([])
            XM.append([])
            XFF.append([])
            for k in range(len(f)):
                C = np.cos(2*np.pi*f[k]*t)
                S = np.sin(2*np.pi*f[k]*t)
                Cx = np.multiply(rx[r, :], C)
                Sx = np.multiply(rx[r, :], S)
                xrf = np.trapz(Cx, dx=1/fs)[0, 0]
                xif = np.trapz(Sx, dx=1/fs)[0, 0]
                XF[r].append(complex(xrf, -xif))
                XM[r].append((xrf**2 + xif**2)**(1/2))
                XFF[r].append(np.arctan2(xif, xrf))

        direct_fourier_transform = DirectFourierTransform(f, XF, XM, XFF)
        return direct_fourier_transform

    @staticmethod
    def filter_high_pass_1_low_pass_freq_max_and_normalize(XM, f, freq_max):
        """signal_filtering
        """

        #aplica filtro passa banda de 1 a freq_max
        xm_filtered = np.matrix(XM)

        for i in range(len(f)):
            if f[i] < 1 or f[i] >= freq_max:
                xm_filtered[:, i] = 0

        #xm_filtered = xm_filtered.tolist()

        xm_filtered_normalized = []
        for i in range(xm_filtered.shape[0]):
            pot = np.trapz(np.power(xm_filtered[i], 2).tolist()[0], f)
            line_normalized = xm_filtered[i]/(pot**(1/2))
            xm_filtered_normalized.append(line_normalized.tolist()[0])

        return xm_filtered_normalized

    """
        %Potencia do sinal normalizado: Teste para validação da normalização do
        %sinal. O resultado da normalização está correto caso todos os valores de
        %potencia da variavel Potencia_sinal_N for igual a 1.
        XQ_N = XM_1_filt_N.^2;
        P_N = zeros(L,1);
        for j=1:L
            P_N(j) = integralMod(XQ_N(j,:),f_teste);
        end
        powerSignalN{1,i}=P_N;
    """

    @staticmethod
    def separate_bands(xn, f, frequence_band_list):
        """separate_bands

        """

        band_xn_list = []

        band_noise = list(filter(lambda band: band.name ==
                                 'ruido', frequence_band_list))
        hasNoise = len(band_noise) > 0

        xn = np.matrix(xn)
        for band in frequence_band_list:
            xn_band = []
            f_band = []
            for freq_index in range(len(f)):
                if f[freq_index] < band.init:
                    continue
                elif f[freq_index] >= band.end:
                    break
                elif hasNoise and band.name != 'ruido' and f[freq_index] >= band_noise[0].init and f[freq_index] < band_noise[0].end:
                    continue
                else:
                    xn_band.append(
                        np.squeeze(np.asarray(xn[:, freq_index].tolist())))
                    f_band.append(f[freq_index])

            xn_band = np.matrix(xn_band).transpose().tolist()
            band_xn_list.append(
                BandFrequenceSignal(band, xn_band, f_band, [], []))

        return band_xn_list

    def compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal_bands(self, tau_max, freq_max, band_frequence_list):
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
        xn = np.matrix(self.__xn)
        mean_xn_line = xn.mean(1)

        deviation_mean_xn = xn - mean_xn_line
        variance_xn = np.power(deviation_mean_xn, 2).sum(1) / (xn[0].size - 1)
        #standard_deviation_epoch = signal_epoch.std(1,ddof=1)
        standard_deviation_xn = np.power(variance_xn, 1/2)

        #ANALISAR# calculo da potencia total deveria ser com o sinal puro? No matlab esta se utilizando o desvio da media
        #epoch._power_signal = np.trapz(np.power(signal_epoch,2),dx=self._T).tolist()
        self._power_signal = np.trapz(
            np.power(standard_deviation_xn, 2), dx=self.T).tolist()
        
        #rms
        #epoch._power_signal = (np.power(signal_epoch,2)/len(signal_epoch)).sum(1)**(1/2)

        #formula da autocorrelação esta correta uma vez Rxx = somatorio( f(x)*f(x+taul) )
        #http://eceweb1.rutgers.edu/~gajic/solmanual/slides/chapter9_CORR.pdf

        #outras referencias diz que eh: somatorio( [f(x)- media(fx)] * [f(x+taul)-media(fx)]) dividido pela varianca de fx
        #https://dsp.stackexchange.com/questions/15658/autocorrelation-function-of-a-discrete-signal

        #A transformada de fourrier inversa da Densidade espectral de potencia é igual a autocorrelação do sinal
        #A transformada de fourrier do sinal vezes a fft do sinal asterixo é igual a densidade espectral de potencia

        #resultado da FFT divergente do esperado para o calculo função pronta
        rx = self.auto_correlation(deviation_mean_xn, self.__fs, tau_max)

        direct_fourier_transform = self.dft_matriz_v3(rx, self.__fs)
        xm_filtered_normalized = self.filter_high_pass_1_low_pass_freq_max_and_normalize(
            direct_fourier_transform.XM, direct_fourier_transform.f, freq_max)

        band_xm_filtered_normalized_list = self.separate_bands(
            xm_filtered_normalized,  direct_fourier_transform.f, band_frequence_list)
        

        return band_xm_filtered_normalized_list


    @staticmethod
    def _process_compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal(epoch, tau_max, freq_max):
        epoch.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal(tau_max, freq_max)

    def compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal(self, tau_max, freq_max):
        self._band_xm_filtered_normalized_list = self.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal_bands(
            tau_max, freq_max, self.__frequence_band_list)

        return self._band_xm_filtered_normalized_list

    def compute_pcp(self):
        """pcp

        """
        if not hasattr(self, '_band_xm_filtered_normalized_list'):
            raise ValueError(
                "band_xm_filtered_normalized_list ainda nao calculada")

        number_channels = len(self._band_xm_filtered_normalized_list[0].xn)
        sum_pcp_channel = np.zeros(number_channels)
        remove_band_list = []
        for i, band_frequence_signal in enumerate(self._band_xm_filtered_normalized_list):
            xn = np.array(band_frequence_signal.xn)
            pot_xd = np.power(xn, 2).tolist()

            pcp = np.squeeze(np.asarray(
                np.trapz(pot_xd, band_frequence_signal.f, axis=1)))
            sum_pcp_channel = sum_pcp_channel + pcp
            self._band_xm_filtered_normalized_list[i] = band_frequence_signal._replace(
                pcp=pcp.tolist())

        for band_remove in remove_band_list:
            self._band_xm_filtered_normalized_list.remove(band_remove)

        #normalizando o PCP
        for i, band_frequence_signal in enumerate(self._band_xm_filtered_normalized_list):
            pcp = np.array(band_frequence_signal.pcp)

            for j, sum_pcp in enumerate(sum_pcp_channel):
                if sum_pcp == 0.0:
                    pcp[j] = pcp[j] * 0
                else:
                    pcp[j] = 100 * pcp[j] / sum_pcp

            self._band_xm_filtered_normalized_list[i] = band_frequence_signal._replace(
                pcp=pcp.tolist())

    def compute_fm(self):
        """fm

        """
        if not hasattr(self, '_band_xm_filtered_normalized_list'):
            raise ValueError(
                "band_xm_filtered_normalized_list ainda nao calculada")

        for i, band_frequence_signal in enumerate(self._band_xm_filtered_normalized_list):

            xn = np.array(band_frequence_signal.xn)
            number_channels = xn.shape[0]

            amount_frequences = len(band_frequence_signal.f)
            median = np.zeros(number_channels)
            for r in range(number_channels):
                s1 = 0
                for q in range(amount_frequences):
                    s1 = s1 + \
                        (xn[r, q]
                         * band_frequence_signal.f[q])

                #ANALISAR# porque se divide pela soma da linha, dissertacao camila pagina 62
                median[r] = s1/sum(xn[r])

            self._band_xm_filtered_normalized_list[i] = band_frequence_signal._replace(
                fm=median.tolist())

    def compute_coherence_side_left_with_right(self, channel_list):
        """coherence

        """
        left_electrodes_name = ['FP1', 'F7',
                                'F3', 'T3', 'C3', 'T5', 'P3', 'O1']

        right_electrodes_name = ['FP2', 'F8',
                                 'F4', 'T4', 'C4', 'T6', 'P4', 'O2']

        n_left = len(left_electrodes_name)
        n_right = len(right_electrodes_name)
        if n_left != n_right:
            raise ValueError(
                "right_electrodes_name possui tamanho diferente de  left_electrodes_name")

        if len(channel_list) != len(self.__xn):
            raise ValueError(
                "channel_list possui tamanho diferente dos canais de xn")

        n_sample = len(self.__xn[0])
        left_side_sample = np.zeros((n_left, n_sample))
        right_side_sample = np.zeros((n_right, n_sample))

        xn = np.array(self.__xn)
        cor_pairs_of_electrodes = []
        frequences_coherence = []
        for i in range(n_left):
            signal_channel_left = xn[self.find_position_channel_name(
                channel_list, left_electrodes_name[i]), :]
            mean_zero_left = signal_channel_left - signal_channel_left.mean()

            signal_channel_right = xn[self.find_position_channel_name(
                channel_list, right_electrodes_name[i]), :]
            mean_zero_rigth = signal_channel_right - signal_channel_right.mean()

            # %   When WINDOW and NOVERLAP are not specified, MSCOHERE divides X into
            # %   eight sections with 50% overlap and windows each section with a Hamming
            # %   window. MSCOHERE computes and averages the periodogram of each section
            # %   to produce the estimate.
            #ANALISAR# a coerencia nao deveria ser relacionada linha por linha e nao coluna por coluna?
            # L = fix(M./4.5);  noverlap = fix(0.5.*L); options.nfft = max(256,2^nextpow2(N));

            #k = (M-noverlap)./(L-noverlap);
            #m= n_sample L = int( n_sample // 4.5 ) noverlap= n/2
            
            L = int( n_sample // 4.5 )
            nfft = max( 256, 2 ** math.ceil( math.log2( L ) ) )
            frequences, coherence = signal.coherence(mean_zero_left, mean_zero_rigth, fs=self.__fs, nfft=nfft, window=signal.get_window('hamming', L, fftbins=False), noverlap=int(L//2))
            # coherence(x,y,'hann',noverlap=3,nperseg=6,fs=1,detrend=False)
            # coherence2 = mlab.cohere(mean_zero_left, mean_zero_rigth, window=np.hamming(L), NFFT=nfft, noverlap=int(L//2), Fs=self.__fs)
            cor_pairs_of_electrodes.append(coherence)
            frequences_coherence.append(frequences)

        self._cor_pairs_of_electrodes = cor_pairs_of_electrodes
        self._frequences_coherence = frequences_coherence

    @staticmethod
    def find_position_channel_name(channel_list, channel_name):

        for i, channel in enumerate(channel_list):
            if channel.name.upper() == channel_name.upper():
                return i

    def validate_channels(self,  freq_max, tau_max):

        #ANALISAR# deveria-se validar o sinal ruido para cada epoca, ou para o exame como um todo.
        frequence_band_list = [BandFrequence('signal', 1, 40),
                               BandFrequence('noise', 58, 62)]

        band_xm_filtered_normalized_list = self.compute_meanZero_powerSignal_RX_TX_Ftest_XM_XF_xmFiltered_XmNorm_xkSignal_bands(
            tau_max, freq_max, frequence_band_list)

        number_channels = len(band_xm_filtered_normalized_list[0].xn)
        max_value_row_list = np.zeros([2, number_channels])

        for i, band_frequence_signal in enumerate(band_xm_filtered_normalized_list):
            xn = np.array(band_frequence_signal.xn)
            xn = np.nan_to_num(xn)
            if (len(xn) != 0):
                max_value_row_list[i] = np.max(xn, 1)

        idx_noise_channel_list = []
        for i in range(number_channels):
            if (max_value_row_list[0, i]/2 <= max_value_row_list[1, i]):
                idx_noise_channel_list.append(i)

        return idx_noise_channel_list

    def set_zeros_in_channel(self, channel_position):

        if channel_position < 0 or channel_position >= len(self.__xn):
            raise ValueError(
                "channel_position não é uma posição valida")

        self.__xn[channel_position] = np.zeros(
            (1, len(self.__xn[channel_position]))).tolist()[0]
