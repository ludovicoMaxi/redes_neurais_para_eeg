from enum import Enum
from .channels import Channels

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
    COMMON_SPATIAL_PATTERNS_4 = (Channels.ALL, 'Padrões Espaciais Comum 4 Valores')
    COMMON_SPATIAL_PATTERNS_10 = (Channels.ALL, 'Padrões Espaciais Comum 10 Valores')
    COMMON_SPATIAL_PATTERNS_20 = (Channels.ALL, 'Padrões Espaciais Comum 20 Valores')
    COMMON_SPATIAL_PATTERNS_100 = (Channels.ALL, 'Padrões Espaciais Comum 100 Valores')
    COMMON_SPATIAL_PATTERNS_AND_CLINICAL = (Channels.ALL, 'Padrões Espaciais Comum e Clinico')
    WAVELET_DECOMPOSITION = (Channels.ALL, 'Decomposicao Estatistica')

    def __new__(cls, channels, description):
        obj = object.__new__(cls)
        obj.channels = channels
        obj.description = description
        return obj