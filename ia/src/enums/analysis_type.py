from enum import Enum

class AnalysisType(Enum):
    DEATH_PROGNOSTIC = ('Prognostico de Morte para coma', 'label')
    ETIOLOGY = ('Etiologia do Coma', 'Etiology')
    MUSIC_LIKE = ('Apreciação Estimulação Musical', 'label')
    RIGHT_ARM = ('Base de dados BCI competition III Data set IVa ‹motor imagery, small training sets›', 'label')
    DATA_MOTOR_IMAGINARY = ('Base de dados BCI Competition IV Data sets 2a ‹4-class motor imagery›', 'Description')

    def __new__(cls, description, assumption_label):
        obj = object.__new__(cls)
        obj.description = description
        obj.assumption_label = assumption_label
        return obj