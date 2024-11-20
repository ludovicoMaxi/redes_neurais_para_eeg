from enum import Enum

class Channels(Enum):
    ALL = (['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'],
           [   0,     1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19])
    FRONTAL = (['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8'],
               [   0,     1,    2,    3,    4,    5,    6])
    CENTRAL = (['C3', 'CZ', 'C4'],
               [  8,    9,    10])
    PARIENTAL = (['P3', 'PZ', 'P4'],
                 [ 13,   14,   15])
    OCCIPITAL = (['O1', 'OZ', 'O2'],
                 [ 17,   18,   19])
    TEMPORAL = (['T3', 'T4', 'T5', 'T6'],
                [  7,   11,   12,   16])
    T3_T4_Pz_O2_Oz = (['T3', 'T4', 'PZ', 'OZ', 'O2'],
                      [  7,    11,   14,   18,  19])

    def __new__(cls, channels_name, positions):
        obj = object.__new__(cls)
        obj.channels_name = channels_name
        obj.positions = positions
        return obj