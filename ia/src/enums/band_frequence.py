from enum import Enum

class BandFrequence(Enum):   
    DELTA = ('Banda Delta de 1 a 3.5 Hz', 0)
    TETA = ('Banda Teta de 3.5 a 7.5', 1)
    ALFA = ('Banda Alfa de 7.5 a 12.5', 2)
    BETA = ('Banda Beta de 12.5 a 30', 3)
    GAMA = ('Banda Gama de 30 a 80', 4)
    SUPERGAMA = ('Banda Super gama de 80 a 100', 5)
    RUIDO = ('Banda Ruido de 58 a 62', 6)

    def __new__(cls, description, position):
        obj = object.__new__(cls)
        obj.description = description
        obj.position = position
        return obj