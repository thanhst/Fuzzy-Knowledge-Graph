from FIS import FIS
from FIS_raw_data import FIS_raw_data

for i in range(10):
    FIS_raw_data(Turn=i)
    FIS(Turn = i)
# FIS_raw_data()
# FIS()