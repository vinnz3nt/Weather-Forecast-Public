import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr 
import numpy as np
from pathlib import Path
import pickle
import os
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

path = DATA_DIR / "raw" / "stockholm_data"
with open(path, 'rb') as f:
    d = pickle.load(f)

print(d['t2m'].shape)


# data = xr.open_dataset(f"/Users/vincentdahlberg/Documents/My programming folder/WeatherForecast/DataFolder/Copenhagen_zip/2m_dewpoint_temperature_Copenhagen_1979_1981.nc",engine="netcdf4").to_dataframe()
# print(data)
