import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

locations = ["Berlin","Copenhagen","Gdansk","Helsinki", "Oslo", "Stockholm", "Sundsvall", "Trondheim", "Visby"]

variables = [
    '2m_temperature',
    'total_precipitation',
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind',
    '100m_u_component_of_wind', 
    '100m_v_component_of_wind',
    'zero_degree_level',
    'surface_pressure',
    'sea_surface_temperature',
    'instantaneous_10m_wind_gust',
    '2m_dewpoint_temperature',
    'mean_sea_level_pressure',
    'low_cloud_cover',
    'medium_cloud_cover',
    'high_cloud_cover',
    'instantaneous_surface_sensible_heat_flux',
    'skin_temperature']

def plot_data(location,variables:list):
    df = pd.read_pickle(f"DataCombined/{locations[0]}_data")
    # print(df)
    data_vec = df[variables].values
    print(data_vec.shape)
    # print(data_vec)
    fig, axs = plt.subplots(len(variables),1)
    
    for row_idx,var in enumerate(variables):
        ax = axs[row_idx]
        ax.plot(data_vec[100:100+365*24,row_idx],label="")
        ax.set_title(f"{var}")
    plt.show()
    
    # plt.plot(data_vec, label=variable)
    # plt.legend()
    # plt.title('Missing Data: Training Set')
    # plt.show()



if __name__ == "__main__":
    loc = locations[5]
    var = ["t2m","sp","lcc","hcc"]
    print(loc)
    
    plot_data(loc,var)
    