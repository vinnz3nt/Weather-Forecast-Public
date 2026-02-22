import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

locations = ["Berlin","Copenhagen","Gdansk","Helsinki", "Oslo", "Stockholm", "Sundsvall", "Trondheim", "Visby"]


def plot_missing(location):
    df = pd.read_pickle(f"DataCombined/{locations[0]}_data")
    sns.heatmap(df.isnull(),yticklabels = False, cbar = False,cmap = 'tab20c_r')
    plt.title('Missing Data: Training Set')
    plt.show()

# print(df.head())


if __name__ == "__main__":
    loc = locations[5]
    print(loc)
    # for loc in locations:
    plot_missing(loc)
    # break