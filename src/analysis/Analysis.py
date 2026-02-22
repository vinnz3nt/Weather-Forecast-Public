import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from helper import loc_ind2name, var_ind2name, get_vars, get_locs,get_short_vars
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

locations = get_locs()
variables = get_vars()
short_vars = get_short_vars()

folder = 'DataCombined'
dataset_list = os.listdir(folder)

data= []
for location in locations:
    PATH = os.path.join(folder,f"{location}_data")
    df = pd.read_pickle(PATH)
    df_array = df.values
    df_array = df_array[:,:,None]
    data.append(df_array)
    
X = np.concat(data,axis=-1)
print(X.shape)

def print_zeros():
    for var in [1,6,11,12,13]:
        tot_loc = 0
        for loc in range(X.shape[-1]):
            tot_loc += np.count_nonzero(X[:,var,loc]==0.0)/X.shape[0]
        print(f"{var_ind2name(var)} {100*tot_loc/9:.2f}")
        

def plot_corr():
    total_corr = np.zeros((16,16))
    for x in range(X.shape[-1]):
        total_corr += np.corrcoef(X[:,:,x], rowvar=False)
    norm_corr = total_corr/9
    # print(np.max(total_corr))
    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(
        norm_corr,
        xticklabels=short_vars,
        yticklabels=short_vars,
        # cmap="crest",
        # vmin=-1,
        # vmax=1,
        square=True,
        cbar=True
    )

    ax.set_title("Correlation matrix")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()
    filtered_corr = total_corr * (total_corr != 1.0)
    print(np.argsort(filtered_corr,))


    
def plot_fourier():
    for var in range(len(variables)):
        x = X[:,var,0]
        N = X.shape[0]
        # freq = np.fft.fftfreq(N, d=1.0)
        fft_vals = np.fft.rfft(x)
        freq = np.fft.rfftfreq(N, d=1.0)
        freq = freq[1:]
        power = np.abs(fft_vals[1:])**2
        period_hours = 1 / freq
        # x_fft = np.fft.fft(x)
        # fft_shift = np.fft.fftshift(x_fft)
        plt.figure(figsize=(10, 5))
        plt.plot(period_hours, power)
        for T, label in [
            (24, "Day"),
            (168, "Week"),
            (8760, "Year"),
        ]:
            plt.axvline(T, linestyle="--", alpha=0.7)
            plt.text(T, plt.ylim()[1]*0.8, label, rotation=90)
        plt.xscale("log")
        plt.xlabel("Period (hours)")
        plt.ylabel("Power")
        plt.title(f"{var_ind2name(var)} Power spectrum (period domain)")
        plt.grid(True, which="both")
        plt.show()

def print_metrics():
    for var in range(len(variables)):
        print(f'{var_ind2name(var)} mean: {np.mean(X[:,var,0])}')
        print(f'{var_ind2name(var)} std: {np.std(X[:,var,0])}')
              

# print_zeros()
plot_corr()
# plot_fourier()
# print_metrics()