import seaborn as sns
import matplotlib.pyplot as plt

# Custom Style Configuration
def set_custom_style():
    sns.set_style('whitegrid')
    sns.set_context('paper')

    custom_params = {
        "axes.spines.right": False, 
        "axes.spines.top": False
    }
    sns.set_theme(style="ticks", rc=custom_params)

    plt.rcParams.update({
        'font.size': 6,
        'axes.linewidth': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'legend.fontsize': 5,
        'legend.title_fontsize': 5,
        'legend.frameon': False,
        'legend.markerscale': 0.5,
        'xtick.major.size': 1,
        'ytick.major.size': 1,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.pad': 1,
        'ytick.major.pad': 1,
    })