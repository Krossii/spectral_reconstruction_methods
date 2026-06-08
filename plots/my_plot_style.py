import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import seaborn as sns

def initialize_my_color_setup(fraction_of_pagesize, size_1 = 1, size_2 = 1):
    page_size_in_inches = [size_1*5.3, size_2*5.3] # default 5.3
    #fraction_of_pagesize = 0.6 # default 0.7
    plt.rcParams["figure.figsize"] = (fraction_of_pagesize * page_size_in_inches[0], fraction_of_pagesize * page_size_in_inches[1])
    plt.rcParams["font.size"] = 8
    plt.rcParams["errorbar.capsize"] = 3 # default 5
    plt.rcParams["lines.markersize"] = 3 # default 4
    plt.rcParams["scatter.marker"] = "s" # default "s"
    plt.rcParams["lines.linewidth"] = 1

def set_color_palette_using_keys(pal,vector):
    color_dict = dict()
    my_palette = mpl.colormaps[pal]
    index_vec = np.linspace(0,1,len(vector))
    return {vector[n]:my_palette(index_vec[n]) for n in range(len(vector))}

#def set_color_palette(pal,palette_size):
#    color_dict = dict()
#    my_palette = sns.color_palette(pal,palette_size)
#    return {n:my_palette[n] for n in range(palette_size)}
#
#def set_color_palette_using_keys(pal,vector):
#    color_dict = dict()
#    my_palette = sns.color_palette(pal,len(vector))
#    return {key:my_palette[np.where(vector==key)[0][0]] for key in vector}
