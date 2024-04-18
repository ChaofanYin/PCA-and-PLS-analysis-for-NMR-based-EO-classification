# ###
# Principal Component Analysis for NMR data
# Author: Chaofan Yin
# Date: 2024-04-18
# Note: 
#     This is part of Supporting Informatioin for the Year 3 Project Report 
#     at the University of Manchester, Department of Chemistry. 
# ###

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os

import warnings
from matplotlib.patches import Ellipse
from scipy.stats import chi2

#plt.style.use('ggplot')

# Define function for PCA. 
# The function takes the following arguments:
# Name: Name of the output files
# PCA_df: Dataframe of NMR data, with samples as columns and variables as rows. 
        #   The first row is the index of sample, from 1 to 32
        #   The second row is the name of sample, e.g., 1TT, 2OR
        #   The third row is the class of sample, e.g. TT, OR
# upper_test_ppm: The upper limit of the range want to be used in ppm
# lower_test_ppm: The lower limit of the range want to be used in ppm
# ppm_upper: The upper limit of the input data in ppm
# ppm_lower: The lower limit of the imput data in ppm
# confidence_level: The confidence level of ellipse in score plot
# plot_cumulative_variance: True-->plot the cumulative variance
# show_name: True-->label the name of sample in the score plot
# subplot: True-->a subplot of expansions in the scoreplot
# weights: True-->plot the first two weight vectors

def plot_pca_and_variance(Name, PCA_df, upper_test_ppm, lower_test_ppm, ppm_upper, ppm_lower, confidence_level=0.9, plot_cumulative_variance=False, show_names=False, subplot=False, weights=False):
    
    # Adjust data frame
    PCA_df_ppm = PCA_df.iloc[int(PCA_df.shape[0]*(ppm_upper-upper_test_ppm)/(ppm_upper-ppm_lower)):int(PCA_df.shape[0]*(ppm_upper-lower_test_ppm)/(ppm_upper-ppm_lower))]

    # Convert data frame to numpy array
    array1 = np.transpose(PCA_df_ppm.values)

    # Centralize the data
    array1_mean = np.mean(array1, axis=0)
    centralised_array = array1 - array1_mean

    # Perform PCA with SVD
    U, S, Vt = np.linalg.svd(centralised_array)

    # Project the data onto the first two principal components
    projected_data = np.dot(centralised_array, Vt.T[:, :2])

    # Initialize legend handles and labels
    legend_handles = []
    legend_labels = []

    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axes object


    #Adjust the position and size of subplot, if needed
    if subplot:
        ax_sub = fig.add_axes([0.6, 0.4, 0.2, 0.2])  # [left, bottom, width, height]

    for sample_class in PCA_df.columns.get_level_values('Class').unique():
        class_name = sample_class.strip()  # Remove leading/trailing spaces
        color = class_colors[class_name]
        class_indices = (PCA_df.columns.get_level_values('Class') == sample_class)
        class_data = projected_data[class_indices]
        
        # Skip ellipse drawing if there are less than 2 samples in the class
        if len(class_data) < 2:
            ax.scatter(class_data[:, 0], class_data[:, 1], c=color, label=class_name)
            continue
        
        # Calculate covariance matrix for the class
        cov_matrix = np.cov(class_data, rowvar=False)
        # Get mean of the class
        class_mean = np.mean(class_data, axis=0)
        
        # Calculate eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Get the angle of rotation for the ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        # Calculate the confidence interval for each eigenvalue
        chi_squared_value = np.sqrt(chi2.ppf(confidence_level, 2))
        # Handle the case where eigenvalues are negative (or NaN)
        if eigenvalues[0] >= 0:
            semi_major_axis = chi_squared_value * np.sqrt(eigenvalues[0])
        else:
            semi_major_axis = 0
            
        if eigenvalues[1] >= 0:
            semi_minor_axis = chi_squared_value * np.sqrt(eigenvalues[1])
        else:
            semi_minor_axis = 0
        
        # Create ellipse
        ellipse = Ellipse(xy=class_mean, width=2*semi_major_axis, height=2*semi_minor_axis,
                          angle=angle, edgecolor=color, facecolor='none')
        ax.add_patch(ellipse)
        
        # Scatter plot of the class data points with names
        for i, (x, y) in enumerate(class_data):
            ax.scatter(x, y, c=color)
            if show_names:
                ax.text(x, y, PCA_df_ppm.columns[class_indices][i][1], fontsize=6, color=color, ha='right', va='baseline')

        # Add to legend
        scatter = ax.scatter([], [], c=color, label=class_name)  # Empty scatter for adding to legend
        legend_handles.append(scatter)
        legend_labels.append(class_name)

        if subplot:
        
            # Scatter plot of the class data points with names for subplot
            for i, (x, y) in enumerate(class_data):
                ax_sub.scatter(x, y, c=color)
                if show_names:
                    ax_sub.text(x, y, PCA_df_ppm.columns[class_indices][i][1], fontsize=6, color=color, ha='right', va='baseline')

            ellipse_sub = Ellipse(xy=class_mean, width=2*semi_major_axis, height=2*semi_minor_axis,
                                angle=angle, edgecolor=color, facecolor='none')
            ax_sub.add_patch(ellipse_sub)
    
    # Get explained variance of each principal component
    explained_variance_ratio = S**2 / np.sum(S**2)

    if subplot:
        # Add rectangle
        rect = plt.Rectangle((rect_x[0], rect_y[0]), rect_x[1]-rect_x[0], rect_y[1]-rect_y[0], linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
        ax.add_patch(rect)

        ax_sub.set_xlim(rect_x[0], rect_x[1])
        ax_sub.set_ylim(rect_y[0], rect_y[1])
        ax_sub.set_xlabel('PC1')
        ax_sub.set_ylabel('PC2')
    
    
    # Add legend
    ax.legend(legend_handles, legend_labels, loc=4)
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
    #ax.set_xlim(-100000, 100000)
    #ax.set_ylim(-20000, 100000)
    #ax.set_title(f'{upper_test_ppm}-{lower_test_ppm} PCA Score Plot with {confidence_level*100:.0f}% Confidence Ellipses')
    
    plt.savefig(f'{Name}_PCA_Score.pdf')
    plt.show()

    if plot_cumulative_variance:
        # Calculate cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance_ratio) * 100

        # Plot cumulative explained variance against the number of principal components
        plt.figure()
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.xticks(range(1, len(cumulative_explained_variance) + 1))
        #plt.grid(True)
        plt.title(f'{upper_test_ppm}-{lower_test_ppm} Cumulative Explained Variance')
        plt.savefig(f'{Name}_PCA_Variance.pdf')
        plt.show()

    if weights:
        # Get the weights of the original features in the principal components
        #weights = Vt.T[:, :2]
        #weights_df = pd.DataFrame(weights, index=PCA_df.columns, columns=['PC1', 'PC2'])
        ppm_interval = np.linspace(upper_test_ppm, lower_test_ppm, num=PCA_df_ppm.shape[0])
        # Plot the weights of the original features in the principal components
        plt.figure()
        for i in range(2):
            plt.plot(ppm_interval, Vt.T[:, i],label='PC{}'.format(i+1))
        #plt.xlabel('\u03B4 / ppm')
        plt.xlabel('Chemical Shift / ppm')
        plt.ylabel('Weight / a.u.')
        #plt.xlim(upper_test_ppm, lower_test_ppm)
        plt.xlim(10.5,0.5)
        #plt.xticks(range(2, 11, 2))
        plt.legend()
        #plt.title(f'{upper_test_ppm}-{lower_test_ppm} Weights')
        #plt.xticks(range(1, int(upper_test_ppm)+1))
        plt.xticks(range(1, 11))
        #plt.grid(axis='y')
        plt.savefig(f'{Name}_PCA_Weights.pdf')
        plt.show()



# Weighting functions: 

def df_divmean(DataFrame):
    return DataFrame.div(DataFrame.mean(axis=1),axis=0)
        
def df_normalise(DataFrame):
    return DataFrame.sub(DataFrame.mean(axis=1),axis=0).div(DataFrame.std(axis=1),axis=0)

def df_filter(DataFrame,limit):
    DataFrame[DataFrame<limit]=0
    return DataFrame

def df_sqrt(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return DataFrame.applymap(np.sqrt)

def df_cbrt(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return DataFrame.applymap(np.cbrt)

def df_log(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    def loge_shift(x):
        return np.log(x+1)
    return DataFrame.applymap(loge_shift)

def df_log10(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    def log10_shift(x):
        return np.log(x+1)
    return DataFrame.applymap(log10_shift)

def df_sigmoid(DataFrame):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return DataFrame.applymap(sigmoid)

def df_remove(DataFrame, unselected):
    # sample_list = DataFrame.columns.to_list()
    # sample_dict = {}
    # for item in sample_list:
    #     key = item[2]  # Third column as the key
    #     value = [x-1 for x in item[0]]  # First column as the value
    #     if key in sample_dict:
    #         sample_dict[key].append(value)
    #     else:
    #         sample_dict[key] = [value]
    
    # columns_to_remove = []
    # for key in unselected:
    #     if key in sample_dict:
    #         columns_to_remove.extend(sample_dict[key])

    DataFrame.drop(DataFrame.columns[unselected], axis=1, inplace=True)
    return DataFrame

# Define colors for each class
class_colors = { 
    
    'PE': '#d62728',  # red    
    'LA': '#1f77b4',  # blue   
    'LG': '#2ca02c',  # green   
    'TT': '#9467bd',  # purple  
    'EU': '#8c564b',  # brown   
    'GE': '#e377c2',  # pink    
    'BE': '#7f7f7f',  # gray    
    'LE': '#ff7f0e',  # orange  
    'OR': '#bcbd22',  # olive 
}

# {
#     'TT': [0, 1, 2, 3],
#     'PE': [4, 5, 6, 7],
#     'OR': [8, 9, 10],
#     'LG': [11, 12, 13, 14],
#     'LE': [15, 16, 17],
#     'LA': [18, 19, 20, 21],
#     'GE': [22, 23, 24],
#     'EU': [25, 26, 27, 28],
#     'BE': [29, 30, 31]
# }

#baseline problem: LE > BE > GE > OR
#overla: TT

# Read the input data file. The format of input file is given above. 
df1 = pd.read_csv('ref+1Hz_0.02_sum.csv',index_col=0, header=[0,1,2])

# Remove the class containing problamtic samples
columns_to_remove = [15, 16, 17, 29, 30, 31, 22, 23, 24, 8, 9, 10]

# Remove the problamtic samples only
#columns_to_remove = [x for x in range(8,32)]

# All samples
columns_to_remove = []

# Define the weighting function, filters, samples
df = (df_filter((df_remove(df1, columns_to_remove)),13))

# Position of subplot
rect_x = [-2500, -1000]
rect_y = [-500,  1500]

# Run
plot_pca_and_variance("10-0_all", df, 
                      10.5, 0.5, 10.5, 0.5, 
                      0.9, 
                      False, False, False, True)


# plot_pca_and_variance(Name, PCA_df, 
#                       upper_test_ppm, lower_test_ppm, ppm_upper, ppm_lower, 
#                       confidence_level=0.9, 
#                       plot_cumulative_variance=True, 
#                       show_names=False, 
#                       subplot=False, 
#                       weights=False)

        
        
        
        