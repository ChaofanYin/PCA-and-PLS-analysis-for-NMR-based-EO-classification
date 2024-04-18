# ###
# Partial Least Squares analysis for NMR data
# Author: Chaofan Yin
# Date: 2024-04-18
# Note: 
#     This is part of Supporting Informatioin for the Year 3 Project Report 
#     at the University of Manchester, Department of Chemistry. 
# ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define function for PLS analysis. 
# The function takes the following arguments:
# Name: Name of the output files
# PLS_df: Dataframe of NMR data, with samples as columns and variables as rows. 
        #   The first row is the index of sample, from 1 to 32
        #   The second row is the name of sample, e.g., 1TT, 2OR
        #   The third row is the class of sample, e.g. TT, OR
# PLS_model: R for single column resonse variable, DA for (n_sample, n_class)
# upper_test_ppm: The upper limit of the range want to be used in ppm
# lower_test_ppm: The lower limit of the range want to be used in ppm
# ppm_upper: The upper limit of the input data in ppm
# ppm_lower: The lower limit of the imput data in ppm
# confidence_level: The confidence level of ellipse in score plot
# plot_weight: True-->plot the first two weight vectors
# show_name: True-->label the name of sample in the score plot
def plot_pls_and_variance(Name, PLS_df, PLS_model,
                          upper_test_ppm, lower_test_ppm, 
                          ppm_upper, ppm_lower, 
                          confidence_level=0.9, plot_weight=False, show_names=False):
    
    ppm_range = ppm_upper - ppm_lower
    data = PLS_df.iloc[int(PLS_df.shape[0]*(ppm_upper-upper_test_ppm)
                   /ppm_range):int(PLS_df.shape[0]*(ppm_upper-lower_test_ppm)/ppm_range)]
    
    # Step 2: Prepare data
    X = data.values.T  # NMR data (samples as rows, variables as columns)

    if PLS_model == 'R':
        ######## Create 1D array as response variable

        # Extract class labels from column names
        class_labels = [col[2] for col in data.columns]

        # Create sample indices
        sample_indices = [int(col[0]) for col in data.columns]

        # Create a dictionary to map sample indices to class labels
        sample_class_mapping = {}
        for i, label in zip(sample_indices, class_labels):
            if '-' in str(i):  # Convert i to string before checking if it contains '-'
                start, end = map(int, str(i).split('-'))
                for j in range(start, end + 1):
                    sample_class_mapping[j] = label
            else:
                sample_class_mapping[i] = label

        # Create response variable (class labels) from sample indices
        y = [sample_class_mapping[i] for i in sample_indices]

        # Convert class labels to numerical values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        ########

    elif PLS_model == 'DA':
        ####### Create a (n_samples, n_targets) array as response variable

        sample_list = [col[2] for col in data.columns]

        n_samples = len(sample_list)
        unique_classes = sorted(set(sample_list))
        n_classes = len(unique_classes)

        y_encoded = np.zeros((n_samples, n_classes))

        current_row = 0
        for target in unique_classes:
            occurrences = sample_list.count(target)
            y_encoded[current_row:current_row+occurrences, unique_classes.index(target)] = 1
            current_row += occurrences
        
        #######

    else:
        raise ValueError('PLS_model must be either "R" or "DA"')

    # Step 3: PLS-R model
    plsr = PLSRegression(n_components=2)  # You can adjust the number of components as needed
    plsr.fit(X, y_encoded)
    
    # Step 4: Visualization - Score plot
    
    # Initialize legend handles and labels
    legend_handles = []
    legend_labels = []

    for class_label in data.columns.get_level_values('Class').unique():
        class_name = class_label.strip()  # Remove leading/trailing spaces
        color = colormap[class_name]
        class_indices = (data.columns.get_level_values('Class') == class_label)
        class_data = plsr.x_scores_[class_indices]

        # Skip ellipse drawing if there are less than 2 samples in the class
        if len(class_data) < 2:
            plt.scatter(class_data[:, 0], class_data[:, 1], c=colormap[class_name], label=class_name)
            continue

        # Calculate covariance matrix for the class
        cov_matrix = np.cov(class_data, rowvar=False)
        class_mean = np.mean(class_data, axis=0)
        
        # Calculate eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        # Calculate the confidence interval for each eigenvalue
        chi_squared_value = np.sqrt(chi2.ppf(confidence_level, 2))
        semi_major_axis = chi_squared_value * np.sqrt(eigenvalues[0])
        semi_minor_axis = chi_squared_value * np.sqrt(eigenvalues[1])
        
        # Create ellipse
        ellipse = Ellipse(xy=class_mean, width=2*semi_major_axis, height=2*semi_minor_axis,
                          angle=angle, edgecolor=colormap[class_name], facecolor='none')
        plt.gca().add_patch(ellipse)
        
        # Scatter plot of the class data points
        for i, (x, y) in enumerate(class_data):
            plt.scatter(x, y, c=color)
            if show_names:
                plt.text(x, y, data.columns[class_indices][i][1], fontsize=8, color=color, ha='center', va='bottom')
        # Add to legend
        scatter = plt.scatter([], [], c=color, label=class_name)  # Empty scatter for adding to legend
        legend_handles.append(scatter)
        legend_labels.append(class_name)

    # Add legend
    plt.legend(legend_handles, legend_labels, loc=1)
        
    plt.xlabel('LV1')
    plt.ylabel('LV2')
    #plt.title(f'PLS-{PLS_model} Score Plot with {confidence_level*100:.0f}% Confidence Ellipses')
    
    plt.savefig(f'{Name}_PLS-{PLS_model}_Score.pdf')
    plt.show()
    
    if plot_weight:
        ppm_interval = np.linspace(upper_test_ppm, lower_test_ppm, num=plsr.x_weights_.shape[0])
        
        plt.figure(figsize=(8, 6))
        for i in range(plsr.x_weights_.shape[1]):
            plt.plot(ppm_interval, plsr.x_weights_[:, i],label='LV {}'.format(i+1))
        plt.title(f'PLS-{PLS_model}Weight Matircx')
        plt.xlabel('Chemical Shift / ppm')
        plt.ylabel('Weight / a.u.')
        plt.xlim(10.5, 0.5)
        plt.legend(loc=4)
        #plt.grid(True)
        plt.savefig(f'{Name}_PLS-{PLS_model}_Variance.pdf')
        plt.show()


#Weighting functions: 

def df_normalise(DataFrame):
    return DataFrame.sub(DataFrame.mean(axis=1),axis=0).div(DataFrame.std(axis=1),axis=0)

def df_filter(DataFrame,limit):
    DataFrame[DataFrame<limit]=0.001
    return DataFrame

def df_sqrt(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return DataFrame.applymap(np.sqrt)

def df_sigmoid(DataFrame):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return DataFrame.applymap(sigmoid)

def df_cbrt(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return DataFrame.applymap(np.cbrt)

def df_log(DataFrame):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    def loge_shift(x):
        return np.log(x+1)
    return DataFrame.applymap(loge_shift)

def df_remove(DataFrame, unselected):
    DataFrame.drop(DataFrame.columns[unselected], axis=1, inplace=True)
    return DataFrame    

# Define colors for each class
colormap = { 
    
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


# Read the input data file. The format of input file is given above. 
df = pd.read_csv('ref+1Hz_0.02_sum.csv', index_col=0, header=[0,1,2])


# Remove the class containing problamtic samples
columns_to_remove = [15, 16, 17, 29, 30, 31, 22, 23, 24, 8, 9, 10]

# Remove the problamtic samples only
#columns_to_remove = [x for x in range(8,32)]

# All samples
#columns_to_remove = []

# Define the weighting function, filters, samples
PLS_df = df_sigmoid(df_filter(df_remove(df, columns_to_remove), 13))

#PLS_df = df

# Run
plot_pls_and_variance('10-3_less_sigmoid(13)', PLS_df, 'DA',
                      10.5, 3.0, 
                      10.5, 0.5, 
                      0.90, False, False)
    
    
    

    
    
    
    
    
    
    
    
    
    
    