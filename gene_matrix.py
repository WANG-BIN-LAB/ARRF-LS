import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

# Data loading and preprocessing
sub_num = 969  # Number of samples per state
state = 2  # Number of states
data = ['REST1', 'GAMBLING']
xTrain_0 = np.zeros((64620, 1))  # Store functional connectivity data of all states and samples
char_head = '/home/data2/data'
index = pd.read_excel('/home/data2/index.xlsx').values.flatten()  # Extract indices
index = index.astype(int)  # Ensure indices are integer type
m = 0  # Python uses 0-based indexing

# Extract lower triangular elements  rows
for i in range(state):
    char = data[i]  # Select state name
    char_new = os.path.join(char_head, char)  # Construct state folder path
    train_files = os.listdir(char_new)  # Get all files under this state
    for file_name in train_files[2:]:  # Iterate through files
        file_path = os.path.join(char_new, file_name)
        data_dict = loadmat(file_path)  # Load functional connectivity matrix FC
        R_train = data_dict['FC']  
        
        # Get linear indices of lower triangular elements
        lower_tri_indices = np.tril_indices(360, -1)  # Get row and column indices of lower triangle
        lower_tri_values = R_train[lower_tri_indices]  # Extract lower triangular elements

        # Extract required lower triangular elements
        R_train_values = lower_tri_values[index]  # Use indices directly

        # Store in xTrain_0 matrix
        xTrain_0[:, m] = R_train_values.flatten()  # Store in xTrain_0 matrix
        m += 1

# Save matrix xTrain_0 directly as a .csv file
np.savetxt('/home/data2/1.csv', xTrain_0, delimiter=',')
print('Identification finished')
