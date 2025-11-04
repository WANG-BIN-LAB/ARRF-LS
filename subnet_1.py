import os
import numpy as np
import pandas as pd
from scipy.io import savemat

# Clear variables (equivalent to clc, clear in MATLAB)
os.system('cls' if os.name == 'nt' else 'clear')

# Define base path
char_head = '/data2/home/HCP_test/REST'

# Get list of files in the directory (equivalent to dir in MATLAB)
train_lujin = os.listdir(char_head)
# Filter out '.' and '..' (equivalent to starting from index 3 in MATLAB)
m = [name for name in train_lujin if name not in ('.', '..')]

# Load embedding residual data
embedding_residual = np.genfromtxt('/data2/home/c_residual_embedding.csv', delimiter=',')

# Read index from Excel file (equivalent to xlsread)
index_df = pd.read_excel('/data2/home/index_1.xlsx', sheet_name='Sheet1', usecols='A1:A64620')
index = index_df.values.flatten().astype(int)  # Convert to 1D array and ensure integer type
# Note: MATLAB uses 1-based indexing, while Python uses 0-based. If index is 1-based, uncomment the following line:
# index -= 1

print('1.data_prepared finished ')

# Define parameters
num = 297
conditions = [
    'REST',
    'EMOTION',
    'GAMBLING',
    'LANGUAGE',
    'MOTOR',
    'RELATIONAL',
    'SOCIAL',
    'TWMEMORY'
]

# Process each condition
for cond_idx, condition in enumerate(conditions):
    # Calculate index range for current condition
    start_idx = cond_idx * num
    end_idx = (cond_idx + 1) * num
    
    # Create output directory if it doesn't exist
    output_dir = f'/data2/home/data_for_research/C_AE_residual_1/{condition}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample in current condition
    for k in range(num):
        i = start_idx + k
        # Get the embedding residual column (MATLAB is column-major)
        a = embedding_residual[:, i]
        
        # Initialize FC matrix with ones
        FC = np.ones((360, 360))
        
        # Assign values using the index (handle potential out-of-bounds)
        valid_indices = index < FC.size  # Ensure indices are within matrix size
        FC_flat = FC.flatten()
        FC_flat[index[valid_indices]] = a[valid_indices]
        FC = FC_flat.reshape(360, 360)
        
        # Transpose and assign again to make symmetric (equivalent to MATLAB's FC=FC'; FC(index)=a;)
        FC = FC.T
        FC_flat = FC.flatten()
        FC_flat[index[valid_indices]] = a[valid_indices]
        FC = FC_flat.reshape(360, 360)
        
        # Get corresponding filename
        file_name = m[k]
        save_path = os.path.join(output_dir, file_name)
        
        # Save as .mat file
        savemat(save_path, {'FC': FC})

print('2.C_AE reconstruction finished ')
