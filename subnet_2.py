import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

os.system('cls' if os.name == 'nt' else 'clear')

# Read network division index (S: network division corresponding to 360 nodes)
# Note: Excel rows A2:A361 correspond to 360 elements (indices 0-359 in Python)
S_df = pd.read_excel(
    '/data2/home/data_for_research/net_divide.xlsx',
    sheet_name='sheet1',
    usecols='A2:A361'  # Read rows 2 to 361 (360 elements)
)
S = S_df.values.flatten()  # Convert to 1D array (shape: (360,))
# Convert to 0-based index (since MATLAB uses 1-based, adjust if necessary)
S = S - 1  # Uncomment this line if the original data is 1-based


# Define the emotion data directory
emotion_dir = '/data2/home/data_for_research/C_AE_residual_1/EMOTION/'
# Get all files in the directory (filter out '.' and '..' which are index 0 and 1 in MATLAB)
emotion_files = [f for f in os.listdir(emotion_dir) if f not in ('.', '..')]

# Define sub-network output directories and their index ranges (start, end)
# Note: All ranges are 0-based (adjusted from MATLAB's 1-based)
subnet_info = [
    ('Primary Visual', 0, 5),           
    ('Secondary Visual', 6, 59),        
    ('Somatomotor', 60, 98),           
    ('Cingulo-Opercular', 99, 154),    
    ('Dorsal-attention', 155, 177),    
    ('Language', 178, 200),             
    ('Frontoparietal', 201, 250),      
    ('Auditory', 251, 265),             
    ('Default', 266, 342),             
    ('Posterior Multimodal', 343, 349), 
    ('Ventral Multimodal', 350, 353),  
    ('Orbito-Affective', 354, 359)      
]

# Create output directories if they don't exist
for subnet_name, _, _ in subnet_info:
    output_dir = f'/data2/home/data_for_research/sub_net_divide/{subnet_name}/EMOTION/'
    os.makedirs(output_dir, exist_ok=True)

# Process each file (equivalent to MATLAB's for i=3:length(EMOTION))
for file_name in emotion_files:
    # Load FC matrix from .mat file
    file_path = os.path.join(emotion_dir, file_name)
    mat_data = loadmat(file_path)
    FC = mat_data['FC']  # Get FC matrix (shape: (360, 360))
    
    # Extract submatrix A = FC(S, S) (network division-based submatrix)
    A = FC[S, :][:, S]  # Equivalent to MATLAB's A = FC(S, S)
    
    # Extract and save each sub-network
    for subnet_name, start_idx, end_idx in subnet_info:
        # Extract submatrix (rows and columns from start_idx to end_idx)
        subnet = A[start_idx:end_idx+1, start_idx:end_idx+1]  # +1 because Python slicing is exclusive
        
        # Define save path
        save_dir = f'/data2/home/data_for_research/sub_net_divide/{subnet_name}/EMOTION/'
        save_path = os.path.join(save_dir, file_name)
        
        # Save sub-network as .mat file
        savemat(save_path, {f'S{subnet_info.index((subnet_name, start_idx, end_idx)) + 1}': subnet})

print("Sub-network extraction completed.")


