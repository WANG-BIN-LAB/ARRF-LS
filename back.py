import numpy as np
import os
import pandas as pd
from scipy.io import savemat


def get_index(x, y):
    m = 0
    while y > 0:
        m += 360 - y - 1
        y -= 1
    m += (x - 1)
    return m

# Read the large matrix
large_matrix = pd.read_csv('/home/data2/S.csv', header=None).to_numpy()

# Check the number of rows and columns
num_rows, num_cols = large_matrix.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

# Set output folders
output_folder1 = '/home/data2/output_folder1REST1S'
output_folder2 = '/home/data2/output_folder2GAMBLINGS'

os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

# Generate corresponding 360x360 matrices for each column
for col in range(num_cols):
    # Get data of the current column
    data_column = large_matrix[:, col]

    # Initialize 360x360 matrix
    mat = np.zeros((360, 360))

    # Fill the lower triangular matrix
    for i in range(360):#i represents columns
        for j in range(i + 1, 360):#j represents rows
            index = get_index(j, i)
            if index < num_rows:  # Ensure the index is within bounds
                mat[j, i] = data_column[index]

    # Fill diagonal elements
    np.fill_diagonal(mat, 1)

    # Since it's a symmetric matrix, upper triangular elements equal lower triangular elements
    mat = mat + mat.T - np.diag(mat.diagonal())

    # Save the file to the corresponding folder in .mat format
    if col < 969:
        mat_path = os.path.join(output_folder1, f'matrix_{col + 1}.mat')
    else:
        mat_path = os.path.join(output_folder2, f'matrix_{col + 1}.mat')

    # Save the matrix as a .mat file
    savemat(mat_path, {'matrix': mat})  # Save in dictionary form

print("Finished generating matrices.")
