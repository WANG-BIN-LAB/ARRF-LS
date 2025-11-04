import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from fused_ls_lambda import fused_ls_lambda  
from sub_iden import sub_iden  
# Read data
xTrain_0 = pd.read_csv('/home/data2/day.csv', header=None).values
rows, cols = xTrain_0.shape

# Check if xTrain_0 is a valid NumPy array
if not isinstance(xTrain_0, np.ndarray):
    raise ValueError("xTrain_0 should be a NumPy array")
# Check the dimension of xTrain_0
if xTrain_0.ndim != 2:
    raise ValueError("xTrain_0 should be a 2D NumPy array")

sub_num = 969   
state = 4
N = 360  # Number of nodes
M = sub_num
N_UP = N * (N - 1) / 2
d = M - 1
lambda1 = 1 / np.sqrt(max(N_UP, M))
lambda2 = 0.001
max_iter = 5000
tol = 1e-3

# Initialize L and S matrices
L = np.zeros((64620, M * state))
S = np.zeros((64620, M * state))

"""for i in range(state):
    a = xTrain_0[:, sub_num * i:sub_num * (i + 1)]
    mu = 1 / np.linalg.norm(a)
    Lthat, Sthat, rhat, err = fused_ls(a, lambda1, lambda2, mu, tol, max_iter)"""
for i in range(1, state + 1): 
    start_index = 1 + sub_num * (i - 1) - 1 
    end_index = sub_num * i
    a = xTrain_0[:, start_index:end_index]
    mu = 1 / np.linalg.norm(a)
    Lthat, Sthat, rhat, err_history = fused_ls_lambda(a, lambda1, lambda2, mu, tol, max_iter)
    L[:, start_index:end_index] = Lthat  
    S[:, start_index:end_index] = Sthat

print('Identification finished_2')

# Calculate identification accuracy
Identification_accuracy_L = sub_iden(L, sub_num, state)
Identification_accuracy_S = sub_iden(S, sub_num, state)

# Save results to CSV files
np.savetxt('/home/data2/Identification_accuracy_L_day.csv', Identification_accuracy_L, delimiter=',')
np.savetxt('/home/data2/Identification_accuracy_S_day.csv', Identification_accuracy_S, delimiter=',')
np.savetxt('/home/data2/L_day.csv', L, delimiter=',')
np.savetxt('/home/data2/S_day.csv', S, delimiter=',')
print('Identification finished')
