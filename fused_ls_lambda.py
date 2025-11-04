import numpy as np  
from scipy.linalg import svd  
from scipy.sparse import eye,kron

  
def fused_ls_lambda2(Z, lambda1, lambda2, mu, tol, max_iter):  
    N, M = Z.shape  
    unobserved = np.isnan(Z)  
    Z[unobserved] = 0  
    d = M - 1  
    e = np.eye(M)  
    D = np.zeros((d, M))  
    normZ = np.linalg.norm(Z, 'fro')  
      
    r = 0  
    for i in range(1, M):  
        D[r, :] = (e[:, i] - e[:, i - 1]).T  
        r += 1  


    A = kron(D, eye(N, format='csr'))
    v = 1.5 * np.max(np.abs(np.linalg.eigvals(D.T @ D)))  
      
    L = np.zeros((N, M))  
    S = np.zeros((N, M))  
    alpha = np.zeros(N * d)  
    U = np.zeros((N, M))  
    V = np.zeros(N * d)  
    AL = A @ L.flatten()  
      
    tauL = 1 / (mu * (v + 1))  
    tauS = lambda1 / mu  
    taualpha = lambda2 / mu  
      
    oldL = L.copy()  
    oldS = S.copy()  

    lambda2_min = 1e-6; 
    lambda2_max = 1;  
    lambda2_decay_factor = 0.8; 

    #err_history = np.zeros(max_iter)
    mu_max = 1e3; 
    mu_min = 1e-6; 
    mu_adapt_threshold = 2;# Threshold for mu adjustment (set to 2x)
    mu_adjust_factor = 1.1; # Adjustment factor for mu
    err_history = []
    for iter in range(1, max_iter + 1):  
        C = L.flatten() - A.T @ (AL - alpha + V) / v  
        Ctilde = C.reshape((N, M))  
        Q = (Z - S + v * Ctilde - U) / (v + 1)  
        L = svt_python(tauL, Q)  
        AL = A @ L.flatten()  
        S = shrink_python(tauS, Z - L - U)  
        alpha = shrink_python(taualpha, AL + V)  
          
        U = U + L + S - Z  
        V = V + AL - alpha  
          
        del_val = np.linalg.norm(np.vstack((L, S)) - np.vstack((oldL, oldS)), 'fro') / max(np.linalg.norm(np.vstack((L, S)), 'fro'), 1)  
        err = np.linalg.norm(Z - L - S, 'fro') / normZ  
        err_history.append(err)  
        #err_history[iter] = err
            
        # Calculate L1 norm of S (flatten the array and apply L1 norm)
        l1_norm_S = np.sum(np.abs(S))

        # Adjust lambda2 based on L1 norm of S
        if l1_norm_S > 0.1:  # If sparsity is low (i.e., many non-zero elements in S)
            lambda2 = min(lambda2 * lambda2_decay_factor, lambda2_max)  # Decrease lambda2
        elif l1_norm_S < 0.01:  # If sparsity is high (i.e., few non-zero elements in S)
            lambda2 = max(lambda2 / lambda2_decay_factor, lambda2_min)  # Increase lambda2
        # Calculate primal residual (reconstruction error) and dual residual (optimality condition)
        primal_res = np.linalg.norm(Z - L - S, 'fro')
        dual_res = mu * (np.linalg.norm(L - oldL, 'fro') + np.linalg.norm(S - oldS, 'fro'))

        # Adaptive adjustment of mu to balance primal and dual residuals
        if (primal_res > mu_adapt_threshold * dual_res) and (mu * mu_adjust_factor <= mu_max):
            mu *= mu_adjust_factor
            print(f'Mu increased to {mu:.2e}')
        elif (dual_res > mu_adapt_threshold * primal_res) and (mu / mu_adjust_factor >= mu_min):
            mu /= mu_adjust_factor
            print(f'Mu decreased to {mu:.2e}')

        # Update threshold parameters with the new mu
        tauL = 1.0 / (mu * (v + 1))
        tauS = lambda1 / mu
        taualpha = lambda2 / mu
  
        if del_val < tol and iter > 10:  
           # err_history = err_history[:iter]
            break  
          
        oldL = L.copy()  
        oldS = S.copy()  

   # if iter == max_iter:
    #    err_history = err_history[:max_iter]  
        
    rL = np.linalg.matrix_rank(L)  
      
    return L, S, rL, err_history  
  
def shrink_python(tau, Y):  
    return np.sign(Y) * np.maximum(np.abs(Y) - tau, 0)  
  

def svt_python(tau, Y):
    """Singular Value Thresholding."""
    U, S, VT = svd(Y, full_matrices=False)  # 'econ' in MATLAB is equivalent to full_matrices=False
    S_shrink = shrink_python(tau, S)  # Apply the shrinkage operator
    X = U @ np.diag(S_shrink) @ VT  # Reconstruct the matrix using the thresholded singular values
    return X
