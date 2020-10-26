"""
Created on Wed Oct  7 22:06:06 2020

Nicolas Gachancipa
MA410 - Linear Optimization
Primal Simplex Method.

"""
# Imports
import numpy as np

# Inputs
z = np.array([-3, -13, -13, 0, 0, 0])
A = np.array([[1, 1, 0, 1, 0, 0], 
              [1, 3, 2, 0, 1, 0], 
              [0, 2, 3, 0, 0, 1]])
b = np.array([[7, 10, 9]])
XB = [3, 4, 5]

# Functions.
def compute_c_hat_and_B(A, z, XN):
    
    # Identify B, B^-1, N.
    B = A[:, np.r_[XB]]
    B_inv = np.linalg.inv(B)
    N = A[:, np.r_[XN]]
    
    # Compute CBT and CNT.
    CBT = [z[i] if i < len(z) else 0 for i in XB]
    CNT = [z[i] if i < len(z) else 0 for i in XN]
    YT = np.dot(CBT, B_inv)
    C_hat_NT = CNT - np.dot(YT, N)
    return C_hat_NT, B, B_inv

# Code.
XN = [i for i in range(len(A[0])) if i not in XB] 
C_hat_NT, B, B_inv = compute_c_hat_and_B(A, z, XN)
original_b = b[:]
while any([x < 0 for x in C_hat_NT]):
    
    # Find the index of the column that will enter the basis.
    idx = [e for e, i in enumerate(C_hat_NT) if i < 0][0]
    idx = XN[idx]
    
    # Compute Aj.
    Aj = np.dot(B_inv, np.array([[i] for i in A[:, idx]]))
    if all(x[0] <= 0 for x in Aj):
        break
    else:
        # Min-ratio test.
        minimum = 1e6
        for e, i in enumerate(Aj):
            if i[0] > 0:
                min_ratio = b[0][e]/i[0]
                if min_ratio < minimum:
                    minimum = min_ratio
                    min_index = int(e)
        leaving_base = XB[min_index]
        
        # Update XB, XN, B, B_inv, and C_hat_NT
        XB[min_index] = idx
        XN[XN.index(idx)] = leaving_base
        C_hat_NT, B, B_inv = compute_c_hat_and_B(A, z, XN)
        b = np.dot(B_inv, original_b.T).T
        
# Print solution.
if all([x >= 0 for x in C_hat_NT]):
    print('Optimal Solution Found:')
    for e, i in zip(XB, b[0]):
        print("Variable X%d: %2f" %(e + 1, i))
    print('All other variables are zero.')
    print('Plug these values into the objective function to get min/max.')
else:
    print("Solution is unbounded.")
