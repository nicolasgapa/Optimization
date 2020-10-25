"""
Created on Wed Oct  7 22:06:06 2020

Nicolas Gachancipa
MA410 - Linear Optimization
Primal Simplex Method.

"""
# Imports
import numpy as np

# Inputs
z = np.array([3, -2, -4])
A = np.array([[4,5, -2, 1, 0], [1, -2, 1, 0, 1]])
b = np.array([[22, 30]])
XB = [3, 4]

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
while any([x < 0 for x in C_hat_NT]):
    
    # Find the index of the column that will enter the basis.
    idx = [e for e, i in enumerate(C_hat_NT) if i < 0][0]
    
    # Compute Aj.
    Aj = np.dot(B_inv, np.array([[i] for i in A[:, idx]]))
    if all(x < 0 for x in [i[0] for i in Aj]):
        break
    else:
        # Min-ratio test.
        minimum = 1e6
        for e, i in enumerate(Aj):
            if i > 0:
                min_ratio = b[0][e]/i[0]
                if min_ratio < minimum:
                    minimum = int(e)
        leaving_base = minimum + len(XN)
        
        # Update XB, XN, B, B_inv, and C_hat_NT
        XB[XB.index(leaving_base)] = idx
        XN[XN.index(idx)] = leaving_base
        C_hat_NT, B, B_inv = compute_c_hat_and_B(A, z, XN)
        
# Print solution.
if all([x >= 0 for x in C_hat_NT]):
    print('Optimal Solution Found:')
    for e, i in zip(XB, np.linalg.solve(B, b.T)):
        print("Variable X%d: %2f" %(e + 1, i[0]))
else:
    print("Solution is unbounded.")
        

            
