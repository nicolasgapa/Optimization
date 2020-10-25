# Optimization: Linear programming optimization techniques

The primal and dual simplex algorithms are techniques to solve linear programming (optimization) problems. 
These methods are also useful for sensitivity analysis. 

The primal method is used to reach optimality starting from a basic feasible solution, while the dual simplex method is used to reach feasibility starting with an infeasible solution. 

In order to use the algorithms, edit the inputs section at the beginning of the code, where the following four variables have to be defined:

1. z: The objective function coefficients.
2. A: The constraints matrix (solving for the basic variables).
3. b: Right hand side of the constraints.
4. XB: The index of the starting basic variables (python-based index starting at 0).
    
        For example, for the following minimization problem:
        
        minimize    Z = 2(X1) + 3(X2)
        subject to  3(X1) + 2(X2) >= 4
                    X1 + 2(X2) >= 3
                    X1, X2 >= 0
              
        First, convert to standard form:
        
        minimize    Z = 2(X1) + 3(X2)
        subject to  3(X1) + 2(X2) - e1 = 4
                    X1 + 2(X2) - e2 = 3
                    X1, X2, e1, e2 >= 0
         
         Inputs:
         z = np.array([[2, 3, 0, 0]])
         A = np.array([[3, -2, -1, 0], [1, 2, 0, -1]])
         b = np.array([[4, 3]])
         XB = [2, 3]  (The second and third basic variables, e1 and e2, are th initial basis) 
         
      
         
      

