import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def f(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2 * x[1]-x[0]**2)**2 

def g(x):
    return -(4 - (x[0]**2 + x[1]**2))


res = minimize(f, x0=[1.75,1.75], constraints={'type': 'ineq', 'fun': g})

# print(res)

print("The optimal value of x is: ", res.x)
print("The optimal value of f(x) is:", res.fun)

n1 = 100
n2 = 99
x1 = np.linspace(0, 2, n1)
x2 = np.linspace(0, 2, n2)

fun_output = np.zeros((n1, n2))
con_output = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        fun_output[i, j] = f([x1[i], x2[j]])
        con_output[i, j] = -g([x1[i], x2[j]])

plt.figure()
# You MUST transpose the output of the function to match the dimensions of x1 and x2 when using contour. Otherwise, the contour plot will be incorrect.
plt.contour(x1, x2, fun_output.T, levels=500, linewidths=0.5, cmap='jet', alpha=0.5)
plt.contourf(x1, x2, con_output.T, levels=[0, 1000], colors='black', alpha=0.95)

# Include a red dot at the optimal point found by the optimizer
plt.plot(res.x[0], res.x[1], 'ro', label='Optimal Point')

plt.show()