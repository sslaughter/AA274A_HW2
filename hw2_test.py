
import numpy as np


x2 = np.array([0,-5])
x1 = np.array([0,0])
steering_vector = x2-x1
print(steering_vector)
eps = 1.0
norm = np.linalg.norm(steering_vector)
print(norm)
if norm < eps:
    print("x2")
    print(x2) # return x2 if it's within steering distance
else:
    unit_steering_vector = steering_vector/np.linalg.norm(steering_vector)
    ans = x1 + eps*unit_steering_vector #return new point that is the steering distance between x1 and x2.

print(np.linalg.norm(unit_steering_vector))
print(ans)