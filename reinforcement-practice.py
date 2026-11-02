import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



a = np.array([1, 2, 3, 4])
print(a)
m = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(m)

print(np.zeros((3, 4)))
print(np.ones((2, 2)))
print(np.arange(0, 10, 1))
print(np.linspace(0, 2, 5))
print(np.random.rand(3, 3))
print("end of A1")


print(m.shape)
print(m.size)
print(len(m))
print("end of A2")

a = np.array(5)                     
b = np.array([1, 2, 3])             
c = np.array([[1, 2], [3, 4]])      
d = np.zeros((28, 28, 3))          

print(a.ndim, b.ndim, c.ndim, d.ndim)   
print("end of A3")


