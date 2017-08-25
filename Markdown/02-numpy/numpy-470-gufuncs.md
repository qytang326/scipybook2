

```python
%matplotlib_svg
import numpy as np
```

### 广义ufunc函数

> **TIP**

> NumPy中的线性代数模块`linalg`中提供的函数大都为广义ufunc函数。在SciPy中也提供了线性代数模块`linalg`，但其中的函数都是一般函数，只能对单个矩阵进行计算。关于线性代数函数库的用法将在下一章进行详细介绍。


```python
a = np.random.rand(10, 20, 3, 3)
ainv = np.linalg.inv(a)
ainv.shape
```




    (10, 20, 3, 3)




```python
i, j = 3, 4
np.allclose(np.dot(a[i, j], ainv[i, j]), np.eye(3))
```




    True




```python
adet = np.linalg.det(a)
adet.shape
```




    (10, 20)




```python
n = 10000
np.random.seed(0)
beta = np.random.rand(n, 3)
x = np.random.rand(n, 10)
y = beta[:,2, None] + x*beta[:, 1, None] + x**2*beta[:, 0, None]
```


```python
print((beta[42]))
print((np.polyfit(x[42], y[42], 2)))
```

    [ 0.0191932   0.30157482  0.66017354]
    [ 0.0191932   0.30157482  0.66017354]



```python
%time beta2 = np.vstack([np.polyfit(x[i], y[i], 2) for i in range(n)])
```

    Wall time: 1.52 s



```python
np.allclose(beta, beta2)
```




    True




```python
xx = np.column_stack(([x[42]**2, x[42], np.ones_like(x[42])]))
print((np.linalg.lstsq(xx, y[42])[0]))
```

    [ 0.0191932   0.30157482  0.66017354]



```python
%%time
X = np.dstack([x**2, x, np.ones_like(x)])
Xt = X.swapaxes(-1, -2)

import numpy.core.umath_tests as umath
A = umath.matrix_multiply(Xt, X)
b = umath.matrix_multiply(Xt, y[..., None]).squeeze()

beta3 = np.linalg.solve(A, b)

print((np.allclose(beta3, beta2)))
```

    True
    Wall time: 30 ms



```python
M = np.array([[[np.cos(t), -np.sin(t)], 
               [np.sin(t), np.cos(t)]]
             for t in np.linspace(0, np.pi, 4, endpoint=False)])

x = np.linspace(-1, 1, 100)
points = np.array((np.c_[x, x], np.c_[x, x**3], np.c_[x**3, x]))
rpoints = umath.matrix_multiply(points, M[:, None, ...])

print((points.shape, M.shape, rpoints.shape))
```

    (3, 100, 2) (4, 2, 2) (4, 3, 100, 2)



```python
#%figonly=使用矩阵乘积的广播运算将3条曲线分别旋转4个角度
import pylab as pl

pl.figure(figsize=(6, 6))
for t in rpoints.reshape(-1, 100, 2):
    pl.plot(t[:,0], t[:,1], color="gray", lw=2)
ax = pl.gca()
ax.set_aspect("equal")
ax.axis("off");
```


![svg](numpy-470-gufuncs_files/numpy-470-gufuncs_13_0.svg)

