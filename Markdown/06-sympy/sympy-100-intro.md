

```python
%matplotlib_svg
import pylab as pl
from sympy import *
import numpy as np
%init_sympy_printing
import sympy
sympy.__version__
```




    '0.7.6'



# SymPy-符号运算好帮手

## 从例子开始

### 封面上的经典公式


```python
from sympy import *
E**(I*pi) + 1
```




$$0$$




```python
x = symbols("x")
expand( E**(I*x) )
```




$$e^{i x}$$




```python
expand(exp(I*x), complex=True)
```




$$i e^{- \Im{x}} \sin{\left (\Re{x} \right )} + e^{- \Im{x}} \cos{\left (\Re{x} \right )}$$




```python
x = Symbol("x", real=True)
expand(exp(I*x), complex=True)    
```




$$i \sin{\left (x \right )} + \cos{\left (x \right )}$$




```python
tmp = series(exp(I*x), x, 0, 10)
tmp
```




$$1 + i x - \frac{x^{2}}{2} - \frac{i x^{3}}{6} + \frac{x^{4}}{24} + \frac{i x^{5}}{120} - \frac{x^{6}}{720} - \frac{i x^{7}}{5040} + \frac{x^{8}}{40320} + \frac{i x^{9}}{362880} + \mathcal{O}\left(x^{10}\right)$$




```python
re(tmp)
```




$$\frac{x^{8}}{40320} - \frac{x^{6}}{720} + \frac{x^{4}}{24} - \frac{x^{2}}{2} + \Re {\left (\mathcal{O}\left(x^{10}\right) \right )} + 1$$




```python
series(cos(x), x, 0, 10)
```




$$1 - \frac{x^{2}}{2} + \frac{x^{4}}{24} - \frac{x^{6}}{720} + \frac{x^{8}}{40320} + \mathcal{O}\left(x^{10}\right)$$




```python
im(tmp)
```




$$\frac{x^{9}}{362880} - \frac{x^{7}}{5040} + \frac{x^{5}}{120} - \frac{x^{3}}{6} + x + \Im {\left ( \mathcal{O}\left(x^{10}\right) \right )}$$




```python
series(sin(x), x, 0, 10)
```




$$x - \frac{x^{3}}{6} + \frac{x^{5}}{120} - \frac{x^{7}}{5040} + \frac{x^{9}}{362880} + \mathcal{O}\left(x^{10}\right)$$



### 球体体积


```python
integrate(x*sin(x), x)
```




$$- x \cos{\left (x \right )} + \sin{\left (x \right )}$$




```python
integrate(x*sin(x), (x, 0, 2*pi))
```




$$- 2 \pi$$




```python
x, y = symbols('x, y')
r = symbols('r', positive=True)
circle_area = 2 * integrate(sqrt(r**2 - x**2), (x, -r, r))
circle_area
```




$$\pi r^{2}$$




```python
circle_area = circle_area.subs(r, sqrt(r**2 - x**2))
circle_area
```




$$\pi \left(r^{2} - x^{2}\right)$$




```python
integrate(circle_area, (x, -r, r))
```




$$\frac{4 \pi}{3} r^{3}$$



### 数值微分


```python
x = symbols('x', real=True)
h = symbols('h', positive=True)
f = symbols('f', cls=Function)
```


```python
f_diff = f(x).diff(x, 1)
f_diff
```




$$\frac{d}{d x} f{\left (x \right )}$$




```python
expr_diff = as_finite_diff(f_diff, [x, x-h, x-2*h, x-3*h])
expr_diff
```




$$\frac{11}{6 h} f{\left (x \right )} - \frac{1}{3 h} f{\left (- 3 h + x \right )} + \frac{3}{2 h} f{\left (- 2 h + x \right )} - \frac{3}{h} f{\left (- h + x \right )}$$




```python
sym_dexpr = f_diff.subs(f(x), x*exp(-x**2)).doit()
sym_dexpr
```




$$- 2 x^{2} e^{- x^{2}} + e^{- x^{2}}$$




```python
sym_dfunc = lambdify([x], sym_dexpr, modules="numpy")
sym_dfunc(np.array([-1, 0, 1]))
```




    array([-0.36787944,  1.        , -0.36787944])




```python
print((expr_diff.args))
```

    (-3*f(-h + x)/h, -f(-3*h + x)/(3*h), 3*f(-2*h + x)/(2*h), 11*f(x)/(6*h))



```python
w = Wild("w")
c = Wild("c")
patterns = [arg.match(c * f(w)) for arg in expr_diff.args]
```


```python
print((patterns[0]))
```

    {w_: -h + x, c_: -3/h}



```python
coefficients = [t[c] for t in sorted(patterns, key=lambda t:t[w])]
print(coefficients)
```

    [-1/(3*h), 3/(2*h), -3/h, 11/(6*h)]



```python
coeff_arr = np.array([float(coeff.subs(h, 1e-3)) for coeff in coefficients])
print(coeff_arr)
```

    [ -333.33333333  1500.         -3000.          1833.33333333]



```python
def moving_window(x, size):
    from numpy.lib.stride_tricks import as_strided    
    x = np.ascontiguousarray(x)
    return as_strided(x, shape=(x.shape[0] - size + 1, size), 
                      strides=(x.itemsize, x.itemsize))

x_arr = np.arange(-2, 2, 1e-3)
y_arr = x_arr * np.exp(-x_arr * x_arr)
num_res = (moving_window(y_arr, 4) * coeff_arr).sum(axis=1)
sym_res = sym_dfunc(x_arr[3:])
print((np.max(abs(num_res - sym_res))))
```

    4.08944167418e-09



```python
def finite_diff_coefficients(f_diff, order, h):
    v = f_diff.variables[0]
    points = [x - i * h for i in range(order)]
    expr_diff = as_finite_diff(f_diff, points)
    w = Wild("w")
    c = Wild("c")
    patterns = [arg.match(c*f(w)) for arg in expr_diff.args]
    coefficients = np.array([float(t[c]) 
                             for t in sorted(patterns, key=lambda t:t[w])])
    return coefficients
```


```python
#%figonly=比较不同点数的数值微分的误差
fig, ax = pl.subplots(figsize=(8, 4))

for order in range(2, 5):
    c = finite_diff_coefficients(f_diff, order, 1e-3)
    num_diff = (moving_window(y_arr, order) * c).sum(axis=1)
    sym_diff = sym_dfunc(x_arr[order-1:])
    error = np.abs(num_diff - sym_diff)
    ax.semilogy(x_arr[order-1:], error, label=str(order))
    
ax.legend(loc="best");
```


![svg](sympy-100-intro_files/sympy-100-intro_32_0.svg)

