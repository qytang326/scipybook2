

```python
import numpy as np
from sympy import *
import sympy
from IPython.display import Latex
%init_sympy_printing

x, y, z = symbols("x, y, z")
a, b = symbols("a, b")
f = Function("f")
```

## 输出符号表达式

### `lambdify`


```python
a, b, c, x = symbols("a, b, c, x", real=True)
quadratic_roots = solve(a*x**2 + b*x + c, x)
lam_quadratic_roots_real = lambdify([a, b, c], quadratic_roots)
lam_quadratic_roots_real(2, -3, 1)
```




    [1.0, 0.5]




```python
import cmath
lam_quadratic_roots_complex = lambdify((a, b, c), quadratic_roots, modules=cmath)
lam_quadratic_roots_complex(2, 2, 1)
```




    [(-0.5+0.5j), (-0.5-0.5j)]




```python
lam_quadratic_roots_numpy = lambdify((a, b, c), quadratic_roots, modules="numpy")
A = np.array([2, 2, 1, 2], np.complex) 
B = np.array([1, 4, 2, 1], np.complex) 
C = np.array([1, 1, 1, 2], np.complex)
lam_quadratic_roots_numpy(A, B, C)
```




    [array([-0.25000000+0.66143783j, -0.29289322+0.j        ,
            -1.00000000+0.j        , -0.25000000+0.96824584j]),
     array([-0.25000000-0.66143783j, -1.70710678+0.j        ,
            -1.00000000+0.j        , -0.25000000-0.96824584j])]



### 用`autowrap()`编译表达式


```python
from sympy.utilities.autowrap import autowrap
matrix_roots = Matrix(quadratic_roots)
quadratic_roots_f2py   = autowrap(matrix_roots, args=[a, b, c], tempdir=r".\tmp")
quadratic_roots_cython = autowrap(matrix_roots, args=[a, b, c], tempdir=r".\tmp",
                                 backend="cython", flags=["-I" + np.get_include()])
```


```python
%C quadratic_roots_f2py(2, -3, 1); quadratic_roots_cython(2, -3, 1)
```

    quadratic_roots_f2py(2, -3, 1)  quadratic_roots_cython(2, -3, 1)
    ------------------------------  --------------------------------
    [[ 1. ],                        [[ 1. ],                        
     [ 0.5]]                         [ 0.5]]                        



```python
from sympy.utilities.autowrap import ufuncify
quadratic_roots_ufunc = ufuncify((a, b, c), quadratic_roots[0], tempdir=r".\tmp")

quadratic_roots_ufunc([1, 2, 10.0], [6, 7, 12.0], [4, 5, 1.0])
```




    array([-0.76393202, -1.        , -0.09009805])




```python
from sympy.utilities.codegen import codegen
(c_name, c_code), (h_name, c_header) = codegen(
    [("root0", quadratic_roots[0]), 
     ("root1", quadratic_roots[1]), 
     ("roots", matrix_roots)], 
    language="C", 
    prefix="quadratic_roots", 
    header=False)
print(h_name)
print(("-" * 40))
print(c_header)
print()
print(c_name)
print(("-" * 40))
print(c_code)
```

    quadratic_roots.h
    ----------------------------------------
    
    #ifndef PROJECT__QUADRATIC_ROOTS__H
    #define PROJECT__QUADRATIC_ROOTS__H
    
    double root0(double a, double b, double c);
    double root1(double a, double b, double c);
    void roots(double a, double b, double c, double *out_1451769269);
    
    #endif
    
    
    
    quadratic_roots.c
    ----------------------------------------
    #include "quadratic_roots.h"
    #include <math.h>
    
    double root0(double a, double b, double c) {
    
       double root0_result;
       root0_result = (1.0L/2.0L)*(-b + sqrt(-4*a*c + pow(b, 2)))/a;
       return root0_result;
    
    }
    
    double root1(double a, double b, double c) {
    
       double root1_result;
       root1_result = -1.0L/2.0L*(b + sqrt(-4*a*c + pow(b, 2)))/a;
       return root1_result;
    
    }
    
    void roots(double a, double b, double c, double *out_1451769269) {
    
       out_1451769269[0] = (1.0L/2.0L)*(-b + sqrt(-4*a*c + pow(b, 2)))/a;
       out_1451769269[1] = -1.0L/2.0L*(b + sqrt(-4*a*c + pow(b, 2)))/a;
    
    }
    



```python
print((ccode(matrix_roots, assign_to="y")))
```

    y[0] = (1.0L/2.0L)*(-b + sqrt(-4*a*c + pow(b, 2)))/a;
    y[1] = -1.0L/2.0L*(b + sqrt(-4*a*c + pow(b, 2)))/a;


### 使用`cse()`分步输出表达式


```python
replacements, reduced_exprs = cse(quadratic_roots)
%sympy_latex replacements
```


$$\left [ \left ( x_{0}, \quad \frac{1}{2 a}\right ), \quad \left ( x_{1}, \quad \sqrt{- 4 a c + b^{2}}\right )\right ]$$



```python
%sympy_latex reduced_exprs
```


$$\left [ x_{0} \left(- b + x_{1}\right), \quad - x_{0} \left(b + x_{1}\right)\right ]$$



```python
replacements, reduced_exprs = cse(quadratic_roots, symbols=numbered_symbols("tmp"))
%sympy_latex replacements
```


$$\left [ \left ( tmp_{0}, \quad \frac{1}{2 a}\right ), \quad \left ( tmp_{1}, \quad \sqrt{- 4 a c + b^{2}}\right )\right ]$$



```python
from scpy2.sympy.cseprinter import cse2func
code = cse2func("cse_quadratic_roots(a, b, c)", quadratic_roots)
exec(code)
print(code)
```

    def cse_quadratic_roots(a, b, c):
        from math import sqrt
        _tmp0 = 0.5/a
        _tmp1 = sqrt((b)**(2.0) - 4.0*a*c)
        return (_tmp0*(_tmp1 - b), -_tmp0*(_tmp1 + b))



```python
cse_quadratic_roots(1, -4, 2)
```




    (3.41421356237, 0.585786437627)




```python
import cmath
exec(cse2func("cse_quadratic_roots(a, b, c)", quadratic_roots, module=cmath))
cse_quadratic_roots(1, -4, 10)
```




    ((2+2.449489742783178j), (2-2.449489742783178j))


