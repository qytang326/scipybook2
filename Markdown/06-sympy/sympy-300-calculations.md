

```python
from sympy import *
import sympy
from IPython.display import Latex
%init_sympy_printing
x, y, z = symbols("x, y, z")
a, b = symbols("a, b")
f = Function("f")
```

## 符号运算

### 表达式变换和化简


```python
simplify((x + 2) ** 2 - (x + 1) ** 2)
```




$$2 x + 3$$




```python
radsimp(1 / (sqrt(5) + 2 * sqrt(2)))
```




$$\frac{1}{3} \left(- \sqrt{5} + 2 \sqrt{2}\right)$$




```python
radsimp(1 / (y * sqrt(x) + x * sqrt(y)))
```




$$\frac{- \sqrt{x} y + x \sqrt{y}}{x y \left(x - y\right)}$$




```python
ratsimp(x / (x + y) + y / (x - y))
```




$$\frac{2 y^{2}}{x^{2} - y^{2}} + 1$$




```python
%sympy_latex fraction(ratsimp(1 / x + 1 / y))
```


$$\left ( x + y, \quad x y\right )$$



```python
%sympy_latex fraction(1 / x + 1 / y)
```


$$\left ( \frac{1}{y} + \frac{1}{x}, \quad 1\right )$$



```python
cancel((x ** 2 - 1) / (1 + x))
```




$$x - 1$$




```python
s = symbols("s")
trans_func = 1/(s**3 + s**2 + s + 1)
apart(trans_func)
```




$$- \frac{s - 1}{2 s^{2} + 2} + \frac{1}{2 s + 2}$$




```python
trigsimp(sin(x) ** 2 + 2 * sin(x) * cos(x) + cos(x) ** 2)
```




$$\sin{\left (2 x \right )} + 1$$




```python
expand_trig(sin(2 * x + y))
```




$$\left(2 \cos^{2}{\left (x \right )} - 1\right) \sin{\left (y \right )} + 2 \sin{\left (x \right )} \cos{\left (x \right )} \cos{\left (y \right )}$$




```python
#%hide
from tabulate import tabulate
from IPython.display import Markdown, display_markdown
flags = ["mul", "log", "multinomial", "power_base", "power_exp"]
expressions = [x * (y + z), log(x * y ** 2), (x + y) ** 3, (x * y) ** z, x ** (y + z)]
infos =["展开乘法", "展开对数函数的参数中的乘积和幂运算", 
        "展开加减法表达式的整数次幂", "展开幂函数的底数乘积", "展开对幂函数的指数和"]
table = []
for flag, expression, info in zip(flags, expressions, infos):
    table.append(["`{}`".format(flag), 
                  "`expand({})`".format(expression), 
                  "${}$".format(latex(expand(expression))),
                 info])

display_markdown(Markdown(tabulate(table, ["标志", "表达式", "结果", "说明"], "pipe")))
```


| 标志            | 表达式                   | 结果                                      | 说明                |
|:--------------|:----------------------|:----------------------------------------|:------------------|
| `mul`         | `expand(x*(y + z))`   | $x y + x z$                             | 展开乘法              |
| `log`         | `expand(log(x*y**2))` | $\log{\left (x y^{2} \right )}$         | 展开对数函数的参数中的乘积和幂运算 |
| `multinomial` | `expand((x + y)**3)`  | $x^{3} + 3 x^{2} y + 3 x y^{2} + y^{3}$ | 展开加减法表达式的整数次幂     |
| `power_base`  | `expand((x*y)**z)`    | $\left(x y\right)^{z}$                  | 展开幂函数的底数乘积        |
| `power_exp`   | `expand(x**(y + z))`  | $x^{y} x^{z}$                           | 展开对幂函数的指数和        |



```python
x, y, z = symbols("x,y,z", positive=True)
expand(x * log(y * z), mul=False)
```




$$x \left(\log{\left (y \right )} + \log{\left (z \right )}\right)$$




```python
#%hide
from tabulate import tabulate
from IPython.display import Markdown
flags = ["complex", "func", "trig"]
expressions = [x * y, gamma(1 + x), sin(x + y)]
infos =["展开乘法", "展开对数函数的参数中的乘积和幂运算", 
        "展开加减法表达式的整数次幂", "展开幂函数的底数乘积", "展开对幂函数的指数和"]
table = []
for flag, expression, info in zip(flags, expressions, infos):
    table.append(["`{}`".format(flag), 
                  "`expand({})`".format(expression), 
                  "${}$".format(latex(expand(expression))),
                 info])

display_markdown(Markdown(tabulate(table, ["标志", "表达式", "结果", "说明"], "pipe")))
```


| 标志        | 表达式                    | 结果                            | 说明                |
|:----------|:-----------------------|:------------------------------|:------------------|
| `complex` | `expand(x*y)`          | $x y$                         | 展开乘法              |
| `func`    | `expand(gamma(x + 1))` | $\Gamma{\left(x + 1 \right)}$ | 展开对数函数的参数中的乘积和幂运算 |
| `trig`    | `expand(sin(x + y))`   | $\sin{\left (x + y \right )}$ | 展开加减法表达式的整数次幂     |



```python
x, y = symbols("x,y", complex=True)
expand(x * y, complex=True)
```




$$\Re{x} \Re{y} + i \Re{x} \Im{y} + i \Re{y} \Im{x} - \Im{x} \Im{y}$$




```python
expand(gamma(1 + x), func=True)
```




$$x \Gamma{\left(x \right)}$$




```python
expand(sin(x + y), trig=True)
```




$$\sin{\left (x \right )} \cos{\left (y \right )} + \sin{\left (y \right )} \cos{\left (x \right )}$$




```python
factor(15 * x ** 2 + 2 * y - 3 * x - 10 * x * y)
```




$$\left(3 x - 2 y\right) \left(5 x - 1\right)$$




```python
eq = (1 + a * x) ** 3 + (1 + b * x) ** 2
eq2 = expand(eq)
collect(eq2, x)
```




$$a^{3} x^{3} + x^{2} \left(3 a^{2} + b^{2}\right) + x \left(3 a + 2 b\right) + 2$$




```python
p = collect(eq2, x, evaluate=False)
%C p[S(1)]; p[x**2]
```

    p[S(1)]     p[x**2]   
    -------  -------------
    2        3*a**2 + b**2



```python
%C eq2.coeff(x, 0); eq2.coeff(x, 2)
```

    eq2.coeff(x, 0)  eq2.coeff(x, 2)
    ---------------  ---------------
    2                3*a**2 + b**2  



```python
collect(a * sin(2 * x) + b * sin(2 * x), sin(2 * x))
```




$$\left(a + b\right) \sin{\left (2 x \right )}$$



### 方程


```python
a, b, c = symbols("a,b,c")
%sympy_latex solve(a * x ** 2 + b * x + c, x)
```


$$\left [ \frac{1}{2 a} \left(- b + \sqrt{- 4 a c + b^{2}}\right), \quad - \frac{1}{2 a} \left(b + \sqrt{- 4 a c + b^{2}}\right)\right ]$$



```python
%sympy_latex solve((x ** 2 + x * y + 1, y ** 2 + x * y + 2), x, y)
```


$$\left [ \left ( - \frac{\sqrt{3} i}{3}, \quad - \frac{2 i}{3} \sqrt{3}\right ), \quad \left ( \frac{\sqrt{3} i}{3}, \quad \frac{2 i}{3} \sqrt{3}\right )\right ]$$



```python
%sympy_latex roots(x**3 - 3*x**2 + x + 1)
```


$$\left \{ 1 : 1, \quad 1 + \sqrt{2} : 1, \quad - \sqrt{2} + 1 : 1\right \}$$


### 微分


```python
t = Derivative(sin(x), x)
t
```




$$\frac{d}{d x} \sin{\left (x \right )}$$




```python
t.doit()
```




$$\cos{\left (x \right )}$$




```python
diff(sin(2*x), x)
```




$$2 \cos{\left (2 x \right )}$$




```python
Derivative(f(x), x)
```




$$\frac{d}{d x} f{\left (x \right )}$$




```python
Derivative(f(x), x, x, x) # 也可以写作Derivative(f(x), x, 3)
```




$$\frac{d^{3}}{d x^{3}}  f{\left (x \right )}$$




```python
Derivative(f(x, y), x, 2, y, 3)
```




$$\frac{\partial^{5}}{\partial x^{2}\partial y^{3}}  f{\left (x,y \right )}$$




```python
diff(sin(x * y), x, 2, y, 3)
```




$$x \left(x^{2} y^{2} \cos{\left (x y \right )} + 6 x y \sin{\left (x y \right )} - 6 \cos{\left (x y \right )}\right)$$



### 微分方程


```python
x=symbols('x')
f=symbols('f', cls=Function)
dsolve(Derivative(f(x), x) - f(x), f(x))
```




$$f{\left (x \right )} = C_{1} e^{x}$$




```python
eq = Eq(f(x).diff(x) + f(x), (cos(x) - sin(x)) * f(x)**2)
classify_ode(eq, f(x))
```




    ('1st_power_series', 'lie_group')




```python
dsolve(eq, f(x))
```




$$f{\left (x \right )} = C_{1} - \frac{C_{1} x^{2}}{2} - \frac{C_{1} x^{3}}{6} + \frac{C_{1} x^{4}}{4} + \frac{C_{1} x^{5}}{120} \left(- C_{1} \left(C_{1} - 3\right) - C_{1} \left(C_{1} + 1\right) + 4 C_{1} + 12\right) + \mathcal{O}\left(x^{6}\right)$$




```python
dsolve(eq, f(x), hint="lie_group")
```




$$f{\left (x \right )} = \frac{1}{C_{1} e^{x} - \sin{\left (x \right )}}$$




```python
dsolve(eq, f(x), hint="all")
```




    {'1st_power_series': f(x) == C1 - C1*x**2/2 - C1*x**3/6 + C1*x**4/4 + C1*x**5*(-C1*(C1 - 3) - C1*(C1 + 1) + 4*C1 + 12)/120 + O(x**6),
     'best': f(x) == C1 - C1*x**2/2 - C1*x**3/6 + C1*x**4/4 + C1*x**5*(-C1*(C1 - 3) - C1*(C1 + 1) + 4*C1 + 12)/120 + O(x**6),
     'best_hint': '1st_power_series',
     'default': '1st_power_series',
     'lie_group': f(x) == 1/(C1*exp(x) - sin(x)),
     'order': 1}




```python
%omit sympy.ode.allhints
```

    ('separable',
     '1st_exact',
     '1st_linear',
     'Bernoulli',
    ...


### 积分


```python
e = Integral(x*sin(x), x)
e    
```




$$\int x \sin{\left (x \right )}\, dx$$




```python
e.doit()
```




$$- x \cos{\left (x \right )} + \sin{\left (x \right )}$$




```python
e2 = Integral(sin(x)/x, (x, 0, 1))
e2.doit()
```




$$\operatorname{Si}{\left (1 \right )}$$




```python
print((e2.evalf()))
print((e2.evalf(50))) # 可以指定精度
```

    0.946083070367183
    0.94608307036718301494135331382317965781233795473811



```python
e3 = Integral(sin(x)/x, (x, 0, oo))
e3.evalf()
```




$$-4.0$$




```python
e3.doit()
```




$$\frac{\pi}{2}$$


