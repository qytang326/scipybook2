

```python
%load_ext cython
```

# Cython-编译Python程序

## 配置编译器


```python
from scpy2.utils import show_compiler, set_compiler
set_compiler("mingw32")
show_compiler()
```

    mingw32  defined by C:\WinPython-32bit-2.7.9.2\python-2.7.9\lib\distutils\distutils.cfg


> **LINK**

> http://www.microsoft.com/en-us/download/confirmation.aspx?id=44266
>
> 微软提供的编译Python 2.7的编译器


```python
import setuptools #先载入setuptools
import distutils
from distutils.msvc9compiler import find_vcvarsall
find_vcvarsall(9.0)
```




    u'C:\\Users\\RY\\AppData\\Local\\Programs\\Common\\Microsoft\\Visual C++ for Python\\9.0\\vcvarsall.bat'



> **TIP**

> `%%cython`魔法命令缺省没有载入到IPyhton的运算核中，需要先通过`%load_ext cython`命令载入该命令。


```cython
%%cython
 
def add(a, b):
    return a + b
```


```python
from scpy2.utils import set_msvc_version
set_compiler("msvc")
set_msvc_version(12)
show_compiler()
```

    msvc 12.0 defined by C:\WinPython-32bit-2.7.9.2\python-2.7.9\lib\distutils\distutils.cfg



```python
set_msvc_version(9)
set_compiler("mingw32")
```
