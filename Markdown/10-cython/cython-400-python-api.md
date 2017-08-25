

```python
%load_ext cython
import numpy as np
```

## 使用Python标准对象和API

### 操作`list`对象


```cython
%%cython
#cython: boundscheck=False, wraparound=False
from cpython.list cimport PyList_New, PyList_SET_ITEM #❶
from cpython.ref cimport Py_INCREF

def my_range(int n):
    cdef int i
    cdef object obj #❷
    cdef list result
    result = PyList_New(n)
    for i in range(n):
        obj = i
        PyList_SET_ITEM(result, i, obj)
        Py_INCREF(obj)
    return result

def my_range2(int n):
    cdef int i
    cdef list result
    result = []
    for i in range(n):
        result.append(i)
    return result
```


```python
%timeit range(100)
%timeit my_range(100)
%timeit my_range2(100)
```

    1000000 loops, best of 3: 1.24 µs per loop
    1000000 loops, best of 3: 1.04 µs per loop
    100000 loops, best of 3: 2.29 µs per loop


### 创建`tuple`对象


```cython
%%cython
#cython: boundscheck=False, wraparound=False
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.ref cimport Py_INCREF

def to_tuple_list(double[:, :] arr):
    cdef int m, n
    cdef int i, j
    cdef list result
    cdef tuple t
    cdef object obj
    
    m, n = arr.shape[0], arr.shape[1]
    result = PyList_New(m)
    for i in range(m):
        t = PyTuple_New(n)
        for j in range(n):
            obj = arr[i, j]
            PyTuple_SET_ITEM(t, j, obj)
            Py_INCREF(obj)
        PyList_SET_ITEM(result, i, t)
        Py_INCREF(t)
    return result
```


```python
import numpy as np
arr = np.random.randint(0, 10, (5, 2)).astype(np.double)
print to_tuple_list(arr)

arr = np.random.rand(100, 5)
%timeit to_tuple_list(arr)
%timeit arr.tolist()
```

    [(0.0, 4.0), (5.0, 7.0), (7.0, 0.0), (5.0, 5.0), (5.0, 9.0)]
    100000 loops, best of 3: 13 µs per loop
    10000 loops, best of 3: 20.5 µs per loop


### 用`array.array`作动态数组


```cython
%%cython -c-Ofast
#cython: boundscheck=False, wraparound=False
import numpy as np
from cpython cimport array

def in_circle(double[:, :] points, double cx, double cy, double r):
    cdef array.array[double] res = array.array("d") #❶
    cdef double r2 = r * r
    cdef double p[2] #❷
    cdef int i 
    for i in range(points.shape[0]):
        p[0] = points[i, 0]
        p[1] = points[i, 1]
        if (p[0] - cx)**2 + (p[1] - cy)**2 < r2:
            array.extend_buffer(res, <char*>p, 2) #❸
    return np.frombuffer(res, np.double).copy().reshape(-1, 2) #❹
```

> **TIP**

> 本例的目的是为了演示`array.array`动态扩容，实际上使用布尔数组有可能得到更快的运算速度。


```python
points = np.random.rand(10000, 2)
cx, cy, r = 0.3, 0.5, 0.05

%timeit points[(points[:, 0] - cx)**2 + (points[:, 1] - cy)**2 < r**2, :]
%timeit in_circle(points, cx, cy, r)
```

    10000 loops, best of 3: 97.7 µs per loop
    10000 loops, best of 3: 38.6 µs per loop

