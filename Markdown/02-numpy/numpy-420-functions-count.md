

```python
import numpy as np
```

### 统计函数


```python
np.random.seed(42)
a = np.random.randint(0, 8, 10)
%C a; np.unique(a)
```

                  a                    np.unique(a)   
    ------------------------------  ------------------
    [6, 3, 4, 6, 2, 7, 4, 4, 6, 1]  [1, 2, 3, 4, 6, 7]



```python
x, index = np.unique(a, return_index=True)
%C x; index; a[index]
```

            x                 index              a[index]     
    ------------------  ------------------  ------------------
    [1, 2, 3, 4, 6, 7]  [9, 4, 1, 2, 0, 5]  [1, 2, 3, 4, 6, 7]



```python
x, rindex = np.unique(a, return_inverse=True)
%C rindex; x[rindex]
```

                rindex                        x[rindex]           
    ------------------------------  ------------------------------
    [4, 2, 3, 4, 1, 5, 3, 3, 4, 0]  [6, 3, 4, 6, 2, 7, 4, 4, 6, 1]



```python
np.bincount(a)
```




    array([0, 1, 1, 1, 3, 0, 3, 1])




```python
x = np.array([0  ,   1,   2,   2,   1,   1,   0])
w = np.array([0.1, 0.3, 0.2, 0.4, 0.5, 0.8, 1.2])
np.bincount(x, w)
```




    array([ 1.3,  1.6,  0.6])




```python
np.bincount(x, w) / np.bincount(x)
```




    array([ 0.65      ,  0.53333333,  0.3       ])




```python
a = np.random.rand(100)
np.histogram(a, bins=5, range=(0, 1))
```




    (array([28, 18, 17, 19, 18]), array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ]))




```python
np.histogram(a, bins=[0, 0.4, 0.8, 1.0])
```




    (array([46, 36, 18]), array([ 0. ,  0.4,  0.8,  1. ]))




```python
d = np.loadtxt("height.csv", delimiter=",")
%C d.shape; np.min(d[:, 0]); np.max(d[:, 0])
```

    d.shape    np.min(d[:, 0])     np.max(d[:, 0])  
    --------  ------------------  ------------------
    (100, 2)  7.0999999999999996  19.899999999999999



```python
sums = np.histogram(d[:, 0], bins=list(range(7, 21)), weights=d[:, 1])[0]
cnts = np.histogram(d[:, 0], bins=list(range(7, 21)))[0]
sums / cnts
```




    array([ 125.96      ,  132.06666667,  137.82857143,  143.8       ,
            148.14      ,  153.44      ,  162.15555556,  166.86666667,
            172.83636364,  173.3       ,  175.275     ,  174.19166667,  175.075     ])



### 分段函数


```python
x = np.arange(10)
np.where(x < 5, 9 - x, x)
```




    array([9, 8, 7, 6, 5, 5, 6, 7, 8, 9])




```python
np.where(x > 6, 2 * x, 0)
```




    array([ 0,  0,  0,  0,  0,  0,  0, 14, 16, 18])




```python
def triangle_wave1(x, c, c0, hc):
    x = x - x.astype(np.int) # 三角波的周期为1，因此只取x坐标的小数部分进行计算
    return np.where(x >= c, 
                    0, 
                    np.where(x < c0, 
                             x / c0 * hc, 
                             (c - x) / (c - c0) * hc))
```


```python
def triangle_wave2(x, c, c0, hc):
    x = x - x.astype(np.int)
    return np.select([x >= c, x < c0 , True            ], 
                      [0     , x/c0*hc, (c-x)/(c-c0)*hc])
```


```python
def triangle_wave3(x, c, c0, hc):
    x = x - x.astype(np.int)
    return np.piecewise(x, 
        [x >= c, x < c0],
        [0,  # x>=c 
        lambda x: x / c0 * hc, # x<c0
        lambda x: (c - x) / (c - c0) * hc])  # else
```


```python
x = np.linspace(0, 2, 10000) 
y1 = triangle_wave1(x, 0.6, 0.4, 1.0)
y2 = triangle_wave2(x, 0.6, 0.4, 1.0)
y3 = triangle_wave3(x, 0.6, 0.4, 1.0)
np.all(y1 == y2), np.all(y1 == y3)
```




    (True, True)




```python
%timeit triangle_wave1(x, 0.6, 0.4, 1.0)
%timeit triangle_wave2(x, 0.6, 0.4, 1.0)
%timeit triangle_wave3(x, 0.6, 0.4, 1.0)
```

    1000 loops, best of 3: 614 µs per loop
    1000 loops, best of 3: 736 µs per loop
    1000 loops, best of 3: 311 µs per loop

