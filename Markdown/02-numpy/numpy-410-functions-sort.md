

```python
import numpy as np
```

### 大小与排序


```python
a = np.array([1, 3, 5, 7])
b = np.array([2, 4, 6])
np.maximum(a[None, :], b[:, None])
```




    array([[2, 3, 5, 7],
           [4, 4, 5, 7],
           [6, 6, 6, 7]])




```python
np.random.seed(42)
a = np.random.randint(0, 10, size=(4, 5))
max_pos = np.argmax(a)
max_pos
```




    5




```python
%C a.ravel()[max_pos]; np.max(a)
```

    a.ravel()[max_pos]  np.max(a)
    ------------------  ---------
    9                   9        



```python
idx = np.unravel_index(max_pos, a.shape)
%C idx; a[idx]
```

     idx    a[idx]
    ------  ------
    (1, 0)  9     



```python
idx = np.argmax(a, axis=1)
idx
```




    array([2, 0, 1, 2])




```python
a[np.arange(a.shape[0]), idx]
```




    array([7, 9, 7, 7])




```python
%C np.sort(a); np.sort(a, axis=0)
```

        np.sort(a)     np.sort(a, axis=0)
    -----------------  ------------------
    [[3, 4, 6, 6, 7],  [[3, 1, 6, 2, 1], 
     [2, 4, 6, 7, 9],   [4, 2, 7, 4, 4], 
     [2, 3, 5, 7, 7],   [6, 3, 7, 5, 5], 
     [1, 1, 4, 5, 7]]   [9, 7, 7, 7, 6]] 



```python
sort_axis1 = np.argsort(a)
sort_axis0 = np.argsort(a, axis=0)
%C sort_axis1; sort_axis0
```

        sort_axis1         sort_axis0   
    -----------------  -----------------
    [[1, 3, 0, 4, 2],  [[2, 3, 1, 2, 3],
     [1, 4, 2, 3, 0],   [3, 1, 0, 0, 1],
     [3, 0, 4, 1, 2],   [0, 0, 2, 3, 2],
     [1, 4, 0, 3, 2]]   [1, 2, 3, 1, 0]]



```python
axis0, axis1 = np.ogrid[:a.shape[0], :a.shape[1]]
```


```python
%C a[axis0, sort_axis1]; a[sort_axis0, axis1]
```

    a[axis0, sort_axis1]  a[sort_axis0, axis1]
    --------------------  --------------------
    [[3, 4, 6, 6, 7],     [[3, 1, 6, 2, 1],   
     [2, 4, 6, 7, 9],      [4, 2, 7, 4, 4],   
     [2, 3, 5, 7, 7],      [6, 3, 7, 5, 5],   
     [1, 1, 4, 5, 7]]      [9, 7, 7, 7, 6]]   



```python
names = ["zhang", "wang", "li", "wang", "zhang"]
ages = [37, 33, 32, 31, 36]
idx = np.lexsort([ages, names])
sorted_data = np.array(zip(names, ages), "O")[idx]
%C idx; sorted_data
```

          idx          sorted_data  
    ---------------  ---------------
    [2, 3, 1, 4, 0]  [['li', 32],   
                      ['wang', 31], 
                      ['wang', 33], 
                      ['zhang', 36],
                      ['zhang', 37]]



```python
b = np.random.randint(0, 10, (5, 3))
%C b; b[np.lexsort(b[:, ::-1].T)]
```

         b       b[np.lexsort(b[:, ::-1].T)]
    -----------  ---------------------------
    [[4, 0, 9],  [[3, 8, 2],                
     [5, 8, 0],   [4, 0, 9],                
     [9, 2, 6],   [4, 2, 6],                
     [3, 8, 2],   [5, 8, 0],                
     [4, 2, 6]]   [9, 2, 6]]                



```python
r = np.random.randint(10, 1000000, 100000)
%C np.sort(r)[:5]; np.partition(r, 5)[:5]
```

       np.sort(r)[:5]     np.partition(r, 5)[:5]
    --------------------  ----------------------
    [15, 23, 25, 37, 47]  [15, 47, 25, 37, 23]  



```python
%timeit np.sort(r)[:5]
%timeit np.sort(np.partition(r, 5)[:5])
```

    100 loops, best of 3: 6.02 ms per loop
    1000 loops, best of 3: 348 µs per loop



```python
np.median(a, axis=1)
```




    array([ 6.,  6.,  5.,  4.])




```python
r = np.abs(np.random.randn(100000))
np.percentile(r, [68.3, 95.4, 99.7])
```




    array([ 1.00029686,  1.99473003,  2.9614485 ])




```python
a = [2, 4, 8, 16, 16, 32]
v = [1, 5, 33, 16]
%C np.searchsorted(a, v); np.searchsorted(a, v, side="right")
```

    np.searchsorted(a, v)  np.searchsorted(a, v, side="right")
    ---------------------  -----------------------------------
    [0, 2, 6, 3]           [0, 2, 6, 5]                       



```python
x = np.array([3, 5, 7, 1, 9, 8, 6, 10])
y = np.array([2, 1, 5, 10, 100, 6])

def get_index_searchsorted(x, y):
    index = np.argsort(x)  # ❶
    sorted_x = x[index]  # ❷
    sorted_index = np.searchsorted(sorted_x, y)  # ❸
    yindex = np.take(index, sorted_index, mode="clip")  # ❹
    mask = x[yindex] != y  # ❺
    yindex[mask] = -1
    return yindex

get_index_searchsorted(x, y)
```




    array([-1,  3,  1,  7, -1,  6])




```python
x = np.random.permutation(1000)[:100]
y = np.random.randint(0, 1000, 2000)
xl, yl = x.tolist(), y.tolist()

def get_index_dict(x, y):
    idx_map = {v:i for i,v in enumerate(x)}
    yindex = [idx_map.get(v, -1) for v in y]
    return yindex

yindex1 = get_index_searchsorted(x, y)
yindex2 = get_index_dict(xl, yl)
print np.all(yindex1 == yindex2)

%timeit get_index_searchsorted(x, y)
%timeit get_index_dict(xl, yl)
```

    True
    10000 loops, best of 3: 122 µs per loop
    1000 loops, best of 3: 368 µs per loop

