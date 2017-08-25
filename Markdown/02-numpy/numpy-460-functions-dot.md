

```python
import numpy as np
```

### 各种乘积运算


```python
a = np.array([1, 2, 3])
%C a[:, None]; a[None, :]
```

    a[:, None]   a[None, :]
    ----------  -----------
    [[1],       [[1, 2, 3]]
     [2],                  
     [3]]                  



```python
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(12, 24).reshape(2, 2, 3)
c = np.dot(a, b)
c.shape
```




    (2, 3, 2, 3)




```python
for i, j in np.ndindex(2, 2):
    assert np.alltrue( c[i, :, j, :] == np.dot(a[i], b[j]) )
```


```python
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(12, 24).reshape(2, 3, 2)
c = np.inner(a, b)
c.shape
```




    (2, 3, 2, 3)




```python
for i, j, k, l in np.ndindex(2, 3, 2, 3):
    assert c[i, j, k, l] == np.inner(a[i, j], b[k, l])
```


```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])
%C np.outer(a, b); np.dot(a[:, None], b[None, :])
```

      np.outer(a, b)    np.dot(a[:, None], b[None, :])
    ------------------  ------------------------------
    [[ 4,  5,  6,  7],  [[ 4,  5,  6,  7],            
     [ 8, 10, 12, 14],   [ 8, 10, 12, 14],            
     [12, 15, 18, 21]]   [12, 15, 18, 21]]            



```python
a = np.random.rand(3, 4)
b = np.random.rand(4, 5)

c1 = np.tensordot(a, b, axes=[[1], [0]]) #❶
c2 = np.tensordot(a, b, axes=1)          #❷
c3 = np.dot(a, b)
assert np.allclose(c1, c3)
assert np.allclose(c2, c3)
```


```python
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(12, 24).reshape(2, 2, 3)
c1 = np.tensordot(a, b, axes=[[-1], [-2]])
c2 = np.dot(a, b)
assert np.alltrue(c1 == c2)
```


```python
a = np.random.rand(4, 5, 6, 7)
b = np.random.rand(6, 5, 2, 3)
c = np.tensordot(a, b, axes=[[1, 2], [1, 0]])

for i, j, k, l in np.ndindex(4, 7, 2, 3):
    assert np.allclose(c[i, j, k, l], np.sum(a[i, :, :, j] * b[:, :, k, l].T))
    
c.shape
```




    (4, 7, 2, 3)


