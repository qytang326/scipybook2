

```python
%matplotlib_svg
import numpy as np
from matplotlib import pyplot as plt
```

## 使用泊松混合合成图像

### 泊松混合算法

###编写代码 


```python
import cv2

offset_x, offset_y = -36, 42
src = cv2.imread("vinci_src.png", 1)
dst = cv2.imread("vinci_target.png", 1)
mask = cv2.imread("vinci_mask.png", 0)
src_mask = (mask > 128).astype(np.uint8)

src_y, src_x = np.where(src_mask) #❶
src_laplacian = cv2.Laplacian(src, cv2.CV_16S, ksize=1)[src_y, src_x, :] #❷
```


```python
dst_mask = np.zeros(dst.shape[:2], np.uint8)
dst_x, dst_y = src_x + offset_x, src_y + offset_y
dst_mask[dst_y, dst_x] = 1  #❶

kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
dst_mask2 = cv2.dilate(dst_mask, kernel=kernel)   #❷

dst_y2, dst_x2 = np.where(dst_mask2)              #❸
dst_ye, dst_xe = np.where(dst_mask2 - dst_mask)   #❹
```


```python
variable_count = len(dst_x2)
variable_index = np.arange(variable_count)  #❶

variables = np.zeros(dst.shape[:2], np.int)
variables[dst_y2, dst_x2] = variable_index

x0 = variables[dst_y  , dst_x  ]  #❷
x1 = variables[dst_y-1, dst_x  ]
x2 = variables[dst_y+1, dst_x  ]
x3 = variables[dst_y  , dst_x-1]
x4 = variables[dst_y  , dst_x+1]
x_edge = variables[dst_ye, dst_xe]  #❸
```


```python
from scipy.sparse import coo_matrix
inner_count = len(x0)
edge_count = len(x_edge)

r = np.r_[x0, x0, x0, x0, x0, x_edge]   
c = np.r_[x0, x1, x2, x3, x4, x_edge]   
v = np.ones(inner_count*5 + edge_count) 
v[:inner_count] = -4                    

A = coo_matrix((v, (r, c))).tocsc()     
```


```python
from scipy.sparse.linalg import spsolve
order = np.argsort(np.r_[variables[dst_y, dst_x], variables[dst_ye, dst_xe]]) #❶

result = dst.copy()

for ch in (0, 1, 2): #❷
    b = np.r_[src_laplacian[:,ch], dst[dst_ye, dst_xe, ch]] #❸
    u = spsolve(A, b[order]) #❹
    u = np.clip(u, 0, 255)
    result[dst_y2, dst_x2, ch] = u #❺
```


```python
#%fig=使用泊松混合算法将吉内薇拉·班琪肖像中的眼睛和鼻子部分复制到蒙娜丽莎的肖像之上
fig, axes = plt.subplots(1, 4, figsize=(10, 4))
ax1, ax2, ax3, ax4 = axes.ravel()
ax1.imshow(src[:, :, ::-1])
ax2.imshow(mask, cmap="gray")
ax3.imshow(dst[:, :, ::-1])
ax4.imshow(result[:, :, ::-1])

for ax in axes.ravel():
    ax.axis("off")
    
fig.subplots_adjust(wspace=0.05)
```


![svg](examples-100-possion_files/examples-100-possion_9_0.svg)


###演示程序

> **SOURCE**

> `scpy2.examples.possion`：使用TraitsUI编写的泊松混合演示程序。该程序使用`scpy2.matplotlib.freedraw_widget`中提供的`ImageMaskDrawer`在图像上绘制半透明的白色区域。


```python
#%hide
%exec_python -m scpy2.examples.possion
```
