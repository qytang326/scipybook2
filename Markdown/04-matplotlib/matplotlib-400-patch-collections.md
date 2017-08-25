

```python
%matplotlib_svg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
```

## 块、路径和集合

### Path与Patch


```python
rect_patch = plt.Rectangle((0, 1), 2, 1)
rect_path = rect_patch.get_path()
%C rect_path.vertices; rect_path.codes
```

    rect_path.vertices    rect_path.codes   
    ------------------  --------------------
    [[ 0.,  0.],        [ 1,  2,  2,  2, 79]
     [ 1.,  0.],                            
     [ 1.,  1.],                            
     [ 0.,  1.],                            
     [ 0.,  0.]]                            



```python
tran = rect_patch.get_patch_transform()
tran.transform(rect_path.vertices)
```




    array([[ 0.,  1.],
           [ 2.,  1.],
           [ 2.,  2.],
           [ 0.,  2.],
           [ 0.,  1.]])



> **SOURCE**

> `scpy2.matplotlib.svg_path`：从SVG文件中获取简单的路径信息。可以使用该模块将矢量绘图软件创建的图形转换为`Patch`对象。


```python
#%fig=使用本书提供的read_svg_path()读入SVG文件中的路径并显示为Patch对象
from scpy2.matplotlib.svg_path import read_svg_path

ax = plt.gca()
patches = read_svg_path("python-logo.svg")
for patch in patches:
    ax.add_patch(patch)

ax.set_aspect("equal")
ax.invert_yaxis()
ax.autoscale();
```


![svg](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_6_0.svg)


### 集合

#### 曲线集合(LineCollection)


```python
#%fig=使用LineCollection显示大量曲线
from matplotlib import collections as mc
lines = []
with open("butterfly.txt", "r") as f:
    for line in f:
        points = line.strip().split()
        points.extend(points[:2]) # ❶
        points = np.array(points).reshape(-1, 2) # ❷
        lines.append(points)
        
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
lc1 = mc.LineCollection(lines, colors="k", linewidths=1) # ❸
lc2 = mc.LineCollection(lines, cmap="Paired", linewidths=1, # ❹
                        array=np.log2(np.array([len(line) for line in lines])))
ax1.add_collection(lc1)
ax2.add_collection(lc2)

for ax in ax1, ax2:
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
```


![svg](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_9_0.svg)



```python
print(("number of lc1 paths:", len(lc1.get_paths())))
print(("number of lc1 colors:", len(lc1.get_edgecolors())))
print(("number of lc2 colors:", len(lc2.get_edgecolors())))
print((np.all(lc2.get_edgecolors() == lc2.cmap(lc2.norm(lc2.get_array())))))
```

    number of lc1 paths: 145
    number of lc1 colors: 1
    number of lc2 colors: 145
    True



```python
print((lc1.get_transforms())) # 路径变换
print((lc1.get_transform() is ax1.transData)) # 主变换为数据坐标变换对象
print((lc1.get_offset_transform(), lc1.get_offsets()))
```

    []
    True
    IdentityTransform() [[ 0.  0.]]



```python
from scipy.integrate import odeint

def field(s, t):
    x, y = s
    return 0.3 * x - y, 0.3 * y + x
    return [u, v]

X, Y = np.mgrid[-2:2:5j, -2:2:5j]
init_pos = np.c_[X.ravel(), Y.ravel()]
t = np.linspace(0, 5, 50)

streams = []
for pos in init_pos:
    r = odeint(field, pos, t)
    streams.append(r)

print((len(streams), streams[0].shape))
```

    25 (50, 2)



```python
#%hide
%matplotlib_png
```


```python
#%fig=使用LineCollection绘制颜色渐变的曲线
lines = np.concatenate([
    np.concatenate((r[:-1, None, :], r[1:, None, :]), axis=1)
    for r in streams], axis=0)

time_value = np.concatenate([t[:-1]] * len(streams))
x, y = lines.mean(axis=1).T
u, v = field([x, y], 0)
speed_value = np.sqrt(u ** 2 + v ** 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
fig.subplots_adjust(0, 0, 1, 1)
ax1.plot(init_pos[:, 0], init_pos[:, 1], "x")
ax2.plot(init_pos[:, 0], init_pos[:, 1], "x")

lc1 = mc.LineCollection(lines, linewidths=2, array=time_value)
lc2 = mc.LineCollection(lines, linewidths=2, array=speed_value)

ax1.add_collection(lc1)
ax2.add_collection(lc2)

plt.colorbar(ax=ax1, mappable=lc1, label="时间")
plt.colorbar(ax=ax2, mappable=lc2, label="速度")

for ax in ax1, ax2:
    ax.plot(init_pos[:, 0], init_pos[:, 1], "x")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
```


![png](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_14_0.png)



```python
#%hide
%matplotlib_svg
```

#### 多边形集合(PolyCollection)


```python
#%fig=用PolyCollection绘制大量多边形
from numpy.random import randint, rand, uniform

def star_polygon(x, y, r, theta, n, s): 
    angles = np.arange(0, 2*np.pi, 2*np.pi/2/n) + theta
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)
    xs[1::2] *= s
    ys[1::2] *= s
    xs += x
    ys += y
    return np.vstack([xs, ys]).T

stars = []
for i in range(1000):
    star = star_polygon(randint(800), randint(500), 
                        uniform(5, 20), uniform(0, 2*np.pi),
                        randint(3, 9), uniform(0.1, 0.7))
    stars.append(star)

fig, ax = plt.subplots(figsize=(10, 5))
polygons = mc.PolyCollection(stars, alpha=0.5, array=np.random.rand(len(stars)))
ax.add_collection(polygons)
ax.autoscale()
ax.margins(0)
ax.set_aspect("equal")
```


![svg](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_17_0.svg)



```python
print(("length of facecolors:", len(polygons.get_facecolors())))
print(("length of edgecolors:", len(polygons.get_edgecolors())))
```

    length of facecolors: 1000
    length of edgecolors: 1


#### 路径集合(PathCollection)


```python
#%nofig
N = 30
np.random.seed(42)
x = np.random.rand(N)
y = np.random.rand(N)
size = np.random.randint(20, 60, N)
value = np.random.rand(N)

fig, ax = plt.subplots()
pc = ax.scatter(x, y, s=size, c=value)
```


![svg](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_20_0.svg)



```python
print((pc.get_transforms().shape))
print((pc.get_transforms()[0])) #下标为0的点对应的缩放矩阵
```

    (30, 3, 3)
    [[ 5.91607978  0.          0.        ]
     [ 0.          5.91607978  0.        ]
     [ 0.          0.          1.        ]]



```python
print((pc.get_offsets()[0])) #下标为0的点对应的中心坐标
#计算下标为0的点对应的屏幕坐标
print((pc.get_offset_transform().transform(pc.get_offsets())[0]))
print((pc.get_offset_transform() is ax.transData))
```

    [ 0.37454012  0.60754485]
    [ 212.66351729  134.74900826]
    True



```python
print((pc.get_transform()))
```

    IdentityTransform()



```python
pc.get_offset_position()
```




    u'screen'



#### 椭圆集合(EllipseCollection)


```python
#%fig=`EllipseColletion`的`unit`参数：`unit='x'`（左图）、`unit='xy'`（右图）
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
offsets = np.c_[3*np.cos(angles), 2*np.sin(angles)]
angles_deg = np.rad2deg(angles)
widths = np.full_like(angles, 2)
heights = np.full_like(angles, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ec0 = mc.EllipseCollection(widths, heights, angles_deg, units="x", array=angles,
                          offsets=offsets, transOffset=axes[0].transData)
axes[0].add_collection(ec0)
axes[0].axis((-5, 5, -5, 5))

ec1 = mc.EllipseCollection(widths, heights, angles_deg, units="xy", array=angles,
                          offsets=offsets, transOffset=axes[1].transData)
axes[1].add_collection(ec1)
axes[1].axis((-5, 5, -5, 5))
#axes[1].set_aspect("equal");
```


![png](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_26_0.png)


#### 数据空间中的圆形集合对象


```python
from matplotlib.collections import CircleCollection, Collection
from matplotlib.transforms import Affine2D

class DataCircleCollection(CircleCollection):

    def set_sizes(self, sizes):
        self._sizes = sizes

    def draw(self, render):
        ax = self.axes
        ms = np.zeros((len(self._sizes), 3, 3))
        ms[:, 0, 0] = self._sizes
        ms[:, 1, 1] = self._sizes
        ms[:, 2, 2] = 1
        self._transforms = ms #❶

        m = ax.transData.get_affine().get_matrix().copy()
        m[:2, 2:] = 0
        self.set_transform(Affine2D(m)) #❷

        return Collection.draw(self, render)
```


```python
#%fig=使用DataCircleCollection绘制大量的圆形
data = np.loadtxt("venus-face.csv", delimiter=",")
offsets = data[:, :2]
sizes = data[:, 2] * 1.05
colors = data[:, 3:] / 256.0

fig, axe = plt.subplots(figsize=(8, 8))
axe.set_rasterized(True)
cc = DataCircleCollection(sizes, facecolors=colors, edgecolors="w", linewidths=0.1,
                          offsets=offsets, transOffset=axe.transData)

axe.add_collection(cc)
axe.axis((0, 512, 512, 0))
axe.axis("off");
```


![svg](matplotlib-400-patch-collections_files/matplotlib-400-patch-collections_29_0.svg)

