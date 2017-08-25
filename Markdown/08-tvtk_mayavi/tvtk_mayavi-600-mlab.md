

```python
%gui qt
from tvtk.api import tvtk
import numpy as np 
from mayavi import mlab
from scpy2.tvtk import fix_mayavi_bugs
fix_mayavi_bugs()
```

    WARNING:traits.has_traits:DEPRECATED: traits.has_traits.wrapped_class, 'the 'implements' class advisor has been deprecated. Use the 'provides' class decorator.


## 用mlab快速绘图

> **WARNING**

> 最新版本的Mayavi 4.4.0中存在GUI操作不更新3D场景的问题，可以通过本书提供的`scipy.tvtk.fix_mayavi_bugs()`修复这些问题。

### 点和线


```python
from scipy.integrate import odeint 

def lorenz(w, t, p, r, b): 
    x, y, z = w
    return np.array([p*(y-x), x*(r-z)-y, x*y-b*z]) 

t = np.arange(0, 30, 0.01) 
track1 = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0)) #❶

from mayavi import mlab
X, Y, Z = track1.T
mlab.plot3d(X, Y, Z, t, tube_radius=0.2) #❷
mlab.show()
```

### Mayavi的流水线


```python
s = mlab.gcf() # 首先获得当前的场景
print(s)
print((s.scene.background))
```

    <mayavi.core.scene.Scene object at 0x13A320F0>
    (0.5, 0.5, 0.5)



```python
source = s.children[0] # 获得场景的第一个子节点，也就是LineSource
print((repr(source)))
print((source.name)) # 节点的名字，也就流水线中显示的文字
print((repr(source.data.points))) # LineSource中的坐标点
print((repr(source.data.point_data.scalars))) #每个点所对应的标量数组
```

    <mayavi.sources.vtk_data_source.VTKDataSource object at 0x13A06CF0>
    LineSource
    [(0.0, 1.0, 0.0), ..., (0.021550891680468726, 1.6938271906706417, 20.31711497016887)], length = 3000
    [0.0, ..., 29.99], length = 3000



```python
stripper = source.children[0]
print((stripper.filter.maximum_length))
print((stripper.outputs[0].number_of_points))
print((repr(stripper.outputs[0])))
print((stripper.outputs[0].number_of_lines))
```

    1000
    3000
    <tvtk.tvtk_classes.poly_data.PolyData object at 0x0CD527B0>
    3



```python
tube = stripper.children[0] # 获得Tube对象
print((repr(tube.outputs[0]))) # tube的输出是一个PolyData对象，它是一个三维圆管
```

    <tvtk.tvtk_classes.poly_data.PolyData object at 0x0CD52210>



```python
manager = tube.children[0]
manager.scalar_lut_manager.lut_mode = 'Blues'
manager.scalar_lut_manager.show_legend = True
```


```python
surface = manager.children[0]
surface.actor.property.representation = 'wireframe'    
surface.actor.property.opacity = 0.6    
```


```python
surface.actor.property.line_width    
```




    2.0



### 二维图像的可视化


```python
x, y = np.ogrid[-2:2:20j, -2:2:20j] #❶
z = x * np.exp( - x**2 - y**2) #❷

face = mlab.surf(x, y, z, warp_scale=2) #❸
axes = mlab.axes(xlabel='x', ylabel='y', zlabel='z', color=(0, 0, 0)) #❹
outline = mlab.outline(face, color=(0, 0, 0))
#%hide
fig = mlab.gcf()
fig.scene.background = 1, 1, 1
axis_color = 0.4, 0.4, 0.4
outline.actor.property.color = axis_color
axes.actors[0].property.color = axis_color
axes.title_text_property.color = axis_color
axes.label_text_property.color = axis_color
mlab.show()
```


```python
data = mlab.gcf().children[0]
img = data.outputs[0]
img
```




    <tvtk.tvtk_classes.image_data.ImageData at 0x13a243f0>




```python
print((img.origin)) # X,Y,Z轴的起点
print((img.spacing)) # X,Y,Z轴上的点的间隔
print((img.dimensions)) # X,Y,Z轴上的点的个数
print((repr(img.point_data.scalars))) # 每个点所对应的标量值
```

    [-2. -2.  0.]
    [ 0.21052632  0.21052632  1.        ]
    [20 20  1]
    [-0.000670925255805, ..., 0.000670925255805], length = 400



```python
data.children[0].outputs[0]
```




    <tvtk.tvtk_classes.poly_data.PolyData at 0x14263720>




```python
x, y = np.ogrid[-10:10:100j, -1:1:100j]
z = np.sin(5*((x/10)**2+y**2))
```


```python
mlab.surf(x, y, z)
mlab.axes();
```


```python
mlab.surf(x, y, z, extent=(-1,1,-1,1,-0.5,0.5))
mlab.axes(nb_labels=5);
```


```python
mlab.surf(x, y, z, extent=(-1,1,-1,1,-0.5,0.5))
mlab.axes(ranges=(x.min(),x.max(),y.min(),y.max(),z.min(),z.max()), nb_labels=5);
```


```python
x, y = np.ogrid[-2:2:20j, -2:2:20j]
z = x * np.exp( - x**2 - y**2)

mlab.imshow(x, y, z)
mlab.show()
```


```python
mlab.contour_surf(x,y,z,warp_scale=2,contours=20);
```


```python
face.enable_contours = True
face.contour.number_of_contours = 20
```

### 网格面mesh


```python
from numpy import sin, cos
dphi, dtheta = np.pi/80.0, np.pi/80.0
phi, theta = np.mgrid[0:np.pi+dphi*1.5:dphi, 0:2*np.pi+dtheta*1.5:dtheta]
m0, m1, m2, m3, m4, m5, m6, m7 = 4,3,2,3,6,2,6,4
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7 #❶
x = r*sin(phi)*cos(theta) #❷
y = r*cos(phi)
z = r*sin(phi)*sin(theta)
s = mlab.mesh(x, y, z) #❸

mlab.show()
```


```python
x = [[-1,1,1,-1,-1],
     [-1,1,1,-1,-1]]

y = [[-1,-1,-1,-1,-1],
     [ 1, 1, 1, 1, 1]]

z = [[1,1,-1,-1,1],
     [1,1,-1,-1,1]]

box = mlab.mesh(x, y, z, representation="surface")
mlab.axes(xlabel='x', ylabel='y', zlabel='z')
mlab.outline(box)
mlab.show()
```


```python
rho, theta = np.mgrid[0:1:40j, 0:2*np.pi:40j] #❶

z = rho*rho #❷

x = rho*np.cos(theta) #❸
y = rho*np.sin(theta) 

s = mlab.mesh(x,y,z)
mlab.show()
```


```python
x, y = np.mgrid[-2:2:20j, -2:2:20j] #❶
z = x * np.exp( - x**2 - y**2)
z *= 2
c = 2*x + y #❷

pl = mlab.mesh(x, y, z, scalars=c) #❸
mlab.axes(xlabel='x', ylabel='y', zlabel='z')
mlab.outline(pl)
mlab.show()
```

### 修改和创建流水线


```python
x, y = np.ogrid[-2:2:20j, -2:2:20j]
z = x * np.exp( - x**2 - y**2)

face = mlab.surf(x, y, z, warp_scale=2)
mlab.axes(xlabel='x', ylabel='y', zlabel='z')
mlab.outline(face);
```


```python
source = mlab.gcf().children[0]
print(source)
img = source.image_data
print((repr(img)))
```

    <mayavi.sources.array_source.ArraySource object at 0x127FCE70>
    <tvtk.tvtk_classes.image_data.ImageData object at 0x127C5960>



```python
c = 2*x + y # 表示颜色的标量数组
array_id = img.point_data.add_array(c.T.ravel())
img.point_data.get_array(array_id).name = "color"
```


```python
source.update()
source.pipeline_changed = True
```


```python
print((z[:3,:3])) # 原始的二维数组中元素
# ImageData中的标量值的顺序
print((img.point_data.scalars.to_array()[:3])) # 和数组z的第0列的数值相同
```

    [[-0.00067093 -0.00148987 -0.00302777]
     [-0.00133304 -0.00296016 -0.00601578]
     [-0.00239035 -0.00530804 -0.01078724]]
    [-0.00067093 -0.00133304 -0.00239035]



```python
normals = mlab.gcf().children[0].children[0].children[0]
```


```python
normals.outputs[0].point_data.scalars.to_array()[:3]
```




    array([-0.00067093, -0.00133304, -0.00239035])




```python
surf = normals.children[0]
del normals.children[0]
```


```python
active_attr = mlab.pipeline.set_active_attribute(normals, point_scalars="color")
```


```python
active_attr.children.append(surf)    
```


```python
normals.children[0].outputs[0].point_data.scalars.to_array()[:3]
```




    array([-6.        , -5.57894737, -5.15789474])




```python
src = mlab.pipeline.array2d_source(x, y, z) #创建ArraySource数据源
#添加color数组
image = src.image_data
array_id = image.point_data.add_array(c.T.ravel())
image.point_data.get_array(array_id).name = "color"
src.update() #更新数据源的输出

# 创建流水线上后续对象
warp = mlab.pipeline.warp_scalar(src, warp_scale=2.0)
normals = mlab.pipeline.poly_data_normals(warp)
active_attr = mlab.pipeline.set_active_attribute(normals,
    point_scalars="color")
surf = mlab.pipeline.surface(active_attr)
mlab.axes()
mlab.outline()
mlab.show()
```

### 标量场

> **SOURCE**

> `scpy2.tvtk.mlab_scalar_field`：使用等值面、体素呈像和切面可视化标量场


```python
#%hide
%exec_python -m scpy2.tvtk.mlab_scalar_field
```


```python
x, y, z = np.ogrid[-2:2:40j, -2:2:40j, -2:0:40j]
s = 2/np.sqrt((x-1)**2 + y**2 + z**2) + 1/np.sqrt((x+1)**2 + y**2 + z**2)
```


```python
surface = mlab.contour3d(s)
```


```python
surface.contour.maximum_contour = 15 # 等值面的上限值为15
surface.contour.number_of_contours = 10 # 在最小值到15之间绘制10个等值面
surface.actor.property.opacity = 0.4 # 透明度为0.4
```


```python
field = mlab.pipeline.scalar_field(s)
mlab.pipeline.volume(field);
```


```python
mlab.pipeline.volume(field, vmin=1.5, vmax=10);
```


```python
cut = mlab.pipeline.scalar_cut_plane(field.children[0], plane_orientation="y_axes")
```


```python
cut.enable_contours = True # 开启等高线显示
cut.contour.number_of_contours = 40 # 等高线的数目为40
```

### 矢量场

> **SOURCE**

> `scpy2.tvtk.mlab_vector_field`：使用矢量箭头、切片、等梯度面和流线显示矢量场


```python
#%hide
%exec_python -m scpy2.tvtk.mlab_vector_field
```


```python
p, r, b = (10.0, 28.0, 3.0)
x, y, z = np.mgrid[-17:20:20j, -21:28:20j, 0:48:20j]
u, v, w = p*(y-x), x*(r-z)-y, x*y-b*z
```


```python
vectors = mlab.quiver3d(x, y, z, u, v, w)
```


```python
vectors.glyph.mask_input_points = True  # 开启使用部分数据的选项
vectors.glyph.mask_points.on_ratio = 20 # 随机选择原始数据中的1/20个点进行描绘
vectors.glyph.glyph.scale_factor = 5.0 # 设置箭头的缩放比例
```


```python
src = mlab.pipeline.vector_field(x, y, z, u, v, w)
cut_plane = mlab.pipeline.vector_cut_plane(src, scale_factor=3)
cut_plane.glyph.mask_points.maximum_number_of_points = 10000
cut_plane.glyph.mask_points.on_ratio = 2
cut_plane.glyph.mask_input_points = True
```


```python
magnitude = mlab.pipeline.extract_vector_norm(src)
```


```python
surface = mlab.pipeline.iso_surface(magnitude)
surface.actor.property.opacity = 0.3
```


```python
print((repr(magnitude.outputs[0].point_data.scalars)))
print((repr(magnitude.outputs[0].point_data.vectors)))
```

    [579.71887207, ..., 602.195983887], length = 8000
    [(-40.0, -455.0, 357.0), ..., (80.0, -428.0, 416.0)], length = 8000



```python
mlab.flow(x, y, z, u, v, w);
```


```python

```
