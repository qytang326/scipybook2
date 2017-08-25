

```python
from tvtk.api import tvtk
import numpy as np
```

## 数据集

### ImageData


```python
img = tvtk.ImageData(spacing=(0.1,0.1,0.1), origin=(0.1,0.2,0.3), dimensions=(3,4,5))
```


```python
for n in range(6):
    print(("%.1f, %.1f, %.1f" % img.get_point(n)))
```

    0.1, 0.2, 0.3
    0.2, 0.2, 0.3
    0.3, 0.2, 0.3
    0.1, 0.3, 0.3
    0.2, 0.3, 0.3
    0.3, 0.3, 0.3



```python
img.point_data
```




    <tvtk.tvtk_classes.point_data.PointData at 0xa187780>




```python
print((img.point_data.scalars)) # 没有数据
img.point_data.scalars = np.arange(0.0, img.number_of_points)
print((type(img.point_data.scalars)))
img.point_data.scalars
```

    None
    <class 'tvtk.tvtk_classes.double_array.DoubleArray'>





    [0.0, ..., 59.0], length = 60




```python
a = img.point_data.scalars.to_array()
print(a)
a[:2] = 10, 11
print((img.point_data.scalars[0], img.point_data.scalars[1]))
```

    [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
      15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.  29.
      30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  42.  43.  44.
      45.  46.  47.  48.  49.  50.  51.  52.  53.  54.  55.  56.  57.  58.  59.]
    10.0 11.0



```python
img.point_data.scalars.number_of_tuples
```




    60




```python
img.point_data.scalars.name = 'scalars'
```


```python
data = tvtk.DoubleArray() # 创建一个空的DoubleArray数组
data.from_array(np.zeros(img.number_of_points))
```


```python
data.name = "zerodata"
```


```python
print((img.point_data.add_array(data)))
print((repr(img.point_data.get_array(1)))) # 获得第1个数组
print((img.point_data.get_array_name(1))) # 获得第1个数组的名字
print((repr(img.point_data.get_array(0)))) # 获得第0个数组
print((img.point_data.get_array_name(0))) # 获得第0个数组的名字
```

    1
    [0.0, ..., 0.0], length = 60
    zerodata
    [10.0, ..., 59.0], length = 60
    scalars



```python
img.point_data.remove_array("zerodata") # 删除名为"zerodata"的数组
img.point_data.number_of_arrays
```




    1




```python
vectors = np.arange(0.0, img.number_of_points*3).reshape(-1, 3)
img.point_data.vectors = vectors
print((repr(img.point_data.vectors)))
print((type(img.point_data.vectors)))
print((img.point_data.vectors[0]))
```

    [(0.0, 1.0, 2.0), ..., (177.0, 178.0, 179.0)], length = 60
    <class 'tvtk.tvtk_classes.double_array.DoubleArray'>
    (0.0, 1.0, 2.0)



```python
%C img.point_data.vectors.number_of_tuples; img.point_data.vectors.number_of_components
```

    img.point_data.vectors.number_of_tuples  img.point_data.vectors.number_of_components
    ---------------------------------------  -------------------------------------------
    60                                       3                                          


> **SOURCE**

> `scpy2.tvtk.figure_imagedata`：绘制`ref:fig-prev`的程序。


```python
#%hide
%exec_python -m scpy2.tvtk.figure_imagedata
```


```python
cell = img.get_cell(0)
print repr(cell)
%C cell.number_of_points; cell.number_of_edges; cell.number_of_faces
```

    <tvtk.tvtk_classes.voxel.Voxel object at 0x0A184C30>
    cell.number_of_points  cell.number_of_edges  cell.number_of_faces
    ---------------------  --------------------  --------------------
    8                      12                    6                   



```python
print((repr(cell.point_ids)))
cell.points.to_array()
```

    [0, 1, 3, 4, 12, 13, 15, 16]





    array([[ 0.1,  0.2,  0.3],
           [ 0.2,  0.2,  0.3],
           [ 0.1,  0.3,  0.3],
           [ 0.2,  0.3,  0.3],
           [ 0.1,  0.2,  0.4],
           [ 0.2,  0.2,  0.4],
           [ 0.1,  0.3,  0.4],
           [ 0.2,  0.3,  0.4]])




```python
img.number_of_cells
```




    24




```python
a = tvtk.IdList()
img.get_point_cells(3, a)
print(("cells of point 3:", repr(a)))
img.get_cell_points(0, a)
print(("points of cell 0:", repr(a))) # 和cell.point_ids的值相同
```

    cells of point 3: [2, 0]
    points of cell 0: [0, 1, 3, 4, 12, 13, 15, 16]



```python
a = tvtk.IdList()
a.from_array([1,2,3])
a.append(4)
a.extend([5,6])
print((repr(a)))
```

    [1, 2, 3, 4, 5, 6]



```python
img.cell_data
```




    <tvtk.tvtk_classes.cell_data.CellData at 0xa285c60>



### RectilinearGrid

> **SOURCE**

> `scpy2.tvtk.figure_rectilineargrid`：绘制`ref:fig-prev`的程序。


```python
#%hide
%exec_python -m scpy2.tvtk.figure_rectilineargrid
```


```python
x = np.array([0,3,9,15])
y = np.array([0,1,5])
z = np.array([0,2,3])
r = tvtk.RectilinearGrid()
r.x_coordinates = x #❶
r.y_coordinates = y
r.z_coordinates = z
r.dimensions = len(x), len(y), len(z) #❷

r.point_data.scalars = np.arange(0.0,r.number_of_points) #❸
r.point_data.scalars.name = 'scalars'
```


```python
for i in range(6):
    print((r.get_point(i)))
```

    (0.0, 0.0, 0.0)
    (3.0, 0.0, 0.0)
    (9.0, 0.0, 0.0)
    (15.0, 0.0, 0.0)
    (0.0, 1.0, 0.0)
    (3.0, 1.0, 0.0)



```python
c = r.get_cell(1)
print(("points of cell 1:", repr(c.point_ids)))
print((c.points.to_array()))
```

    points of cell 1: [1, 2, 5, 6, 13, 14, 17, 18]
    [[ 3.  0.  0.]
     [ 9.  0.  0.]
     [ 3.  1.  0.]
     [ 9.  1.  0.]
     [ 3.  0.  2.]
     [ 9.  0.  2.]
     [ 3.  1.  2.]
     [ 9.  1.  2.]]


### StructuredGrid

> **SOURCE**

> `scpy2.tvtk.figure_structuredgrid`：绘制`ref:fig-prev`的程序。


```python
#%hide
%exec_python -m scpy2.tvtk.figure_structuredgrid
```


```python
def make_points_array(x, y, z):
    return np.c_[x.ravel(), y.ravel(), z.ravel()]
    
z, y, x = np.mgrid[:3.0, :5.0, :4.0] #❶
x *= (4-z)/3 #❷
y *= (4-z)/3 
s1 = tvtk.StructuredGrid()
s1.points = make_points_array(x, y, z) #❸
s1.dimensions = x.shape[::-1] #❹
s1.point_data.scalars = np.arange(0, s1.number_of_points)
s1.point_data.scalars.name = 'scalars'
```


```python
s1.get_cell(2).point_ids
```




    [2, 3, 7, 6, 22, 23, 27, 26]




```python
c = s1.get_cell(2)
print(("cell type:", type(c)))
print(("number_of_faces:", c.number_of_faces)) #单元的面数
f = c.get_face(0) #获得第0个面
print(("face type:", type(f))) #每个面用一个Quad对象表示
print(("points of face 0:", repr(f.point_ids))) #构成第0面的四个点的下标
print(("edge count of cell:", c.number_of_edges)) # 单元的边数
e = c.get_edge(0) #获得第0个边
print(("edge type:", type(e)))
print(("points of edge 0:", repr(e.point_ids))) #构成第0边的两个点的下标
```

    cell type: <class 'tvtk.tvtk_classes.hexahedron.Hexahedron'>
    number_of_faces: 6
    face type: <class 'tvtk.tvtk_classes.quad.Quad'>
    points of face 0: [2, 22, 26, 6]
    edge count of cell: 12
    edge type: <class 'tvtk.tvtk_classes.line.Line'>
    points of edge 0: [2, 3]



```python
r, theta, z2 = np.mgrid[2:3:3j, -np.pi/2:np.pi/2:6j, 0:4:7j]
x2 = np.cos(theta)*r
y2 = np.sin(theta)*r

s2 = tvtk.StructuredGrid(dimensions=x2.shape[::-1])
s2.points = make_points_array(x2, y2, z2)
s2.point_data.scalars = np.arange(0, s2.number_of_points)
s2.point_data.scalars.name = 'scalars'
```

### PolyData


```python
source = tvtk.ConeSource(resolution = 4)
source.update() # 让source计算其输出数据
cone = source.output
type(cone)
```




    tvtk.tvtk_classes.poly_data.PolyData




```python
print((np.array_str(cone.points.to_array(), suppress_small=True)))
```

    [[ 0.5  0.   0. ]
     [-0.5  0.5  0. ]
     [-0.5  0.   0.5]
     [-0.5 -0.5  0. ]
     [-0.5 -0.  -0.5]]



```python
print((type(cone.polys)))
print((cone.polys.number_of_cells)) # 圆锥有5个面
print((cone.polys.to_array()))
```

    <class 'tvtk.tvtk_classes.cell_array.CellArray'>
    5
    [4 4 3 2 1 3 0 1 2 3 0 2 3 3 0 3 4 3 0 4 1]



```python
p1 = tvtk.PolyData()
p1.points = [(1,1,0),(1,-1,0),(-1,-1,0),(-1,1,0),(0,0,2)] #❶
faces = [ 
    4,0,1,2,3,
    3,4,0,1,
    3,4,1,2,
    3,4,2,3,
    3,4,3,0
    ]
cells = tvtk.CellArray() #❷
cells.set_cells(5, faces) #❸ 
p1.polys = cells
p1.point_data.scalars = np.linspace(0.0, 1.0, len(p1.points))
```

> **SOURCE**

> `scpy2.tvtk.figure_polydata`：绘制`ref:fig-prev`的程序。


```python
#%hide
%exec_python -m scpy2.tvtk.figure_polydata
```


```python
print((repr(p1.get_cell(0).point_ids)))
print((repr(p1.get_cell(1).point_ids)))
```

    [0, 1, 2, 3]
    [4, 0, 1]



```python
N = 10
a, b = np.mgrid[0:np.pi:N*1j, 0:np.pi:N*1j]
x = np.sin(a)*np.cos(b)
y = np.sin(a)*np.sin(b)
z = np.cos(a)

points = make_points_array(x, y, z) #❶
faces = np.zeros(((N-1)**2, 4), np.int) #❷
t1, t2 = np.mgrid[:(N-1)*N:N, :N-1]
faces[:,0] = (t1+t2).ravel()
faces[:,1] = faces[:,0] + 1
faces[:,2] = faces[:,1] + N
faces[:,3] = faces[:,0] + N

p2 = tvtk.PolyData(points = points, polys = faces)
p2.point_data.scalars = np.linspace(0.0, 1.0, len(p2.points))
```


```python
p2.polys.to_array()[:20]
```




    array([ 4,  0,  1, 11, 10,  4,  1,  2, 12, 11,  4,  2,  3, 13, 12,  4,  3,
            4, 14, 13])


