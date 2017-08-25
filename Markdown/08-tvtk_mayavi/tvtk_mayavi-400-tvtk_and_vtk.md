

```python
%gui qt
import numpy as np
from tvtk.api import tvtk
```

## TVTK的改进


```python
%%python

# -*- coding: utf-8 -*-
import vtk

# 创建一个圆锥数据源
cone = vtk.vtkConeSource( )
cone.SetHeight( 3.0 )
cone.SetRadius( 1.0 )
cone.SetResolution(10)
# 使用PolyDataMapper将数据转换为图形数据
coneMapper = vtk.vtkPolyDataMapper( )
coneMapper.SetInputConnection( cone.GetOutputPort( ) )
# 创建一个Actor
coneActor = vtk.vtkActor( )
coneActor.SetMapper ( coneMapper )
# 用线框模式显示圆锥
coneActor.GetProperty( ).SetRepresentationToWireframe( )
# 创建Renderer和窗口
ren1 = vtk.vtkRenderer( )
ren1.AddActor( coneActor )
ren1.SetBackground( 0.1 , 0.2 , 0.4 )
renWin = vtk.vtkRenderWindow( )
renWin.AddRenderer( ren1 )
renWin.SetSize(300 , 300)
# 创建交互工具
iren = vtk.vtkRenderWindowInteractor( )
iren.SetRenderWindow( renWin )
iren.Initialize( )
iren.Start( )
```

### TVTK的基本用法


```python
from tvtk.api import tvtk

cs = tvtk.ConeSource(height=3.0, radius=1.0, resolution=36)
m = tvtk.PolyDataMapper(input_connection = cs.output_port)
a = tvtk.Actor(mapper=m)
ren = tvtk.Renderer(background=(1, 1, 1))
ren.add_actor(a)
rw = tvtk.RenderWindow(size=(300,300))
rw.add_renderer(ren)
rwi = tvtk.RenderWindowInteractor(render_window=rw) 
rwi.initialize()
rwi.start()
```

### Trait属性


```python
p = tvtk.Property()
p.set(opacity=0.5, color=(1,0,0), representation="w")
```


```python
p.edit_traits()
```


```python
print((p.representation))
p_vtk = tvtk.to_vtk(p)
p_vtk.SetRepresentationToSurface()
print((p.representation))
```

    wireframe
    surface


### 序列化


```python
import pickle
p = tvtk.Property()
p.representation = "w"
s = pickle.dumps(p)
del p
q = pickle.loads(s)
q.representation
```




    'wireframe'




```python
p = tvtk.Property()
p.interpolation = "flat"
d = p.__getstate__()
del p
q = tvtk.Property()
print((q.interpolation))
q.__setstate__(d)
print((q.interpolation))
```

    gouraud
    flat


### 集合迭代


```python
ac = tvtk.ActorCollection()
print((len(ac)))
ac.append(tvtk.Actor())
ac.append(tvtk.Actor())
print((len(ac)))

for a in ac:
    print((repr(a)))

del ac[0]
print((len(ac)))
```

    0
    2
    <tvtk.tvtk_classes.open_gl_actor.OpenGLActor object at 0x0A24A690>
    <tvtk.tvtk_classes.open_gl_actor.OpenGLActor object at 0x0A174ED0>
    1



```python
import vtk
ac = vtk.vtkActorCollection()
print((ac.GetNumberOfItems()))
ac.AddItem(vtk.vtkActor())
ac.AddItem(vtk.vtkActor())
print((ac.GetNumberOfItems()))

ac.InitTraversal()
for i in range(ac.GetNumberOfItems()):
    print((repr(ac.GetNextItem())))
    
ac.RemoveItem(0)
print((ac.GetNumberOfItems()))
```

    0
    2
    (vtkOpenGLActor)0A24AF90
    (vtkOpenGLActor)0A24AF00
    1


### 数组操作


```python
pts = tvtk.Points()
p_array = np.eye(3)
pts.from_array(p_array)
pts.print_traits()
pts.to_array()
```

    _in_set:                 0
    _vtk_obj:                (vtkPoints)0A2D82D0
    actual_memory_size:      1
    bounds:                  (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    class_name:              'vtkPoints'
    data:                    [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    data_type:               'double'
    data_type_:              11
    debug:                   0
    debug_:                  0
    global_warning_display:  1
    global_warning_display_: 1
    m_time:                  44927
    number_of_points:        3
    reference_count:         1





    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])




```python
points = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], 'f')
triangles = np.array([[0,1,3],[0,3,2],[1,2,3],[0,2,1]])
values = np.array([1.1, 1.2, 2.1, 2.2])
mesh = tvtk.PolyData(points=points, polys=triangles)
mesh.point_data.scalars = values
print((repr(mesh.points)))
print((repr(mesh.polys)))
print((mesh.polys.to_array()))
print((mesh.point_data.scalars.to_array()))
```

    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    <tvtk.tvtk_classes.cell_array.CellArray object at 0x0A2E5360>
    [3 0 1 3 3 0 3 2 3 1 2 3 3 0 2 1]
    [ 1.1  1.2  2.1  2.2]

