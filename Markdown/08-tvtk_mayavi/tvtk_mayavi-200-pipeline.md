

```python
%gui qt
```

## VTK的流水线(Pipeline)

### 显示圆锥


```python
%%python

#coding=utf-8
from tvtk.api import tvtk #❶

# 创建一个圆锥数据源，并且同时设置其高度，底面半径和底面圆的分辨率(用36边形近似)
cs = tvtk.ConeSource(height=3.0, radius=1.0, resolution=36) #❷
# 使用PolyDataMapper将数据转换为图形数据
m = tvtk.PolyDataMapper(input_connection=cs.output_port) #❸
# 创建一个Actor
a = tvtk.Actor(mapper=m) #❹
# 创建一个Renderer，将Actor添加进去
ren = tvtk.Renderer(background=(1, 1, 1)) #❺
ren.add_actor(a)
# 创建一个RenderWindow(窗口)，将Renderer添加进去
rw = tvtk.RenderWindow(size=(300,300)) #❻
rw.add_renderer(ren)
# 创建一个RenderWindowInteractor（窗口的交互工具)
rwi = tvtk.RenderWindowInteractor(render_window=rw) #❼
# 开启交互
rwi.initialize()
rwi.start()
```


```python
from tvtk.api import tvtk
cs = tvtk.ConeSource(height=3.0, radius=1.0, resolution=36)
m = tvtk.PolyDataMapper(input_connection=cs.output_port)
a = tvtk.Actor(mapper=m)
ren = tvtk.Renderer(background=(1, 1, 1))
ren.add_actor(a)

%omit cs.trait_names()
```

    ['number_of_output_ports',
     'abort_execute_',
     'class_name',
     'executive',
    ...



```python
%C cs.height; cs.radius; cs.resolution
```

    cs.height  cs.radius  cs.resolution
    ---------  ---------  -------------
    3.0        1.0        36           



```python
print((type(cs.output), cs.output is m.input))
```

    <class 'tvtk.tvtk_classes.poly_data.PolyData'> True



```python
print((a.mapper is m))
print((a.scale)) # Actor对象的scale属性表示各个轴的缩放比例
```

    True
    [ 1.  1.  1.]



```python
ren.actors
```




    ['<tvtk.tvtk_classes.actor.Actor object at 0x0D7C7BD0>']



### 用ivtk观察流水线


```python
from tvtk.api import tvtk
from scpy2.tvtk.tvtkhelp import ivtk_scene, event_loop

cs = tvtk.ConeSource(height=3.0, radius=1.0, resolution=36)
m = tvtk.PolyDataMapper(input_connection=cs.output_port)
a = tvtk.Actor(mapper=m)

window = ivtk_scene([a]) #❶
window.scene.isometric_view()
event_loop() #❷
```

#### 照相机


```python
#%hide
camera.clipping_range = [4.22273550264, 12.6954685402]
```


```python
camera = window.scene.renderer.active_camera
print((camera.clipping_range))
camera.view_up = 0, 1, 0
camera.edit_traits() # 显示编辑照相机属性的窗口;
```

    [  4.2227355   12.69546854]


#### 光源


```python
lights = window.scene.renderer.lights
lights[0].edit_traits() # 显示编辑光源属性的窗口;
```


```python
camera = window.scene.renderer.active_camera
light = tvtk.Light(color=(1,0,0))
light.position=camera.position
light.focal_point=camera.focal_point
window.scene.renderer.add_light(light)
```

#### 实体


```python
a.edit_traits() # a是表示圆锥的Actor对象
window.scene.renderer.actors[0].edit_traits();
```


```python
axe = tvtk.AxesActor(total_length=(3,3,3)) # 在场景中添加坐标轴
window.scene.add_actor( axe )
```


```python
a.property.edit_traits() # a是表示圆锥的Actor对象;
```
