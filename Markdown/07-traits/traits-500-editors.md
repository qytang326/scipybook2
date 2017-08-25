

```python
%gui qt
from traits.api import *
from traitsui.api import *
```

## 属性编辑器

> **SOURCE**

> `traitsuidemo.demo`：TraitsUI官方提供的演示程序


```python
#%hide
%exec_python -m traitsuidemo.demo
```

### 编辑器演示程序

> **SOURCE**

> `scpy2.traits.traitsui_editors`：演示TraitsUI提供的各种编辑器的用法。


```python
#%hide
%exec_python -m scpy2.traits.traitsui_editors
```


```python
%%include python traits/traitsui_editors.py 1
class EditorDemoItem(HasTraits):
    code = Code()
    view = View(
        Group(
            Item("item", style="simple", label="simple", width=-300), #❶
            "_",  #❷
            Item("item", style="custom", label="custom"),
            "_",
            Item("item", style="text", label="text"),
            "_",
            Item("item", style="readonly", label="readonly"),
        ),
    )
```


```python
%%include python traits/traitsui_editors.py 2
class EditorDemo(HasTraits):
    codes = List(Str)
    selected_item = Instance(EditorDemoItem)  
    selected_code = Str 
    view = View(
        HSplit(
            Item("codes", style="custom", show_label=False,  #❶
                editor=ListStrEditor(editable=False, selected="selected_code")), 
            Item("selected_item", style="custom", show_label=False),
        ),
        resizable=True,
        width = 800,
        height = 400,
        title="各种编辑器演示"
    )

    def _selected_code_changed(self):
        item = EditorDemoItem(code=self.selected_code)
        item.add_trait("item", eval(self.selected_code)) #❷
        self.selected_item = item
```


```python
%%include python traits/traitsui_editors.py 3
employee = Employee()
demo_list = ["低通", "高通", "带通", "带阻"]

trait_defines ="""
    Array(dtype="int32", shape=(3,3)) #{1}
    Bool(True)
    Button("Click me")
    List(editor=CheckListEditor(values=demo_list))
    Code("print 'hello world'")
    Color("red")
    RGBColor("red")
    Trait(*demo_list)
    Directory(os.getcwd())
    Enum(*demo_list)
    File()
    Font()
    HTML('<b><font color="red" size="40">hello world</font></b>')
    List(Str, demo_list)
    Range(1, 10, 5)
    List(editor=SetEditor(values=demo_list))
    List(demo_list, editor=ListStrEditor())
    Str("hello")
    Password("hello")
    Str("Hello", editor=TitleEditor())
    Tuple(Color("red"), Range(1,4), Str("hello"))
    Instance(EditorDemoItem, employee)    
    Instance(EditorDemoItem, employee, editor=ValueEditor())
    Instance(time, time(), editor=TimeEditor())
"""
demo = EditorDemo()
demo.codes = [s.split("#")[0].strip() for s in trait_defines.split("\n") if s.strip()!=""]
demo.configure_traits()
```

### 对象编辑器

> **SOURCE**

> `scpy2.traits.traitsui_component`：TraitsUI的组件演示程序。


```python
#%hide
%exec_python -m scpy2.traits.traitsui_component
```


```python
%%include python traits/traitsui_component.py 1 -r
class Point(HasTraits):
    x = Int
    y = Int
    view = View(HGroup(Item("x"), Item("y")))
```


```python
%%include python traits/traitsui_component.py 2 -r
class Shape(HasTraits):
    info = Str #❶
    
    def __init__(self, **traits):
        super(Shape, self).__init__(**traits)
        self.set_info() #❷


class Triangle(Shape):
    a = Instance(Point, ()) #❸
    b = Instance(Point, ())
    c = Instance(Point, ())
    
    view = View(
        VGroup(
            Item("a", style="custom"), #❹
            Item("b", style="custom"),
            Item("c", style="custom"),
        )
    )
    
    @on_trait_change("a.[x,y],b.[x,y],c.[x,y]")
    def set_info(self):
        a,b,c = self.a, self.b, self.c
        l1 = ((a.x-b.x)**2+(a.y-b.y)**2)**0.5
        l2 = ((c.x-b.x)**2+(c.y-b.y)**2)**0.5
        l3 = ((a.x-c.x)**2+(a.y-c.y)**2)**0.5
        self.info = "edge length: %f, %f, %f" % (l1,l2,l3)
    
class Circle(Shape):
    center = Instance(Point, ())
    r = Int
    
    view = View(
        VGroup(
            Item("center", style="custom"), 
            Item("r"),
        )
    )
    
    @on_trait_change("r")
    def set_info(self):
        from math import pi
        self.info = "area:%f" % (pi*self.r**2)
```


```python
Triangle().configure_traits()
Circle().configure_traits();
```


```python
%%include python traits/traitsui_component.py 3 -r
class ShapeSelector(HasTraits):
    select = Enum(*[cls.__name__ for cls in Shape.__subclasses__()]) #❶
    shape = Instance(Shape) #❷
    
    view = View(
        VGroup(
            Item("select"),
            Item("shape", style="custom"), #❸
            Item("object.shape.info", style="custom"), #❹
            show_labels = False
        ),
        width = 350, height = 300, resizable = True
    )
    
    def __init__(self, **traits):
        super(ShapeSelector, self).__init__(**traits)
        self._select_changed()
    
    def _select_changed(self):    #❺
        klass =  [c for c in Shape.__subclasses__() if c.__name__ == self.select][0]
        self.shape = klass()
```

> **SOURCE**

> `scpy2.traits.traitsui_component_multi_view`：使用多个视图显示组件。


```python
#%hide
%exec_python -m scpy2.traits.traitsui_component_multi_view 
```


```python
%%include python traits/traitsui_component_multi_view.py 1
class Shape(HasTraits):
    info = Str
    view_info = View(Item("info", style="custom", show_label=False))

    def __init__(self, **traits):
        super(Shape, self).__init__(**traits)
        self.set_info()
```


```python
%%include python traits/traitsui_component_multi_view.py 2
    view = View(
        VGroup(
            Item("select", show_label=False),
            VSplit( #❶
                Item("shape", style="custom", editor=InstanceEditor(view="view")), #❷
                Item("shape", style="custom", editor=InstanceEditor(view="view_info")), 
                show_labels = False
            )

        ),
        width = 350, height = 300, resizable = True
    )
```

###自定义编辑器 


```python
%%include python traits/mpl_figure_editor.py 1
import matplotlib
from traits.api import Bool
from traitsui.api import toolkit
from traitsui.basic_editor_factory import BasicEditorFactory
from traits.etsconfig.api import ETSConfig

if ETSConfig.toolkit == "wx":
    # matplotlib采用WXAgg为后台，这样才能将绘图控件嵌入以wx为后台界面库的traitsUI窗口中
    import wx
    matplotlib.use("WXAgg")
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
    from matplotlib.backends.backend_wx import NavigationToolbar2Wx as Toolbar
    from traitsui.wx.editor import Editor
    
elif ETSConfig.toolkit == "qt4":
    matplotlib.use("Qt4Agg")
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as Toolbar
    from traitsui.qt4.editor import Editor
    from pyface.qt import QtGui
```


```python
%%include python traits/mpl_figure_editor.py 3
class _QtFigureEditor(Editor):
    scrollable = True

    def init(self, parent): #❶
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        
        panel = QtGui.QWidget()
        
        def mousemoved(event):           
            if event.xdata is not None:
                x, y = event.xdata, event.ydata
                name = "Axes"
            else:
                x, y = event.x, event.y
                name = "Figure"
                
            panel.info.setText("%s: %g, %g" % (name, x, y))
            
        panel.mousemoved = mousemoved
        vbox = QtGui.QVBoxLayout()
        panel.setLayout(vbox)
        
        mpl_control = FigureCanvas(self.value) #❷
        vbox.addWidget(mpl_control)
        if hasattr(self.value, "canvas_events"):
            for event_name, callback in self.value.canvas_events:
                mpl_control.mpl_connect(event_name, callback)

        mpl_control.mpl_connect("motion_notify_event", mousemoved)  

        if self.factory.toolbar: #❸
            toolbar = Toolbar(mpl_control, panel)
            vbox.addWidget(toolbar)       

        panel.info = QtGui.QLabel(panel)
        vbox.addWidget(panel.info)
        return panel
```


```python
%%include python traits/mpl_figure_editor.py 4
class MPLFigureEditor(BasicEditorFactory):
    """
    相当于traits.ui中的EditorFactory，它返回真正创建控件的类
    """    
    if ETSConfig.toolkit == "wx":
        klass = _WxFigureEditor
    elif ETSConfig.toolkit == "qt4":
        klass = _QtFigureEditor  #❶
        
    toolbar = Bool(True)  #❷
```


```python
import numpy as np
from matplotlib.figure import Figure
from scpy2.traits import MPLFigureEditor


class SinWave(HasTraits):
    figure = Instance(Figure, ())
    view = View(
        Item("figure", editor=MPLFigureEditor(toolbar=True), show_label=False),
        width = 400,
        height = 300,
        resizable = True)

    def __init__(self, **kw):
        super(SinWave, self).__init__(**kw)
        self.figure.canvas_events = [
            ("button_press_event", self.figure_button_pressed)
        ]
        axes = self.figure.add_subplot(111)
        t = np.linspace(0, 2*np.pi, 200)
        axes.plot(np.sin(t))

    def figure_button_pressed(self, event):
        print((event.xdata, event.ydata))
        
model = SinWave()
model.edit_traits();
```
