

```python
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)
%gui qt
```

## 用Handler控制界面和模型

### 用Handler处理事件


```python
from traits.api import HasTraits, Str, Int
from traitsui.api import View, Item, Group, Handler
from traitsui.menu import ModalButtons

g1 = [Item('department', label="部门"),
      Item('name', label="姓名")]
g2 = [Item('salary', label="工资"),
      Item('bonus', label="奖金")]

class Employee(HasTraits):
    name = Str
    department = Str
    salary = Int
    bonus = Int
    
    def _department_changed(self): #❶
        print((self, "department changed to ", self.department))
        
    def __str__(self): #❷
        return "<Employee at 0x%x>" % id(self)

view1 = View(
    Group(*g1, label = '个人信息', show_border = True),
    Group(*g2, label = '收入', show_border = True),
    title = "外部视图",
    kind = "modal",   #❸
    buttons = ModalButtons
)

class EmployeeHandler(Handler): #❹
    def init(self, info):
        super(EmployeeHandler, self).init(info)
        print("init called")

    def init_info(self, info):
        super( EmployeeHandler, self).init_info(info)
        print("init info called")
        
    def position(self, info):
        super(EmployeeHandler, self).position(info)
        print("position called")
        
    def setattr(self, info, obj, name, value):
        super(EmployeeHandler, self).setattr(info, obj, name, value)
        print(("setattr called:%s.%s=%s" % (obj, name, value)))
        
    def apply(self, info):
        super(EmployeeHandler, self).apply(info)
        print("apply called")
        
    def close(self, info, is_ok):
        super(EmployeeHandler, self).close(info, is_ok)
        print(("close called: %s" % is_ok))
        return True
        
    def closed(self, info, is_ok):
        super(EmployeeHandler, self).closed(info, is_ok)
        print(("closed called: %s" % is_ok))
        
    def revert(self, info):
        super(EmployeeHandler, self).revert(info)
        print("revert called")
           
zhang = Employee(name="Zhang")
print(("zhang is ", zhang))
zhang.configure_traits(view=view1, handler=EmployeeHandler()) #❺
```

    zhang is  <Employee at 0x91efcf0>
    init info called
    init called
    position called
    <Employee at 0x96223c0> department changed to  开发
    setattr called:<Employee at 0x96223c0>.department=开发
    <Employee at 0x96223c0> department changed to  开发部门
    setattr called:<Employee at 0x96223c0>.department=开发部门
    <Employee at 0x91efcf0> department changed to  开发部门
    apply called
    close called: True
    closed called: True





    True



### Controller和UIInfo对象


```python
from traitsui.api import Controller

view1.kind = "nonmodal"
zhang = Employee(name="Zhang")
c = Controller(zhang)
c.edit_traits(view=view1);
```


```python
c.get()
```




    {'_ipython_display_': None,
     '_repr_html_': None,
     '_repr_javascript_': None,
     '_repr_jpeg_': None,
     '_repr_json_': None,
     '_repr_latex_': None,
     '_repr_pdf_': None,
     '_repr_png_': None,
     '_repr_svg_': None,
     'info': <traitsui.ui_info.UIInfo at 0x5614810>,
     'model': <__main__.Employee at 0x55b71e0>}




```python
c.info.get()
```




    {'initialized': True, 'ui': <traitsui.ui.UI at 0x55b7570>}




```python
%omit c.info.ui.get()
```

    {'_active_group': 0,
     '_checked': [],
     '_context': {'controller': <traitsui.handler.Controller at 0x5665870>,
      'handler': <traitsui.handler.Controller at 0x5665870>,
    ...



```python
ui = c.info.ui
ui.context
```




    {'controller': <traitsui.handler.Controller at 0x56143f0>,
     'handler': <traitsui.handler.Controller at 0x360b090>,
     'object': <__main__.Employee at 0x5b58030>}




```python
ui.control # ui对象所表示的实际界面控件
```




    <traitsui.qt4.ui_base._StickyDialog at 0x5658780>




```python
%omit ui.view
```

    ( Group(
        Item( 'department'
              object     = 'object',
              label      = u'\u90e8\u95e8',
    ...



```python
%omit ui._editors
```

    [<traitsui.qt4.text_editor.SimpleEditor at 0x5abe480>,
     <traitsui.qt4.text_editor.SimpleEditor at 0x5b00510>,
    ...


### 响应Trait属性的事件


```python
from traits.api import HasTraits, Bool
from traitsui.api import View, Handler

class MyHandler(Handler):
    def setattr(self, info, object, name, value): #❶
        Handler.setattr(self, info, object, name, value)
        info.object.updated = True #❷
        print(("setattr", name))
        
    def object_updated_changed(self, info): #❸
        print(("updated changed", "initialized=%s" % info.initialized))
        if info.initialized:
            info.ui.title += "*"

class TestClass(HasTraits):
    b1 = Bool
    b2 = Bool
    b3 = Bool
    updated = Bool(False)

view1 = View('b1', 'b2', 'b3',
             handler=MyHandler(),
             title = "Test",
             buttons = ['OK', 'Cancel'])

tc = TestClass()
tc.configure_traits(view=view1);
```

    setattr b2
    updated changed initialized=False

