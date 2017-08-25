

```python
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)
%gui qt
```

## TraitsUI入门

### 缺省界面


```python
from traits.api import HasTraits, Str, Int

class Employee(HasTraits):
    name = Str
    department = Str
    salary = Int
    bonus = Int

Employee().configure_traits();
```

### 用View定义界面

#### 外部视图和内部视图


```python
from traits.api import HasTraits, Str, Int
from traitsui.api import View, Item #❶

class Employee(HasTraits):
    name = Str
    department = Str
    salary = Int
    bonus = Int
    
    view = View(  #❷
        Item('department', label="部门", tooltip="在哪个部门干活"), #❸
        Item('name', label="姓名"),
        Item('salary', label="工资"),
        Item('bonus', label="奖金"),
        title="员工资料", width=250, height=150, resizable=True   #❹
    )
    
p = Employee()
p.configure_traits();
```


```python
from traits.api import HasTraits, Str, Int
from traitsui.api import View, Group, Item #❶

g1 = [Item('department', label="部门", tooltip="在哪个部门干活"), #❷
      Item('name', label="姓名")]
g2 = [Item('salary', label="工资"),
      Item('bonus', label="奖金")]

class Employee(HasTraits):
    name = Str
    department = Str
    salary = Int
    bonus = Int

    traits_view = View( #❸
        Group(*g1, label = '个人信息', show_border = True),
        Group(*g2, label = '收入', show_border = True),
        title = "缺省内部视图")    

    another_view = View( #❹
        Group(*g1, label = '个人信息', show_border = True),
        Group(*g2, label = '收入', show_border = True),
        title = "另一个内部视图")    
        
global_view = View( #❺
    Group(*g1, label = '个人信息', show_border = True),
    Group(*g2, label = '收入', show_border = True),
    title = "外部视图")    
    
p = Employee()

# 使用内部视图traits_view 
p.edit_traits() #❻;
```


```python
list(Employee.__view_traits__.content.keys())
```




    ['another_view', 'traits_view']




```python
# 使用内部视图another_view 
p.edit_traits(view="another_view")
```


```python
# 使用外部视图view1
p.edit_traits(view=global_view)
```

> **TIP**

> 用TraitsUI库创建的界面可以选择后台界面库，目前支持的有qt4和wx两种。在启动程序时添加`-toolkit qt4`或者`-toolkit wx`选择使用何种界面库生成界面。本书中全部使用Qt作为后台界面库。

#### 多模型视图


```python
from traits.api import HasTraits, Str, Int
from traitsui.api import View, Group, Item

class Employee(HasTraits):
    name = Str
    department = Str
    salary = Int
    bonus = Int

comp_view = View( #❶
    Group(
        Group(
            Item('p1.department', label="部门"),
            Item('p1.name', label="姓名"),
            Item('p1.salary', label="工资"),
            Item('p1.bonus', label="奖金"),
            show_border=True
        ),
        Group(
            Item('p2.department', label="部门"),
            Item('p2.name', label="姓名"),
            Item('p2.salary', label="工资"),
            Item('p2.bonus', label="奖金"),
            show_border=True
        ),
        orientation = 'horizontal'
    ),
    title = "员工对比"    
)

employee1 = Employee(department = "开发", name = "张三", salary = 3000, bonus = 300) #❷
employee2 = Employee(department = "销售", name = "李四", salary = 4000, bonus = 400)

HasTraits().configure_traits(view=comp_view, context={"p1":employee1, "p2":employee2}) #❸;
```


```python
comp_view.ui({"p1":employee1, "p2":employee2});
```

#### Group对象


```python
from traits.api import HasTraits, Str, Int
from traitsui.api import View, Item, Group, VGrid, VGroup, HSplit, VSplit

class SimpleEmployee(HasTraits):
    first_name = Str
    last_name = Str
    department = Str

    employee_number = Str
    salary = Int
    bonus = Int
    
items1 = [Item(name = 'employee_number', label='编号'),
          Item(name = 'department', label="部门", tooltip="在哪个部门干活"),
          Item(name = 'last_name', label="姓"),
          Item(name = 'first_name', label="名")]

items2 = [Item(name = 'salary', label="工资"),
          Item(name = 'bonus', label="奖金")]

view1 = View(
    Group(*items1, label = '个人信息', show_border = True),
    Group(*items2, label = '收入', show_border = True),
    title = "标签页方式",
    resizable = True    
)
    
view2 = View(
    VGroup(
        VGrid(*items1, label = '个人信息', show_border = True, scrollable = True),
        VGroup(*items2, label = '收入', show_border = True),
    ), 
    resizable = True, width = 400, height = 250, title = "垂直分组"    
)

view3 = View(
    HSplit(
        VGroup(*items1, show_border = True, scrollable = True),
        VGroup(*items2, show_border = True, scrollable = True),
    ), 
    resizable = True, width = 400, height = 150, title = "水平分组(带调节栏)"    
)

view4 = View(
    VSplit(
        VGroup(*items1, show_border = True, scrollable = True),
        VGroup(*items2, show_border = True, scrollable = True),
    ), 
    resizable = True, width = 200, height = 300, title = "垂直分组(带调节栏)"    
)

sam = SimpleEmployee()
sam.configure_traits(view=view1)
sam.configure_traits(view=view2)
sam.configure_traits(view=view3)
sam.configure_traits(view=view4);
```

> **TIP**

> `Item`也提供了`visible_when`和`enabled_when`属性，其用法和`Group`完全相同。


```python
from traits.api import HasTraits, Int, Bool, Enum, Property
from traitsui.api import View, HGroup, VGroup, Item

class Shape(HasTraits):
    shape_type = Enum("rectangle", "circle")
    editable = Bool
    x, y, w, h, r = [Int]*5
    
    view = View(
        VGroup(
            HGroup(Item("shape_type"), Item("editable")),
            VGroup(Item("x"), Item("y"), Item("w"), Item("h"), 
                visible_when="shape_type=='rectangle'", enabled_when="editable"),
            VGroup(Item("x"), Item("y"), Item("r"), 
                visible_when="shape_type=='circle'",  enabled_when="editable"),
        ), resizable = True)
    
shape = Shape()
shape.configure_traits();
```

#### 配置视图


```python
from traitsui import menu
[btn.name for btn in menu.ModalButtons]
```




    [u'Apply', u'Revert', u'OK', u'Cancel', u'Help']


