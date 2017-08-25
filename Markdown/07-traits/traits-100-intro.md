

```python
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)
%gui qt
```

# Traits & TraitsUI-轻松制作图形界面

## Traits类型入门

### 什么是Traits属性


```python
from traits.api import HasTraits, Color #❶

class Circle(HasTraits): #❷
    color = Color #❸
```


```python
c = Circle()
Circle.color    #Circle类没有color属性
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-4-8335a9908186> in <module>()
          1 c = Circle()
    ----> 2 Circle.color    #Circle类没有color属性
    

    AttributeError: type object 'Circle' has no attribute 'color'



```python
print((c.color))
print((c.color.getRgb()))
```

    <PyQt4.QtGui.QColor object at 0x0542F270>
    (255, 255, 255, 255)



```python
c.color = "red"
print((c.color.getRgb()))
c.color = 0x00ff00
print((c.color.getRgb()))
c.color = (0, 255, 255)
print((c.color.getRgb()))

from traits.api import TraitError
try:
    c.color = 0.5
except TraitError as ex:
    print((ex[0][:350], "..."))
```

    (255, 0, 0, 255)
    (0, 255, 0, 255)
    (0, 255, 255, 255)
    The 'color' trait of a Circle instance must be a string of the form (r,g,b) or (r,g,b,a) where r, g, b, and a are integers from 0 to 255, a QColor instance, a Qt.GlobalColor, an integer which in hex is of the form 0xRRGGBB, a string of the form #RGB, #RRGGBB, #RRRGGGBBB or #RRRRGGGGBBBB or 'aliceblue' or 'antiquewhite' or 'aqua' or 'aquamarine' or  ...



```python
c.configure_traits();
```

> **WARNING**

> 当使用wxPython作为后台界面库时，由于TraitsUI 4.4.0中的一个错误，程序退出时会导致进程崩溃。请读者将本书提供的`scpy2\patches\toolkit.py`复制到`site-packages\traitsui\wx`目录下，覆盖原有的`toolkit.py`文件。

> **TIP**

> 如果在Notebook中运行`c.configure_traits()`，它会立即返回`False`，而不会等待对话框关闭。当程序单独运行时`configure_traits()`会等待界面关闭，并根据用户点击的按钮返回`True`或`False`。


```python
c.color.getRgb()
```




    (83, 120, 255, 255)



### Trait属性的功能


```python
from traits.api import Delegate, HasTraits, Instance, Int, Str

class Parent ( HasTraits ):
    # 初始化: last_name被初始化为'Zhang'
    last_name = Str( 'Zhang' ) #❶

class Child ( HasTraits ):          
    age = Int

    # 验证: father属性的值必须是Parent类的实例
    father = Instance( Parent ) #❷

    # 代理： Child的实例的last_name属性代理给其father属性的last_name
    last_name = Delegate( 'father' ) #❸

    # 监听: 当age属性的值被修改时，下面的函数将被运行
    def _age_changed ( self, old, new ): #❹
        print(('Age changed from %s to %s ' % ( old, new )))
        
p = Parent()
c = Child()
```


```python
p.last_name
```




    'Zhang'




```python
c.last_name   
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-10-fff30c984f1b> in <module>()
    ----> 1 c.last_name
    

    AttributeError: 'NoneType' object has no attribute 'last_name'



```python
c.father = p
print((c.last_name))
p.last_name = "ZHANG"
print((c.last_name))
```

    Zhang
    ZHANG



```python
c.age = 4
```

    Age changed from 0 to 4 



```python
c.configure_traits();
```


```python
c.print_traits()
```

    age:       4
    father:    <__main__.Parent object at 0x05D9CC90>
    last_name: 'ZHANG'



```python
c.get()
```




    {'age': 4, 'father': <__main__.Parent at 0x5d9cc90>, 'last_name': 'ZHANG'}




```python
c.set(age = 6)
```

    Age changed from 4 to 6 





    <__main__.Child at 0x5d9c600>




```python
c2 = Child(father=p, age=3)
```

    Age changed from 0 to 3 



```python
c.trait("age")
```




    <traits.traits.CTrait at 0x9e23870>




```python
p.trait("last_name").default
```




    'Zhang'




```python
try:
    c.trait("father").validate(c, "father", 2)
except TraitError as ex:
    print(ex)
```

    The 'father' trait of a Child instance must be a Parent or None, but a value of 2 <type 'int'> was specified.



```python
c.trait("father").validate(c, "father", p)
```




    <__main__.Parent at 0x5d9cc90>




```python
c.trait_property_changed("age", 8, 10)
c.age # age属性值没有发生变化
```

    Age changed from 8 to 10 





    6




```python
print((c.trait("age").trait_type))
print((c.trait("father").trait_type))
```

    <traits.trait_types.Int object at 0x09DC0490>
    <traits.trait_types.Instance object at 0x09DC0830>


### Trait类型对象


```python
from traits.api import Float, Int, HasTraits

class Person(HasTraits):
    age = Int(30)
    weight = Float
```


```python
p1 = Person()
p2 = Person()
print((p1.trait("age") is p2.trait("age")))
print((p1.trait("weight").trait_type is p2.trait("weight").trait_type)) 
```

    True
    True



```python
from traits.api import HasTraits, Range

coefficient = Range(-1.0, 1.0, 0.0)

class Quadratic(HasTraits):
    c2 = coefficient
    c1 = coefficient
    c0 = coefficient

class Quadratic2(HasTraits):
    c2 = Range(-1.0, 1.0, 0.0)
    c1 = Range(-1.0, 1.0, 0.0)
    c0 = Range(-1.0, 1.0, 0.0)
```


```python
q = Quadratic()

print((coefficient is q.trait("c0").trait_type))
print((coefficient is q.trait("c1").trait_type))
```

    True
    True



```python
q2 = Quadratic2()
q2.trait("c0").trait_type is q2.trait("c1").trait_type
```




    False



### Trait的元数据


```python
from traits.api import HasTraits, Int, Str, Array, List
   
class MetadataTest(HasTraits):
    i = Int(99, myinfo="test my info") #❶
    s = Str("test", label="字符串")    #❷
    # NumPy的数组
    a = Array         #❸
    # 元素为Int的列表
    list = List(Int)  #❹

test = MetadataTest()
```


```python
test.traits()
```




    {'a': <traits.traits.CTrait at 0x9e4fbe0>,
     'i': <traits.traits.CTrait at 0x9e4f9d0>,
     'list': <traits.traits.CTrait at 0x9e4fb30>,
     's': <traits.traits.CTrait at 0x9e4fa80>,
     'trait_added': <traits.traits.CTrait at 0x4fc2c38>,
     'trait_modified': <traits.traits.CTrait at 0x4fc2be0>}




```python
print((test.trait("i").default))
print((test.trait("i").myinfo))
print((test.trait("i").trait_type))
```

    99
    test my info
    <traits.trait_types.Int object at 0x05DA4F50>



```python
print((test.trait("s").label))
```

    字符串



```python
test.trait("a").array
```




    True




```python
print((test.trait("list")))
print((test.trait("list").trait_type))
print((test.trait("list").inner_traits)) # list属性的内部元素所对应的CTrait对象
print((test.trait("list").inner_traits[0].trait_type)) # 内部元素所对应的Trait类型对象
```

    <traits.traits.CTrait object at 0x09E4FB30>
    <traits.trait_types.List object at 0x05DA46D0>
    (<traits.traits.CTrait object at 0x09E4FC38>,)
    <traits.trait_types.Int object at 0x05DA4E50>



```python

```
