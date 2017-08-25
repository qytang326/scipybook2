

```python
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)
%gui qt
```

## Trait类型

### 预定义的Trait类型


```python
from traits.api import HasTraits, CFloat, Float, TraitError

class Person(HasTraits):
    cweight = CFloat(50.0)
    weight = Float(50.0)
```


```python
p = Person()
p.cweight = "90"
print((p.cweight))
try:
    p.weight = "90"
except TraitError as ex:
    print(ex)
```

    90.0
    The 'weight' trait of a Person instance must be a float, but a value of '90' <type 'str'> was specified.



```python
from traits.api import Enum, List

class Items(HasTraits):
    count = Enum(None, 0, 1, 2, 3, "many")
    # 或者：
    # count = Enum([None, 0, 1, 2, 3, "many"])    
```


```python
item = Items()
item.count = 2
item.count = "many"
try:
    item.count = 5
except TraitError as ex:
    print(ex)
```

    The 'count' trait of an Items instance must be None or 0 or 1 or 2 or 3 or 'many', but a value of 5 <type 'int'> was specified.



```python
class Items(HasTraits):
    count_list = List([None, 0, 1, 2, 3, "many"])
    count = Enum(values="count_list")
```


```python
item = Items()

try:
    item.count = 5    #由于候选值列表中没有5，因此赋值失败
except TraitError as ex:
    print(ex)
    
item.count_list.append(5)
item.count = 5       #由于候选值列表中有5，因此赋值成功
item.count
```

    The 'count' trait of an Items instance must be None or 0 or 1 or 2 or 3 or 'many', but a value of 5 <type 'int'> was specified.





    5



### Property属性


```python
from traits.api import HasTraits, Float, Property, cached_property

class Rectangle(HasTraits):
    width = Float(1.0) 
    height = Float(2.0)

    #area是一个属性，当width,height的值变化时，它对应的_get_area函数将被调用
    area = Property(depends_on=['width', 'height'])  #❶

    # 通过cached_property修饰器缓存_get_area()的输出
    @cached_property     #❷
    def _get_area(self): #❸
        "area的get函数，注意此函数名和对应的Proerty名的关系"
        print('recalculating')
        return self.width * self.height
```


```python
r = Rectangle()
print((r.area))  # 第一次取得area，需要进行运算
r.width = 10
print((r.area)) # 修改width之后，取得area，需要进行计算
print((r.area)) # width和height都没有发生变化，因此直接返回缓存值，没有重新计算
```

    recalculating
    2.0
    recalculating
    20.0
    20.0



```python
#%hide
r.edit_traits()
r.edit_traits();
```


```python
t = r.trait("area") #获得与area属性对应的CTrait对象
t._notifiers(True) # _notifiers方法返回所有的通知对象，当aera属性改变时，这里对象将被通知
```




    [<traits.trait_notifiers.FastUITraitChangeNotifyWrapper at 0x8b9e3f0>,
     <traits.trait_notifiers.FastUITraitChangeNotifyWrapper at 0x8bd4e10>]



### Trait属性监听


```python
from traits.api import HasTraits, Str, Int

class Child ( HasTraits ):          
    name = Str
    age = Int 
    doing = Str

    def __str__(self):
        return "%s<%x>" % (self.name, id(self))

    # 当age属性的值被修改时，下面的函数将被运行
    def _age_changed ( self, old, new ): #❶
        print(("%s.age changed: form %s to %s" % (self, old, new)))

    def _anytrait_changed(self, name, old, new): #❷
        print(("anytrait changed: %s.%s from %s to %s" % (self, name, old, new)))

def log_trait_changed(obj, name, old, new): #❸
    print(("log: %s.%s changed from %s to %s" % (obj, name, old, new)))
    
h = Child(name = "HaiYue", age=9)
k = Child(name = "KaiWen", age=2)
h.on_trait_change(log_trait_changed, name="doing") #❹
```

    anytrait changed: <8b823f0>.age from 0 to 9
    <8b823f0>.age changed: form 0 to 9
    anytrait changed: HaiYue<8b823f0>.name from  to HaiYue
    anytrait changed: <8b823c0>.age from 0 to 2
    <8b823c0>.age changed: form 0 to 2
    anytrait changed: KaiWen<8b823c0>.name from  to KaiWen



```python
h.age = 10
h.doing = "sleeping"
k.doing = "playing"
```

    anytrait changed: HaiYue<8b823f0>.age from 9 to 10
    HaiYue<8b823f0>.age changed: form 9 to 10
    anytrait changed: HaiYue<8b823f0>.doing from  to sleeping
    log: HaiYue<8b823f0>.doing changed from  to sleeping
    anytrait changed: KaiWen<8b823c0>.doing from  to playing



```python
from traits.api import HasTraits, Str, Int, Instance, List, on_trait_change

class HasName(HasTraits):
    name = Str()
    
    def __str__(self):
        return "<%s %s>" % (self.__class__.__name__, self.name)

class Inner(HasName):
    x = Int
    y = Int

class Demo(HasName):
    x = Int
    y = Int
    z = Int(monitor=1) # 有元数据属性monitor的Int
    inner = Instance(Inner)
    alist = List(Int)
    test1 = Str()
    test2 = Str()
    
    def _inner_default(self):
        return Inner(name="inner1")
            
    @on_trait_change("x,y,inner.[x,y],test+,+monitor,alist[]")
    def event(self, obj, name, old, new):
        print((obj, name, old, new))
```


```python
d = Demo(name="demo")
d.x = 10 # 与x匹配
d.y = 20 # 与y匹配
d.inner.x = 1 # 与inner.[x,y]匹配
d.inner.y = 2 # 与inner.[x,y]匹配
d.inner = Inner(name="inner2") # 与inner.[x,y]匹配
d.test1 = "ok" #与 test+匹配
d.test2 = "hello" #与test+匹配
d.z = 30  # 与+monitor匹配
d.alist = [3] # 与alist[]匹配
d.alist.extend([4,5]) #与alist[]匹配
d.alist[2] = 10 # 与alist[]匹配
```

    <Demo demo> x 0 10
    <Demo demo> y 0 20
    <Inner inner1> x 0 1
    <Inner inner1> y 0 2
    <Demo demo> inner <Inner inner1> <Inner inner2>
    <Demo demo> test1  ok
    <Demo demo> test2  hello
    <Demo demo> z 0 30
    <Demo demo> alist [] [3]
    <Demo demo> alist_items [] [4, 5]
    <Demo demo> alist_items [5] [10]


### Event和Button属性


```python
from traits.api import HasTraits, Float, Event, on_trait_change

class Point(HasTraits):       #❶
    x = Float(0.0)
    y = Float(0.0)
    updated = Event
            
    @on_trait_change( "x,y" )
    def pos_changed(self):    #❷
        self.updated = True

    def _updated_fired(self): #❸
        self.redraw()
    
    def redraw(self):         #❹
        print(("redraw at %s, %s" % (self.x, self.y)))
```


```python
p = Point()
p.x = 1
p.y = 1
p.x = 1 # 由于x的值已经为1，因此不触发事件
p.updated = True
p.updated = 0 # 给updated赋任何值都能触发
```

    redraw at 1.0, 0.0
    redraw at 1.0, 1.0
    redraw at 1.0, 1.0
    redraw at 1.0, 1.0


### 动态添加Trait属性


```python
a = HasTraits()  
a.add_trait("x", Float(3.0))
a.x
```




    3.0




```python
b = HasTraits()
b.add_trait("a", Instance(HasTraits))
b.a = a
```


```python
from traits.api import Delegate
b.add_trait("y", Delegate("a", "x", modify=True))    
print((b.y))
b.y = 10    
print((a.x))
```

    3.0
    10.0



```python
class A(HasTraits):
    pass

a = A()
a.x = 3
a.y = "string"
a.traits()
```




    {'trait_added': <traits.traits.CTrait at 0x3927c90>,
     'trait_modified': <traits.traits.CTrait at 0x3927c38>,
     'x': <traits.traits.CTrait at 0x3927f50>,
     'y': <traits.traits.CTrait at 0x3927f50>}




```python
 a.trait("x").trait_type
```




    <traits.trait_types.Python at 0x39399b0>


