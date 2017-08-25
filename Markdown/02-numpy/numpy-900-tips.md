

```python
import numpy as np
import math
```

## 实用技巧

### 动态数组


```python
import numpy as np
from array import array
a = array("d", [1,2,3,4])   # 创建一个array数组
# 通过np.frombuffer()创建一个和a共享内存的NumPy数组
na = np.frombuffer(a, dtype=np.float) 
print(a)
print(na)
na[1] = 20  # 修改NumPy数组中的第一个元素
print(a)
```

    array('d', [1.0, 2.0, 3.0, 4.0])
    [ 1.  2.  3.  4.]
    array('d', [1.0, 20.0, 3.0, 4.0])



```python
import math
buf = array("d")
for i in range(5):
    buf.append(math.sin(i*0.1)) 
    buf.append(math.cos(i*0.1))

data = np.frombuffer(buf, dtype=np.float).reshape(-1, 2)
print(data)
```

    [[ 0.          1.        ]
     [ 0.09983342  0.99500417]
     [ 0.19866933  0.98006658]
     [ 0.29552021  0.95533649]
     [ 0.38941834  0.92106099]]



```python
a = array("d")
for i in range(10):
    a.append(i)
    if i == 2:
        na = np.frombuffer(a, dtype=float)
    print(a.buffer_info(), end=' ')
    if i == 4:
        print()
```

    (83088512, 1) (83088512, 2) (83088512, 3) (83088512, 4) (31531848, 5)
    (31531848, 6) (31531848, 7) (31531848, 8) (34405776, 9) (34405776, 10)



```python
print((na.ctypes.data))
print(na)
```

    83088512
    [  2.11777767e+161   6.24020631e-085   8.82069697e+199]


> **TIP**

> `bytearray`对象的`+=`运算与其`extend()`方法的功能相同，但`+=`的运行速度要比`extend()`快许多，读者可以使用`%timeit`自行验证。


```python
import struct
buf = bytearray()
for i in range(5):
    buf += struct.pack("=hdd", i, math.sin(i*0.1), math.cos(i*0.1)) #❶

dtype = np.dtype({"names":["id","sin","cos"], "formats":["h", "d", "d"]}) #❷
data = np.frombuffer(buf, dtype=dtype) #❸
print(data)
```

    [(0, 0.0, 1.0) (1, 0.09983341664682815, 0.9950041652780258)
     (2, 0.19866933079506122, 0.9800665778412416)
     (3, 0.2955202066613396, 0.955336489125606)
     (4, 0.3894183423086505, 0.9210609940028851)]


### 和其它对象共享内存


```python
from PyQt4.QtGui import QImage, qRgb
img = QImage("lena.png")
print(("width & height:", img.width(), img.height()))
print(("depth:", img.depth())) #每个像素的比特数
print(("format:", img.format(), QImage.Format_RGB32)) 
print(("byteCount:", img.byteCount())) #图像的总字节数
print(("bytesPerLine:", img.bytesPerLine())) #每行的字节数
print(("bits:", int(img.bits()))) #图像第一个字节的地址
```

    width & height: 512 393
    depth: 32
    format: 4 4
    byteCount: 804864
    bytesPerLine: 2048
    bits: 156041248



```python
import ctypes
addr = int(img.bits())
pointer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8)) #❶
arr = np.ctypeslib.as_array(pointer, (img.height(), img.width(), img.depth()//8)) #❷
```


```python
x, y = 100, 50
b, g, r, a = arr[y, x]
print((qRgb(r, g, b)))
print((img.pixel(x, y)))
```

    4289282380
    4289282380



```python
arr[y, x, :3] = 0x12, 0x34, 0x56
print((hex(img.pixel(x, y))))
```

    0xff563412L



```python
interface = {
    'shape': (img.height(), img.width(), 4),
    'data': (int(img.bits()), False),
    'strides': (img.bytesPerLine(), 4, 1),
    'typestr': "|u1",
    'version': 3,
}

img.__array_interface__ = interface #❶

arr2 = np.array(img, copy=False)  #❷
del img.__array_interface__ #❸
print((np.all(arr2 == arr), arr2.base is img))  #❹
```

    True True



```python
class ArrayProxy(object):
    def __init__(self, base, interface):
        self.base = base
        self.__array_interface__ = interface
        
arr3 = np.array(ArrayProxy(img, interface), copy=False)
print((np.all(arr3 == arr)))
```

    True


### 与结构数组共享内存


```python
persontype = np.dtype({
    'names':['name', 'age', 'weight', 'height'],
    'formats':['S30','i', 'f', 'f']}, align= True )
a = np.array([("Zhang", 32, 72.5, 167.0), 
              ("Wang", 24, 65.2, 170.0)], dtype=persontype)

print((a["age"].base is a))  #视图
print((a[["age", "height"]].base is None)) #复制
```

    True
    True



```python
def fields_view(arr, fields):
    dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)

v = fields_view(a, ["age", "weight"])
print((v.base is a))

v["age"] += 10
print(a)
```

    True
    [('Zhang', 42, 72.5, 167.0) ('Wang', 34, 65.19999694824219, 170.0)]



```python
print((a.dtype.fields))
print((a.dtype))
print((v.dtype))
```

    {'age': (dtype('int32'), 32), 'name': (dtype('S30'), 0), 'weight': (dtype('float32'), 36), 'height': (dtype('float32'), 40)}
    {'names':['name','age','weight','height'], 'formats':['S30','<i4','<f4','<f4'], 'offsets':[0,32,36,40], 'itemsize':44, 'aligned':True}
    {'names':['age','weight'], 'formats':['<i4','<f4'], 'offsets':[32,36], 'itemsize':40}

