

```python
%matplotlib_svg
import numpy as np
import pylab as pl
import cv2
from cv2 import cv
```

## 类型转换

### 分析cv2的源程序

> **SOURCE**

> `codes\pyopencv_src`：为了方便读者查看`cv2`模块的源代码，本书提供了自动生成的源代码。若读者遇到参数类型不确定的情况，可以查看这些文件中相应的函数。


```python
%%language c++
static PyObject* pyopencv_line(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:line", 
                                    (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, 
                                    &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::line(img, pt1, pt2, color, thickness, lineType, shift));
        Py_RETURN_NONE;
    }

    return NULL;
}
```


```python
%%language c++
static int pyopencv_to(const PyObject* o, Mat& m, const ArgInfo info, bool allowND=true);
static inline bool pyopencv_to(PyObject* obj, Point& p, const char* name = "<unknown>");
static bool pyopencv_to(PyObject *o, Scalar& s, const char *name = "<unknown>");
```


```python
%%language c++
static inline bool pyopencv_to(PyObject* obj, Point& p, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyComplex_CheckExact(obj))
    {
        Py_complex c = PyComplex_AsCComplex(obj);
        p.x = saturate_cast<int>(c.real);
        p.y = saturate_cast<int>(c.imag);
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &p.x, &p.y) > 0;
}
```

> **QUESTION**

> 请读者使用同样的方法找到与`Scalar`类型对应的`pyopencv_to()`函数，并分析它能将何种类型的对象转换成`Scalars`对象。


```python
%%language c++
static inline PyObject* pyopencv_from(const RotatedRect& src)
{
    return Py_BuildValue("((ff)(ff)f)", src.center.x, src.center.y, 
                         src.size.width, src.size.height, src.angle);
}
```


```python
points = np.random.rand(20, 2).astype(np.float32)
(x, y), (w, h), angle = cv2.minAreaRect(points)
```

### `Mat`对象


```python
cvmat = cv.CreateMat(200, 100, cv2.CV_16UC3)
%C cvmat.height; cvmat.width; cvmat.channels; cvmat.type; cvmat.step
```

    cvmat.height  cvmat.width  cvmat.channels  cvmat.type  cvmat.step
    ------------  -----------  --------------  ----------  ----------
    200           100          3               18          600       



```python
%C cv2.CV_16U; cv2.CV_16UC3
```

    cv2.CV_16U  cv2.CV_16UC3
    ----------  ------------
    2           18          



```python
cv2.normalize(1)
```




    array([[ 1.],
           [ 0.],
           [ 0.],
           [ 0.]])



### 在`cv`和`cv2`之间转换图像对象


```python
img = cv2.imread("lena.jpg")
iplimage = cv.LoadImage("lena.jpg")
cvmat = cv.LoadImageM("lena.jpg")
print(iplimage)
print(cvmat)
```

    <iplimage(nChannels=3 width=512 height=512 widthStep=1536 )>
    <cvmat(type=42424010 8UC3 rows=512 cols=512 step=1536 )>



```python
import numpy as np
np.all(img == np.asarray(iplimage[:]))
```




    True




```python
iplimage2 = cv.GetImage(cv.fromarray(img[::2,::2,:].copy()))
```
