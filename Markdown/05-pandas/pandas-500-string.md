

```python
import pandas as pd
import numpy as np
```

### 字符串处理


```python
s_abc = pd.Series(["a", "b", "c"])
print((s_abc.str.upper()))
```

    0    A
    1    B
    2    C
    dtype: object



```python
s_utf8 = pd.Series([b"北京", b"北京市", b"北京地区"])
s_unicode = s_utf8.str.decode("utf-8")
s_gb2312 = s_unicode.str.encode("gb2312")

%C s_utf8.str.len(); s_unicode.str.len(); s_gb2312.str.len()
```

    s_utf8.str.len()  s_unicode.str.len()  s_gb2312.str.len()
    ----------------  -------------------  ------------------
    0     6           0    2               0    4            
    1     9           1    3               1    6            
    2    12           2    4               2    8            
    dtype: int64      dtype: int64         dtype: int64      



```python
print((s_unicode.str[:2]))
```

    0    北京
    1    北京
    2    北京
    dtype: object



```python
print((s_unicode + "-" + s_abc * 2))
```

    0      北京-aa
    1     北京市-bb
    2    北京地区-cc
    dtype: object



```python
print((s_unicode.str.cat(s_abc, sep="-")))
```

    0      北京-a
    1     北京市-b
    2    北京地区-c
    dtype: object



```python
print((s_unicode.str.len().astype(str)))
```

    0    2
    1    3
    2    4
    dtype: object



```python
s = pd.Series(["a|bc|de", "x|xyz|yz"])
s_list = s.str.split("|")
s_comma = s_list.str.join(",")
%C s; s_list; s_comma
```

          s              s_list          s_comma   
    -------------  -----------------  -------------
    0     a|bc|de  0     [a, bc, de]  0     a,bc,de
    1    x|xyz|yz  1    [x, xyz, yz]  1    x,xyz,yz
    dtype: object  dtype: object      dtype: object



```python
s_list.str[1]
```




    0     bc
    1    xyz
    dtype: object




```python
print((pd.DataFrame(s_list.tolist(), columns=["A", "B", "C"])))
```

       A    B   C
    0  a   bc  de
    1  x  xyz  yz



```python
df_extract1 = s.str.extract(r"(\w+)\|(\w+)\|(\w+)")
df_extract2 = s.str.extract(r"(?P<A>\w+)\|(?P<B>\w+)|")
%C df_extract1; df_extract2
```

     df_extract1   df_extract2
    -------------  -----------
       0    1   2     A    B  
    0  a   bc  de  0  a   bc  
    1  x  xyz  yz  1  x  xyz  



```python
import io
text = """A, B|C|D
B, E|F
C, A
D, B|C
"""

df = pd.read_csv(io.BytesIO(text), skipinitialspace=True, header=None)
print(df)
```

       0      1
    0  A  B|C|D
    1  B    E|F
    2  C      A
    3  D    B|C



```python
nodes = df[1].str.split("|") #❶
from_node = df[0].values.repeat(nodes.str.len().astype(np.int32)) #❷
to_node = np.concatenate(nodes) #❸

print((pd.DataFrame({"from_node":from_node, "to_node":to_node})))
```

      from_node to_node
    0         A       B
    1         A       C
    2         A       D
    3         B       E
    4         B       F
    5         C       A
    6         D       B
    7         D       C



```python
print((df[1].str.get_dummies(sep="|")))
```

       A  B  C  D  E  F
    0  0  1  1  1  0  0
    1  0  0  0  0  1  1
    2  1  0  0  0  0  0
    3  0  1  1  0  0  0



```python
df[1].map(lambda s:max(s.split("|")))
```




    0    D
    1    F
    2    A
    3    C
    Name: 1, dtype: object




```python
df_soil = pd.read_csv("Soils.csv", usecols=[2, 3, 4, 6])
print((df_soil.dtypes))
```

    Contour     object
    Depth       object
    Gp          object
    pH         float64
    dtype: object



```python
for col in ["Contour", "Depth", "Gp"]:
    df_soil[col] = df_soil[col].astype("category")
print((df_soil.dtypes))
```

    Contour    category
    Depth      category
    Gp         category
    pH          float64
    dtype: object



```python
Gp = df_soil.Gp
print((Gp.cat.categories))
```

    Index([u'D0', u'D1', u'D3', u'D6', u'S0', u'S1', u'S3', u'S6', u'T0', u'T1',
           u'T3', u'T6'],
          dtype='object')



```python
%C Gp.head(5); Gp.cat.codes.head(5)
```

                              Gp.head(5)                            Gp.cat.codes.head(5)
    --------------------------------------------------------------  --------------------
    0    T0                                                         0    8              
    1    T0                                                         1    8              
    2    T0                                                         2    8              
    3    T0                                                         3    8              
    4    T1                                                         4    9              
    Name: Gp, dtype: category                                       dtype: int8         
    Categories (12, object): [D0, D1, D3, D6, ..., T0, T1, T3, T6]                      



```python
depth = df_soil.Depth
%C depth.cat.as_ordered().head()
```

                depth.cat.as_ordered().head()             
    ------------------------------------------------------
    0     0-10                                            
    1     0-10                                            
    2     0-10                                            
    3     0-10                                            
    4    10-30                                            
    dtype: category                                       
    Categories (4, object): [0-10 < 10-30 < 30-60 < 60-90]



```python
contour = df_soil.Contour
categories = ["Top", "Slope", "Depression"]
%C contour.cat.reorder_categories(categories, ordered=True).head()
```

    contour.cat.reorder_categories(categories, ordered=True).head()
    ---------------------------------------------------------------
    0    Top                                                       
    1    Top                                                       
    2    Top                                                       
    3    Top                                                       
    4    Top                                                       
    dtype: category                                                
    Categories (3, object): [Top < Slope < Depression]             

