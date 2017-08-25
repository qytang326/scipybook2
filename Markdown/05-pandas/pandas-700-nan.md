

```python
import numpy as np
import pandas as pd
```

### 与`NaN`相关的函数


```python
np.random.seed(41)
df_int = pd.DataFrame(np.random.randint(0, 10, (10, 3)), columns=list("ABC"))
df_int["A"] += 10
df_nan = df_int.where(df_int > 2)
#%hide
%C df_int.dtypes; df_nan.dtypes
print
%C 4 df_int; df_nan
```

    df_int.dtypes  df_nan.dtypes
    -------------  -------------
    A    int32     A      int32 
    B    int32     B    float64 
    C    int32     C    float64 
    dtype: object  dtype: object
    
       df_int          df_nan   
    -----------    -------------
        A  B  C        A   B   C
    0  10  3  2    0  10   3 NaN
    1  10  1  3    1  10 NaN   3
    2  19  7  5    2  19   7   5
    3  18  3  3    3  18   3   3
    4  12  6  0    4  12   6 NaN
    5  14  6  9    5  14   6   9
    6  13  8  4    6  13   8   4
    7  17  6  1    7  17   6 NaN
    8  15  2  1    8  15 NaN NaN
    9  15  3  2    9  15   3 NaN



```python
%C 4 df_nan.isnull(); df_nan.notnull()
```

       df_nan.isnull()           df_nan.notnull()  
    ----------------------    ---------------------
           A      B      C          A      B      C
    0  False  False   True    0  True   True  False
    1  False   True  False    1  True  False   True
    2  False  False  False    2  True   True   True
    3  False  False  False    3  True   True   True
    4  False  False   True    4  True   True  False
    5  False  False  False    5  True   True   True
    6  False  False  False    6  True   True   True
    7  False  False   True    7  True   True  False
    8  False   True   True    8  True  False  False
    9  False  False   True    9  True   True  False



```python
%C 4 df_nan.count(); df_nan.count(axis=1)
```

    df_nan.count()    df_nan.count(axis=1)
    --------------    --------------------
    A    10           0    2              
    B     8           1    2              
    C     5           2    3              
    dtype: int64      3    3              
                      4    2              
                      5    3              
                      6    3              
                      7    2              
                      8    1              
                      9    2              
                      dtype: int64        



```python
%C df_nan.dropna(); df_nan.dropna(thresh=2)
```

    df_nan.dropna()  df_nan.dropna(thresh=2)
    ---------------  -----------------------
        A  B  C          A   B   C          
    2  19  7  5      0  10   3 NaN          
    3  18  3  3      1  10 NaN   3          
    5  14  6  9      2  19   7   5          
    6  13  8  4      3  18   3   3          
                     4  12   6 NaN          
                     5  14   6   9          
                     6  13   8   4          
                     7  17   6 NaN          
                     9  15   3 NaN          



```python
%C df_nan.ffill(); df_nan.bfill(); df_nan.interpolate()
```

    df_nan.ffill()  df_nan.bfill()  df_nan.interpolate()
    --------------  --------------  --------------------
        A  B   C        A  B   C        A    B   C      
    0  10  3 NaN    0  10  3   3    0  10  3.0 NaN      
    1  10  3   3    1  10  7   3    1  10  5.0   3      
    2  19  7   5    2  19  7   5    2  19  7.0   5      
    3  18  3   3    3  18  3   3    3  18  3.0   3      
    4  12  6   3    4  12  6   9    4  12  6.0   6      
    5  14  6   9    5  14  6   9    5  14  6.0   9      
    6  13  8   4    6  13  8   4    6  13  8.0   4      
    7  17  6   4    7  17  6 NaN    7  17  6.0   4      
    8  15  6   4    8  15  3 NaN    8  15  4.5   4      
    9  15  3   4    9  15  3 NaN    9  15  3.0   4      



```python
s = pd.Series([3, np.NaN, 7], index=[0, 8, 9])
%C s.interpolate(); s.interpolate(method="index")
```

    s.interpolate()  s.interpolate(method="index")
    ---------------  -----------------------------
    0    3           0    3.000000                
    8    5           8    6.555556                
    9    7           9    7.000000                
    dtype: float64   dtype: float64               



```python
print((df_nan.fillna({"B":-999, "C":0})))
```

        A    B  C
    0  10    3  0
    1  10 -999  3
    2  19    7  5
    3  18    3  3
    4  12    6  0
    5  14    6  9
    6  13    8  4
    7  17    6  0
    8  15 -999  0
    9  15    3  0



```python
%C df_nan.sum(); df_nan.sum(skipna=False); df_nan.dropna().sum()
```

     df_nan.sum()   df_nan.sum(skipna=False)  df_nan.dropna().sum()
    --------------  ------------------------  ---------------------
    A    143        A    143                  A    64              
    B     42        B    NaN                  B    24              
    C     24        C    NaN                  C    21              
    dtype: float64  dtype: float64            dtype: float64       



```python
df_other = pd.DataFrame(np.random.randint(0, 10, (4, 2)), 
                        columns=["B", "C"], 
                        index=[1, 2, 8, 9])
print((df_nan.combine_first(df_other)))
```

        A  B   C
    0  10  3 NaN
    1  10  4   3
    2  19  7   5
    3  18  3   3
    4  12  6 NaN
    5  14  6   9
    6  13  8   4
    7  17  6 NaN
    8  15  4   5
    9  15  3   5

