

```python
import pandas as pd
import numpy as np
pd.set_option("display.show_dimensions", False)
pd.set_option("display.float_format", "{:4.2g}".format)
```

## 分组运算


```python
dose_df = pd.read_csv("dose.csv")
print((dose_df.head(3)))
```

       Dose  Response1  Response2 Tmt  Age Gender
    0    50        9.9         10   C  60s      F
    1    15      0.002      0.004   D  60s      F
    2    25       0.63        0.8   C  50s      M


### `groupby()`方法

> **TIP**

> `groupby()`并不立即执行分组操作，而只是返回保存源数据和分组数据的`GroupBy`对象。在需要获取每个分组的实际数据时，`GroupBy`对象才会执行分组操作。


```python
tmt_group = dose_df.groupby("Tmt")
print((type(tmt_group)))
```

    <class 'pandas.core.groupby.DataFrameGroupBy'>



```python
tmt_age_group = dose_df.groupby(["Tmt", "Age"])
```


```python
random_values = np.random.randint(0, 5, dose_df.shape[0])
random_group = dose_df.groupby(random_values)
```


```python
alternating_group = dose_df.groupby(lambda n:n % 3)
```


```python
crazy_group = dose_df.groupby(["Gender", lambda n: n % 2, random_values])
```

### `GroupBy`对象


```python
print((len(tmt_age_group), len(crazy_group)))
```

    10 20



```python
for key, df in tmt_age_group:
    print(("key =", key, ", shape =", df.shape))
```

    key = ('A', '50s') , shape = (39, 6)
    key = ('A', '60s') , shape = (26, 6)
    key = ('B', '40s') , shape = (13, 6)
    key = ('B', '50s') , shape = (13, 6)
    key = ('B', '60s') , shape = (39, 6)
    key = ('C', '40s') , shape = (13, 6)
    key = ('C', '50s') , shape = (13, 6)
    key = ('C', '60s') , shape = (39, 6)
    key = ('D', '50s') , shape = (52, 6)
    key = ('D', '60s') , shape = (13, 6)



```python
(_, df_A), (_, df_B), (_, df_C), (_, df_D) = tmt_group
```

> **TIP**

> 由于`GroupBy`对象有`keys`属性，因此无法通过`dict(tmt_group)`直接将其转换为字典，可以先将其转换为迭代器，再转换为字典`dict(iter(tmt_group))`。


```python
%C tmt_group.get_group("A").head(3);; tmt_age_group.get_group(("A", "50s")).head(3)
```

           tmt_group.get_group("A").head(3)       
    ----------------------------------------------
        Dose  Response1  Response2 Tmt  Age Gender
    6      1          0          0   A  50s      F
    10    15        5.2        5.2   A  60s      F
    12     5          0      0.001   A  60s      F
    
    tmt_age_group.get_group(("A", "50s")).head(3) 
    ----------------------------------------------
        Dose  Response1  Response2 Tmt  Age Gender
    6      1          0          0   A  50s      F
    17     5          0      0.003   A  50s      M
    34    40         11         10   A  50s      M



```python
print((tmt_group["Dose"]))
print((tmt_group[["Response1", "Response2"]]))
```

    <pandas.core.groupby.SeriesGroupBy object at 0x0C6076F0>
    <pandas.core.groupby.DataFrameGroupBy object at 0x0C6077F0>



```python
print((tmt_group.Dose))
```

    <pandas.core.groupby.SeriesGroupBy object at 0x05D96B70>


### 分组－运算－合并

#### `agg()`－聚合


```python
agg_res1 = tmt_group.agg(np.mean) #❶
agg_res2 = tmt_group.agg(lambda df:df.loc[df.Response1.idxmax()]) #❷
%C 4 agg_res1; agg_res2
```

                agg_res1                                 agg_res2                 
    -------------------------------    -------------------------------------------
         Dose  Response1  Response2         Dose  Response1  Response2  Age Gender
    Tmt                                Tmt                                        
    A      34        6.7        6.9    A      80         11         10  60s      F
    B      34        5.6        5.5    B   1e+02         11         10  50s      M
    C      34          4        4.1    C      60         10         11  50s      M
    D      34        3.3        3.2    D      80         11        9.9  60s      F


#### `transform()`－转换


```python
transform_res1 = tmt_group.transform(lambda s:s - s.mean()) #❶
transform_res2 = tmt_group.transform(
    lambda df:df.assign(Response1=df.Response1 - df.Response1.mean())) #❷
%C transform_res1.head(5); transform_res2.head(5)
```

        transform_res1.head(5)               transform_res2.head(5)         
    -----------------------------  -----------------------------------------
       Dose  Response1  Response2     Dose  Response1  Response2  Age Gender
    0    16        5.8        5.9  0    50        5.8         10  60s      F
    1   -19       -3.3       -3.2  1    15       -3.3      0.004  60s      F
    2  -8.5       -3.4       -3.3  2    25       -3.4        0.8  50s      M
    3  -8.5       -2.7       -2.6  3    25       -2.7        1.6  60s      F
    4   -19         -4       -4.1  4    15         -4       0.02  60s      F


#### `filter()`－过滤


```python
print((tmt_group.filter(lambda df:df.Response1.max() < 11).head()))
```

       Dose  Response1  Response2 Tmt  Age Gender
    0    50        9.9         10   C  60s      F
    1    15      0.002      0.004   D  60s      F
    2    25       0.63        0.8   C  50s      M
    3    25        1.4        1.6   C  60s      F
    4    15       0.01       0.02   C  60s      F


#### `apply()`－运用

> **WARNING**

> 注意目前的版本采用`is`判断索引是否相同，很容易引起混淆，未来的版本可能会对这一点进行修改。


```python
%C 4 tmt_group.apply(pd.DataFrame.max); tmt_group.apply(pd.DataFrame.mean)
```

           tmt_group.apply(pd.DataFrame.max)           tmt_group.apply(pd.DataFrame.mean)
    -----------------------------------------------    ----------------------------------
         Dose  Response1  Response2 Tmt  Age Gender         Dose  Response1  Response2   
    Tmt                                                Tmt                               
    A   1e+02         11         11   A  60s      M    A      34        6.7        6.9   
    B   1e+02         11         10   B  60s      M    B      34        5.6        5.5   
    C   1e+02         10         11   C  60s      M    C      34          4        4.1   
    D   1e+02         11        9.9   D  60s      M    D      34        3.3        3.2   



```python
sample_res1 = tmt_group.apply(lambda df:df.Response1.sample(2)) #❶
sample_res2 = tmt_group.apply(
    lambda df:df.Response1.sample(2).reset_index(drop=True)) #❷
%C 4 sample_res1; sample_res2
```

              sample_res1                  sample_res2     
    -------------------------------    --------------------
    Tmt                                Response1     0    1
    A    248      10                   Tmt                 
         164      10                   A            10   10
    B    113    0.19                   B            10   10
         26      9.4                   C         0.004  9.9
    C    191      10                   D          0.33   11
         236     1.7                                       
    D    188   0.061                                       
         8     0.001                                       
    Name: Response1, dtype: float64                        



```python
group = tmt_group[["Response1", "Response1"]]
apply_res1 = group.apply(lambda df:df - df.mean())
apply_res2 = group.apply(lambda df:(df - df.mean())[:])

%C 4 apply_res1.head(); apply_res2.head()
```

       apply_res1.head()            apply_res2.head()      
    -----------------------    ----------------------------
       Response1  Response1            Response1  Response1
    0        5.8        5.8    Tmt                         
    1       -3.3       -3.3    A   6        -6.7       -6.7
    2       -3.4       -3.4        10       -1.5       -1.5
    3       -2.7       -2.7        12       -6.7       -6.7
    4         -4         -4        17       -6.7       -6.7
                                   32        2.6        2.6



```python
print((tmt_group.apply(lambda df:None if df.Response1.mean() < 5 else df.sample(2))))
```

             Dose  Response1  Response2 Tmt  Age Gender
    Tmt                                                
    A   235    60        9.8         10   A  50s      M
        164    20         10         10   A  50s      F
    B   9      40         11         10   B  60s      F
        16     30        9.8         10   B  60s      F



```python
%C 4 tmt_group.mean(); tmt_group.quantile(q=0.75)
```

            tmt_group.mean()              tmt_group.quantile(q=0.75)  
    -------------------------------    -------------------------------
         Dose  Response1  Response2         Dose  Response1  Response2
    Tmt                                Tmt                            
    A      34        6.7        6.9    A      50         10         10
    B      34        5.6        5.5    B      50        9.8         10
    C      34          4        4.1    C      50        9.6        9.6
    D      34        3.3        3.2    D      50        8.9        8.4

