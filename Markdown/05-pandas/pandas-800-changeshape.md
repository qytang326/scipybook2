

```python
import pandas as pd
import numpy as np
pd.set_option("display.show_dimensions", False)
pd.set_option("display.float_format", "{:4.2g}".format)
```

### 改变DataFrame的形状


```python
soils = pd.read_csv("Soils.csv", index_col=0)[["Depth", "Contour", "Group", "pH", "N"]]
soils_mean = soils.groupby(["Depth", "Contour"]).mean()
%C soils.head(); soils_mean.head()
```

               soils.head()                    soils_mean.head()        
    ---------------------------------  ---------------------------------
       Depth Contour  Group   pH    N                    Group   pH    N
    1   0-10     Top      1  5.4 0.19  Depth Contour                    
    2   0-10     Top      1  5.7 0.17  0-10  Depression      9  5.4 0.18
    3   0-10     Top      1  5.1 0.26        Slope           5  5.5 0.22
    4   0-10     Top      1  5.1 0.17        Top             1  5.3  0.2
    5  10-30     Top      2  5.1 0.16  10-30 Depression     10  4.9 0.08
                                             Slope           6  5.3  0.1


#### 添加删除列或行


```python
soils["N_percent"] = soils.eval("N * 100")
```


```python
print((soils.assign(pH2 = soils.pH + 1).head()))
```

       Depth Contour  Group   pH    N  N_percent  pH2
    1   0-10     Top      1  5.4 0.19         19  6.4
    2   0-10     Top      1  5.7 0.17         16  6.7
    3   0-10     Top      1  5.1 0.26         26  6.1
    4   0-10     Top      1  5.1 0.17         17  6.1
    5  10-30     Top      2  5.1 0.16         16  6.1



```python
def random_dataframe(n):
    columns = ["A", "B", "C"]
    for i in range(n):
        nrow = np.random.randint(10, 20)
        yield pd.DataFrame(np.random.randint(0, 100, size=(nrow, 3)), columns=columns)

df_list = list(random_dataframe(1000))
```


```python
%%time
df_res1 = pd.DataFrame([])
for df in df_list:
    df_res1 = df_res1.append(df)
```

    Wall time: 1.37 s



```python
%%time
df_res2 = pd.concat(df_list, axis=0)
```

    Wall time: 118 ms



```python
df_res3 = pd.concat(df_list, axis=0, keys=list(range(len(df_list))))
df_res3.loc[30].equals(df_list[30])
```




    True




```python
print((soils.drop(["N", "Group"], axis=1).head()))
```

       Depth Contour   pH  N_percent
    1   0-10     Top  5.4         19
    2   0-10     Top  5.7         16
    3   0-10     Top  5.1         26
    4   0-10     Top  5.1         17
    5  10-30     Top  5.1         16


#### 行索引与列之间相互转换


```python
print((soils_mean.reset_index(level="Contour").head()))
```

              Contour  Group   pH    N
    Depth                             
    0-10   Depression      9  5.4 0.18
    0-10        Slope      5  5.5 0.22
    0-10          Top      1  5.3  0.2
    10-30  Depression     10  4.9 0.08
    10-30       Slope      6  5.3  0.1



```python
print((soils_mean.set_index("Group", append=True).head()))
```

                             pH    N
    Depth Contour    Group          
    0-10  Depression 9      5.4 0.18
          Slope      5      5.5 0.22
          Top        1      5.3  0.2
    10-30 Depression 10     4.9 0.08
          Slope      6      5.3  0.1


#### 行和列的索引相互转换


```python
print((soils_mean.unstack(1)[["Group", "pH"]].head()))
```

                 Group                   pH           
    Contour Depression Slope Top Depression Slope  Top
    Depth                                             
    0-10             9     5   1        5.4   5.5  5.3
    10-30           10     6   2        4.9   5.3  4.8
    30-60           11     7   3        4.4   4.3  4.2
    60-90           12     8   4        4.2   3.9  3.9



```python
print((soils_mean.stack().head(10)))
```

    Depth  Contour          
    0-10   Depression  Group      9
                       pH       5.4
                       N       0.18
           Slope       Group      5
                       pH       5.5
                       N       0.22
           Top         Group      1
                       pH       5.3
                       N        0.2
    10-30  Depression  Group     10
    dtype: float64


#### 交换索引的等级


```python
print((soils_mean.swaplevel(0, 1).sort_index()))
```

                      Group   pH     N
    Contour    Depth                  
    Depression 0-10       9  5.4  0.18
               10-30     10  4.9  0.08
               30-60     11  4.4 0.051
               60-90     12  4.2  0.04
    Slope      0-10       5  5.5  0.22
               10-30      6  5.3   0.1
               30-60      7  4.3 0.061
               60-90      8  3.9 0.043
    Top        0-10       1  5.3   0.2
               10-30      2  4.8  0.12
               30-60      3  4.2  0.08
               60-90      4  3.9 0.058


#### 透视表


```python
df = soils_mean.reset_index()[["Depth", "Contour", "pH", "N"]]
df_pivot_pH = df.pivot("Depth", "Contour", "pH")
%C df; df_pivot_pH
```

                   df                           df_pivot_pH          
    --------------------------------  -------------------------------
        Depth     Contour   pH     N  Contour  Depression  Slope  Top
    0    0-10  Depression  5.4  0.18  Depth                          
    1    0-10       Slope  5.5  0.22  0-10            5.4    5.5  5.3
    2    0-10         Top  5.3   0.2  10-30           4.9    5.3  4.8
    3   10-30  Depression  4.9  0.08  30-60           4.4    4.3  4.2
    4   10-30       Slope  5.3   0.1  60-90           4.2    3.9  3.9
    5   10-30         Top  4.8  0.12                                 
    6   30-60  Depression  4.4 0.051                                 
    7   30-60       Slope  4.3 0.061                                 
    8   30-60         Top  4.2  0.08                                 
    9   60-90  Depression  4.2  0.04                                 
    10  60-90       Slope  3.9 0.043                                 
    11  60-90         Top  3.9 0.058                                 



```python
print((df.pivot("Depth", "Contour")))
```

                    pH                     N            
    Contour Depression Slope  Top Depression Slope   Top
    Depth                                               
    0-10           5.4   5.5  5.3       0.18  0.22   0.2
    10-30          4.9   5.3  4.8       0.08   0.1  0.12
    30-60          4.4   4.3  4.2      0.051 0.061  0.08
    60-90          4.2   3.9  3.9       0.04 0.043 0.058



```python
df_before_melt = df_pivot_pH.reset_index()
df_after_melt = pd.melt(df_before_melt, id_vars="Depth", value_name="pH")
%C df_before_melt; df_after_melt
```

                df_before_melt                    df_after_melt       
    --------------------------------------  --------------------------
    Contour  Depth  Depression  Slope  Top      Depth     Contour   pH
    0         0-10         5.4    5.5  5.3  0    0-10  Depression  5.4
    1        10-30         4.9    5.3  4.8  1   10-30  Depression  4.9
    2        30-60         4.4    4.3  4.2  2   30-60  Depression  4.4
    3        60-90         4.2    3.9  3.9  3   60-90  Depression  4.2
                                            4    0-10       Slope  5.5
                                            5   10-30       Slope  5.3
                                            6   30-60       Slope  4.3
                                            7   60-90       Slope  3.9
                                            8    0-10         Top  5.3
                                            9   10-30         Top  4.8
                                            10  30-60         Top  4.2
                                            11  60-90         Top  3.9

