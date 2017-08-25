

```python
import pandas as pd
import numpy as np
pd.set_option("display.show_dimensions", False)
pd.set_option("display.float_format", "{:4.2g}".format)
```

## 下标存取


```python
np.random.seed(42)
df = pd.DataFrame(np.random.randint(0, 10, (5, 3)), 
                  index=["r1", "r2", "r3", "r4", "r5"], 
                  columns=["c1", "c2", "c3"])
```

### `[]`操作符


```python
%C 5 df; df[2:4]; df["r2":"r4"]
```

          df              df[2:4]         df["r2":"r4"] 
    --------------     --------------     --------------
        c1  c2  c3         c1  c2  c3         c1  c2  c3
    r1   6   3   7     r3   2   6   7     r2   4   6   9
    r2   4   6   9     r4   4   3   7     r3   2   6   7
    r3   2   6   7                        r4   4   3   7
    r4   4   3   7                                      
    r5   7   2   5                                      



```python
%C 5 df[df.c1 > 4]; df[df > 2]
```

    df[df.c1 > 4]         df[df > 2]   
    --------------     ----------------
        c1  c2  c3          c1   c2  c3
    r1   6   3   7     r1    6    3   7
    r5   7   2   5     r2    4    6   9
                       r3  nan    6   7
                       r4    4    3   7
                       r5    7  nan   5


### `.loc[]`和`.iloc[]`存取器


```python
%C 5 df.loc["r2"]; df.loc["r2","c2"]
```

         df.loc["r2"]          df.loc["r2","c2"]
    ----------------------     -----------------
    c1    4                    6                
    c2    6                                     
    c3    9                                     
    Name: r2, dtype: int32                      



```python
%C 5 df.loc[["r2","r3"]]; df.loc[["r2","r3"],["c1","c2"]]
```

    df.loc[["r2","r3"]]     df.loc[["r2","r3"],["c1","c2"]]
    -------------------     -------------------------------
        c1  c2  c3              c1  c2                     
    r2   4   6   9          r2   4   6                     
    r3   2   6   7          r3   2   6                     



```python
%C 5 df.loc["r2":"r4", ["c2","c3"]]; df.loc[df.c1>2, ["c1","c2"]]
```

    df.loc["r2":"r4", ["c2","c3"]]     df.loc[df.c1>2, ["c1","c2"]]
    ------------------------------     ----------------------------
        c2  c3                             c1  c2                  
    r2   6   9                         r1   6   3                  
    r3   6   7                         r2   4   6                  
    r4   3   7                         r4   4   3                  
                                       r5   7   2                  



```python
%C 5 df.iloc[2]; df.iloc[[2,4]]; df.iloc[[1,3]]; df.iloc[[1,3],[0,2]]
```

          df.iloc[2]           df.iloc[[2,4]]     df.iloc[[1,3]]     df.iloc[[1,3],[0,2]]
    ----------------------     --------------     --------------     --------------------
    c1    2                        c1  c2  c3         c1  c2  c3         c1  c3          
    c2    6                    r3   2   6   7     r2   4   6   9     r2   4   9          
    c3    7                    r5   7   2   5     r4   4   3   7     r4   4   7          
    Name: r3, dtype: int32                                                               



```python
%C 5 df.iloc[2:4, [0,2]]; df.iloc[df.c1.values>2, [0,1]]
```

    df.iloc[2:4, [0,2]]     df.iloc[df.c1.values>2, [0,1]]
    -------------------     ------------------------------
        c1  c3                  c1  c2                    
    r3   2   7              r1   6   3                    
    r4   4   7              r2   4   6                    
                            r4   4   3                    
                            r5   7   2                    



```python
%C 5 df.ix[2:4, ["c1", "c3"]]; df.ix["r1":"r3", [0, 2]]
```

    df.ix[2:4, ["c1", "c3"]]     df.ix["r1":"r3", [0, 2]]
    ------------------------     ------------------------
        c1  c3                       c1  c3              
    r3   2   7                   r1   6   7              
    r4   4   7                   r2   4   9              
                                 r3   2   7              


### 获取单个值


```python
%C 3 df.at["r2", "c2"]; df.iat[1, 1]; df.get_value("r2", "c2")
```

    df.at["r2", "c2"]   df.iat[1, 1]   df.get_value("r2", "c2")
    -----------------   ------------   ------------------------
    6                   6              6                       



```python
df.lookup(["r2", "r4", "r3"], ["c1", "c2", "c1"])
```




    array([4, 3, 2])



### 多级标签的存取


```python
soil_df = pd.read_csv("data/Soils-simple.csv", index_col=[0, 1], parse_dates=["Date"])
```


```python
%C soil_df.loc["10-30", ["pH", "Ca"]]
```

    soil_df.loc["10-30", ["pH", "Ca"]]
    ----------------------------------
                 pH   Ca              
    Contour                           
    Depression  4.9  7.5              
    Slope       5.3  9.5              
    Top         4.8   10              



```python
%C soil_df.loc[np.s_[:, "Top"], ["pH", "Ca"]]
```

    soil_df.loc[np.s_[:, "Top"], ["pH", "Ca"]]
    ------------------------------------------
                    pH   Ca                   
    Depth Contour                             
    0-10  Top      5.3   13                   
    10-30 Top      4.8   10                   


### `query()`方法


```python
print((soil_df.query("pH > 5 and Ca < 11")))
```

                       pH  Dens   Ca  Conduc       Date   Name
    Depth Contour                                             
    0-10  Depression  5.4  0.98   11     1.5 2015-05-26   Lois
    10-30 Slope       5.3   1.3  9.5     4.9 2015-02-06  Diana



```python
#%hide_output
pH_low = 5
Ca_hi = 11
print((soil_df.query("pH > @pH_low and Ca < @Ca_hi")))
```

                       pH  Dens   Ca  Conduc       Date   Name
    Depth Contour                                             
    0-10  Depression  5.4  0.98   11     1.5 2015-05-26   Lois
    10-30 Slope       5.3   1.3  9.5     4.9 2015-02-06  Diana



```python

```
