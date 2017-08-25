

```python
import pandas as pd
import numpy as np
pd.set_option("display.show_dimensions", False)
pd.set_option("display.float_format", "{:4.2g}".format)
```

## 文件的输入输出

### CSV文件

> **LINK**

> http://air.epmap.org/

> 空气质量数据来源：青悦空气质量历史数据库


```python
df_list = []

for df in pd.read_csv(
        u"data/aqi/上海市_201406.csv", 
        encoding="utf-8-sig",  #文件编码
        chunksize=100,         #一次读入的行数
        usecols=[u"时间", u"监测点", "AQI", "PM2.5", "PM10"], #只读入这些列
        na_values=["-", "—"],  #这些字符串表示缺失数据
        parse_dates=[0]):      #第一列为时间列
    df_list.append(df)  #在这里处理数据

%C df_list[0].count(); df_list[0].dtypes
```

    df_list[0].count()     df_list[0].dtypes   
    ------------------  -----------------------
    时间       100        时间       datetime64[ns]
    监测点       90        监测点              object
    AQI      100        AQI               int64
    PM2.5    100        PM2.5             int64
    PM10      98        PM10            float64
    dtype: int64        dtype: object          



```python
print((type(df.loc[0, "监测点"])))
```

    <type 'unicode'>


### HDF5文件

> **LINK**

> http://www.nsmc.cma.gov.cn/FENGYUNCast/docs/HDF5.0_chinese.pdf

> 中文的HDF5使用简介


```python
store = pd.HDFStore("a.hdf5", complib="blosc", complevel=9)
```


```python
df1 = pd.DataFrame(np.random.rand(100000, 4), columns=list("ABCD"))
df2 = pd.DataFrame(np.random.randint(0, 10000, (10000, 3)), 
                   columns=["One", "Two", "Three"])
s1 = pd.Series(np.random.rand(1000))
store["dataframes/df1"] = df1
store["dataframes/df2"] = df2
store["series/s1"] = s1
print((list(store.keys())))
print((df1.equals(store["dataframes/df1"])))
```

    ['/dataframes/df1', '/dataframes/df2', '/series/s1']
    True


> **LINK**

> http://pytables.github.io/usersguide/libref/hierarchy_classes.html
>
> `pytables`官方文档


```python
root = store.get_node("//")
for node in root._f_walknodes():
    print(node)
```

    /dataframes (Group) u''
    /series (Group) u''
    /dataframes/df1 (Group) u''
    /dataframes/df2 (Group) u''
    /series/s1 (Group) u''
    /series/s1/index (CArray(1000,), shuffle, blosc(9)) ''
    /series/s1/values (CArray(1000,), shuffle, blosc(9)) ''
    /dataframes/df1/axis0 (CArray(4,), shuffle, blosc(9)) ''
    /dataframes/df1/axis1 (CArray(100000,), shuffle, blosc(9)) ''
    /dataframes/df1/block0_items (CArray(4,), shuffle, blosc(9)) ''
    /dataframes/df1/block0_values (CArray(100000, 4), shuffle, blosc(9)) ''
    /dataframes/df2/axis0 (CArray(3,), shuffle, blosc(9)) ''
    /dataframes/df2/axis1 (CArray(10000,), shuffle, blosc(9)) ''
    /dataframes/df2/block0_items (CArray(3,), shuffle, blosc(9)) ''
    /dataframes/df2/block0_values (CArray(10000, 3), shuffle, blosc(9)) ''



```python
store.append('dataframes/df_dynamic1', df1, append=False) #❶
df3 = pd.DataFrame(np.random.rand(100, 4), columns=list("ABCD"))
store.append('dataframes/df_dynamic1', df3) #❷
store['dataframes/df_dynamic1'].shape
```




    (100100, 4)




```python
print((store.select('dataframes/df_dynamic1', where='index > 97 & index < 102')))
```

            A     B    C     D
    98   0.95 0.072 0.78  0.18
    99   0.19 0.043 0.24 0.075
    100  0.21  0.78 0.86  0.47
    101  0.71  0.87 0.63  0.74
    98  0.058  0.18 0.91 0.083
    99   0.47  0.81 0.71  0.59



```python
store.append('dataframes/df_dynamic1', df1, append=False, data_columns=["A", "B"])
print((store.select('dataframes/df_dynamic1', where='A > 0.99 & B < 0.01')))
```

             A       B    C     D
    3656  0.99  0.0018 0.67  0.47
    5091     1   0.004 0.43  0.15
    17671    1  0.0042 0.99  0.31
    41052    1 0.00081  0.9  0.32
    45307    1  0.0093 0.72 0.065
    67976 0.99  0.0096 0.93  0.79
    69078    1  0.0055 0.97  0.88
    87871    1   0.008 0.59  0.35
    94421 0.99  0.0049 0.36   0.9


> **WARNING**

> 由于所有从CSV文件读入`DataFrame`对象的行索引都为缺省值，因此HDF5文件中的数据的行索引并不是唯一的。


```python
def read_aqi_files(fn_pattern):
    from glob import glob
    from os import path
    
    UTF8_BOM = b"\xEF\xBB\xBF"
    
    cols = "时间,城市,监测点,质量等级,AQI,PM2.5,PM10,CO,NO2,O3,SO2".split(",")
    float_dtypes = {col:float for col in "AQI,PM2.5,PM10,CO,NO2,O3,SO2".split(",")}
    names_map = {"时间":"Time", 
                 "监测点":"Position", 
                 "质量等级":"Level", 
                 "城市":"City", 
                 "PM2.5":"PM2_5"}
    
    for fn in glob(fn_pattern):
        with open(fn, "rb") as f:
            sig = f.read(3) #❶
            if sig != UTF8_BOM:
                f.seek(0, 0)
            df = pd.read_csv(f, 
                             parse_dates=[0], 
                             na_values=["-", "—"], 
                             usecols=cols, 
                             dtype=float_dtypes) #❷
        df.rename_axis(names_map, axis=1, inplace=True)  
        df.dropna(inplace=True)
        yield df


store = pd.HDFStore("data/aqi/aqi.hdf5", complib="blosc", complevel=9)
string_size = {"City": 12, "Position": 30, "Level":12}

for idx, df in enumerate(read_aqi_files("data/aqi/*.csv")):
    store.append('aqi', df, append=idx!=0, min_itemsize=string_size, data_columns=True) #❸
    
store.close()
```


```python
store = pd.HDFStore("data/aqi/aqi.hdf5")
df_aqi = store.select("aqi")
print((len(df_aqi)))
```

    337250



```python
df_polluted = store.select("aqi", where="PM2_5 > 500")
print((len(df_polluted)))
```

    87


### 读写数据库


```python
from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/aqi/aqi.db')
```


```python
try:
    engine.execute("DROP TABLE aqi")
except:
    pass
```


```python
str_cols = ["Position", "City", "Level"]

for df in read_aqi_files("data/aqi/*.csv"):
    for col in str_cols:
        df[col] = df[col].str.decode("utf8")
    df.to_sql("aqi", engine, if_exists="append", index=False)
```


```python
df_aqi = pd.read_sql("aqi", engine)
```


```python
df_polluted = pd.read_sql("select * from aqi where PM2_5 > 500", engine)
print((len(df_polluted)))
```

    87


### 使用Pickle序列化


```python
df_aqi.to_pickle("data/aqi/aqi.pickle")
df_aqi2 = pd.read_pickle("data/aqi/aqi.pickle")
df_aqi.equals(df_aqi2)
```




    True


