

```python
import pandas as pd
import numpy as np
```

## 时间序列

### 时间点、时间段、时间间隔


```python
now = pd.Timestamp.now()
now_shanghai = now.tz_localize("Asia/Shanghai")
now_tokyo = now_shanghai.tz_convert("Asia/Tokyo")
print(("本地时间:", now))
print(("上海时区:", now_shanghai))
print(("东京时区:", now_tokyo))
```

    本地时间: 2015-07-25 11:50:46.264000
    上海时区: 2015-07-25 11:50:46.264000+08:00
    东京时区: 2015-07-25 12:50:46.264000+09:00



```python
now_shanghai == now_tokyo
```




    True




```python
import pytz
%omit pytz.common_timezones
```

    ['Africa/Abidjan',
     'Africa/Accra',
     'Africa/Addis_Ababa',
     'Africa/Algiers',
    ...



```python
now_day = pd.Period.now(freq="D")
now_hour = pd.Period.now(freq="H")
%C now_day; now_hour
```

             now_day                       now_hour           
    -------------------------  -------------------------------
    Period('2015-07-25', 'D')  Period('2015-07-25 11:00', 'H')



```python
from pandas.tseries import frequencies
list(frequencies._period_code_map.keys())
frequencies._period_alias_dictionary();
```


```python
now_week_sun = pd.Period.now(freq="W")
now_week_mon = pd.Period.now(freq="W-MON")
%C now_week_sun; now_week_mon
```

                  now_week_sun                              now_week_mon              
    ----------------------------------------  ----------------------------------------
    Period('2015-07-20/2015-07-26', 'W-SUN')  Period('2015-07-21/2015-07-27', 'W-MON')



```python
%C now_day.start_time; now_day.end_time
```

           now_day.start_time                      now_day.end_time             
    --------------------------------  ------------------------------------------
    Timestamp('2015-07-25 00:00:00')  Timestamp('2015-07-25 23:59:59.999999999')



```python
now_shanghai.to_period("H")
```




    Period('2015-07-25 11:00', 'H')




```python
%C now.year; now.month; now.day; now.dayofweek; now.dayofyear; now.hour
```

    now.year  now.month  now.day  now.dayofweek  now.dayofyear  now.hour
    --------  ---------  -------  -------------  -------------  --------
    2015      7          25       5              206            11      



```python
national_day = pd.Timestamp("2015-10-1")
td = national_day - pd.Timestamp.now()
td
```




    Timedelta('67 days 12:09:04.039000')




```python
national_day + pd.Timedelta("20 days 10:20:30") 
```




    Timestamp('2015-10-21 10:20:30')




```python
%C td.days; td.seconds; td.microseconds
```

    td.days  td.seconds  td.microseconds
    -------  ----------  ---------------
    67L      43744L      39000L         



```python
print((pd.Timedelta(days=10, hours=1, minutes=2, seconds=10.5)))
print((pd.Timedelta(seconds=100000)))
```

    10 days 01:02:10.500000
    1 days 03:46:40


### 时间序列


```python
def random_timestamps(start, end, freq, count):
    index = pd.date_range(start, end, freq=freq)
    locations = np.random.choice(np.arange(len(index)), size=count, replace=False)
    locations.sort()
    return index[locations]

np.random.seed(42)
ts_index = random_timestamps("2015-01-01", "2015-10-01", freq="Min", count=5)
pd_index = ts_index.to_period("M")
td_index = pd.TimedeltaIndex(np.diff(ts_index))

print((ts_index, "\n"))
print((pd_index, "\n"))
print((td_index, "\n"))
```

    DatetimeIndex(['2015-01-15 16:12:00', '2015-02-15 08:04:00',
                   '2015-02-28 12:30:00', '2015-08-06 02:40:00',
                   '2015-08-18 13:13:00'],
                  dtype='datetime64[ns]', freq=None, tz=None) 
    
    PeriodIndex(['2015-01', '2015-02', '2015-02', '2015-08', '2015-08'], dtype='int64', freq='M') 
    
    TimedeltaIndex(['30 days 15:52:00', '13 days 04:26:00', '158 days 14:10:00',
                    '12 days 10:33:00'],
                   dtype='timedelta64[ns]', freq=None) 
    



```python
%C ts_index.dtype; pd_index.dtype; td_index.dtype
```

     ts_index.dtype   pd_index.dtype   td_index.dtype 
    ----------------  --------------  ----------------
    dtype('<M8[ns]')  dtype('int64')  dtype('<m8[ns]')



```python
%C ts_index.weekday; pd_index.month; td_index.seconds
```

    ts_index.weekday   pd_index.month        td_index.seconds      
    ----------------  ---------------  ----------------------------
    [3, 6, 5, 3, 1]   [1, 2, 2, 8, 8]  [57120, 15960, 51000, 37980]



```python
ts_index.shift(1, "H")
```




    DatetimeIndex(['2015-01-15 17:12:00', '2015-02-15 09:04:00',
                   '2015-02-28 13:30:00', '2015-08-06 03:40:00',
                   '2015-08-18 14:13:00'],
                  dtype='datetime64[ns]', freq=None, tz=None)




```python
ts_index.shift(1, "M")
```




    DatetimeIndex(['2015-01-31 16:12:00', '2015-02-28 08:04:00',
                   '2015-03-31 12:30:00', '2015-08-31 02:40:00',
                   '2015-08-31 13:13:00'],
                  dtype='datetime64[ns]', freq=None, tz=None)




```python
ts_index.normalize()
```




    DatetimeIndex(['2015-01-15', '2015-02-15', '2015-02-28', '2015-08-06',
                   '2015-08-18'],
                  dtype='datetime64[ns]', freq=None, tz=None)




```python
ts_index.to_period("H").to_timestamp()
```




    DatetimeIndex(['2015-01-15 16:00:00', '2015-02-15 08:00:00',
                   '2015-02-28 12:00:00', '2015-08-06 02:00:00',
                   '2015-08-18 13:00:00'],
                  dtype='datetime64[ns]', freq=None, tz=None)




```python
ts_series = pd.Series(list(range(5)), index=ts_index)
```


```python
ts_series.between_time("9:00", "18:00")
```




    2015-01-15 16:12:00    0
    2015-02-28 12:30:00    2
    2015-08-18 13:13:00    4
    dtype: int64




```python
ts_series.tshift(1, freq="D")
```




    2015-01-16 16:12:00    0
    2015-02-16 08:04:00    1
    2015-03-01 12:30:00    2
    2015-08-07 02:40:00    3
    2015-08-19 13:13:00    4
    dtype: int64




```python
pd_series = pd.Series(range(5), index=pd_index)
td_series = pd.Series(range(4), index=td_index)
%C pd_series.tshift(1); td_series.tshift(10, freq="H")
```

     pd_series.tshift(1)   td_series.tshift(10, freq="H")
    ---------------------  ------------------------------
    2015-02    0           31 days 01:52:00     0        
    2015-03    1           13 days 14:26:00     1        
    2015-03    2           159 days 00:10:00    2        
    2015-09    3           12 days 20:33:00     3        
    2015-09    4           dtype: int64                  
    Freq: M, dtype: int64                                



```python
ts_data = pd.Series(ts_index)
pd_data = pd.Series(pd_index)
td_data = pd.Series(td_index)
%C ts_data.dtype; pd_data.dtype; td_data.dtype
```

     ts_data.dtype    pd_data.dtype   td_data.dtype  
    ----------------  -------------  ----------------
    dtype('<M8[ns]')  dtype('O')     dtype('<m8[ns]')



```python
%C ts_data.dt.hour; pd_data.dt.month; td_data.dt.days
```

    ts_data.dt.hour  pd_data.dt.month  td_data.dt.days
    ---------------  ----------------  ---------------
    0    16          0    1            0     30       
    1     8          1    2            1     13       
    2    12          2    2            2    158       
    3     2          3    8            3     12       
    4    13          4    8            dtype: int64   
    dtype: int64     dtype: int64                     

