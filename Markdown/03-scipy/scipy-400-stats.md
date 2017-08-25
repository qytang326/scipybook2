

```python
%matplotlib_svg
import numpy as np
import pylab as pl
from scipy import stats
```

## 统计-stats

### 连续概率分布


```python
from scipy import stats
%col 4 [k for k, v in stats.__dict__.items() if isinstance(v, stats.rv_continuous)]
```

    ['genhalflogistic',    'triang',             'rayleigh',           'betaprime',         
     'levy',               'foldnorm',           'genlogistic',        'gilbrat',           
     'lognorm',            'anglit',             'truncnorm',          'erlang',            
     'norm',               'nakagami',           'weibull_min',        'cosine',            
     'logistic',           'fisk',               'genpareto',          'tukeylambda',       
     'dgamma',             'pareto',             'halflogistic',       'semicircular',      
     'ksone',              'mielke',             'ncx2',               'gengamma',          
     'johnsonsu',          'powernorm',          'powerlaw',           'burr',              
     'johnsonsb',          'beta',               'gamma',              'wald',              
     'arcsine',            'maxwell',            'invgauss',           'gausshyper',        
     'rice',               'vonmises_line',      'loglaplace',         'levy_stable',       
     'exponweib',          'pearson3',           'chi',                't',                 
     'cauchy',             'truncexpon',         'kstwobign',          'recipinvgauss',     
     'frechet_l',          'foldcauchy',         'wrapcauchy',         'ncf',               
     'genexpon',           'expon',              'reciprocal',         'f',                 
     'lomax',              'loggamma',           'invgamma',           'powerlognorm',      
     'laplace',            'vonmises',           'frechet_r',          'dweibull',          
     'rdist',              'gumbel_r',           'gompertz',           'halfcauchy',        
     'invweibull',         'exponpow',           'weibull_max',        'gumbel_l',          
     'halfnorm',           'fatiguelife',        'chi2',               'nct',               
     'uniform',            'genextreme',         'alpha',              'hypsecant',         
     'bradford',           'levy_l']            



```python
stats.norm.stats()
```




    (array(0.0), array(1.0))




```python
X = stats.norm(loc=1.0, scale=2.0)
X.stats()
```




    (array(1.0), array(4.0))




```python
x = X.rvs(size=10000) # 对随机变量取10000个值
np.mean(x), np.var(x) # 期望值和方差
```




    (1.0043406567303883, 3.8899572813426553)




```python
stats.norm.fit(x) # 得到随机序列期望值和标准差
```




    (1.0043406567303883, 1.9722974626923433)




```python
pdf, t = np.histogram(x, bins=100, normed=True)  #❶
t = (t[:-1] + t[1:]) * 0.5 #❷
cdf = np.cumsum(pdf) * (t[1] - t[0]) #❸
p_error = pdf - X.pdf(t)
c_error = cdf - X.cdf(t)
print(("max pdf error: {}, max cdf error: {}".format(
    np.abs(p_error).max(), np.abs(c_error).max())))
```

    max pdf error: 0.0217211429624, max cdf error: 0.0209887986472



```python
#%figonly=正态分布的概率密度函数（左）和累积分布函数（右）
fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(7, 2))
ax1.plot(t, pdf, label="统计值")
ax1.plot(t, X.pdf(t), label="理论值", alpha=0.6)
ax1.legend(loc="best")
ax2.plot(t, cdf)
ax2.plot(t2, X.cdf(t), alpha=0.6); 
```


![svg](scipy-400-stats_files/scipy-400-stats_9_0.svg)



```python
print((stats.gamma.stats(1.0)))
print((stats.gamma.stats(2.0)))
```

    (array(1.0), array(1.0))
    (array(2.0), array(2.0))



```python
stats.gamma.stats(2.0, scale=2)
```




    (array(4.0), array(8.0))




```python
x = stats.gamma.rvs(2, scale=2, size=4)
x   
```




    array([ 2.47613445,  1.93667652,  0.85723572,  9.49088092])




```python
stats.gamma.pdf(x, 2, scale=2)
```




    array([ 0.17948513,  0.18384555,  0.13960273,  0.02062186])




```python
X = stats.gamma(2, scale=2) 
X.pdf(x)
```




    array([ 0.17948513,  0.18384555,  0.13960273,  0.02062186])



### 离散概率分布


```python
x = list(range(1, 7))    
p = (0.4, 0.2, 0.1, 0.1, 0.1, 0.1)
```


```python
dice = stats.rv_discrete(values=(x, p))
dice.rvs(size=20)
```




    array([1, 6, 3, 1, 2, 2, 4, 1, 1, 1, 2, 5, 6, 2, 4, 2, 5, 2, 1, 4])




```python
np.random.seed(42)
samples = dice.rvs(size=(20000, 50))
samples_mean = np.mean(samples, axis=1)
```

### 核密度估计


```python
#%fig=核密度估计能更准确地表示随机变量的概率密度函数
_, bins, step = pl.hist(
    samples_mean, bins=100, normed=True, histtype="step", label="直方图统计")
kde = stats.kde.gaussian_kde(samples_mean)
x = np.linspace(bins[0], bins[-1], 100)
pl.plot(x, kde(x), label="核密度估计")
mean, std = stats.norm.fit(samples_mean)
pl.plot(x, stats.norm(mean, std).pdf(x), alpha=0.8, label="正态分布拟合")
pl.legend();
```


![svg](scipy-400-stats_files/scipy-400-stats_20_0.svg)



```python
#%fig=`bw_method`参数越大核密度估计曲线越平滑
for bw in [0.2, 0.3, 0.6, 1.0]:
    kde = stats.gaussian_kde([-1, 0, 1], bw_method=bw)
    x = np.linspace(-5, 5, 1000)
    y = kde(x)
    pl.plot(x, y, lw=2, label="bw={}".format(bw), alpha=0.6)
pl.legend(loc="best");
```


![svg](scipy-400-stats_files/scipy-400-stats_21_0.svg)


### 二项、泊松、伽玛分布


```python
stats.binom.pmf(list(range(6)), 5, 1/6.0)
```




    array([  4.01877572e-01,   4.01877572e-01,   1.60751029e-01,
             3.21502058e-02,   3.21502058e-03,   1.28600823e-04])




```python
#%fig=当n足够大时二项分布和泊松分布近似相等
lambda_ = 10.0
x = np.arange(20)

n1, n2 = 100, 1000

y_binom_n1 = stats.binom.pmf(x, n1, lambda_ / n1)
y_binom_n2 = stats.binom.pmf(x, n2, lambda_ / n2)
y_poisson = stats.poisson.pmf(x, lambda_)
print((np.max(np.abs(y_binom_n1 - possion))))
print((np.max(np.abs(y_binom_n2 - possion))))
#%hide
fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(7.5, 2.5))

ax1.plot(x, y_binom_n1, label="binom", lw=2)
ax1.plot(x, y_poisson, label="poisson", lw=2, color="red")
ax2.plot(x, y_binom_n2, label="binom", lw=2)
ax2.plot(x, y_poisson, label="poisson", lw=2, color="red")
for n, ax in zip((n1, n2), (ax1, ax2)):
    ax.set_xlabel("次数")
    ax.set_ylabel("概率")
    ax.set_title("n={}".format(n))
    ax.legend()
fig.subplots_adjust(0.1, 0.15, 0.95, 0.90, 0.2, 0.1);
```

    0.00675531110335
    0.000630175404978



![svg](scipy-400-stats_files/scipy-400-stats_24_1.svg)



```python
#%fig=模拟泊松分布
np.random.seed(42)

def sim_poisson(lambda_, time):
    t = np.random.uniform(0, time, size=lambda_ * time) #❶
    count, time_edges = np.histogram(t, bins=time, range=(0, time))  #❷
    dist, count_edges = np.histogram(count, bins=20, range=(0, 20), density=True) #❸
    x = count_edges[:-1]
    poisson = stats.poisson.pmf(x, lambda_)
    return x, poisson, dist

lambda_ = 10      
times = 1000, 50000
x1, poisson1, dist1 = sim_poisson(lambda_, times[0])
x2, poisson2, dist2 = sim_poisson(lambda_, times[1])
max_error1 = np.max(np.abs(dist1 - poisson1))
max_error2 = np.max(np.abs(dist2 - poisson2))         
print(("time={}, max_error={}".format(times[0], max_error1))) 
print(("time={}, max_error={}".format(times[1], max_error2))) 
#%hide
fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(7.5, 2.5))

ax1.plot(x1, dist1, "-o", lw=2, label="统计结果")
ax1.plot(x1, poisson1, "->", lw=2, label="泊松分布", color="red", alpha=0.6)
ax2.plot(x2, dist2, "-o", lw=2, label="统计结果")
ax2.plot(x2, poisson2, "->", lw=2, label="泊松分布", color="red", alpha=0.6)

for ax, time in zip((ax1, ax2), times):
    ax.set_xlabel("次数")
    ax.set_ylabel("概率")
    ax.set_title("time = {}".format(time))
    ax.legend(loc="lower center")
    
fig.subplots_adjust(0.1, 0.15, 0.95, 0.90, 0.2, 0.1);
```

    time=1000, max_error=0.019642302016
    time=50000, max_error=0.00179801289496



![svg](scipy-400-stats_files/scipy-400-stats_25_1.svg)



```python
#%fig=模拟伽玛分布
def sim_gamma(lambda_, time, k):
    t = np.random.uniform(0, time, size=lambda_ * time) #❶
    t.sort()  #❷
    interval = t[k:] - t[:-k] #❸
    dist, interval_edges = np.histogram(interval, bins=100, density=True) #❹
    x = (interval_edges[1:] + interval_edges[:-1])/2  #❺
    gamma = stats.gamma.pdf(x, k, scale=1.0/lambda_) #❺
    return x, gamma, dist

lambda_ = 10.0
time = 1000
ks = 1, 2
x1, gamma1, dist1 = sim_gamma(lambda_, time, ks[0])
x2, gamma2, dist2 = sim_gamma(lambda_, time, ks[1])
#%hide
fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(7.5, 2.5))

ax1.plot(x1, dist1,  lw=2, label="统计结果")
ax1.plot(x1, gamma1, lw=2, label="伽玛分布", color="red", alpha=0.6)
ax2.plot(x2, dist2,  lw=2, label="统计结果")
ax2.plot(x2, gamma2, lw=2, label="伽玛分布", color="red", alpha=0.6)

for ax, k in zip((ax1, ax2), ks):
    ax.set_xlabel("时间间隔")
    ax.set_ylabel("概率密度")
    ax.set_title("k = {}".format(k))
    ax.legend(loc="upper right")
    
fig.subplots_adjust(0.1, 0.15, 0.95, 0.90, 0.2, 0.1);
```


![svg](scipy-400-stats_files/scipy-400-stats_26_0.svg)



```python
T = 100000
A_count = T / 5
B_count = T / 10

A_time = np.random.uniform(0, T, A_count) #❶
B_time = np.random.uniform(0, T, B_count)

bus_time = np.concatenate((A_time, B_time)) #❷
bus_time.sort()

N = 200000
passenger_time = np.random.uniform(bus_time[0], bus_time[-1], N) #❸

idx = np.searchsorted(bus_time, passenger_time) #❹
np.mean(bus_time[idx] - passenger_time) * 60    #❺
```




    199.12512768644049




```python
np.mean(np.diff(bus_time)) * 60
```




    199.98208112933918




```python
#%figonly=观察者偏差
import matplotlib.gridspec as gridspec
pl.figure(figsize=(7.5, 3))

G = gridspec.GridSpec(10, 1)
ax1 = pl.subplot(G[:2,  0])
ax2 = pl.subplot(G[3:, 0])

ax1.vlines(bus_time[:10], 0, 1, lw=2, color="blue", label="公交车")
ptime = np.random.uniform(bus_time[0], bus_time[9], 100)
ax1.vlines(ptime, 0, 1, lw=1, color="red", alpha=0.2, label="乘客")
ax1.legend()
count, bins = np.histogram(passenger_time, bins=bus_time)
ax2.plot(np.diff(bins), count, ".", alpha=0.3, rasterized=True)
ax2.set_xlabel("公交车的时间间隔")
ax2.set_ylabel("等待人数");
```


![svg](scipy-400-stats_files/scipy-400-stats_29_0.svg)



```python
from scipy import integrate
t = 10.0 / 3  # 两辆公交车之间的平均时间间隔
bus_interval = stats.gamma(1, scale=t)
n, _ = integrate.quad(lambda x: 0.5 * x * x * bus_interval.pdf(x), 0, 1000)
d, _ = integrate.quad(lambda x: x * bus_interval.pdf(x), 0, 1000)
n / d * 60
```




    200.0



### 学生t-分布与t检验


```python
#%fig=模拟学生t-分布
mu = 0.0
n = 10
samples = stats.norm(mu).rvs(size=(100000, n)) #❶
t_samples = (np.mean(samples, axis=1) - mu) / np.std(samples, ddof=1, axis=1) * n**0.5 #❷
sample_dist, x = np.histogram(t_samples, bins=100, density=True) #❸
x = 0.5 * (x[:-1] + x[1:])
t_dist = stats.t(n-1).pdf(x)
print(("max error:", np.max(np.abs(sample_dist - t_dist))))
#%hide
pl.plot(x, sample_dist, lw=2, label="样本分布")
pl.plot(x, t_dist, lw=2, alpha=0.6, label="t分布")
pl.xlim(-5, 5)
pl.legend(loc="best");
```

    max error: 0.00658734287935



![svg](scipy-400-stats_files/scipy-400-stats_32_1.svg)



```python
#%figonly=当`df`增大，学生t-分布趋向于正态分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.5))
ax1.plot(x, stats.t(6-1).pdf(x), label="df=5", lw=2)
ax1.plot(x, stats.t(40-1).pdf(x), label="df=39", lw=2, alpha=0.6)
ax1.plot(x, stats.norm.pdf(x), "k--", label="norm")
ax1.legend()

ax2.plot(x, stats.t(6-1).sf(x), label="df=5", lw=2)
ax2.plot(x, stats.t(40-1).sf(x), label="df=39", lw=2, alpha=0.6)
ax2.plot(x, stats.norm.sf(x), "k--", label="norm")
ax2.legend();
```


![svg](scipy-400-stats_files/scipy-400-stats_33_0.svg)



```python
n = 30
np.random.seed(42)
s = stats.norm.rvs(loc=1, scale=0.8, size=n)
```


```python
t = (np.mean(s) - 0.5) / (np.std(s, ddof=1) / np.sqrt(n))
print((t, stats.ttest_1samp(s, 0.5)))
```

    2.65858434088 (2.6585843408822241, 0.012637702257091229)



```python
print(((np.mean(s) - 1) / (np.std(s, ddof=1) / np.sqrt(n))))
print((stats.ttest_1samp(s, 1), stats.ttest_1samp(s, 0.9)))
```

    -1.14501736704
    (-1.1450173670383303, 0.26156414618801477) (-0.38429702545421962, 0.70356191034252025)



```python
#%fig=红色部分为`ttest_1samp()`计算的p值
x = np.linspace(-5, 5, 500)
y = stats.t(n-1).pdf(x)
plt.plot(x, y, lw=2)
t, p = stats.ttest_1samp(s, 0.5)
mask = x > np.abs(t)
plt.fill_between(x[mask], y[mask], color="red", alpha=0.5)
mask = x < -np.abs(t)
plt.fill_between(x[mask], y[mask], color="red", alpha=0.5)
plt.axhline(color="k", lw=0.5)
plt.xlim(-5, 5);
```


![svg](scipy-400-stats_files/scipy-400-stats_37_0.svg)



```python
from scipy import integrate
x = np.linspace(-10, 10, 100000)
y = stats.t(n-1).pdf(x)
mask = x >= np.abs(t)
integrate.trapz(y[mask], x[mask])*2
```




    0.012633433707685974




```python
m = 200000
mean = 0.5
r = stats.norm.rvs(loc=mean, scale=0.8, size=(m, n))
ts = (np.mean(s) - mean) / (np.std(s, ddof=1) / np.sqrt(n))
tr = (np.mean(r, axis=1) - mean) / (np.std(r, ddof=1, axis=1) / np.sqrt(n))
np.mean(np.abs(tr) > np.abs(ts))
```




    0.012695




```python
np.random.seed(42)

s1 = stats.norm.rvs(loc=1, scale=1.0, size=20)
s2 = stats.norm.rvs(loc=1.5, scale=0.5, size=20)
s3 = stats.norm.rvs(loc=1.5, scale=0.5, size=25)

print((stats.ttest_ind(s1, s2, equal_var=False))) #❶
print((stats.ttest_ind(s2, s3, equal_var=True)))  #❷
```

    (-2.2391470627176755, 0.033250866086743665)
    (-0.59466985218561719, 0.55518058758105393)


### 卡方分布和卡方检验


```python
#%fig=使用随机数验证卡方分布
a = np.random.normal(size=(300000, 4))
cs = np.sum(a**2, axis=1)

sample_dist, bins = np.histogram(cs, bins=100, range=(0, 20), density=True)
x = 0.5 * (bins[:-1] + bins[1:])
chi2_dist = stats.chi2.pdf(x, 4) 
print(("max error:", np.max(np.abs(sample_dist - chi2_dist))))
#%hide
pl.plot(x, sample_dist, lw=2, label="样本分布")
pl.plot(x, chi2_dist, lw=2, alpha=0.6, label="$\chi ^{2}$分布")
pl.legend(loc="best");
```

    max error: 0.00340194486328



![svg](scipy-400-stats_files/scipy-400-stats_42_1.svg)



```python
#%fig=模拟卡方分布
repeat_count = 60000
n, k = 100, 5

np.random.seed(42)
ball_ids = np.random.randint(0, k, size=(repeat_count, n)) #❶
counts = np.apply_along_axis(np.bincount, 1, ball_ids, minlength=k) #❷
cs2 = np.sum((counts - n/k)**2.0/(n/k), axis=1) #❸
k = stats.kde.gaussian_kde(cs2) #❹
x = np.linspace(0, 10, 200)
pl.plot(x, stats.chi2.pdf(x, 4), lw=2, label="$\chi ^{2}$分布")
pl.plot(x, k(x), lw=2, color="red", alpha=0.6, label="样本分布")
pl.legend(loc="best")
pl.xlim(0, 10);
```


![svg](scipy-400-stats_files/scipy-400-stats_43_0.svg)



```python
def choose_balls(probabilities, size):
    r = stats.rv_discrete(values=(range(len(probabilities)), probabilities))
    s = r.rvs(size=size)
    counts = np.bincount(s)    
    return counts

np.random.seed(42)
counts1 = choose_balls([0.18, 0.24, 0.25, 0.16, 0.17], 400)
counts2 = choose_balls([0.2]*5, 400)

%C counts1; counts2
```

          counts1               counts2       
    --------------------  --------------------
    [80, 93, 97, 64, 66]  [89, 76, 79, 71, 85]



```python
chi1, p1 = stats.chisquare(counts1)
chi2, p2 = stats.chisquare(counts2)

print(("chi1 =", chi1, "p1 =", p1))
print(("chi2 =", chi2, "p2 =", p2))
```

    chi1 = 11.375 p1 = 0.0226576012398
    chi2 = 2.55 p2 = 0.635705452704



```python
#%figonly=卡方检验计算的概率为阴影部分的面积
x = np.linspace(0, 30, 200)
CHI2 = stats.chi2(4)
pl.plot(x, CHI2.pdf(x), "k", lw=2)
pl.vlines(chi1, 0, CHI2.pdf(chi1))
pl.vlines(chi2, 0, CHI2.pdf(chi2))
pl.fill_between(x[x>chi1], 0, CHI2.pdf(x[x>chi1]), color="red", alpha=1.0)
pl.fill_between(x[x>chi2], 0, CHI2.pdf(x[x>chi2]), color="green", alpha=0.5)
pl.text(chi1, 0.015, r"$\chi^2_1$", fontsize=14)
pl.text(chi2, 0.015, r"$\chi^2_2$", fontsize=14)
pl.ylim(0, 0.2)
pl.xlim(0, 20);
```


![svg](scipy-400-stats_files/scipy-400-stats_46_0.svg)



```python
table = [[43, 9], [44, 4]]
chi2, p, dof, expected = stats.chi2_contingency(table)
%C chi2; p
```

           chi2                  p         
    ------------------  -------------------
    1.0724852071005921  0.30038477039056899



```python
stats.fisher_exact(table)
```




    (0.43434343434343436, 0.23915695682225618)


