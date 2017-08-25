# -*- coding: utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
t = np.linspace(0, 10, 1000)
y = np.sin(t)
plt.plot(t, y)
plt.xlabel("时间", fontproperties=font) 
plt.ylabel("振幅", fontproperties=font)
plt.title("正弦波", fontproperties=font)
plt.show()