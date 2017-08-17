# -*- coding: utf-8 -*-
import pickle
import numpy as np

f = file("IFS.data", "rb")
names = pickle.load(f)

data = []
for i in range(len(names)):
    data.append( np.load(f).tolist())
    
f2 = file("ifs2.data", "wb")   
pickle.dump(list(zip(names, data)), f2)
