# -*- coding: utf-8 -*-
import numpy as np
from visual import *
scene.background = 1,1,1 # 场景背景颜色
scene.foreground = 0.7,0.7,0.7 # 模型颜色

box(size=(9,6,0.2), pos=(0,0,-0.1), color=(1,1,1)) # 地板
xy = ((x,y) for x in range(-3,4,2) for y in range(-2,3,2)) 
idx = 0
x,y = next(xy)
sphere(pos=(x,y,0.5)) # 球体
x,y = next(xy)
cone(pos=(x,y,0), axis=(0,0,1)) # 圆锥
x,y = next(xy)
cylinder(pos=(x,y,0), axis=(0,0,1)) # 圆柱
x,y = next(xy)
arrow(pos=(x,y,0), axis=(0,0,2), shaftwidth=1) # 箭头
x,y = next(xy)
convex(pos=[(0,0,-1),(0.5,0.5,0),(-0.5,0.5,0),  # 凸多面体 
			(0.5,-0.5,0),(-0.5,-0.5,0),(0,0,1)], frame=frame(pos=(x,y,1))) 
x,y = next(xy)
t = np.linspace(0,6*pi,100)
pos =np.array([0.05*t*np.sin(t), 0.05*t*np.cos(t), t/3/pi])
curve(pos=pos.T, radius=0.05, frame=frame(pos=(x,y,0)))  # 曲线 
x,y = next(xy)
ellipsoid(pos=(x,y,0.5), size=(2,1.5,1)) # 椭球
x,y = next(xy)
helix(pos=(x,y,0), axis=(0,0,2), thickness =0.1) # 弹簧、线圈
x,y = next(xy)
pos = np.random.normal(scale=0.5, size=(50,3))
points(pos=pos, frame=frame(pos=(x,y,1)))  # 散布点 
x,y = next(xy)
pyramid(pos=(x,y,0), axis=(0,0,2)) # 方锥
x,y = next(xy)
ring(pos=(x,y,0.3), axis=(0,0,1), thickness=0.3, radius=0.8) # 圆环
x,y = next(xy)
text(pos=(x,y,0), text="Text", depth=0.3, up=(0,0,1), align="center") # 三维文字
