
##函数曲线绘制工具 

> **SOURCE**

> `scpy2.traits.traitsui_function_plotter`：采用TraitsUI编写的函数曲线绘制工具。


```python
#%hide
%exec_python -m scpy2.traits.traitsui_function_plotter
```

> **WARNING**

> `Code`对应的编辑器代码存在BUG，请读者将`patches\pygments_highlighter.py`复制到`site-packages\pyface\ui\qt4\code_editor`下覆盖原有的文件。


```python
%%include python traits/traitsui_function_plotter.py 2
class FunctionPlotter(HasTraits):
    figure = Instance(Figure, ()) #❶
    code = Code()  #❷
    points = List(Instance(Point), [])  #❸
    draw_button = Button("Plot")

    view = View(
        VSplit(
            Item("figure", editor=MPLFigureEditor(toolbar=True), show_label=False), 
            HSplit(
                VGroup(
                    Item("code", style="custom"), 
                    HGroup(
                        Item("draw_button", show_label=False),
                    ),
                    show_labels=False
                ),
                Item("points", editor=point_table_editor, show_label=False) 
            )
        ),
        width=800, height=600, title="Function Plotter", resizable=True
    )
```


```python
%%include python traits/traitsui_function_plotter.py 1
class Point(HasTraits):
    x = Float()
    y = Float()


point_table_editor = TableEditor(
    columns=[ObjectColumn(name='x', width=100, format="%g"),
             ObjectColumn(name='y', width=100, format="%g")],
    editable=True,
    sortable=False,
    sort_model=False,
    auto_size=False,
    row_factory=Point
)
```


```python
%%include python traits/traitsui_function_plotter.py 3
    def __init__(self, **kw):
        super(FunctionPlotter, self).__init__(**kw)
        self.figure.canvas_events = [ #❶
            ("button_press_event", self.memory_location),
            ("button_release_event", self.update_location)
        ]
        self.button_press_status = None #保存鼠标按键按下时的状态
        self.lines = [] #保存所有曲线
        self.functions = [] #保存所有的曲线函数
        self.env = {} #代码的执行环境

        self.axe = self.figure.add_subplot(1, 1, 1)
        self.axe.callbacks.connect('xlim_changed', self.update_data) #❷
        self.axe.set_xlim(0, 1)
        self.axe.set_ylim(0, 1)
        self.points_line, = self.axe.plot([], [], "kx", ms=8, zorder=1000) #数据点
```


```python
%%include python traits/traitsui_function_plotter.py 4
    def memory_location(self, evt):
        if evt.button in (1, 3):
            self.button_press_status = time.clock(), evt.x, evt.y
        else:
            self.button_press_status = None

    def update_location(self, evt):
        if evt.button in (1, 3) and self.button_press_status is not None:
            last_clock, last_x, last_y = self.button_press_status
            if time.clock() - last_clock > 0.5: #❶
                return
            if ((evt.x - last_x) ** 2 + (evt.y - last_y) ** 2) ** 0.5 > 4: #❷
                return

        if evt.button == 1:
            if evt.xdata is not None and evt.ydata is not None:
                point = Point(x=evt.xdata, y=evt.ydata) #❸
                self.points.append(point)
        elif evt.button == 3:
            if self.points:
                self.points.pop() #❹
```


```python
%%include python traits/traitsui_function_plotter.py 5
    @on_trait_change("points[]")
    def _points_changed(self, obj, name, new):
        for point in new:
            point.on_trait_change(self.update_points, name="x, y") #❶
        self.update_points()

    def update_points(self): #❷
        arr = np.array([(point.x, point.y) for point in self.points])
        if arr.shape[0] > 0:
            self.points_line.set_data(arr[:, 0], arr[:, 1])
        else:
            self.points_line.set_data([], [])
        self.update_figure()

    def update_figure(self): #❸
        if self.figure.canvas is not None: #❹
            self.figure.canvas.draw_idle()
```


```python
%%include python traits/traitsui_function_plotter.py 6
    def update_data(self, axe):
        xmin, xmax = axe.get_xlim()
        x = np.linspace(xmin, xmax, 500)
        for line, func in zip(self.lines, self.functions):
            y = func(x)
            line.set_data(x, y)
        self.update_figure()
```


```python
%%include python traits/traitsui_function_plotter.py 7
    def _draw_button_fired(self):
        self.plot_lines()

    def plot_lines(self):
        xmin, xmax = self.axe.get_xlim() #❶
        x = np.linspace(xmin, xmax, 500)
        self.env = {"points": np.array([(point.x, point.y) for point in self.points])} #❷
        exec self.code in self.env

        results = []
        for line in self.lines:
            line.remove()
        self.axe.set_color_cycle(None) #重置颜色循环
        self.functions = []
        self.lines = []
        for name, value in self.env.items(): #❸
            if name.startswith("_"): #忽略以_开头的名字
                continue
            if callable(value):
                try:
                    y = value(x)
                    if y.shape != x.shape: #输出数组应该与输入数组的形状一致
                        raise ValueError("the return shape is not the same as x")
                except Exception as ex:
                    import traceback
                    print "failed when call function {}\n".format(name)
                    traceback.print_exc()
                    continue

                results.append((name, y))
                self.functions.append(value)

        for (name, y), function in zip(results, self.functions):
            #如果函数有plot_parameters属性,则用其作为plot()的参数
            kw = getattr(function, "plot_parameters", {})  #❹
            label = kw.get("label", name)
            line, = self.axe.plot(x, y, label=label, **kw)
            self.lines.append(line)

        points = self.env.get("points", None) #❺
        if points is not None:
            self.points = [Point(x=x, y=y) for x, y in np.asarray(points).tolist()]

        self.axe.legend()
        self.update_figure()
```
