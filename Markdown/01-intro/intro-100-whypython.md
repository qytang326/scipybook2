
# Python科学计算环境的安装与简介

## Python简介

Python是一种解释型、面向对象、动态的高级程序设计语言。自从90年代初Python语言诞生至今，它逐渐被广泛应用于处理系统管理任务和开发Web系统。目前Python已经成为最受欢迎的程序设计语言之一。

由于Python语言的简洁、易读以及可扩展性，在国外用Python做科学计算的研究机构日益增多，一些知名大学已经采用Python教授程序设计课程。众多开源的科学计算软件包都提供了Python的调用接口，例如计算机视觉库OpenCV、三维可视化库VTK、复杂网络分析库igraph等。而Python专用的科学计算扩展库就更多了，例如三个十分经典的科学计算扩展库：NumPy、SciPy和matplotlib，它们分别为Python提供了快速数组处理、数值运算以及绘图等功能。因此Python语言及其众多的扩展库所构成的开发环境十分适合工程技术、科研人员处理实验数据，制作图表，甚至开发科学计算应用程序。近年随着数据分析扩展库Pandas、机器学习扩展库scikit-learn以及IPython Notebook交互环境的日益成熟，Python也逐渐成为数据分析领域的首选工具。

说起科学计算，首先会被提到的可能是MATLAB。然而除了MATLAB的一些专业性很强的工具箱目前还无法替代之外，MATLAB的大部分常用功能都可以在Python世界中找到相应的扩展库。和MATLAB相比，用Python做科学计算有如下优点：

* 首先，MATLAB是一款商用软件，并且价格不菲。而Python完全免费，众多开源的科学计算库都提供了Python的调用接口。用户可以在任何计算机上免费安装Python及其绝大多数扩展库。

* 其次，与MATLAB相比Python是一门更易学、更严谨的程序设计语言。它能让用户编写出更易读、易维护的代码。

* 最后，MATLAB主要专注于工程和科学计算。然而即使在计算领域，也经常会遇到文件管理、界面设计、网络通信等各种需求。而Python有着丰富的扩展库，可以轻易完成各种高阶任务，开发者可以用Python实现完整应用程序所需的各种功能。

### Python2还是Python3

自从2008年发布以来，Python3经过了5个小版本的更迭，无论是语法还是标准库都发展得十分成熟。许多重要的扩展库也已经逐渐同时支持Python2和Python3。但是由于Python3不向下兼容，目前大多数开发者仍然在生产环境中使用Python 2.7。在PyCon2014大会上，Python之父宣布Python 2.7的官方支持延长至2020年。因此本书仍然使用Python 2.7作为开发环境。

在本书涉及的扩展库中，IPython、NumPy、SciPy、matplotlib、Pandas、SymPy、Cython、Spyder和OpenCV等都已经支持Python 3。而Traits、TraitsUI、TVTK、Mayavi等扩展库则尚未着手Python 3的移植。虽然一些新兴的三维可视化扩展库正朝着替代Mayavi的方向努力，但目前Python环境中尚未有能替代VTK和Mayavi的专业级别的三维可视化扩展库，因此本书仍保留第一版中相关的章节。

### 开发环境

和MATLAB等商用软件不同，Python的众多扩展库由许多社区分别维护和发布，因此要一一将其收集齐全并安装到计算机中是一件十分耗费时间和精力的事情。本节介绍两个科学计算用的Python集成软件包。读者只需要下载并执行一个安装程序，就能安装好本书涉及的所有扩展库。



#### WinPython

> **LINK**

> https://winpython.github.io/
>
> WinPython的下载地址

WinPython只能在Windows系统中运行，其安装包不会修改系统的任何配置，各种扩展库的用户配置文件也保存在WinPython的文件夹之下。因此可将整个运行环境复制到U盘中，在任何Windows操作系统的计算机上运行。WinPython提供了一个安装扩展库的WinPython Control Panel界面程序，通过它可以安装Python的各种扩展库。可以通过下面的链接下载已经编译好的二进制扩展库安装包，然后通过WinPython Control Panel安装。

> **LINK**

> http://www.lfd.uci.edu/~gohlke/pythonlibs/
>
> 从该网址可以下载各种Python扩展库的Windows安装文件

下图显示通过WinPython Control Panel安装本书介绍的几个扩展库。通过“Add Packages”按钮添加扩展库的安装程序之后，点击“Install Packages”按钮一次性安装勾选的所有扩展库。

![通过WinPython Control Panel安装扩展库](/files/images/WinPython_Control_Panel.png "")

虽然手动安装扩展库有些麻烦，不过这种方式适合没有网络连接或者网络较慢的计算机。例如在笔者的工作环境中，有大量的实验用计算机不允许连接互联网。

如果读者从WinPython的官方网站下载WinPython开发环境，为了运行本书所有实例程序，还需要安装如下扩展库：

* VTK、Mayavi、pyface、Traits和TraitsUI：在图形界面以及三维可视化章节需要用到这些扩展库。
* OpenCV：在图像处理章节需要用该扩展库。

#### Anaconda

> **LINK**

> https://store.continuum.io/cshop/anaconda/
>
> Anaconda的下载地址

由CONTINUUM开发的Anaconda开发环境支持Windows、Linux和Mac OSX。安装时会提示是否修改`PATH`环境变量和注册表，如果希望手工激活Anaconda环境，请取消选择这两个选项。

在命令行中运行安装路径之下的批处理文件`Scripts\anaconda.bat`启动Anaconda环境，然后就可以输入下面的`conda`命令管理扩展库了。

|           命令           |         说明         |
|:------------------------|:--------------------|
| `conda list`             | 列出所有的扩展库     |
| `conda update 扩展库名`  | 升级扩展库           |
| `conda install 扩展库名` | 安装扩展库           |
| `conda search 模板`      | 搜索符合模板的扩展库 |

`conda`命令本身也是一个扩展库，因此通常在执行上述命令之前，可以先运行`conda update conda`尝试升级到最新版本。`conda`缺省从官方频道下载扩展库，如果未找到指定的扩展库，还可以使用`anaconda`命令从Anaconda网站的其它频道搜索指定的扩展库。例如下面的命令搜索可使用`conda`安装的OpenCV扩展库：

```
binstar search -t conda opencv
```

找到包含目标扩展库的频道之后，输入下面的命令从指定的频道`rsignell`安装：

```
conda install opencv-python -c rsignell
```

还可以使用`pip`命令安装下载的扩展库文件，例如从前面介绍的网址下载文件`opencv_python-2.4.11-cp27-none-win32.whl`之后，切换到该文件所在的路径并输入`pip install opencv_python-2.4.11-cp27-none-win32.whl`即可安装该扩展库。

#### 使用附赠光盘中的开发环境

本书的附赠光盘中包含了能运行本书所有实例程序的`WinPython`压缩包：`winpython.zip`。请读者将之解压到C盘根目录之下，该压缩包会创建`C:\WinPython-32bit-2.7.9.2`目录。

然后将本书附盘中提供的代码目录`scipybook2`复制到计算机的硬盘中，为了保证代码正常运行，请确保该代码目录的完整路径中不包含空格和中文字符。在`scipybook2`中包含三个子目录：

* `codes`：其中的`scpy2`子目录下包含本书提供的示例程序，该示例程序库采用包的形式管理，因此需要将它添加进Python的包搜索路径环境变量`PYTHONPATH`中才能正确运行`scpy2`中的示例程序。在`scipybook2`目录下的批处理文件`run_console.bat`和`run_notebook.bat`中会自动设置该环境变量。

* `notebooks`：本书完全使用IPython Notebook编写，该目录下的Notebook文件中保存了本书所有章节的标题以及示例代码。读者可以通过`run_notebook.bat`批处理文件启动本书的编写环境。为了保护本书版权，除本章之外的其他所有章节的文字解说内容都已被删除。

* `settings`：保存各种扩展库的配置文件。这些文件会保存在`HOME`环境变量所设置的目录之下，缺省值为：`C:\Users\用户名`。为了避免与读者的系统中的配置文件发生冲突，在批处理文件中将`HOME`环境变量修改为该`settings`目录。

为了确认开发环境正确安装，请读者运行`run_console.bat`，然后在命令行中执行`python -m scpy2`，并检查是否打印出开发环境中各个扩展库的版本信息。

> **TIP**

> 如果读者将`winpython.zip`文件解压到别的路径之下，可以修改`env.bat`文件中第二行中的路径。


```python
!python -m scpy2
```

    Welcome to Scpy2
    Python: 2.7.9
    executable: C:\WinPython-32bit-2.7.9.2\python-2.7.9\python.exe
    Cython              : 0.22
    matplotlib          : 1.4.3
    numpy-MKL           : 1.9.1
    opencv_python       : 2.4.11
    pandas              : 0.16.0
    scipy               : 0.15.0
    sympy               : 0.7.6


### 集成开发环境(IDE)

本节介绍两个常用的Python集成开发环境，它们能实现自动完成、定义跳转、自动重构、调试等常用的IDE功能，并集成了IPython的交互环境以及查看数组、绘制图表等科学计算开发中常用的功能。熟练使用这些工具能极大地提高编程效率。

#### Spyder

Spyder是WinPython的作者开发的一个简单的集成开发环境，可通过WinPython的安装目录下的`Spyder.exe`运行。如果读者希望在本书的开发环境中运行Spyder，可以在`run_console.bat`开启的命令行中输入`spyder`命令。

和其它的Python开发环境相比，它最大的优点就是模仿MATLAB的“工作空间”的功能，可以很方便地观察和修改数组的值。`ref:fig-next`是Spyder的界面截图。

![在Spyder中执行图像处理的程序](/files/images/spyder_interface.png "")

Spyder的界面由许多泊坞窗口构成，用户可以根据自己的喜好调整它们的位置和大小。当多个窗口在一个区域中时，将使用标签页的形式显示。例如在`ref:fig-prev`中，可以看到“Editor”、“Variable explorer”、“File explorer”、“IPython Console”等窗口。在View菜单中可以设置是否显示这些窗口。下表列出了Spyder的主要窗口及其作用：

| 窗口名  | 功能  |
|:--|:--|
| Editor  | 编辑程序，以标签页的形式编辑多个程序文件  |
| Console  | 在别的进程中运行的Python控制台  |
| Variable explorer  | 显示Python控制台中的变量列表  |
| Object inspector | 查看对象的说明文档和源程序 |
| File explorer  | 文件浏览器，用来打开程序文件或者切换当前路径  |

按F5将在另外的控制台进程中运行当前编辑器中的程序。第一次运行程序时，将弹出一个如`ref:fig-next`所示的运行配置对话框。在此对话框中可以对程序的运行进行如下配置：

* Command line options：输入程序的运行参数。

* Working directory：输入程序的运行路径。

* Execute in current Python or IPython interpreter：在当前的Python控制台中运行程序。程序可以访问此控制台中的所有全局对象，控制台中已经载入的模块不需要重新载入，因此程序的启动速度较快。

* Execute in a new dedicated Python interpreter：新开一个Python控制台并在其中运行程序，程序的启动速度较慢，但是由于新控制台中没有多余的全局对象，因此更接近真实运行的情况。当选择此项时，还可以勾选“Interact with the Python interpreter after execution”，这样当程序结束运行之后，控制台进程继续运行，可以通过它查看程序运行之后的所有的全局对象。此外，还可以在“Command line options”中输入新控制台的启动参数。

* Execute in an external System terminal：选择该选项则完全脱离Spyder运行程序。

运行配置对话框只会在第一次运行程序时出现，如果想修改程序的运行配置，可以按F6打开运行配置对话框。

![运行配置对话框](/files/images/spyder_run_dialog.png "")

控制台中的全局对象可以在“Variable explorer”窗口中找到。此窗口支持数值、字符串、元组、列表、字典以及NumPy的数组等对象的显示和编辑。`ref:fig-next`（左）是“Variable explorer”窗口的截图，列出了当前运行环境中的变量名、类型、大小以及其内容。右键点击变量名弹出对此变量进行操作的菜单。在菜单中选择Edit选项，弹出`ref:fig-next`（右）所示的数组编辑窗口。此编辑窗口中的单元格的背景颜色直观地显示了数值的大小。当有多个控制台运行时，“Variable explorer”窗口显示当前控制台中的全局对象。

![使用“Variable explorer”查看和编辑变量内容](/files/images/spyder_workspace_01.png "")

选择菜单中的Plot选项，将弹出如`ref:fig-next`所示的绘图窗口。在绘图窗口的工具栏中点击最右边的按钮，将弹出一个编辑绘图对象的对话框。图中使用此对话框修改了曲线的颜色和线宽。

![在“Variable explorer”中将数组绘制成曲线图](/files/images/spyder_workspace_02.png "")

Spyder的功能比较多，这里仅介绍一些常用的功能和技巧：

* 缺省配置下，“Variable explorer”中不显示大写字母开头的变量，可以点击其工具栏中的配置按钮(最后一个按钮)，在菜单中取消“Exclude capitalized references”的勾选状态。

* 在控制台中，可以按Tab按键自动补全。在变量名之后输入“?”，可以在“Object inspector”窗口中查看对象的说明文档。此窗口的Options菜单中的“Show source”选项可以开启显示函数的源程序。

* 可以通过“Working directory”工具栏修改工作路径，用户程序运行时，将以此工作路径作为当前路径。只需要修改工作路径，就可以用同一个程序处理不同文件夹下的数据文件。

* 在程序编辑窗口中按住Ctrl按键，并单击变量名、函数名、类名或者模块名，可以快速跳转到其定义位置。如果是在别的程序文件中定义的，将打开此文件。在学习一个新的模块库的用法时，经常会需要查看模块中的某个函数或者某个类是如何定义的，使用此功能可以帮助我们快速查看和分析各个库的源程序。

#### PyCharm

PyCharm是由JetBrains开发的集成开发环境。它具有项目管理、代码跳转、代码格式化、自动完成、重构、自动导入、调试等功能。虽然专业版价格比较高，但是它提供的免费社区版本具有开发Python程序所需的所有功能。如果读者需要开发较大的应用程序，使用它可以提高开发效率，保证代码的质量。

> **LINK**

> http://www.jetbrains.com/pycharm
>
> PyCharm的官方网站

如果读者使用本书提供的便携WinPython版本，则需要在PyCharm中设置Python解释器。通过菜单“File → Settings”打开配置对话框，在左栏中找到“Project Interpreter”，然后通过右侧的齿轮按钮，并选择弹出的菜单中的“Add Local”选项，即可打开如`ref:fig-next`所示的对话框：

![配置Python解释器的路径](/files/images/pycharm_interpreter.png "")

由于本书提供的代码没有复制到Python的库搜索路径中，可以将`scpy2`的路径添加进`PYTHONPATH`环境变量，或者在PyCharm中将`scpy2`所在的路径添加进Python的搜索路径。按上述的齿轮按钮，并选择“More...”，将打开`ref:fig-next`中左侧的对话框，选择解释器之后，按右侧工具栏中最下方的按钮打开路径配置对话框，通过此对话框添加本书提供的`scpy2`库所在的路径。

![添加库搜索路径](/files/images/pycharm_path.png "")


```python

```
