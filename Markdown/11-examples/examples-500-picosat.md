

```python
%matplotlib_svg
import pylab as pl
import numpy as np
from scpy2.cycosat import CoSAT
```

## 布尔可满足性问题求解器

> **LINK**

> http://fmv.jku.at/picosat/

> PicoSAT的下载地址


```python
from sympy import symbols
from sympy.logic.boolalg import to_cnf

A, B, C, D = symbols("A:D") #❶
S1 = ~A   
S2 = D
S3 = B
S4 = ~D
dnf = ((S1 & ~S2 & ~S3 & ~S4) |    #❷
       (~S1 & S2 & ~S3 & ~S4) | 
       (~S1 & ~S2 & S3 & ~S4) | 
       (~S1 & ~S2 & ~S3 & S4))

cnf = to_cnf(dnf)  #❸
%sympy_latex cnf
```


$$A \wedge \neg B \wedge \left(A \vee D\right) \wedge \left(A \vee \neg B\right) \wedge \left(A \vee \neg D\right) \wedge \left(D \vee \neg B\right) \wedge \left(D \vee \neg D\right) \wedge \left(\neg B \vee \neg D\right)$$



```python
from sympy.logic.inference import satisfiable
satisfiable(cnf)
```




    {B: False, A: True, D: False}




```python
from scpy2.cycosat import CoSAT

sat = CoSAT()
problem = [[1, -4], [1, -2], [1, 4], [-4, -2],
           [-4, 4], [-2, 4], [-1, 4, 2, -4]]

sat.add_clauses(problem)  # ❶
print((sat.solve()))  # ❷
```

    [1, -1, -1, -1]


### 用Cython包装PicoSAT


```python
%%include cython cycosat/cycosat.pyx 1
cdef extern from "picosat.h":
    ctypedef enum:
        PICOSAT_UNKNOWN
        PICOSAT_SATISFIABLE
        PICOSAT_UNSATISFIABLE        

    ctypedef struct PicoSAT:
        pass
    
    PicoSAT * picosat_init ()
    void picosat_reset (PicoSAT *)
    int picosat_add (PicoSAT *, int lit)
    int picosat_add_lits(PicoSAT *, int * lits)
    int picosat_sat (PicoSAT *, int decision_limit)
    int picosat_variables (PicoSAT *)
    int picosat_deref (PicoSAT *, int lit)
    void picosat_assume (PicoSAT *, int lit)
```


```python
%%include cython cycosat/cycosat.pyx 2
from cpython cimport array
    
cdef class CoSAT:
    
    cdef PicoSAT * sat
    cdef public array.array clauses #❶
    cdef int buf_pos   #❷

    def __cinit__(self):
        self.buf_pos = -1
        self.clauses = array.array("i", [0])
        
    def __dealloc__(self):
        self.close_sat()

    cdef close_sat(self):
        if self.sat is not NULL:
            picosat_reset(self.sat)
            self.sat = NULL
```


```python
%%include cython cycosat/cycosat.pyx 3
    cdef build_clauses(self):
        cdef int * p
        cdef int i
        cdef int count = len(self.clauses)
        if count - 1 == self.buf_pos:
            return
        p = self.clauses.data.as_ints #❶
        for i in range(self.buf_pos, count - 1):
            if p[i] == 0:
                picosat_add_lits(self.sat, p+i+1)
        self.buf_pos = count - 1
            
    cdef build_sat(self):  #❷
        if self.buf_pos == -1:
            self.close_sat()
            self.sat = picosat_init()
            if self.sat is NULL:
                raise MemoryError()
            self.buf_pos = 0
        self.build_clauses()
```


```python
%%include cython cycosat/cycosat.pyx 4
    cdef _add_clause(self, clause):
        self.clauses.extend(clause)
        self.clauses.append(0)
        
    def add_clause(self, clause):
        self._add_clause(clause)
        
    def add_clauses(self, clauses):
        for clause in clauses:
            self._add_clause(clause)
```


```python
%%include cython cycosat/cycosat.pyx 5
    def get_solution(self):
        cdef list solution = []
        cdef int i, v
        cdef int max_index
        
        max_index = picosat_variables(self.sat)  #❶
        for i in range(max_index):
            v = picosat_deref(self.sat, i+1)  #❷
            solution.append(v)
        return solution
               
    def solve(self, limit=-1):
        cdef int res
        self.build_sat()  #❸
        res = picosat_sat(self.sat, limit)  #❹
        if res == PICOSAT_SATISFIABLE:
            return self.get_solution()
        elif res == PICOSAT_UNSATISFIABLE:
            return "PICOSAT_UNSATISFIABLE"
        elif res == PICOSAT_UNKNOWN:
            return "PICOSAT_UNKNOWN"
```


```python
%%include cython cycosat/cycosat.pyx 6
    def iter_solve(self):
        cdef int res, i
        cdef list solution
        self.build_sat()
        while True:
            res = picosat_sat(self.sat, -1)
            if res == PICOSAT_SATISFIABLE:
                solution = self.get_solution()
                yield solution
                for i in range(len(solution)):
                    picosat_add(self.sat, -solution[i] * (i+1))   #❶
                picosat_add(self.sat, 0)
            else:
                break
        self.iter_reset()
        
    def iter_reset(self):  #❷
        self.buf_pos = -1
```

### 数独游戏


```python
bools = np.arange(1, 9 * 9 * 9 + 1).reshape(9, 9, 9)
```


```python
from itertools import combinations

def get_conditions(bools):
    conditions = []
    n = bools.shape[-1]
    index = np.array(list(combinations(list(range(n)), 2)))  # ❶
    # 以最后一轴为组
    # 第一个条件: 每组只能有一个为真
    conditions.extend(bools.reshape(-1, n).tolist())  # ❷
    # 第二个条件: 每组中没有两个同时为真
    conditions.extend((-bools[..., index].reshape(-1, 2)).tolist())  # ❸
    return conditions
```


```python
print((get_conditions(np.array([[1, 2, 3], [4, 5, 6]]))))
```

    [[1, 2, 3], [4, 5, 6], [-1, -2], [-1, -3], [-2, -3], [-4, -5], [-4, -6], [-5, -6]]



```python
c1 = get_conditions(bools)  # 每个单元格只能取1-9之中的一个数字
c2 = get_conditions(np.swapaxes(bools, 1, 2))  # 每行的数字不能重复
c3 = get_conditions(np.swapaxes(bools, 0, 2))  # 每列的数字不能重复
```


```python
tmp = np.swapaxes(bools.reshape(3, 3, 3, 3, 9), 1, 2).reshape(9, 9, 9)
c4 = get_conditions(np.swapaxes(tmp, 1, 2))  # 每块的数字不能重复

conditions = c1 + c2 + c3 + c4
```


```python
def format_solution(solution):
    solution = np.array(solution).reshape(9, 9, 9)  # ❶
    return (np.where(solution == 1)[2] + 1).reshape(9, 9)  # ❷

sat = CoSAT()
sat.add_clauses(conditions)
solution = sat.solve()
format_solution(solution)
```




    array([[9, 8, 7, 6, 5, 4, 3, 2, 1],
           [6, 5, 4, 3, 1, 2, 9, 8, 7],
           [3, 2, 1, 9, 8, 7, 6, 5, 4],
           [8, 9, 6, 7, 4, 5, 2, 1, 3],
           [7, 4, 5, 2, 3, 1, 8, 9, 6],
           [2, 1, 3, 8, 9, 6, 7, 4, 5],
           [5, 7, 9, 4, 6, 8, 1, 3, 2],
           [4, 6, 8, 1, 2, 3, 5, 7, 9],
           [1, 3, 2, 5, 7, 9, 4, 6, 8]])




```python
sudoku_str = """
000000185
007030000
000021400
800000020
003905600
050000004
004860000
000040300
931000000"""

sudoku = np.array([[int(x) for x in line]
                   for line in sudoku_str.strip().split()])
r, c = np.where(sudoku != 0)
v = sudoku[r, c] - 1

conditions2 = [[x] for x in bools[r, c, v]]  # ❶
print "conditions2:"
%col 6 conditions2
sat = CoSAT()
sat.add_clauses(conditions + conditions2)  # ❷
solution = sat.solve()
format_solution(solution)
```

    conditions2:
    [[55],     [71],     [77],     [106],    [120],    [200],   
     [208],    [220],    [251],    [308],    [345],    [360],   
     [374],    [384],    [419],    [481],    [508],    [521],   
     [528],    [607],    [624],    [657],    [660],    [667]]   





    array([[3, 6, 2, 7, 9, 4, 1, 8, 5],
           [4, 1, 7, 5, 3, 8, 2, 6, 9],
           [5, 9, 8, 6, 2, 1, 4, 3, 7],
           [8, 7, 9, 4, 1, 6, 5, 2, 3],
           [2, 4, 3, 9, 7, 5, 6, 1, 8],
           [1, 5, 6, 3, 8, 2, 7, 9, 4],
           [7, 2, 4, 8, 6, 3, 9, 5, 1],
           [6, 8, 5, 1, 4, 9, 3, 7, 2],
           [9, 3, 1, 2, 5, 7, 8, 4, 6]])




```python
%%include cython cycosat/cycosat.pyx 7
    def assume_solve(self, assumes):
        self.build_sat()
        for assume in assumes:
            picosat_assume(self.sat, assume)
        return self.solve()
```


```python
sat = CoSAT()
sat.add_clauses(conditions)
solution = sat.assume_solve(bools[r, c, v].tolist())
format_solution(solution)
```




    array([[3, 6, 2, 7, 9, 4, 1, 8, 5],
           [4, 1, 7, 5, 3, 8, 2, 6, 9],
           [5, 9, 8, 6, 2, 1, 4, 3, 7],
           [8, 7, 9, 4, 1, 6, 5, 2, 3],
           [2, 4, 3, 9, 7, 5, 6, 1, 8],
           [1, 5, 6, 3, 8, 2, 7, 9, 4],
           [7, 2, 4, 8, 6, 3, 9, 5, 1],
           [6, 8, 5, 1, 4, 9, 3, 7, 2],
           [9, 3, 1, 2, 5, 7, 8, 4, 6]])



> **SOURCE**

> `scpy2.examples.sudoku_solver`：采用matplotlib制作的数独游戏求解器


```python
#%hide
%exec_python -m scpy2.examples.sudoku_solver
```

### 扫雷游戏

#### 识别雷区中的数字


```python
import cv2

X0, Y0, SIZE, COLS, ROWS = 30, 30, 18, 30, 16
SHAPE = ROWS, SIZE, COLS, SIZE, -1

mine_area = np.s_[Y0:Y0 + SIZE * ROWS, X0:X0 + SIZE * COLS, :]  # ❶

img_init = cv2.imread("mine_init.png")[mine_area]
img_mine = cv2.imread("mine01.png")[mine_area]
img_numbers = cv2.imread("mine_numbers.png")  # ❷
img_numbers.shape
```




    (96, 12, 3)



> **TIP**

> 可以通过`pl.hist()`绘制`mask_mean`数组的直方图，找到最佳的阈值。


```python
#%fig=计算已打开方块的位置
mask = (img_init != img_mine).reshape(SHAPE)
mask_mean = np.mean(mask, axis=(1, 3, 4))
block_mask = mask_mean > 0.3

fig, axes = pl.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(block_mask, interpolation="nearest", cmap="gray")
axes[1].imshow(img_mine[:, :, ::-1])
axes[0].set_axis_off()
axes[1].set_axis_off()
fig.subplots_adjust(wspace=0.01)
```


![svg](examples-500-picosat_files/examples-500-picosat_29_0.svg)



```python
from scipy.spatial import distance

img_mine2 = np.swapaxes(img_mine.reshape(SHAPE), 1, 2)

blocks = img_mine2[block_mask][:, 3:-3, 3:-3, :].copy()
blocks = blocks.reshape(blocks.shape[0], -1)

img_numbers.shape = 8, -1
numbers = np.argmin(distance.cdist(blocks, img_numbers), axis=1)
rows, cols = np.where(block_mask)
```


```python
#%fig=识别扫雷界面中的数字
from scpy2.matplotlib import draw_grid
table = np.full((ROWS, COLS), " ", dtype="unicode")
table[rows, cols] = numbers.astype(str)
draw_grid(table, fontsize=12)
```


![svg](examples-500-picosat_files/examples-500-picosat_31_0.svg)


#### 用SAT扫雷


```python
variables = list(range(1, 9))
from itertools import combinations

clauses = []
for vs in combinations(variables, 4):
    clauses.append([-x for x in vs])

for vs in combinations(variables, 6):
    clauses.append(vs)

sat = CoSAT()
sat.add_clauses(clauses)
sat.solve()
```




    [-1, -1, -1, -1, -1, 1, 1, 1]




```python
from collections import defaultdict

variable_neighbors = defaultdict(list)

directs = [(-1, -1), (-1,  0), (-1,  1), (0, -1),
           (0,  1), (1, -1), (1,  0), (1,  1)]

variables = np.arange(1, COLS * ROWS + 1).reshape(ROWS, COLS)

for (i, j), v in np.ndenumerate(variables):
    for di, dj in directs:
        i2 = i + di
        j2 = j + dj
        if 0 <= i2 < ROWS and 0 <= j2 < COLS:
            variable_neighbors[v].append(variables[i2, j2])
```


```python
variable_neighbors[50]
```




    [19, 20, 21, 49, 51, 79, 80, 81]




```python
def get_clauses(var_id, num):
    clauses = []
    neighbors = variable_neighbors[var_id]
    neg_neighbors = [-x for x in neighbors]
    clauses.extend(combinations(neg_neighbors, num + 1))
    clauses.extend(combinations(neighbors, len(neighbors) - num + 1))
    clauses.append([-var_id])
    return clauses
```


```python
%%include cython cycosat/cycosat.pyx 8
    def get_failed_assumes(self):
        cdef int max_index
        cdef int ret1, ret0
        cdef list assumes = []
        self.build_sat()
        max_index = picosat_variables(self.sat)
        for i in range(1, max_index+1):
            picosat_assume(self.sat, i)
            ret1 = picosat_sat(self.sat, -1)
            picosat_assume(self.sat, -i)
            ret0 = picosat_sat(self.sat, -1)
            if ret0 == PICOSAT_UNSATISFIABLE:
                assumes.append(-i)
            if ret1 == PICOSAT_UNSATISFIABLE:
                assumes.append(i)
        return assumes
```


```python
#%fig=使用SAT求解器推断方格中是否有地雷
sat = CoSAT()
for var_id, num in zip(variables[rows, cols], numbers):
    sat.add_clauses(get_clauses(var_id, num))
failed_assumes = sat.get_failed_assumes()

for v in failed_assumes:
    av = abs(v)
    col = (av - 1) % COLS
    row = (av - 1) // COLS
    if table[row, col] == " ":
        if v > 0:
            table[row, col] = "★"
        else:
            table[row, col] = "●"
draw_grid(table, fontsize=12)
```


![svg](examples-500-picosat_files/examples-500-picosat_38_0.svg)


#### 自动扫雷

> **SOURCE**

> `scpy2.examples.automine`：Windows 7系统下自动扫雷，需将扫雷游戏的难度设置为高级（99个雷），并且关闭“显示动画”、“播放声音”以及“显示提示”等选项。


```python
#%hide
%exec_python -m scpy2.examples.automine
```
