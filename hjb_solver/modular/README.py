"""
HJB方程求解器 - 通用框架使用说明
================================

文件结构:
---------
1. hjb_solver_base.py    - 基础框架和抽象类
2. hjb_solver_crra.py    - CRRA效用实现示例
3. hjb_solver_log.py     - 对数效用实现示例
4. README.py             - 本说明文件


如何修改公式/添加新功能:
------------------------

### 1. 修改效用函数

在 hjb_solver_base.py 中的 UtilityFunction 类添加新的效用函数:

```python
class UtilityFunction:
    @staticmethod
    def my_utility(c: np.ndarray, param: float) -> np.ndarray:
        '''你的自定义效用函数'''
        return ...
    
    @staticmethod
    def my_utility_marginal(c: np.ndarray, param: float) -> np.ndarray:
        '''对应的边际效用'''
        return ...
```


### 2. 创建新的求解器

继承 HJBSolverBase，重写关键方法:

```python
from hjb_solver_base import HJBSolverBase

class HJBSolverMyModel(HJBSolverBase):
    
    def compute_terminal_value(self, r: float, w: float) -> float:
        '''定义终端时刻的值函数'''
        return ...
    
    def compute_optimal_consumption(self, V_w: float, **kwargs) -> float:
        '''定义最优消费规则'''
        # 根据你的效用函数修改
        return ...
    
    def compute_optimal_portfolio(self, **kwargs) -> float:
        '''定义最优投资组合规则'''
        return ...
    
    def solve_time_step(self, V_next: np.ndarray, t_idx: int) -> np.ndarray:
        '''定义时间步进求解方法'''
        # 可以参考CRRA或Log版本的实现
        return ...
```


### 3. 添加新的状态变量

如果需要添加更多状态变量（如收入y、年龄等）:

```python
class HJBSolverExtended(HJBSolverBase):
    
    def _setup_grids(self):
        '''重写网格设置，添加新维度'''
        super()._setup_grids()  # 调用父类设置t, r, w
        
        # 添加收入维度
        self.y_min = self.params.get('y_min', 0.1)
        self.y_max = self.params.get('y_max', 5.0)
        self.Ny = self.params.get('Ny', 20)
        self.y = np.linspace(self.y_min, self.y_max, self.Ny)
        
        # 重新初始化存储数组
        self.V = np.zeros((self.Nt, self.Nr, self.Nw, self.Ny))
        # ... 其他控制变量
```


### 4. 修改约束条件

例如添加借贷约束:

```python
def compute_optimal_consumption(self, V_w: float, w: float = None, **kwargs) -> float:
    c = ...  # 计算理论最优消费
    
    # 添加借贷约束: 消费不能超过财富
    if w is not None:
        c = min(c, w)
    
    # 添加最低消费约束
    c = max(c, self.params.get('c_min', 0.01))
    
    return c
```


### 5. 修改数值方法

可以替换策略迭代为其他方法:

```python
def solve_time_step_newton(self, V_next: np.ndarray, t_idx: int) -> np.ndarray:
    '''使用Newton-Raphson方法'''
    # 实现Newton-Raphson迭代
    pass

def solve_time_step_explicit(self, V_next: np.ndarray, t_idx: int) -> np.ndarray:
    '''使用显式格式'''
    # 实现显式欧拉方法
    pass
```


关键修改点总结:
--------------

1. **效用函数**: 修改 `compute_optimal_consumption` 中的一阶条件
2. **终端条件**: 修改 `compute_terminal_value`
3. **投资组合**: 修改 `compute_optimal_portfolio` 中的优化问题
4. **状态变量**: 重写 `_setup_grids` 和 `_build_differential_operators`
5. **约束条件**: 在控制变量计算中添加约束逻辑
6. **数值方法**: 重写 `solve_time_step`


运行示例:
--------

```python
# CRRA效用
from hjb_solver_crra import HJBSolverCRRA
params = {...}
solver = HJBSolverCRRA(params)
V, c, pi = solver.solve_backward()

# 对数效用
from hjb_solver_log import HJBSolverLog
solver = HJBSolverLog(params)
V, c, pi = solver.solve_backward()
```
"""

if __name__ == "__main__":
    print(__doc__)
