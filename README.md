# 第 10 周：二维格点点源模型中的 SOR 方法

源项为单位中心点源，内部为 `N × N`，外圈零 Dirichlet 边界，格距 `h=1`。模型为经典实标量方程 `(-Δ + m²)u = s`。报告覆盖四个必做任务、小网格直接解、残差曲线与理论值，以及三种迭代法、区域大小和偏心源的比较。

## 根目录运行入口

使用 Python 3.9 或更高版本，在仓库根目录验证任务 1 和小网格直接解。

```sh
python3 -S -m unittest -v
python3 -S main.py --smoke
```

这两条命令仅使用 Python 标准库。第一条检查任务 1 的边界和停止判据，并独立组装 $N=7$ 的矩阵，用 Cholesky 直接解检验两种残差阈值。第二条输出任务 1 的结果，预期用 **33 轮**达到阈值，$\|r\|_\infty=3.88565\times10^{-7}$，峰值 $u_{16,16}=0.37685525$。

安装依赖后，可运行独立的小网格验证和完整实验。

```sh
python3 -m pip install -r requirements.txt
python3 main.py --validate
python3 main.py --output reproduced
python3 -S verify_results.py reproduced
```

完整实验将全部表格数据和五组图写入 `reproduced/`。校验程序逐项比较新输出和仓库中的数据，并检查五组图是否成功导出。直接运行 `python3 main.py` 则写入默认的 `results/`。

GitHub 托管运行的测试、提交版本和关键输出见 [复现记录](verification.md)。

## 结果速查

| $m$ | $e^{-1}$ 插值距离 / 格 | 实测代表 $\omega$ | SOR 轮数 | GS 轮数 | 理论 $\omega$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 1.865 | 1.681 | 63 | 316 | 1.741341 |
| 0.5 | 1.145 | 1.405 | 30 | 68 | 1.491361 |
| 1.0 | 0.861 | 1.158 | 17 | 23 | 1.248999 |

每个 $\omega$ 都使用零初场、字典序逐点扫描及相同的绝对无穷范数残差。收敛阈值为 $10^{-6}$，上限 5000 轮。细扫覆盖 0.005 和 0.001 两种间隔，代表值为达到最少轮数的最小采样值。$N=7$、$m=0.5$ 的小网格以直接线性求解交叉核对，最大解差 $2.55\times10^{-7}$。

## 按题目查看证据

| 题面要求 | 运行数据与图表 |
|---|---|
| 任务 1，$N=31$ 中心点源 | [热图与截面](results/figures/task1_field.pdf) · [场数据](results/data/task1_field_n31.csv) |
| 任务 2，同一网格的三个 $m$ | [归一化截面](results/figures/mass_response.pdf) · [左右截面数据](results/data/normalized_profiles.csv) |
| 任务 3 和 4，粗扫、细扫与 GS 对照 | [扫描图](results/figures/omega_scan.pdf) · [扫描数据](results/data/omega_scan.csv) · [轮数比和并列参数](results/summary.json) |
| 小网格直接验证 | [直接解误差、残差和对称差](validation.json) · [系统组装与 Cholesky](direct.py) · [标准库验证](test_sor.py) · [NumPy 及谱核验](reference.py) |
| 残差曲线与谱理论 | [残差图](results/figures/convergence.pdf) · [逐轮残差](results/data/residual_history.csv) · [理论参数与紧阈值复算](results/summary.json) |
| 区域大小与非中心源 | [边界和源位对照图](results/figures/boundaries.pdf) · [边界截面数据](results/data/boundary_profiles.csv) |

`sor.py` 实现五点更新和全格点残差，`experiments.py` 组织各组对照，`plot_results.py` 从保存的数据生成图。模型推导、停止准则和结果解释见 [计算报告](report.md) 与 [排版 PDF](report.pdf)，实验条件见 [实验说明](assignment.md)。
