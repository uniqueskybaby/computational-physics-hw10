# GitHub 托管运行验证

GitHub Actions 在 Ubuntu 24.04 上检出 `0d429c2be0eb6743e5290355b4fdc61f1d5437aa`，于北京时间 2026 年 11 月 24 日 09:55:04 至 09:55:59 完成。运行环境为 Python 3.12.3、NumPy 2.5.5、Matplotlib 3.11.2，工作流结论为 `success`。[GitHub Actions 运行与日志](https://github.com/uniqueskybaby/computational-physics-hw10/actions/runs/45308543776)

| 项目 | 托管运行输出 |
|---|---|
| 标准库验证 | 中心点源与小网格直接解的两项测试通过 |
| $N=31$ 中心点源 | 33 轮，末轮残差 $3.88564705\times10^{-7}$，峰值 $0.376855245955$ |
| $N=7$ 直接解对照，阈值 $10^{-6}$ | 21 轮，最大解差 $2.55349334\times10^{-7}$ |
| $N=7$ 直接解对照，阈值 $10^{-8}$ | 28 轮，最大解差 $9.30527613\times10^{-10}$ |
| 完整数值实验 | 576 条扫描记录，逐值复核 6 份 CSV，重新导出 5 组图 |

工作流核对 GitHub 提供的提交哈希，依次运行标准库测试、独立参考解计算、完整实验和输出复核。输出复核按浮点容差比较新汇总与归档数据，并检查每组图的 PNG 与 PDF。逐步结果可展开日志中的 `Run standard-library checks`、`Set up the experiment environment` 和 `Rerun the independent checks and all experiments`。[工作流配置](.github/workflows/reproduce.yml)
