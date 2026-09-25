# MySR Formal Benchmark 使用计划

## 1. 目的和题库边界

正式比较回答四个问题：基本公式恢复能力、现实或科学黑箱上的预测与系统识别能力、对噪声/无关变量/删点和随机种子的鲁棒性与敏感性，以及有限资源下的工程效率。

正式题库只有两个表格材料包和一个动态系统扩展：

- **Material A**：SRBench/PMLB 的 12 个 black-box 任务、12 个 first-principles 来源任务，以及 Nguyen、Keijzer、Pagie-1、Vladislavleva、Korns 五类共 51 个经典任务。
- **Material B**：SRSD-Feynman easy/medium/hard 共 120 个原始任务，以及对应的 120 个 dummy-variable 任务。
- **ODEBench 扩展**：63 个系统的无噪声、1% 噪声、5% 噪声、10% 扩展噪声和 5% 噪声加 50% 删点轨迹。

`Benchmark_PySR_Shortcoming` 和旧的 `original-development-v0.1` 只属于历史资料，不进入正式结果。

## 2. 四个报告部分

### A. 基本能力

主结果使用经典 51 个任务和 SRSD-Feynman 120 个任务的 clean 版本，评价 exact recovery、结构恢复、数值等价、变量选择、测试误差和复杂度。PMLB first-principles 任务目前作为预测子轨道；只有在完整公式元数据加入任务清单后才启用 exact recovery。

### B. 现实和科学黑箱

PMLB black-box 作为现实表格黑箱，主要评价 test NRMSE、复杂度和运行成本。ODEBench clean/noiseless 作为科学动态黑箱，评价动力学方程恢复和轨迹预测。两者分开汇报，不合并成一个不加说明的总分。

### C. Robustness 和 Sensitivity

对同一个潜在任务配对运行 clean、1%、5%、10% 输出噪声。SRSD dummy-variable 检查无关输入，ODEBench 的噪声和删点检查动态系统稳定性。地面真值任务的主指标使用干净 test target；black-box 任务只报告预测鲁棒性，不能声称 exact recovery。

每个正式任务使用 10 个预注册 search seeds。报告中位数、均值、分位数、recovery rate、timeout/invalid rate、seed sensitivity 和相对于 clean 的性能下降。不得选择最佳 seed 代表普通结果。

### D. 资源效率

资源不是另一套题库。每个适用任务都在相同硬件和线程策略下运行两条预先冻结的有限预算轨道：

- `constrained_resource`：初始候选 20,000 evaluations、120 秒、8 GiB、单线程；
- `capability_ceiling`：初始候选 500,000 evaluations、900 秒、32 GiB、单线程。

正式值先由 5 个 pilot seeds 和代表任务校准，并从声明的候选网格中冻结；不得根据正式结果临时加预算。启动、Julia 预编译和算法搜索时间分别记录。

## 3. 横向方法

正式基线顺序为：

1. PySR；
2. Operon；
3. DSR；
4. AI-Feynman 2.0；
5. gplearn；
6. TF4SR（Transformer 科学符号回归基线）。

MySR/MySRCore 在所有基线完成并通过代码质量窗口后最后运行。TF4SR 使用官方预训练权重和 SRSD 兼容任务；它不被强行应用到不支持的任意 black-box 任务，结果记录为 `not_applicable`。

每个方法同时提供：

- `common-space`：统一数据、算子、合法域、评分、预处理和预算；
- `native-default`：保留方法默认设置，并记录所有差异。

每次运行输出统一字段：表达式、完整 frontier、train/validation/test 分数、复杂度、wall time、CPU time、peak memory、evaluations、seed 和 failure status。

## 4. 执行顺序

1. 锁定来源 revision、题目清单、许可证、数据 hash 和噪声生成规则；
2. 安装并验证 PySR、Operon、DSR、AI-Feynman、gplearn、TF4SR 环境；
3. 在代表任务上运行 adapter smoke 和 5 个 pilot seeds；
4. 冻结任务、算子、评分、10 个正式 seeds、两档资源和 adapter commits；
5. 先完成六个横向基线的 clean/noisy 与两档资源运行；
6. 在隔离 worktree 修复 MySR/MySRCore，并运行 serial determinism、bridge、scorer 和小任务检查；
7. 锁定 MySR/MySRCore commit、环境 lock 和 execution commit；
8. 对 MySR 运行完全相同的正式矩阵；
9. 发布逐 seed metrics、完整 frontier、失败记录、资源日志、汇总统计和绘图脚本。

## 5. 结果位置和复现

题库和生成记录位于：

```text
/home/taomingyu/taomingyu_5/MySR_Benchmark/20260925-formal-corpus-v1
```

横向代码和环境位于：

```text
/home/taomingyu/taomingyu_5/parallel_comparsion
```

横向结果位于：

```text
/home/taomingyu/taomingyu_5/parallel_comparsion_result/<run-id>
```

GitHub 保存协议、manifest、源码 revision、环境 lock、adapter 和汇总结果。2.8 GB 的完整第三方数据与原始运行产物使用外部版本化归档；仓库中的 hash 和下载/校验脚本保证可追溯。

## 6. Slurm 规则

所有作业显式使用固定环境前缀、单线程变量、任务 seed、资源轨道和结果目录。node2 可立即作为 smoke/pilot 节点；node1 当前有长期占用，作业提交到 node1 后等待调度，不假设登录节点或 SSH 可用。正式作业只通过 `sbatch`/`srun` 在分配节点执行，并保留 `.out`、`.err`、环境信息和失败状态。
