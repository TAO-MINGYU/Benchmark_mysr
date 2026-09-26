# MySR 横向 Symbolic Regression Benchmark 计划

本文是正式结果前冻结的横向比较计划。横向比较的对象、题库、条件和统计单位在 baseline 运行前确定；baseline 结果不能反过来删题或改变预算。

## 1. 横向对象

正式对象固定为六个 baseline，加上最后运行的 MySR：

| method | 类型 | 横向作用 | 正式环境 |
|---|---|---|---|
| PySR | Julia/Python evolutionary SR | MySR 的主要祖先匹配基线 | `env_1_pysr` |
| Operon/PyOperon | C++/Python 高性能 GP | 高吞吐传统 GP | `parallel_operon` |
| DSR | policy-gradient neural SR | 神经符号搜索 | `parallel_dsr` |
| AI-Feynman 2.0 | physics-prior decomposition/search | 物理先验科学公式基线 | `parallel_ai_feynman` |
| gplearn | Python GP | 透明、低门槛 GP 基线 | `parallel_gplearn` |
| TF4SR | Transformer scientific SR | Transformer 代表 | `parallel_tf4sr` |
| MySR | MySR/MySRCore | 最终评估对象 | `env_1_mysr` |

TF4SR 只用于其官方 SRSD 科学公式输入域；AI-Feynman 和 TF4SR 在任意 PMLB 黑箱上不适用时必须写入 `not_applicable`，不能修改题库来迁就它们。

## 2. 每个对象计算哪些题库

`Material A` 和 `Material B` 是正式题库；ODEBench 是动态系统扩展，暂不混入普通 tabular SR 总分。

| 题库层 | 任务内容 | PySR | Operon | DSR | AI-Feynman | gplearn | TF4SR | MySR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A/legacy-classic | Nguyen、Keijzer、Pagie-1、Vladislavleva、Korns，共 51 个 | ✓ | ✓ | ✓ | 仅物理公式适用子集 | ✓ | NA | ✓ |
| A/ground-truth | SRBench/PMLB first-principles 任务，当前材料化 12 个 | ✓ | ✓ | ✓ | ✓ | ✓ | 兼容任务 | ✓ |
| A/black-box | SRBench/PMLB black-box，当前材料化 12 个 | ✓ | ✓ | ✓ | NA | ✓ | NA | ✓ |
| B/SRSD-Feynman | easy/medium/hard，共 120 个 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| B/dummy-variable | 上述 120 个任务加入无关变量后的配对版本 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ODEBench | 63 个动态系统、轨迹、噪声和删点 | 先 NA | 先 NA | 先 NA | 先 NA | 先 NA | 先 NA | 独立 ODE 轨道 |

第一版横向总分只使用前五个 tabular 行。ODEBench 作为 MySR 动态系统能力和鲁棒性轨道；当其他方法拥有经过验证的 ODE adapter 后，才加入横向 ODE 比较。AI-Feynman 的 legacy classic 只在其输入格式和物理假设合法的预注册子集运行，其余任务报告 `not_applicable`。

## 3. 四组正式条件

每个适用的 `method × task` 组合运行四组：

| 数据条件 | constrained-resource | capability-ceiling |
|---|---|---|
| clean | 20,000 evaluations、120 s、8 GiB、1 thread | 500,000 evaluations、900 s、32 GiB、1 thread |
| output noise 5% | 同上 | 同上 |

`output_noise_01` 和 `output_noise_10` 不改变四组主矩阵，而作为 sensitivity 扩展报告。噪声配对使用同一点、同一 split、同一潜在公式；`sigma = 0.05 * std(y_train_clean)`。ground-truth 主指标使用干净 test latent target，同时保留带噪 test 指标；black-box 只报告预测性能，不能声称 exact recovery。

资源数字是预注册的有限预算，不是“无限资源”。在正式运行前用 pilot 校准启动时间、evaluation throughput 和内存；校准只能从 manifest 中的候选网格选择，不能看完正式结果后临时加预算。当前 Slurm 节点的 `RealMemory=1 MiB` 声明异常，在修复前不能把 Slurm 结果作为正式 memory-efficiency 证据。

## 4. split、seed 和重复

- 每个任务固定 `train/validation/test` 三个 split；split、dataset、noise、search 使用独立 seed stream。
- 所有方法共享同一组 10 个正式 search seed：`23654, 15795, 860, 5390, 16850, 29910, 4426, 21962, 14423, 28020`。
- 5 个 pilot seed 只用于 adapter 和资源校准，不进入正式统计。
- GPSR 没有需要验证集调节的超参数时仍保留 validation split；不使用 validation 调参就明确记录。若方法使用 validation 选择 checkpoint/HOF，则只能使用 validation，不能看 test。
- 相同 task 的 10 个 seed 是重复测量；跨任务统计的主要单位是公式家族/题库层。报告每个 seed、median、mean、分位数、recovery rate、timeout/invalid rate、seed sensitivity 和方法间配对差值。
- deterministic repeat 只在 smoke 和代表任务执行两次，用于检查相同 task/seed 的 frontier checksum 和失败元数据。

## 5. 统一输出和指标

每次运行输出 `selected_expression`、完整 Pareto/HOF、train/validation/test score、complexity、wall time、CPU time、peak memory、evaluations、seed、hostname、Slurm job ID 和 `failure_status`。

ground-truth 任务报告 exact recovery、numeric equivalence、structural-near recovery、变量选择、test NRMSE 和复杂度；black-box 报告 test NRMSE、复杂度和资源成本；噪声轨道额外报告相对 clean 的性能下降和 seed 方差。timeout、crash、invalid output 和 not applicable 都保留在分母和结果表中。

## 6. 执行顺序

1. 锁定 corpus manifest、数据 hash、operators、split、四组预算和 10 个 seed。
2. 先运行 PySR、Operon、DSR、AI-Feynman、gplearn、TF4SR 的 adapters 和 pilot。
3. 完成六个 baseline 的四组矩阵；测试结果不能修改题库。
4. 在隔离 worktree 修复 MySR/MySRCore 的 bug、serial determinism、scorer、资源记录和 API ablation。
5. 冻结 MySR/MySRCore commit、环境 lock 和 adapter commit。
6. 最后运行 MySR；MySR 使用完全相同的 task membership、seed、资源、评分器和失败规则。
7. 发布逐 seed 原始 metrics、frontier、资源日志、失败记录、汇总表和绘图脚本。

## 7. Slurm 任务命名

Slurm job 名必须包含 `baseline`、method、material/stratum、noise、resource-track 和 seed。每个 job 输出一条机器可读 JSON；数组索引与 `run-manifest.json` 一一对应。`squeue` 只显示尚未结束的作业；结束后的正式证据使用 `sacct` 和结果 JSON 查询。

本计划只把 baseline calibration 作业作为当前部署对象，不能把 calibration 结果写成最终方法排名。

## 8. 完整横向任务登记

正式外部方法矩阵已经登记为 `20260926-external-horizontal-matrix-v1`：315 个 tabular 任务 × 6 个外部方法 × 4 个 noise variants × 2 个 resource tracks × 10 个 search seeds，共 151,200 个 run records，按方法拆成六个 315-task Slurm arrays。MySR 不在这些 arrays 中。数组 33895–33900 当前保持 `JobHeldUser`，因为统一 solver adapters 尚未全部完成；释放前必须通过表达式导出、评分、资源记录和失败状态 smoke。ODEBench 仍是独立动态系统扩展，不进入这批 tabular arrays。
