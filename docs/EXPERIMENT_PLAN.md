# Time‑ME 实验规划与论文大纲

本文件给出面向“可发 SCI 三区”的完整实验规划与论文大纲，目标是以最短路径补齐证据链，形成强 SOTA 对比、系统消融、鲁棒性与不确定性分析、效率评估与复现实验材料。

## 总体目标

- 在标准多数据集×多步长上，稳定领先主流基线（≥4 个数据集、≥2 个步长显著优势）。
- 完成系统消融、效率、置信区间覆盖率、鲁棒性与可解释性分析。
- 提供一键复现脚本与聚合报告，输出可直接入论文的表格与图。

## 数据与任务设置

- 数据集（建议优先级）：Electricity、ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather、Traffic、Illness（ILI）
- 预测步长：96/192/336/720（ILI 可用 24/36/48 补充）
- 种子：0/1/2（报告均值±方差）
- 评估指标：MSE、MAE、SMAPE、R2、Directional Accuracy（DA）
- 效率指标：训练/验证吞吐（samples/s）、峰值显存（MB）、总时长
- 不确定性：miscoverage@α、平均区间宽度、Coverage–Width 曲线（α∈{0.05,0.10,0.15}）

## 实验矩阵（核心主表）

- 数据集 × 步长 × 3 种子，共计约 8×4×3 套实验。
- 统一训练预算（epochs/早停）、统一数据划分，避免过拟合调参。

### 统一入口与示例命令

- 主入口（多变量数据）：`scripts/train_time_me.py`
- 推荐模板（离线，使用 MAE 特征或零特征回退）：

  ```bash
  python scripts/train_time_me.py \
    --root_path ./dataset \
    --data electricity \
    --seq_len 512 --pred_len 96 \
    --d_model 256 --batch_size 32 --epochs 20 --learning_rate 1e-4 \
    --use_enhanced_fusion --vlm_type mae --offline \
    --save_path checkpoints/electricity_p96
  ```

- 步长切换：`--pred_len {96,192,336,720}`
- 建议固定随机种子（如需，可在训练脚本中补充 `seed` 参数与 cudnn 确定性设置）。

## 基线对比（必须）

- 覆盖主流强基线（同一数据/步长/预算）：PatchTST、iTransformer、TimesNet、DLinear、TiDE、TimeMixer、FEDformer/Informer、Temporal Fusion Transformer、TimeLLM/Time‑VLM。
- 实施建议：
  - 优先复用公开实现或本仓已有实现；若短期不便，先引用权威表并在 2–3 个数据集上自复现关键基线增强可信度。
  - 统一训练预算（epochs/早停）与数据划分，确保公平性。

## 消融实验（必须）

目的：证明各模块必要性与协同增益。

- 去 Coupled‑Mamba（仅记忆/MLP/注意力路径）：不加 `--use_enhanced_fusion`，即 Time‑VLM 路径（`models/time_me.py`）。
- 仅多模态/仅记忆分支：建议新增轻量配置开关（可在模型内添加开关以纯化路径）。
- 记忆库移除/退化：禁用 `PatchMemoryBank` 或设 `top_k=1`。
- 门控去除：将 `fusion_gate_logit` 固定到 0 或 +∞，实现“仅多模态/仅记忆”。
- MAE 变体：`--use_optimized_mae`、`--use_reconstruction_mae`、`--use_dual_path_reconstruction`。
- Mamba 深度：`--mamba_layers {1,2,4}`，分析深度–效能权衡。
- Patch 策略：`--patch_len {8,16,32}`、`--stride {4,8,16}`。
- 离线/在线：`--offline` 开/关，验证可用性与性能变化。

输出：每个消融点相对主模型的 ΔMSE/ΔMAE、统计检验（成对 t/Wilcoxon），条形图可视化。

### Electricity‑p96 上的 Coupled‑Mamba × MAE 变体消融（seq_len=512, d_model=256）

已完成的核心消融实验（数据集 Electricity，预测步长 96，`vlm_type=mae`，`offline=True`，`seq_len=512`，`d_model=256`）如下表。

```markdown
| 组合 ID | 是否 Coupled‑Mamba | MAE 变体            | MSE       | MAE       | R2        | Directional Acc. |
|--------|--------------------|---------------------|----------:|----------:|----------:|------------------:|
| S‑noCM | 否                 | Standard MAE        | 0.12749   | 0.23433   | 0.85074   | 0.74786           |
| S‑CM   | 是                 | Standard MAE        | 0.13686   | 0.24526   | 0.83978   | 0.74224           |
| O‑noCM | 否                 | Optimized MAE       | 0.12888   | 0.23848   | 0.84911   | 0.74606           |
| O‑CM   | 是                 | Optimized MAE       | 0.13882   | 0.24837   | 0.83748   | 0.74095           |
| R‑noCM | 否                 | Reconstruction MAE  | 0.12876   | 0.23668   | 0.84926   | 0.74715           |
| R‑CM   | 是                 | Reconstruction MAE  | 0.13740   | 0.24733   | 0.83915   | 0.74210           |
```

从结果可以看到，在 Electricity‑p96 这种“纯时间序列 + 伪多模态（由同一时序生成的图像）”设定下：

- 不同 MAE 变体本身（standard / optimized / reconstruction）在不引入 Coupled‑Mamba 时已经能达到较强的性能（MSE≈0.128，R2≈0.849）。  
- 在相同的训练预算下，加入 Coupled‑Mamba 后并未带来进一步收益，甚至略有下降（MSE 上升约 0.009–0.011，R2 下降约 0.010–0.012）。

这说明：在“多模态分支与主时序分支信息高度冗余”的场景中，Coupled‑Mamba 更多是在对同源信息做复杂重加权，而不是利用真正互补的模态信号；因此它的优势难以体现。这个观察本身也可以作为论文中对结构适用范围的一个负结果分析。

为凸显 Coupled‑Mamba 的价值，后续专门设计了两类实验：

1. **合成多模态实验**：构造具有互补信息的“时间序列 + 视觉 + 文本”三模态数据，控制变量地对比：无多模态 / 简单融合 / Coupled‑Mamba 三种结构在同一任务上的表现，验证在有真实模态互补时 Coupled‑Mamba 的优势。  
2. **真实多模态案例（可选）**：在带有外部图像或文本描述的实际时序任务上（如电网/交通/气象等），将 Coupled‑Mamba 与简单融合结构进行对比，进一步验证其在真实多模态场景中的作用。


## 不确定性与保序校准（必须）

- 配置：`--conformal_enable` + `--conformal_method {crc,hpd,rcps}` + `--conformal_alpha {0.05,0.10,0.15}`。
- 指标：覆盖率（1–失覆盖）、平均区间宽度；随 α 的曲线；不同 scale 代理（MAD/STD/global MAD）。
- 方法定位：当前实现是一个 **post-hoc conformal calibration 插件**，在 calibration split 上基于残差 `r=|y-\hat y|` 与尺度代理 `s` 拟合单个全局阈值 `\lambda`，再在测试集上构造区间 `[ \hat y-\lambda s,\ \hat y+\lambda s ]`。
- 三种可选校准规则共享同一 `\lambda` 主线，而不是统一的 Bayesian Quadrature 框架：
  - `crc`：基于经验失覆盖率的 conformal risk correction。
  - `hpd`：基于 Dirichlet 随机权重的高概率风险上界。
  - `rcps`：基于 Hoeffding 上界的风险控制。
- 产物：
  - 校准：`scripts/train_time_me.py` 已集成 `ConformalCalibrator`（`utils/conformal_plugin.py`）。
  - 区间：`results/conformal_metrics.txt`、`predictions/*.npy`。

## 鲁棒性与泛化（加分）

- 噪声鲁棒：对输入加高斯噪声（不同 σ），记录性能降幅。
- 缺失观测：随机 mask 输入片段；或“缺模态”模拟（仅时序、无视觉/文本）。
- 分布移位：按时间片划分训练/测试（滚动或跨年），验证时移泛化。
- 子样本训练：1/4、1/2 数据量训练，观察数据效率。

## 效率与资源（必须）

- 统一记录：训练耗时、吞吐、显存峰值（`scripts/train_time_me.py` 已记录）。
- 复杂度对比：参数量、推理时延；与 Transformer/Mamba 基线对照。
- 可视化：吞吐–显存散点、步长维度的耗时曲线。

## 可解释性（建议）

- 门控/注意特征可视化：不同时间步与变量的权重热力图。
- 案例分析：典型序列预测曲线、残差与区间可视化。

## 复现实验与聚合

- 目录规范：
  - 数据：`./dataset/{dataset}.csv`
  - 日志：`./logs/*.log`（含 `Validation Metrics` 与 `Perf` 行）
  - 检查点：`--save_path checkpoints/{tag}`
  - 结果：`./results/conformal_metrics.txt`，`./predictions/*.npy`

- 批量运行（示例）：

  ```bash
  # 伪代码：遍历数据集×步长×种子
  for data in electricity ettm1 etth1 traffic; do
    for pl in 96 192 336 720; do
      for seed in 0 1 2; do
        python scripts/train_time_me.py \
          --root_path ./dataset --data $data \
          --seq_len 512 --pred_len $pl \
          --d_model 256 --batch_size 32 --epochs 20 --learning_rate 1e-4 \
          --use_enhanced_fusion --vlm_type mae --offline \
          --save_path checkpoints/${data}_p${pl}_s${seed}
      done
    done
  done
  ```

- 日志聚合：建议新增 `log_parser.py`，从 `logs/*.log` 提取 `Validation Metrics` 与 `Perf`，汇总为 CSV，生成主表/图（如需我可补实现）。

## 论文大纲

- 标题：
  - Time‑ME: Multimodal Time Series Forecasting via Coupled‑Mamba Fusion with Memory and Conformal Calibration
- 摘要：
  - 问题/方法/贡献/关键结果（SOTA 对比、置信区间覆盖、效率）。
- 1 引言：
  - 多模态时序挑战、现有方法不足（Time‑LLM/Time‑VLM 与纯时序模型）、本文贡献三点。
- 2 相关工作：
  - 时序预测（Transformer/Mamba/混合）、多模态 VLM 融合、不确定性估计（Conformal）。
- 3 方法：
  - 3.1 架构概览（模型图）。
  - 3.2 Coupled‑Mamba 融合（模态并行分支、跨模态增强、加权融合）— 参考 `src/coupled_mamba_fusion.py`。
  - 3.3 记忆模块与门控（局部/全局记忆、门控组合）— 参考 `models/time_me.py`。
  - 3.4 视觉/文本通路与离线模式（MAE 优化、回退策略）— 参考 `src/TimeVLM/vlm_manager.py`。
  - 3.5 训练与推理（损失、复杂度、实现细节）。
  - 3.6 后处理不确定性校准：在 calibration split 上拟合全局阈值 `\lambda` 的 conformal 插件（`crc/hpd/rcps`），其中 `hpd` 仅对应 Dirichlet 随机权重下的高概率风险上界，不应表述为通用 Bayesian Quadrature。
- 4 实验设置：
  - 数据集与预处理、划分、指标、超参、硬件、复现细节。
- 5 主要结果（SOTA 对比主表）：
  - 各数据集×步长表；显著性检验；案例曲线。
- 6 消融研究：
  - 各模块去除/替换，图表与讨论。
- 7 不确定性与区间质量：
  - 覆盖率/区间宽度曲线，方法比较（crc/hpd/rcps），尺度代理（mad/std/global）。
  - 明确其为基于 calibration split 的 post-hoc conformal 校准，而非通用 Bayesian Quadrature 视角。
- 8 鲁棒性与效率：
  - 噪声/缺失/分布移位/子样本；吞吐/显存/参数/延迟。
- 9 可解释性与案例：
  - 门控/注意热力图；成功/失败案例分析。
- 10 局限与未来工作：
  - 模态依赖、计算负担、领域泛化。
- 11 结论。
- 附录：
  - 更多表格、参数、曲线、复现指引。

## 时间与里程碑（建议）

- 第 1 周：数据清单/脚本联调；完成 Electricity/ETTm1/ETTh1 × {96,336} × 3 种子与主表初稿。
- 第 2 周：补全其余数据与步长；完成消融、不确定性、效率；聚合表与主图草图。
- 第 3 周：鲁棒性与可解释性；论文写作与润色；对比与附录收尾。

## 验收标准（自查）

- 主表：≥4 数据集×≥2 步长领先（平均/统计显著）。
- 消融：≥5 个关键组件具正增益或清晰作用解释。
- 复现：一键脚本 + CSV 聚合 + 日志/权重可提供。
- 不确定性：覆盖率达标，报告区间宽度与权衡。

## 可选的工程增强（我可协助落地）

- 新增 `run_matrix.sh` 与 `log_parser.py`，一键批跑与日志聚合制表。
- 在模型中添加 `memory_only` / `multimodal_only` 开关，做更干净的门控消融。
- 固定随机种子与 cudnn 确定性，确保结果严格可复现。

---

如需我先落地批量脚本与日志聚合工具，请告知优先跑的“数据集×步长×种子”集合与 GPU 预算，我会据此补齐对应文件与说明。
