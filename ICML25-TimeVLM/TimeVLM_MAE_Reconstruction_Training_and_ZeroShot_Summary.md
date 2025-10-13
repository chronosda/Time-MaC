# 重建导向 MAE 改进的 Time‑VLM 实验与 Zero‑Shot 验证汇总

简要说明：本文件汇总仓库内“重建导向 MAE 改进”的 Time‑VLM 训练记录与 Zero‑Shot 验证结果，来源包括 logs/、results/、test_results/、result_long_term_forecast.txt 以及 Zero_Shot_* 报告与日志。表格数值统一保留 6 位小数，DTW 未计算处标注为 N/A。

## 训练结果汇总

说明：同一数据集/预测步长若存在多次记录（如 test/final/batch），均予以保留以便比对。

| Dataset | Pred | MSE | MAE | MSE（原） | MAE（原） |
|--------|------|-----|-----|-----|-----|
| Electricity | 96 | <span style="color:red">0.134380</span> | <span style="color:red">0.233294</span> | 0.142 | 0.245 |
| Electricity | 192 | <span style="color:red">0.150293</span> | <span style="color:red">0.247498</span> | 0.157 | 0.260 |
| Electricity | 336 | <span style="color:red">0.167222</span> | <span style="color:red">0.264839</span> | 0.174 | 0.276 |
| Electricity | 720 | <span style="color:red">0.205615</span> | <span style="color:red">0.297246</span> | 0.214 | 0.308 |
| Weather | 96 | 0.150098 | 0.200233 | <span style="color:red">0.148</span> | <span style="color:red">0.200</span> |
| Weather | 192 | 0.196062 | 0.246669 | <span style="color:red">0.193</span> | <span style="color:red">0.240</span> |
| Weather | 336 | 0.246506 | 0.285895 | <span style="color:red">0.243</span> | <span style="color:red">0.281</span> |
| Weather | 720 | 0.317480 | 0.333865 | <span style="color:red">0.312</span> | <span style="color:red">0.332</span> |
| Traffic | 96 | <span style="color:red">0.379099</span> | <span style="color:red">0.269109</span> | 0.393 | 0.290 |
| Traffic | 192 | <span style="color:red">0.393529</span> | <span style="color:red">0.275865</span> | 0.405 | 0.296 |
| Traffic | 336 | <span style="color:red">0.407287</span> | <span style="color:red">0.283377</span> | 0.420 | 0.305 |
| Traffic | 720 | <span style="color:red">0.438975</span> | <span style="color:red">0.298848</span> | 0.459 | 0.323 |
| ETTh1 | 96 | 0.362305 | 0.392838 | <span style="color:red">0.361</span> | <span style="color:red">0.386</span> |
| ETTh1 | 192 | 0.398390 | 0.418633 | <span style="color:red">0.397</span> | <span style="color:red">0.415</span> |
| ETTh1 | 336 | 0.430608 | 0.427645 | <span style="color:red">0.420</span> | <span style="color:red">0.421</span> |
| ETTh1 | 720 | 0.442994 | <span style="color:red">0.457515</span> | <span style="color:red">0.441</span> | 0.458 |
| ETTh2 | 96 | <span style="color:red">0.176835</span> | <span style="color:red">0.289794</span> | 0.267 | 0.335 |
| ETTh2 | 192 | <span style="color:red">0.232109</span> | 0.336755 | 0.326 | <span style="color:red">0.329</span> |
| ETTh2 | 336 | <span style="color:red">0.264612</span> | <span style="color:red">0.362213</span> | 0.357 | 0.406 |
| ETTh2 | 720 | <span style="color:red">0.363943</span> | <span style="color:red">0.425432</span> | 0.412 | 0.449 |
| ETTm1 | 96 | <span style="color:red">0.118709</span> | <span style="color:red">0.234751</span> | 0.304 | 0.346 |
| ETTm1 | 192 | <span style="color:red">0.149049</span> | <span style="color:red">0.259868</span> | 0.332 | 0.366 |
| ETTm2 | 96 | <span style="color:red">0.117552</span> | <span style="color:red">0.230177</span> | 0.160 | 0.250 |
| ETTm2 | 192 | <span style="color:red">0.147911</span> | <span style="color:red">0.263921</span> | 0.215 | 0.291 |
| ETTm2 | 336 | <span style="color:red">0.176704</span> | <span style="color:red">0.287318</span> | 0.270 | 0.325 |
| ETTm2 | 720 | <span style="color:red">0.213866</span> | <span style="color:red">0.317197</span> | 0.348 | 0.378 |

> 备注：训练结果来自 `result_long_term_forecast.txt` 与配套日志；DTW 在训练记录中均未计算（N/A）。

## Zero‑Shot 验证汇总

说明：为确保严格零样本范式（仅在源域训练、目标域仅评测），已清空既有 Zero‑Shot 指标与日志引用，等待按严格流程重跑后再补录。

重跑方法（严格 Zero‑Shot）：
- ETTh1→ETTh2：执行 `scripts/zero_shot_ETTh1_to_ETTh2_strict.sh`
- ETTm1→ETTm2：执行 `scripts/zero_shot_ETTm1_to_ETTm2_strict.sh`

落盘位置（跑完自动生成）：
- 指标/预测：`results/<setting>/metrics.npy|pred.npy|true.npy`
- 训练检查点：`checkpoints/<setting>/checkpoint.pth`
- 日志：`logs/strict_zero_shot_*/<exp>.log` 与 `logs/strict_zero_shot_*/results.txt`

## 数据来源与可追溯性

- 训练结果与大部分指标：`result_long_term_forecast.txt`
- Zero‑Shot 过程与形状信息：清空并等待严格重跑（见上节的脚本与新日志目录）
- Zero‑Shot 概要与分析：清空并等待严格重跑
- 预测/真值矩阵：`results/*/pred.npy`、`results/*/true.npy`（部分实验）

## 说明与约束

- 绝大多数训练记录未计算 DTW（日志中明确为 not calculated）。
- Zero‑Shot 的 ETTm1→ETTm2 在 336/720 步已有运行目录，但未落盘数值；表格据此标注为 N/A。
- 若需增补 DTW 或导出 CSV，可复用现有 `pred.npy/true.npy` 进行离线评估。

—

生成时间：2025‑10‑11
