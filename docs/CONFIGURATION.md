# 配置参考

## 合成场景 JSON

`sources` 为数组，每项包含 `channel`、`x`、`y`、`radius`、`orientation`。方向以弧度表示；`null` 表示全向源。顶层还包含 `noise_seed`、`environment`（`omni` 或 `mixed`）、`family` 和 `mode`（`standard` 或测试用 `custom`）。重复频道、越界、非法数值和未知字段会被拒绝。

## 策略

| 名称 | 环境 | 用途 |
|---|---|---|
| `coverage_route` | `omni` | 覆盖证书与滚动路线搜索 |
| `shared_search` | `omni` | 共享测点的几何参考策略 |
| `adaptive_tour` | `mixed` | 可见性与服务巡游联合规划 |
| `adaptive_tour_no_pickup` | `mixed` | 关闭顺路服务的比较策略 |

混合策略的测试配置位于包内 `data/adaptive_tour.json`。这些配置控制规划器，不改变标准物理模型。配置中的概率用于排序，不能替代几何证据。

## 工作空间与输出

`workspace.json` 为可移动工作空间标记。`RF_SEARCHLAB_HOME` 可显式选择它所在的目录。输入 JSON 默认上限 4 MB，拒绝重复键与非有限数值。

`runs/<id>/` 保存 manifest、解析后的配置、待执行动作、事件、决策、公开状态、指标、CSV 与报告。`evaluation_private/world.json` 只供仿真、恢复与评分。运行 ID 必须为 32 位小写十六进制，不能借其读取任意路径。

命令行 `--max-actions` 和 `--wall-seconds` 控制资源预算。Web 默认只监听 `127.0.0.1:8531`，可通过 `ui --port` 选择其他本地端口。

