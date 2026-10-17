# 架构

项目采用单进程模块化设计。CLI 与 Streamlit 工作台调用同一个 `application.Run`，界面没有第二套搜索算法。

| 模块 | 职责 |
|---|---|
| `geometry` / `core.geometry` | 定位区域、精确谓词、保守裁剪与包围圆 |
| `localization` | 定位分析与主动测点规划 API |
| `core` | 不可变动作/观测类型、状态更新、序列化协议 |
| `algorithms.shared` / `route` | 共享测点与覆盖路线搜索 |
| `algorithms.directional` / `lookahead` | 混合源证据、候选和服务规划 |
| `algorithms.resilient` / `visibility` / `revisit` / `tour` | 可见性、再次测向、恢复与服务巡游 |
| `simulator` | 独立确定性环境及评分真值 |
| `application` / `recovery` | 串行执行、生命周期与日志重建 |
| `storage` / `evaluation` | 原子持久化、互斥、独立计时与只读回放 |
| `benchmark` / `cli` / `ui` | 评测与用户入口 |

包内覆盖数据通过相对模块路径定位。工作空间整体可移动；运行不导入外部源码，不读取外部资料目录。依赖安装与实际仿真分离，仿真无需外网。

输入与配置有哈希，事件顺序和响应在恢复时逐项核对。当前没有密码学事件链；截断记录保持只读取证，不猜测缺失内容。

