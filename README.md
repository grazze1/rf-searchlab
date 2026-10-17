# RF-SearchLab

无线电测向定位与自主搜索仿真平台。输入带误差的测向数据，分析可能的信号源区域、规划下一次测量，或在合成环境中比较自主搜索策略。

项目可独立安装和运行，代码、覆盖数据、合成示例、测试和文档均在本目录中。当前版本为 `1.0.0`，验证状态见 [发布验证](docs/VALIDATION.md)。

## 功能

- **定位分析**：有界误差楔、半平面求交、空集与无界集分类、区域直径、直径圆覆盖和独立最小包围圆。
- **主动测点规划**：筛选可接收的候选位置，并计算未来观测分支的保守定位半径上界。
- **自主搜索**：全向信号环境的覆盖路线与共享测点策略；混合信号环境的可见性、服务巡游与证据驱动停止策略。
- **任务管理**：暂停、继续、取消、进程中断恢复、幂等动作、独立虚拟时间账本。
- **实验与回放**：固定场景配对比较、逐动作时间轴、自动播放、评分真值图层、CSV/JSON/Markdown 导出。

## 快速开始

需要 Python 3.12 或更新版本。首次安装下载依赖，完成安装后日常运行使用本地资源。

Windows 在配置好 Python 3.12+ 后可双击根目录的 `Start.cmd`。它会在首次启动时创建本地环境并安装依赖，然后启动本地工作台。也可在本目录打开 PowerShell：

```powershell
.\scripts\setup.ps1
.\scripts\start_ui.ps1
```

浏览器打开 [本地工作台](http://127.0.0.1:8531)。脚本不要求管理员权限。也可直接执行：

```powershell
python -m venv .venv
$env:PIP_CACHE_DIR = "$PWD\.work\pip-cache"
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m rf_searchlab.cli doctor
.\.venv\Scripts\python.exe -m rf_searchlab.cli ui
```

## 命令示例

```powershell
.\.venv\Scripts\rf-searchlab.exe locate --input examples/bearings.json --out runs/bearing-analysis
.\.venv\Scripts\rf-searchlab.exe plan --input examples/measurement_plan.json --out runs/measurement-plan
.\.venv\Scripts\rf-searchlab.exe run --environment omni --strategy coverage_route --seed 101 --count 10
.\.venv\Scripts\rf-searchlab.exe run --environment mixed --strategy adaptive_tour --seed 101 --count 10
.\.venv\Scripts\rf-searchlab.exe run --environment mixed --scenario examples/mixed_scene.json
.\.venv\Scripts\rf-searchlab.exe benchmark --suite smoke
```

每个搜索任务自动生成唯一运行 ID。用实际 ID 替换占位符：

```text
rf-searchlab replay --run-id <run-id>
rf-searchlab report --run-id <run-id>
rf-searchlab resume-run --run-id <interrupted-run-id>
```

`locate` 和 `plan` 的输出目录必须位于当前工作空间内且尚不存在。已有输出不会被覆盖。

## 目录

```text
src/rf_searchlab/   完整 Python 包、算法和内置覆盖数据
examples/          可直接运行的合成输入
tests/             几何、协议、策略、恢复和界面测试
scripts/           安装、检查、演示与发布打包
docs/              用户、模型、配置、开发和验证文档
runs/              运行日志和导出结果（运行时生成）
benchmarks/        固定实验规格和逐局结果（运行时生成）
```

安装包可在其他目录使用：先执行 `rf-searchlab init --directory <新目录>`，再进入该目录运行命令；或将 `RF_SEARCHLAB_HOME` 设为该目录。工作空间标记不含本机绝对路径，可整体移动。

## 模型与验证

内置标准模型使用 1800 m 源域、20 个频道、最多 16 个源、有界 1° 测向误差，以及 1000–1500 m 接收范围。源域限制不限制探测器的运动。参数和费用的含义见 [模型说明](docs/MODEL.md)。测点规划为离散候选上的保守数值评价，不能宣称连续全局最优。

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.work/manual-tests
.\scripts\check.ps1
.\.venv\Scripts\python.exe scripts/build_release.py
```

严格检查保留整体 80% 和关键模块 90% 覆盖率门槛。实际命令、结果与限制见 [发布验证](docs/VALIDATION.md)。发布包在 `dist/`，包含源码 ZIP、源码分发包、wheel 和校验和。

阅读 [使用指南](docs/USER_GUIDE.md)、[配置参考](docs/CONFIGURATION.md)、[开发指南](docs/DEVELOPMENT.md) 和 [已知限制](docs/LIMITATIONS.md)。项目采用 [Apache License 2.0](LICENSE) 开源许可证，版权声明见 [NOTICE](NOTICE)，第三方依赖授权见 [依赖清单](docs/DEPENDENCIES.md)。

