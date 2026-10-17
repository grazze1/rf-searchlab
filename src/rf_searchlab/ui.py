"""Chinese local custom interface; all operations use the shared application service."""

import json

import plotly.graph_objects as go
import streamlit as st

from rf_searchlab.application import TaskManager
from rf_searchlab.evaluation import ledger, replay
from rf_searchlab.localization import figure, locate, plan_measurement
from rf_searchlab.policies import REGISTRY
from rf_searchlab.simulator import generate
from rf_searchlab.storage import read_json, root, run_path

st.set_page_config(page_title="RF-SearchLab", layout="wide")
st.title("无线电源定位与自主搜索仿真平台")
st.caption("RF-SearchLab · 本地无线电定位与自主搜索仿真")


@st.cache_resource
def manager():
    return TaskManager()


home = root()
page = st.sidebar.radio("工作区", ["概览", "定位分析", "搜索任务", "回放与诊断", "基准比较与导出"])
folders = sorted((home / "runs").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
replay_folders = sorted(
    (home / "runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True
)
if page == "概览":
    st.subheader("从测向几何到完整搜索任务")
    st.write("分析测向定位区域，规划下一次测量，运行自主搜索任务，并用完整事件日志回放和比较策略。")
    c1, c2, c3 = st.columns(3)
    c1.metric("已记录运行", len(folders))
    c2.metric("可用搜索策略", len(REGISTRY))
    c3.metric("运行后端", "离线")
    if folders:
        st.dataframe([read_json(p) for p in folders[:50]], width="stretch")
    else:
        st.info("尚无运行结果。前往场景运行创建第一局。")
    st.info("选择左侧功能开始。场景参数仅用于合成环境与事后评分，搜索策略只接收公开观测。")
elif page == "定位分析":
    kind = st.radio("分析方式", ["区域与覆盖", "主动测点规划"], horizontal=True)
    default = (
        {"sensors_m": [[0, 0], [100, 0], [50, -100]], "bearings_deg": [45, 135, 90], "epsilon_deg": 1}
        if kind == "区域与覆盖"
        else {"sensor_m": [0, 0], "bearing_deg": 25, "precision": "fast"}
    )
    raw = st.text_area(
        "测点与示向 JSON（坐标 m，角度 °）", json.dumps(default, ensure_ascii=False), height=130, key=kind
    )
    if st.button("计算定位实验", type="primary"):
        try:
            if len(raw) > 100000:
                raise ValueError("输入过大")
            st.session_state["localization"] = (
                kind,
                (locate if kind == "区域与覆盖" else plan_measurement)(json.loads(raw)),
            )
        except (ValueError, KeyError, TypeError) as e:
            st.error(str(e))
    saved = st.session_state.get("localization")
    if saved and saved[0] == kind:
        result = saved[1]
        st.plotly_chart(figure(result), width="stretch")
        st.json(result)
        st.download_button(
            "导出计算 JSON", json.dumps(result, ensure_ascii=False, indent=2), "localization.json"
        )
elif page == "搜索任务":
    environment = st.selectbox(
        "信号环境",
        ["omni", "mixed"],
        format_func=lambda value: {"omni": "全向信号源", "mixed": "全向与定向混合信号源"}[value],
    )
    strategy = st.selectbox("策略", [k for k, v in REGISTRY.items() if v == environment])
    c1, c2, c3 = st.columns(3)
    seed = c1.number_input("场景种子", value=101, min_value=0, max_value=1000000)
    count = c2.number_input("信号源数量（评分侧）", value=10, min_value=1, max_value=16)
    budget = c3.number_input("动作上限", value=1500, min_value=10, max_value=20000)
    family = st.selectbox("合成分布", ["uniform", "cluster", "boundary", "radius1000", "outward"])
    if st.button("开始离线任务", type="primary"):
        try:
            st.session_state["active_run"] = manager().start(
                generate(seed, environment, count, family), strategy, max_actions=budget, wall_seconds=120
            )
        except (ValueError, RuntimeError) as e:
            st.error(str(e))

    @st.fragment(run_every="1s")
    def progress():
        rid = st.session_state.get("active_run")
        if not rid:
            st.info("创建任务后显示实时状态。")
            return
        run = manager().runs.get(rid)
        if run is None:
            st.warning("进程状态已丢失，请从头重跑；已保存日志仍可回放。")
            return
        st.code(rid)
        st.write("任务状态：", run.manifest["status"])
        st.metric("已完成动作", len(run.events))
        a, b, c = st.columns(3)
        if a.button("暂停", disabled=run.manifest["status"] != "running"):
            run.pause()
        if b.button("继续", disabled=run.manifest["status"] != "paused"):
            run.resume()
        if c.button("取消", disabled=run.manifest["status"] not in ("running", "paused")):
            run.cancel()
        actual = [e["action"]["position"] for e in list(run.events) if e["action"]["position"]]
        if actual:
            live = go.Figure(
                go.Scatter(
                    x=[p["x_m"] for p in actual],
                    y=[p["y_m"] for p in actual],
                    mode="lines",
                    name="已执行动作",
                )
            )
            live.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(live, width="stretch")
        if run.metrics:
            st.json(run.metrics)

    progress()
elif page == "回放与诊断":

    @st.fragment(run_every="600ms")
    def replay_panel():
        if not replay_folders:
            st.info("没有可回放的运行。")
        else:
            rid = st.selectbox("历史 run ID", [p.parent.name for p in replay_folders])
            result = replay(rid)
            st.write("日志状态：", result["status"])
            for error in result["errors"]:
                st.error(error)
            events = result["events"]
            if events:
                timeline_key = f"timeline-{rid}"
                playing_key = f"playing-{rid}"
                if st.session_state.get(playing_key):
                    current = st.session_state.get(timeline_key, len(events) - 1)
                    if current < len(events) - 1:
                        st.session_state[timeline_key] = current + 1
                    else:
                        st.session_state[playing_key] = False
                index = st.slider("动作时间轴", 0, len(events) - 1, len(events) - 1, key=timeline_key)
                a, b, c = st.columns(3)

                def step(delta):
                    st.session_state[playing_key] = False
                    st.session_state[timeline_key] = max(
                        0, min(len(events) - 1, st.session_state[timeline_key] + delta)
                    )

                def toggle_play():
                    if st.session_state.get(playing_key):
                        st.session_state[playing_key] = False
                    else:
                        if st.session_state[timeline_key] >= len(events) - 1:
                            st.session_state[timeline_key] = 0
                        st.session_state[playing_key] = True

                a.button("上一步", on_click=step, args=(-1,))
                b.button("下一步", on_click=step, args=(1,))
                c.button(
                    "暂停播放" if st.session_state.get(playing_key) else "自动播放", on_click=toggle_play
                )
                prefix = events[: index + 1]
                pts = [[0, 0]] + [
                    [e["action"]["position"]["x_m"], e["action"]["position"]["y_m"]]
                    for e in prefix
                    if e["action"]["position"]
                ]
                fig = go.Figure(
                    go.Scatter(
                        x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name="已执行轨迹"
                    )
                )
                truth = st.checkbox("教学真值视图（不参与决策）", value=False)
                if truth:
                    st.warning("真值视图，不参与决策")
                    sources = read_json(run_path(rid) / "evaluation_private/world.json")["sources"]
                    fig.add_trace(
                        go.Scatter(
                            x=[s["x"] for s in sources],
                            y=[s["y"] for s in sources],
                            mode="markers",
                            name="评分侧源位置",
                        )
                    )
                current_event = events[index]
                states_file = run_path(rid) / "public_states.jsonl"
                if states_file.exists():
                    with states_file.open(encoding="utf-8") as f:
                        state = next((json.loads(line) for n, line in enumerate(f) if n == index), None)
                    if state and current_event["action"]["channel"]:
                        vertices = state["channels"][current_event["action"]["channel"] - 1]["vertices"]
                        if vertices:
                            v = vertices + [vertices[0]]
                            fig.add_trace(
                                go.Scatter(
                                    x=[p[0] for p in v],
                                    y=[p[1] for p in v],
                                    fill="toself",
                                    name="该频道观测外包区域",
                                )
                            )
                if current_event["response"]["result"] == "direction":
                    from math import cos, radians, sin

                    p = current_event["action"]["position"]
                    angle = radians(current_event["response"]["svd_deg"])
                    fig.add_trace(
                        go.Scatter(
                            x=[p["x_m"], p["x_m"] + 1500 * cos(angle)],
                            y=[p["y_m"], p["y_m"] + 1500 * sin(angle)],
                            name="当前测向",
                        )
                    )
                fig.update_layout(xaxis_title="东向 / m", yaxis_title="北向 / m")
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig, width="stretch")
                st.json(current_event)
                st.json(ledger(prefix))
                st.download_button(
                    "导出完整事件", (run_path(rid) / "events.jsonl").read_bytes(), f"{rid}.jsonl"
                )
                report = run_path(rid) / "report.md"
                if report.exists():
                    st.download_button("导出报告", report.read_bytes(), f"{rid}.md")

    replay_panel()
elif page == "基准比较与导出":
    suites = sorted((home / "benchmarks").glob("*/summary.json"))
    if not suites:
        st.info("尚无配对基准。运行 rf-searchlab benchmark --suite smoke 生成。")
    else:
        selected = st.selectbox("冻结配对实验", [p.parent.name for p in suites])
        folder = home / "benchmarks" / selected
        summary = read_json(folder / "summary.json")
        st.dataframe(summary["results"], width="stretch")
        import pandas as pd

        df = pd.read_csv(folder / "cases.csv")
        only_failed = st.checkbox("只看失败局")
        st.dataframe(df[df.status != "completed"] if only_failed else df, width="stretch")
        st.download_button("导出全部逐局 CSV", (folder / "cases.csv").read_bytes(), selected + ".csv")
        st.download_button("导出基准报告", (folder / "report.md").read_bytes(), selected + ".md")
