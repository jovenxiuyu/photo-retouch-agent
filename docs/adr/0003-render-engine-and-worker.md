# ADR-0003：独立渲染引擎与任务 Worker

- 状态：Accepted
- 日期：2026-07-03

## 决策

- 本地实时美化拆为独立、无 UI 依赖的 `retouch-engine`。
- 每类效果通过 Effect Registry 注册。
- 通用 Warp、Mask、Filter 与具体皮肤/眼部/身形算法分离。
- 固定比例定位实现 `HeuristicGeometryProvider`，未来关键点模型实现相同接口。
- AI 修图任务由独立 Worker 执行，API 只提交、查询和取消任务。

## 原因

- Canvas 引擎已超过 1200 行，继续集中修改会产生跨效果回归。
- React/DOM 状态不应进入高频渲染管线。
- 进程内 `asyncio.create_task` 无法在重启后恢复任务。

## 性能预算

- 832×1248 常用参数组合 P95 小于 30ms。
- 参数变化只传递差量。
- 所有模块拆分必须通过 Golden Image 与现有单元测试。

