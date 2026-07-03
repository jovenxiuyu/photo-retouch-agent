# ADR-0002：采用 Ports & Adapters 与单向依赖

- 状态：Accepted
- 日期：2026-07-03

## 决策

后端依赖方向固定为：

```text
Domain <- Application <- Agent / Provider / Infrastructure <- API / Worker
```

## 规则

1. Domain 只依赖标准库。
2. Application 只能依赖 Domain。
3. Agent、Provider、Infrastructure 可以依赖 Application 和 Domain。
4. API 与 Worker 是 Composition Root，可以依赖所有需要组装的外层实现。
5. HTTP、Pydantic、数据库、文件系统和厂商 SDK 类型不能进入 Domain。
6. Route 不直接操作 Repository、Queue 或厂商客户端。
7. 禁止 import-time 创建全局数据库连接、Queue、Registry 或 Factory。

## 验证

CI 使用 AST 架构测试扫描内部 import。后续模块迁移必须同步更新依赖规则，不能通过跳过测试规避。

