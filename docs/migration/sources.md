# 代码来源与许可

派生代码来自智法宝（law-wiz）的 Git 提交 `be0d7727297c2f20775069ea66d24b79d58a52d9`。原项目 MIT 许可及版权通知完整保存在 [许可证文件](../../ai-service/LICENSES/upstream-MIT.txt) 中，并随 AI Service 的 Python 构建产物分发。

## AI Service

下表除单独标明的根目录文件外，路径均相对于各仓库的 `ai-service/`。目标列的“原路径”表示文件名及相对路径保持一致。

| 原文件 | LexNexus 文件 | 调整 |
| --- | --- | --- |
| `LICENSE`（原仓库根目录） | `LICENSES/upstream-MIT.txt` | 保留完整原文。 |
| `app/__init__.py` 及 `app/{agent,agent/core,agent/legal_qa,api,api/v1,llm,schemas,tools}/__init__.py` | 原路径 | 保留 Python 包标记。 |
| `app/agent/core/base_agent.py`、`app/agent/{registry,runtime,exceptions}.py` | 原路径 | 适配类型契约、显式注册和业务输入异常。 |
| `app/agent/legal_qa/{agent,graph,nodes,prompts,state}.py` | 原路径 | 保留核心工作流，精简节点并在生成前校验引用。 |
| `app/api/v1/agent.py`、`app/service/agent_service.py` | `app/api/v1/agent.py` | 统一入口直接调用 Runtime。 |
| `app/api/exception_handlers.py`、`app/schemas/contracts.py`、`app/main.py` | 原路径 | 保留调用契约、处理器和应用入口，适配校验与应用装配。 |
| `app/tools/backend.py`（检索接口及 Fake 组件） | `app/tools/law.py` | 保留异步检索方法签名，以协议和固定测试材料实现离线检索。 |
| `app/llm/deepseek.py`、`app/core/config.py`、`.env.example` | 原路径 | 保留模型适配和配置，明确运行模式。 |
| `pyproject.toml`、`uv.lock` | 原路径 | 适配服务元数据及开发依赖。 |
| `tests/test_{deepseek,exception_handlers,legal_qa,runtime}.py` | 原路径 | 适配现有用例并补充检索与接口边界测试。 |

## 前端

前端使用同一来源提交。原项目 `frontend/` 中的页面、样式、路由、状态、请求封装、工程配置和锁文件迁入 LexNexus 的 `frontend/`。适配范围包括项目品牌、令牌存储键、开发代理和使用说明。

原仓库根目录 `LICENSE` 的原文保存在 [前端上游 MIT 许可](../../frontend/public/upstream-MIT.txt)，构建时随前端静态产物分发。

原项目的公开归属链接待确认。LexNexus 独立新增部分的许可证待项目维护者确认。
