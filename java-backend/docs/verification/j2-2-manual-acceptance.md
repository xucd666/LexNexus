# J2-2 法律问答人工验收记录

本文按项目负责人已完成的人工验收整理。

- Java Backend 以 `local` Profile 启动，Python AI-Service 使用 `fake + vector` 模式。
- 通过 `POST /api/v1/dev/legal-qa` 发起请求。IDEA 断点观察了 Controller → ServiceImpl → AiServiceClient，确认 Java 生成的 `requestId` 正确传入 AI-Service，并与上游回显一致。
- 正常请求返回 HTTP 200，响应包含 `answer` 和 3 条 `citations`。通过 `curl.exe` 获取响应并按 UTF-8 解码，确认中文 JSON 正常。
- 停止 Python 后重新请求，Java 返回 HTTP 503 和 `QA_SERVICE_UNAVAILABLE`，响应未泄露内部异常堆栈。

验收范围为 Fake + Vector。回答使用 Fake 模式，DeepSeek Live 生成未验证。
