# 律枢（LexNexus）Java 后端

Java 后端可独立运行，提供 HTTP 健康检查、AI-Service HTTP 客户端和本地法律问答开发接口。健康检查无需数据库、Redis 或 AI-Service。用户认证、对外法律问答和合同审查接口尚未实现。

## 环境

- JDK 21，确保 `java -version` 和 `mvn -version` 均显示 Java 21。
- Maven 3.9.10（本次验证版本）。
- Spring Boot 3.5.16，使用 Spring Web、Spring Validation 和 Spring Boot Test。

## 启动与健康检查

从仓库根目录进入模块并启动应用。

```powershell
cd java-backend
mvn spring-boot:run
```

应用默认使用 `8080` 端口。在另一个终端发起请求。

```powershell
curl.exe -i http://localhost:8080/api/v1/health
```

响应为 HTTP `200 OK`，内容类型为 `application/json`。

```json
{"service":"lexnexus-java-backend","status":"UP"}
```

`UP` 表示 Java 后端进程已能响应 HTTP 请求。`service` 取自 `spring.application.name`。

## AI-Service 客户端

`AiServiceClient.invokeLegalQa(input, context)` 供 Java 业务层调用。客户端向 `POST /api/v1/agent/invoke` 发送 `legal_qa` 请求，校验响应与回显的 `requestId`，并保留上游错误语义。Agent 调度、引用生成和模型调用由 AI-Service 负责。

| Spring 属性 | 环境变量 | 默认值 |
| --- | --- | --- |
| `ai-service.base-url` | `AI_SERVICE_BASE_URL` | `http://127.0.0.1:8000` |
| `ai-service.connect-timeout` | `AI_SERVICE_CONNECT_TIMEOUT` | `2s` |
| `ai-service.read-timeout` | `AI_SERVICE_READ_TIMEOUT` | `150s` |

连接和读取超时采用正数时长，例如 `2s` 或 `500ms`。客户端通过 JDK HTTP 客户端配置连接超时，通过 Spring 的 `JdkClientHttpRequestFactory` 限制等待和读取响应的时间。POST 失败后由调用方决定下一步，客户端不会自动重发或跟随重定向。Java 健康检查可以在 AI-Service 未运行时独立使用。

### 客户端响应边界与错误分类

成功和错误响应共用 2 MiB（2,097,152 字节）正文上限。客户端最多读取上限加一个字节来判断越界，超过上限即抛出 `RESPONSE_TOO_LARGE`，并关闭响应。大小按实际流读取的字节计算，适用于缺少 Content-Length 或分块传输的响应。

网络错误按异常及其原因链分类。明确的 `HttpTimeoutException`、`SocketTimeoutException` 归为 `TIMEOUT`；连接失败或地址解析失败归为 `NETWORK_UNREACHABLE`；连接中断及其余 I/O 失败归为 `NETWORK_IO_ERROR`。异常保留 requestId、已取得的 HTTP 状态和可用的异常类名。

当前 Spring 6.2.19 / JDK 21 组合在部分正文读取超时场景中只关闭流并抛出普通 IOException，这类情况采用 `NETWORK_IO_ERROR`。读取时间仍由请求工厂限制，分类只依赖异常证据。客户端异常消息使用固定分类，底层异常消息及响应正文不写入日志。

## 本地法律问答

通过 `spring.profiles.active=local` 启用开发接口。从 `java-backend` 目录运行：

```powershell
mvn spring-boot:run '-Dspring-boot.run.arguments=--spring.profiles.active=local'
```

`local` Profile 默认绑定 `127.0.0.1`。Java 在 Web 服务启动前检查实际监听地址，未配置有效回环地址时拒绝启动。此接口无用户鉴权，仅用于本机开发，请勿配置外部反向代理。默认 Profile 下此路径返回 HTTP 404。调用前按[仓库启动说明](../README.md#本地运行)启动 AI-Service，可选用 Fake 模式验证调用。人工联调结果见 [J2-2 验收记录](docs/verification/j2-2-manual-acceptance.md)。

`POST /api/v1/dev/legal-qa` 接收 JSON 对象，其中 `question` 为必填字符串。去除首尾空白后须为 1 至 4000 个 Unicode 字符。每次请求同步等待一次 AI-Service 调用，Java 生成 UUID `requestId` 并传给 AI-Service。

```powershell
$body = @{ question = '合同义务未履行时，通常有哪些违约救济？' } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/api/v1/dev/legal-qa' `
    -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

HTTP 200 返回 `requestId`、`answer` 和 `citations`。引用字段为 `documentId`、`kbChunkId`、`lawName`、`articleNo`、`quotedText` 和 `relevanceScore`。无引用时 `citations` 为 `[]`。

错误响应为 `{code, message, requestId}`。输入不合法返回 400，此时尚未生成 `requestId`，字段为 `null`。上游模型不可用或服务不可达返回 503，超时返回 504，其他已知的上游请求、响应和 I/O 错误返回 502。上游失败带回 Java 的调用标识，消息使用固定文本。

## 测试与打包

在 `java-backend` 目录执行以下命令。

```powershell
mvn test
mvn package
java -jar target/lexnexus-java-backend-0.0.1-SNAPSHOT.jar
```

测试使用随机端口启动 Spring Boot，并通过 HTTP 验证健康检查的状态码、内容类型和响应字段。`mvn package` 会运行测试并生成可执行 JAR。启动 JAR 前请先停止占用默认端口的开发进程。

AI-Service 客户端测试使用 Spring HTTP stub 检查 JSON 契约，并使用本机回环 HTTP 服务验证连接失败、读取超时与 POST 发送次数。本地问答 API 测试使用 MockMvc 和模拟客户端验证分层调用、输入校验、错误响应和 Profile 隔离。J2-0 的向量联调仍为显式启用。
