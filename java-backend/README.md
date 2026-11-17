# 律枢（LexNexus）Java 后端

J1 已提供可独立运行的 Spring Boot 基础工程、HTTP 健康检查和 AI-Service HTTP 客户端。健康检查无需数据库、Redis 或 AI-Service；客户端仅供后续业务层调用。用户认证、对外法律问答和合同审查接口尚未实现。

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

`AiServiceClient.invokeLegalQa(input, context)` 供 Java 业务层调用。客户端向 `POST /api/v1/agent/invoke` 发送 `legal_qa` 请求，校验响应与回显的 `requestId`，并保留上游错误语义。`requestId` 由调用方提供。Agent 调度、引用生成和模型调用由 AI-Service 负责。对外法律问答 API 留待后续阶段。

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

## 测试与打包

在 `java-backend` 目录执行以下命令。

```powershell
mvn test
mvn package
java -jar target/lexnexus-java-backend-0.0.1-SNAPSHOT.jar
```

测试使用随机端口启动 Spring Boot，并通过 HTTP 验证健康检查的状态码、内容类型和响应字段。`mvn package` 会运行测试并生成可执行 JAR。启动 JAR 前请先停止占用默认端口的开发进程。

AI-Service 客户端测试使用 Spring HTTP stub 检查 JSON 契约，并使用本机回环 HTTP 服务验证连接失败、读取超时与 POST 发送次数。测试数据均为固定样例。
