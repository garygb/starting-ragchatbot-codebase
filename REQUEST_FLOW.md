# 用户请求追踪：从前端到后端

以用户输入 "What is MCP?" 为例，追踪完整的请求处理过程。

## 第 1 步：前端发起请求

**`frontend/script.js:45`** — `sendMessage()`

1. 用户点击发送按钮或按 Enter，触发 `sendMessage()`
2. 清空输入框，禁用输入控件，在聊天区显示用户消息和 loading 动画
3. 向 `POST /api/query` 发送 JSON 请求体：
   ```json
   { "query": "What is MCP?", "session_id": null }
   ```
   `session_id` 首次为 `null`，后续请求携带服务端返回的值

## 第 2 步：FastAPI 接收请求

**`backend/app.py:61`** — `query_documents()`

1. 请求体被 Pydantic 模型 `QueryRequest` 解析（`query: str`, `session_id: Optional[str]`）
2. 若 `session_id` 为空，调用 `session_manager.create_session()` 创建新会话（生成 `session_1`、`session_2` ...）
3. 调用 `rag_system.query(request.query, session_id)` 进入 RAG 核心处理

## 第 3 步：RAG 系统编排

**`backend/rag_system.py:102`** — `RAGSystem.query()`

1. 构造提示词：`"Answer this question about course materials: What is MCP?"`
2. 调用 `session_manager.get_conversation_history(session_id)` 获取历史对话（首轮为 `None`）
3. 获取工具定义：`tool_manager.get_tool_definitions()` — 返回 `CourseSearchTool` 的 Anthropic tool schema
4. 调用 `ai_generator.generate_response(query, history, tools, tool_manager)`

## 第 4 步：AI 第一次 API 调用（决定是否搜索）

**`backend/ai_generator.py:46`** — `generate_response()`

构造 Anthropic Messages API 请求：
```json
{
  "model": "glm-5.1",
  "temperature": 0,
  "max_tokens": 800,
  "system": "<SYSTEM_PROMPT + 历史对话>",
  "messages": [{"role": "user", "content": "Answer this question about course materials: What is MCP?"}],
  "tools": [{"name": "search_course_content", "input_schema": {...}}],
  "tool_choice": {"type": "auto"}
}
```

AI 返回 `stop_reason: "tool_use"`，表示需要调用搜索工具：
```json
{
  "content": [{"type": "tool_use", "name": "search_course_content", "input": {"query": "MCP"}}],
  "stop_reason": "tool_use"
}
```

## 第 5 步：执行搜索工具

**`backend/ai_generator.py:92`** — `_handle_tool_execution()`

1. 将 AI 的 tool_use 响应追加到 messages
2. 遍历 content_blocks，对每个 `tool_use` 调用 `tool_manager.execute_tool()`

**`backend/search_tools.py:52`** — `CourseSearchTool.execute()`

1. 调用 `self.store.search(query="MCP")` 执行语义检索

**`backend/vector_store.py:61`** — `VectorStore.search()`

1. 无课程名过滤 → 直接查询 `course_content` 集合
2. ChromaDB 用 `all-MiniLM-L6-v2` 将 "MCP" 转为向量，在 `course_content` 中找最相似的 5 个文本块
3. 返回 `SearchResults`（5 个文档 + 元数据 + 距离）

**`backend/search_tools.py:88`** — `_format_results()`

1. 格式化搜索结果：`"[课程名 - Lesson N]\n文本内容"`
2. 查询每个结果的链接：先查 `get_lesson_link(course_title, lesson_num)`，无结果则回退 `get_course_link(course_title)`
3. 将来源信息存入 `self.last_sources`：`[{"title": "...", "url": "..."}]`
4. 返回格式化的搜索结果文本

## 第 6 步：AI 第二次 API 调用（综合回答）

**`backend/ai_generator.py:92`** — `_handle_tool_execution()` 继续

构造第二次 API 请求，**不带 tools**（强制直接回答）：
```json
{
  "model": "glm-5.1",
  "temperature": 0,
  "max_tokens": 800,
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {"role": "user", "content": "Answer this question..."},
    {"role": "assistant", "content": [<tool_use block>]},
    {"role": "user", "content": [<tool_result block>]}
  ]
}
```

AI 基于搜索结果生成最终回答，`stop_reason: "end_turn"`，返回文本。

## 第 7 步：RAG 系统收尾

**`backend/rag_system.py:129`** — 回到 `query()`

1. `tool_manager.get_last_sources()` 获取来源列表（带链接的字典数组）
2. `tool_manager.reset_sources()` 清空
3. `session_manager.add_exchange(session_id, query, response)` 记录对话历史
4. 返回 `(response, sources)`

## 第 8 步：FastAPI 构造响应

**`backend/app.py:71`** — 回到 `query_documents()`

1. 用 `QueryResponse` 模型序列化：
   ```json
   {
     "answer": "MCP (Model Context Protocol) is...",
     "sources": [
       {"title": "MCP: Build Rich-Context AI Apps with Anthropic - Lesson 1", "url": "https://learn.deeplearning.ai/..."},
       ...
     ],
     "session_id": "session_1"
   }
   ```
2. 返回 HTTP 200

## 第 9 步：前端渲染结果

**`frontend/script.js:76`** — `sendMessage()` 继续

1. 解析 JSON 响应
2. 首次请求时保存 `currentSessionId = data.session_id`
3. 移除 loading 动画
4. 调用 `addMessage(data.answer, 'assistant', data.sources)`

**`frontend/script.js:113`** — `addMessage()`

1. 用 `marked.parse()` 将 Markdown 答案转为 HTML
2. 渲染 Sources 折叠区：有 URL 的来源渲染为 `<a href="..." target="_blank">` 链接，无 URL 的显示纯文本
3. 重新启用输入控件，聚焦输入框

## 流程总览

```
前端 sendMessage()                          后端
    │
    ├─ POST /api/query ──────────────────► query_documents()
    │   {query, session_id}                    │
    │                                          ├─ 创建/复用 session
    │                                          │
    │                                     RAGSystem.query()
    │                                          │
    │                                     AIGenerator.generate_response()
    │                                          │
    │                              ┌──── 第1次 API 调用 ────┐
    │                              │  带tools，AI决定搜索   │
    │                              └──── stop_reason: tool_use ─┘
    │                                          │
    │                                     CourseSearchTool.execute()
    │                                          │
    │                                     VectorStore.search()
    │                                     → ChromaDB 语义检索
    │                                     → _format_results() 查链接
    │                                          │
    │                              ┌──── 第2次 API 调用 ────┐
    │                              │  不带tools，综合回答    │
    │                              └──── stop_reason: end_turn ──┘
    │                                          │
    │                                     记录对话历史
    │                                     返回 (answer, sources)
    │                                          │
    │◄─── {answer, sources, session_id} ──────┘
    │
    ├─ addMessage() 渲染答案 (Markdown → HTML)
    ├─ addMessage() 渲染来源 (可点击链接)
    └─ 恢复输入控件
```
