# 课程材料 RAG 聊天机器人 — 项目概述

## 核心流程

```
用户提问 → FastAPI → RAG系统 → AI调用搜索工具 → ChromaDB语义检索 → AI综合回答 → 返回答案+来源链接
```

## 技术栈

- **后端**: FastAPI + Anthropic Claude API（兼容腾讯云）+ ChromaDB 向量数据库
- **前端**: 纯静态 HTML/JS/CSS，无构建步骤
- **嵌入模型**: SentenceTransformer `all-MiniLM-L6-v2`
- **包管理**: uv

## 项目结构

| 目录/文件 | 作用 |
|---|---|
| `backend/app.py` | FastAPI 入口，2 个 API（`/api/query`、`/api/courses`），启动时加载 `docs/` 文档 |
| `backend/rag_system.py` | 核心编排器，串联所有组件 |
| `backend/ai_generator.py` | Anthropic Messages API 客户端，支持 tool-use 双轮调用 |
| `backend/search_tools.py` | 搜索工具定义（`CourseSearchTool`），AI 通过 tool-use 协议调用 |
| `backend/vector_store.py` | ChromaDB 封装，两个集合：`course_catalog`（元数据）和 `course_content`（文本块） |
| `backend/document_processor.py` | 解析课程 `.txt` 文件（标题/讲师/Lesson 结构），按句子分块 |
| `backend/session_manager.py` | 内存中的会话历史管理 |
| `backend/models.py` | Pydantic 数据模型（Course、Lesson、CourseChunk） |
| `frontend/` | 静态前端，使用 `marked.js` 渲染 Markdown |
| `docs/` | 4 个课程文本文件，启动时自动索引入 ChromaDB |

## 关键机制

- **AI 搜索流程**: 用户提问后，AI 自动判断是否需要搜索课程内容。需要时调用 `search_course_content` 工具，执行语义检索后再综合回答（双次 API 调用）
- **文档格式**: `Course Title:` / `Course Link:` / `Lesson N:` 结构化的 `.txt` 文件
- **去重**: 以课程标题为 ID，已索引的课程不会重复加载
- **来源链接**: 搜索结果附带课程视频 URL，前端渲染为可点击链接

## 当前配置

连接腾讯云 Anthropic 兼容 API，模型 `glm-5.1`，Base URL 指向 `api.lkeap.cloud.tencent.com`。
