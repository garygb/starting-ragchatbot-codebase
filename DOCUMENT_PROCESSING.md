# 文档处理流程

## 1. 解析课程文档 (`process_course_document`)

课程 `.txt` 文件遵循固定格式：

```
Course Title: MCP: Build Rich-Context AI Apps with Anthropic
Course Link: https://www.deeplearning.ai/...
Course Instructor: Elie Schoppik

Lesson 0: Introduction
Lesson Link: https://learn.deeplearning.ai/...
课程正文内容...

Lesson 1: Why MCP
Lesson Link: https://learn.deeplearning.ai/...
课程正文内容...
```

解析逻辑：
- **前 3 行**提取课程元数据（标题、链接、讲师）
- **后续行**按 `Lesson N: <标题>` 正则匹配拆分课时
- 每个 `Lesson Link:` 行紧跟在课时标题后面，被单独解析，不进入正文内容
- 解析结果：1 个 `Course` 对象（含 `Lesson` 列表）+ 若干 `CourseChunk` 文本块

## 2. 文本分块 (`chunk_text`)

每个课时的正文被切成多个小块存入向量数据库：

- **按句子拆分**：用正则 `(?<=\.|\!|\?)\s+(?=[A-Z])` 按句号/叹号/问号后跟大写字母断句
- **按大小拼块**：从当前句子开始，依次累加直到超过 `CHUNK_SIZE`（800 字符）则截止
- **块间重叠**：当前块末尾的若干句子（总长 ≤ `CHUNK_OVERLAP` 即 100 字符）会作为下一块的起始，确保上下文不丢失
- **首块加前缀**：每个课时的第一个块会添加 `"Lesson N content: "` 前缀，提供课时上下文

## 3. 数据流向

```
.txt 文件
  → DocumentProcessor.process_course_document()
    → Course 对象 → VectorStore.add_course_metadata() → ChromaDB course_catalog 集合
    → CourseChunk 列表 → VectorStore.add_course_content() → ChromaDB course_content 集合
```

- `course_catalog` 存课程元数据（标题、讲师、课程链接、课时列表 JSON），用于课程名解析和来源链接查询
- `course_content` 存文本块（content + course_title + lesson_number 元数据），用于语义检索
