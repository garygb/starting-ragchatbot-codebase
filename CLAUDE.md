# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Start the server (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000

# Or use the shell script
./run.sh
```

The app runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

Requires a `.env` file in the project root with:
```
ANTHROPIC_API_KEY=<key>
ANTHROPIC_BASE_URL=https://api.lkeap.cloud.tencent.com/plan/anthropic
ANTHROPIC_MODEL=glm-5.1
```

The project uses `uv` for dependency management (not pip). Python 3.13+.

## Architecture

**Full-stack RAG chatbot** — FastAPI backend serves both the API and the static frontend.

### Data Flow

1. User query hits `POST /api/query` → `RAGSystem.query()`
2. Query sent to `AIGenerator` with tool definitions (Anthropic tool-use protocol)
3. AI decides whether to call `search_course_content` tool via `CourseSearchTool`
4. Tool delegates to `VectorStore.search()` → ChromaDB semantic search → results returned to AI
5. AI synthesizes final answer from search results; sources tracked via `ToolManager`
6. `GET /api/courses` returns course catalog stats (no AI call)

### Key Components (all in `backend/`)

| Component | File | Role |
|---|---|---|
| FastAPI app | `app.py` | HTTP endpoints, serves frontend as static files, loads docs on startup |
| RAG orchestrator | `rag_system.py` | Wires all components together, owns the query pipeline |
| AI client | `ai_generator.py` | Anthropic Messages API with tool-use loop (two API calls when tool used) |
| Vector store | `vector_store.py` | ChromaDB with SentenceTransformer embeddings; two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| Document processor | `document_processor.py` | Parses `.txt` course files (header format: `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson N:` sections), chunks text by sentences |
| Tool system | `search_tools.py` | `Tool` ABC → `CourseSearchTool` implements Anthropic tool definition; `ToolManager` dispatches and tracks sources |
| Session manager | `session_manager.py` | In-memory conversation history per session |
| Models | `models.py` | Pydantic: `Course`, `Lesson`, `CourseChunk` |

### Frontend

Static files in `frontend/` — `index.html`, `script.js`, `style.css`. Uses `marked.js` for markdown rendering. No build step.

### Document Format

Course `.txt` files in `docs/` follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content>

Lesson 1: <title>
<content>
```

On startup, `app.py` loads all `.txt`/`.pdf`/`.docx` files from `../docs` into ChromaDB (skipping already-indexed courses by title). ChromaDB data persists in `backend/chroma_db/`.

## Configuration

All config via `backend/config.py` dataclass, values from `.env` via `python-dotenv`. Key tunables: `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (100), `MAX_RESULTS` (5), `MAX_HISTORY` (2), `EMBEDDING_MODEL` (all-MiniLM-L6-v2).

## Notes

- ChromaDB uses course title as the document ID in `course_catalog` collection — duplicate titles are skipped on reload
- The AI tool-use flow makes **two API calls** per query when search is triggered: first call returns tool_use, second call gets the final answer with tool results
- `AIGenerator._handle_tool_execution` sends the follow-up call **without tools** to force a direct answer
- `SessionManager` stores history in memory only — lost on server restart
