# Frontend Changes

This feature (enhancing the testing framework) did not require any changes to the frontend. All modifications were in the backend test infrastructure:

- `pyproject.toml` — Added `[tool.pytest.ini_options]` config and `httpx` dev dependency
- `backend/tests/conftest.py` — Added API test fixtures (mock RAGSystem, test FastAPI app, TestClient)
- `backend/tests/test_api.py` — New file: 27 API endpoint tests for `/api/query`, `/api/courses`, `/api/session/clear`
