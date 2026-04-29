"""
Shared fixtures and configuration for all tests.

Run tests with: uv run pytest backend/tests/ -v
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


# ==================== Pydantic models (mirrors app.py) ====================

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SourceItem(BaseModel):
    title: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str

class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]

class ClearSessionRequest(BaseModel):
    session_id: str


# ==================== Unit test fixtures ====================

@pytest.fixture
def sample_course_metadata():
    """Sample course metadata for testing"""
    return {
        "course_title": "Introduction to Python",
        "lesson_number": 1,
        "instructor": "Dr. Smith",
        "course_link": "https://example.com/python"
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    from vector_store import SearchResults
    return SearchResults(
        documents=[
            "Python is a versatile programming language.",
            "Variables in Python are dynamically typed."
        ],
        metadata=[
            {"course_title": "Python 101", "lesson_number": 1},
            {"course_title": "Python 101", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def sample_tool_definition():
    """Sample tool definition for testing"""
    return {
        "name": "search_course_content",
        "description": "Search course materials with smart course name matching",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in the course content"
                },
                "course_name": {
                    "type": "string",
                    "description": "Course title (partial matches work)"
                },
                "lesson_number": {
                    "type": "integer",
                    "description": "Specific lesson number"
                }
            },
            "required": ["query"]
        }
    }


# ==================== API test fixtures ====================

@pytest.fixture
def mock_rag_system():
    """Create a fully mocked RAGSystem for API tests"""
    rag = Mock()

    # Mock query method
    rag.query.return_value = (
        "Python is a versatile programming language.",
        [{"title": "Python 101 - Lesson 1", "url": "https://example.com/lesson/1"}]
    )

    # Mock get_course_analytics method
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python 101", "MCP Course"]
    }

    # Mock session_manager
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "test-session-123"
    rag.session_manager.clear_session = Mock()

    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with the same routes as the real app,
    but without static file mounting that would fail in test environment."""
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear")
    async def clear_session(request: ClearSessionRequest):
        mock_rag_system.session_manager.clear_session(request.session_id)
        return {"status": "ok"}

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """FastAPI test client"""
    return TestClient(test_app)


@pytest.fixture
def sample_sources():
    """Sample source references for testing"""
    return [
        {"title": "Python 101 - Lesson 1", "url": "https://example.com/lesson/1"},
        {"title": "Python 101 - Lesson 2", "url": "https://example.com/lesson/2"}
    ]
