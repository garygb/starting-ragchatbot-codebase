"""
Tests for FastAPI endpoints (/api/query, /api/courses, /api/session/clear).

Run with: uv run pytest backend/tests/test_api.py -v
"""

import pytest
from unittest.mock import Mock


class TestQueryEndpoint:
    """Test suite for POST /api/query endpoint"""

    def test_query_returns_200(self, client, mock_rag_system):
        """POST /api/query should return 200 on success"""
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        assert response.status_code == 200

    def test_query_response_structure(self, client, mock_rag_system):
        """POST /api/query response should match QueryResponse schema"""
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_query_returns_answer(self, client, mock_rag_system):
        """POST /api/query should return the answer from RAGSystem"""
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        data = response.json()

        assert data["answer"] == "Python is a versatile programming language."

    def test_query_returns_sources(self, client, mock_rag_system):
        """POST /api/query should return sources from RAGSystem"""
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        data = response.json()

        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Python 101 - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson/1"

    def test_query_creates_session_when_none(self, client, mock_rag_system):
        """POST /api/query should create a new session when session_id not provided"""
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        data = response.json()

        mock_rag_system.session_manager.create_session.assert_called_once()
        assert data["session_id"] == "test-session-123"

    def test_query_uses_existing_session(self, client, mock_rag_system):
        """POST /api/query should use provided session_id"""
        response = client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": "existing-session-456"
        })
        data = response.json()

        mock_rag_system.session_manager.create_session.assert_not_called()
        assert data["session_id"] == "existing-session-456"

    def test_query_passes_query_to_rag(self, client, mock_rag_system):
        """POST /api/query should pass the query string to RAGSystem"""
        client.post("/api/query", json={
            "query": "What is Python?"
        })

        mock_rag_system.query.assert_called_once_with(
            "What is Python?",
            "test-session-123"
        )

    def test_query_passes_session_id_to_rag(self, client, mock_rag_system):
        """POST /api/query should pass session_id to RAGSystem.query"""
        client.post("/api/query", json={
            "query": "Follow up",
            "session_id": "existing-session-456"
        })

        mock_rag_system.query.assert_called_once_with(
            "Follow up",
            "existing-session-456"
        )

    def test_query_missing_query_field(self, client, mock_rag_system):
        """POST /api/query should return 422 when query field is missing"""
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_query_empty_query_string(self, client, mock_rag_system):
        """POST /api/query should accept empty query string (validation is RAGSystem's job)"""
        mock_rag_system.query.return_value = ("Please provide a question.", [])
        response = client.post("/api/query", json={"query": ""})
        assert response.status_code == 200

    def test_query_handles_rag_error(self, client, mock_rag_system):
        """POST /api/query should return 500 when RAGSystem raises exception"""
        mock_rag_system.query.side_effect = Exception("AI service unavailable")

        response = client.post("/api/query", json={
            "query": "What is Python?"
        })

        assert response.status_code == 500
        assert "AI service unavailable" in response.json()["detail"]

    def test_query_with_sources_having_no_url(self, client, mock_rag_system):
        """POST /api/query should handle sources without URL"""
        mock_rag_system.query.return_value = (
            "Answer",
            [{"title": "Python 101 - Lesson 1", "url": None}]
        )

        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        data = response.json()

        assert data["sources"][0]["url"] is None

    def test_query_with_empty_sources(self, client, mock_rag_system):
        """POST /api/query should handle empty sources list"""
        mock_rag_system.query.return_value = ("General answer", [])

        response = client.post("/api/query", json={
            "query": "Hello"
        })
        data = response.json()

        assert data["sources"] == []


class TestCoursesEndpoint:
    """Test suite for GET /api/courses endpoint"""

    def test_courses_returns_200(self, client, mock_rag_system):
        """GET /api/courses should return 200 on success"""
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_response_structure(self, client, mock_rag_system):
        """GET /api/courses response should match CourseStats schema"""
        response = client.get("/api/courses")
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_courses_returns_analytics(self, client, mock_rag_system):
        """GET /api/courses should return data from RAGSystem.get_course_analytics"""
        response = client.get("/api/courses")
        data = response.json()

        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Python 101", "MCP Course"]

    def test_courses_calls_analytics(self, client, mock_rag_system):
        """GET /api/courses should call RAGSystem.get_course_analytics"""
        client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_handles_error(self, client, mock_rag_system):
        """GET /api/courses should return 500 when analytics fails"""
        mock_rag_system.get_course_analytics.side_effect = Exception("DB error")

        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "DB error" in response.json()["detail"]

    def test_courses_with_empty_catalog(self, client, mock_rag_system):
        """GET /api/courses should handle empty course catalog"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []


class TestSessionClearEndpoint:
    """Test suite for POST /api/session/clear endpoint"""

    def test_clear_returns_200(self, client, mock_rag_system):
        """POST /api/session/clear should return 200 on success"""
        response = client.post("/api/session/clear", json={
            "session_id": "session-123"
        })
        assert response.status_code == 200

    def test_clear_returns_ok_status(self, client, mock_rag_system):
        """POST /api/session/clear should return status ok"""
        response = client.post("/api/session/clear", json={
            "session_id": "session-123"
        })
        data = response.json()
        assert data["status"] == "ok"

    def test_clear_calls_session_manager(self, client, mock_rag_system):
        """POST /api/session/clear should call session_manager.clear_session"""
        client.post("/api/session/clear", json={
            "session_id": "session-123"
        })
        mock_rag_system.session_manager.clear_session.assert_called_once_with("session-123")

    def test_clear_missing_session_id(self, client, mock_rag_system):
        """POST /api/session/clear should return 422 when session_id is missing"""
        response = client.post("/api/session/clear", json={})
        assert response.status_code == 422


class TestAPIValidation:
    """Test suite for API request validation"""

    def test_query_rejects_non_string_query(self, client, mock_rag_system):
        """POST /api/query should reject non-string query values"""
        response = client.post("/api/query", json={
            "query": 123
        })
        assert response.status_code == 422

    def test_query_rejects_wrong_content_type(self, client, mock_rag_system):
        """POST /api/query should reject non-JSON content type"""
        response = client.post(
            "/api/query",
            data="query=hello",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422

    def test_courses_rejects_post(self, client, mock_rag_system):
        """GET /api/courses should not accept POST requests"""
        response = client.post("/api/courses")
        assert response.status_code == 405

    def test_query_rejects_get(self, client, mock_rag_system):
        """POST /api/query should not accept GET requests"""
        response = client.get("/api/query")
        assert response.status_code == 405
