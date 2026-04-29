"""
Tests for RAGSystem query handling with content-related questions.

Run with: uv run pytest backend/tests/test_rag_system.py -v
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemQuery:
    """Test suite for RAGSystem query handling"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "test_model"
        config.ANTHROPIC_BASE_URL = ""
        return config

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore"""
        return Mock()

    @pytest.fixture
    def mock_ai_generator(self):
        """Create mock AIGenerator"""
        return Mock()

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock SessionManager"""
        manager = Mock()
        manager.get_conversation_history.return_value = None
        return manager

    @pytest.fixture
    def rag_system(
        self, mock_config, mock_vector_store, mock_ai_generator, mock_session_manager
    ):
        """Create RAGSystem with mocked dependencies"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
        ):

            system = RAGSystem(mock_config)
            system.vector_store = mock_vector_store
            system.ai_generator = mock_ai_generator
            system.session_manager = mock_session_manager
            return system

    # ==================== Content Query Tests ====================

    def test_query_for_course_content(self, rag_system, mock_ai_generator):
        """RAGSystem should handle course content queries"""
        mock_ai_generator.generate_response.return_value = "Course content answer"

        response, sources = rag_system.query("What is MCP?")

        assert response == "Course content answer"
        mock_ai_generator.generate_response.assert_called_once()

    def test_query_passes_tools_to_ai(self, rag_system, mock_ai_generator):
        """RAGSystem should pass tool definitions to AIGenerator"""
        mock_ai_generator.generate_response.return_value = "Answer"

        rag_system.query("Tell me about Python")

        call_args = mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["tools"] is not None
        assert len(call_args.kwargs["tools"]) == 2
        tool_names = [t["name"] for t in call_args.kwargs["tools"]]
        assert "search_content_within_lessons" in tool_names
        assert "list_all_lessons_in_course" in tool_names

    def test_query_passes_tool_manager(self, rag_system, mock_ai_generator):
        """RAGSystem should pass tool_manager to enable tool execution"""
        mock_ai_generator.generate_response.return_value = "Answer"

        rag_system.query("test question")

        call_args = mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["tool_manager"] is not None

    def test_query_retrieves_sources_from_tool(self, rag_system, mock_ai_generator):
        """RAGSystem should retrieve sources from tool manager after query"""
        mock_ai_generator.generate_response.return_value = (
            "Answer based on course content"
        )

        response, sources = rag_system.query("What is lesson 1 about?")

        # Sources should be returned (may be empty if no search was triggered)
        assert isinstance(sources, list)

    def test_query_updates_conversation_history(
        self, rag_system, mock_ai_generator, mock_session_manager
    ):
        """RAGSystem should update session history with original query after response"""
        mock_ai_generator.generate_response.return_value = "Answer"
        mock_session_manager.get_conversation_history.return_value = None

        rag_system.query("First question", session_id="session_1")

        # Session stores the original query (not the modified prompt)
        mock_session_manager.add_exchange.assert_called_once_with(
            "session_1",
            "First question",  # Original query, not the prefixed prompt
            "Answer",
        )

    # ==================== Session Handling Tests ====================

    def test_query_with_session_id(
        self, rag_system, mock_ai_generator, mock_session_manager
    ):
        """RAGSystem should retrieve history when session_id provided"""
        mock_session_manager.get_conversation_history.return_value = (
            "Previous conversation"
        )
        mock_ai_generator.generate_response.return_value = "Follow-up answer"

        rag_system.query("Follow-up question", session_id="session_123")

        mock_session_manager.get_conversation_history.assert_called_once_with(
            "session_123"
        )
        call_args = mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["conversation_history"] == "Previous conversation"

    def test_query_without_session_id(
        self, rag_system, mock_ai_generator, mock_session_manager
    ):
        """RAGSystem should not retrieve history when no session_id"""
        mock_ai_generator.generate_response.return_value = "Answer"

        rag_system.query("Standalone question")

        mock_session_manager.get_conversation_history.assert_not_called()
        call_args = mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["conversation_history"] is None

    # ==================== Tool Execution Integration Tests ====================

    def test_tool_execution_flow(self, rag_system):
        """Test the full flow of tool execution within RAGSystem"""
        # This test verifies the integration between ToolManager and CourseSearchTool
        assert hasattr(rag_system, "tool_manager")
        assert hasattr(rag_system, "search_tool")
        assert hasattr(rag_system, "outline_tool")

        # Verify both tools are registered
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_defs) == 2
        tool_names = [t["name"] for t in tool_defs]
        assert "search_content_within_lessons" in tool_names
        assert "list_all_lessons_in_course" in tool_names

    def test_search_tool_has_vector_store(self, rag_system):
        """CourseSearchTool should have access to VectorStore"""
        assert rag_system.search_tool.store == rag_system.vector_store

    # ==================== Error Handling Tests ====================

    def test_query_handles_ai_error(self, rag_system, mock_ai_generator):
        """RAGSystem should propagate AI errors"""
        mock_ai_generator.generate_response.side_effect = Exception(
            "AI service unavailable"
        )

        with pytest.raises(Exception, match="AI service unavailable"):
            rag_system.query("test question")

    def test_query_handles_empty_response(self, rag_system, mock_ai_generator):
        """RAGSystem should handle empty AI response"""
        mock_ai_generator.generate_response.return_value = ""

        response, sources = rag_system.query("test question")

        assert response == ""
        assert isinstance(sources, list)


class TestRAGSystemToolIntegration:
    """Test tool integration within RAGSystem"""

    @pytest.fixture
    def rag_system(self):
        """Create a more complete RAGSystem mock for integration testing"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            config = Mock()
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_chroma"
            config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2
            config.ANTHROPIC_API_KEY = "test_key"
            config.ANTHROPIC_MODEL = "test_model"
            config.ANTHROPIC_BASE_URL = ""

            return RAGSystem(config)

    def test_tool_manager_executes_search(self, rag_system):
        """ToolManager should be able to execute CourseSearchTool"""
        # Mock the vector store search
        rag_system.search_tool.store = Mock()
        rag_system.search_tool.store.search.return_value = SearchResults(
            documents=["MCP is Model Context Protocol"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
        )
        rag_system.search_tool.store.get_lesson_link.return_value = (
            "https://example.com/lesson/1"
        )

        result = rag_system.tool_manager.execute_tool(
            "search_content_within_lessons", query="What is MCP?"
        )

        assert "MCP" in result
        assert "MCP Course" in result

    def test_tool_manager_returns_sources(self, rag_system):
        """ToolManager should track and return sources after search"""
        rag_system.search_tool.store = Mock()
        rag_system.search_tool.store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )
        rag_system.search_tool.store.get_lesson_link.return_value = (
            "https://example.com/lesson"
        )

        rag_system.tool_manager.execute_tool(
            "search_content_within_lessons", query="test"
        )

        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Course - Lesson 1"

    def test_tool_manager_resets_sources(self, rag_system):
        """ToolManager should reset sources when requested"""
        rag_system.search_tool.last_sources = [{"title": "Old Source"}]

        rag_system.tool_manager.reset_sources()

        assert rag_system.search_tool.last_sources == []


class TestRAGSystemDocumentProcessing:
    """Test document processing within RAGSystem"""

    @pytest.fixture
    def rag_system(self):
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            config = Mock()
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_chroma"
            config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2
            config.ANTHROPIC_API_KEY = "test_key"
            config.ANTHROPIC_MODEL = "test_model"
            config.ANTHROPIC_BASE_URL = ""

            return RAGSystem(config)

    def test_add_course_document_success(self, rag_system):
        """RAGSystem should add course documents successfully"""
        from models import Course, CourseChunk, Lesson

        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="https://example.com",
            lessons=[Lesson(lesson_number=1, title="Intro", content="Content")],
        )
        mock_chunks = [
            CourseChunk(
                content="Chunk content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        rag_system.document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        course, chunk_count = rag_system.add_course_document("test_path.txt")

        rag_system.vector_store.add_course_metadata.assert_called_once()
        rag_system.vector_store.add_course_content.assert_called_once()
        assert chunk_count == 1

    def test_add_course_document_error(self, rag_system):
        """RAGSystem should handle document processing errors"""
        rag_system.document_processor.process_course_document.side_effect = Exception(
            "Parse error"
        )

        course, chunk_count = rag_system.add_course_document("bad_file.txt")

        assert course is None
        assert chunk_count == 0


class TestRAGSystemCourseAnalytics:
    """Test course analytics functionality"""

    @pytest.fixture
    def rag_system(self):
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            config = Mock()
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_chroma"
            config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2
            config.ANTHROPIC_API_KEY = "test_key"
            config.ANTHROPIC_MODEL = "test_model"
            config.ANTHROPIC_BASE_URL = ""

            return RAGSystem(config)

    def test_get_course_analytics(self, rag_system):
        """RAGSystem should return course analytics"""
        rag_system.vector_store.get_course_count.return_value = 5
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5


class TestRAGSystemEndToEnd:
    """End-to-end style tests with real component interactions"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock the Anthropic API response flow"""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    def test_full_query_flow_with_tool_use(self, mock_anthropic):
        """Test complete query flow from user input to response with tool use"""
        # Setup mock responses
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_content_within_lessons"
        tool_use_content.id = "tool_1"
        tool_use_content.input = {"query": "What is MCP?"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        second_response = MagicMock()
        second_response.stop_reason = "end_turn"
        second_response.content = [
            MagicMock(text="MCP stands for Model Context Protocol.")
        ]

        mock_anthropic.messages.create.side_effect = [first_response, second_response]

        # Create mock vector store before patching
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["MCP is Model Context Protocol for AI integration"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        # Create RAGSystem with mocked vector store
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.SessionManager"),
        ):

            config = Mock()
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_chroma"
            config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2
            config.ANTHROPIC_API_KEY = "test_key"
            config.ANTHROPIC_MODEL = "test_model"
            config.ANTHROPIC_BASE_URL = ""

            system = RAGSystem(config)
            response, sources = system.query("What is MCP?")

            # Verify tool was called via vector store
            mock_vector_store.search.assert_called_once()

            # Verify response
            assert "MCP" in response
