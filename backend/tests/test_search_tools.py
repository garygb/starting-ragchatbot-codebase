"""
Tests for CourseSearchTool.execute method output quality and correctness.

Run with: uv run pytest backend/tests/test_search_tools.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore for testing"""
        return Mock()

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool with mocked VectorStore"""
        return CourseSearchTool(mock_vector_store)

    # ==================== Tool Definition Tests ====================

    def test_tool_definition_has_required_fields(self, search_tool):
        """Tool definition should have all required Anthropic fields"""
        definition = search_tool.get_tool_definition()

        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "required" in definition["input_schema"]
        assert "query" in definition["input_schema"]["required"]

    def test_tool_definition_optional_fields(self, search_tool):
        """Tool definition should include optional course_name and lesson_number"""
        definition = search_tool.get_tool_definition()
        properties = definition["input_schema"]["properties"]

        assert "course_name" in properties
        assert properties["course_name"]["type"] == "string"
        assert "lesson_number" in properties
        assert properties["lesson_number"]["type"] == "integer"

    # ==================== Successful Search Tests ====================

    def test_execute_basic_search_success(self, search_tool, mock_vector_store):
        """Execute should return formatted results for successful search"""
        # Setup mock
        mock_vector_store.search.return_value = SearchResults(
            documents=["Python is a programming language."],
            metadata=[{
                "course_title": "Introduction to Python",
                "lesson_number": 1
            }],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        result = search_tool.execute(query="What is Python?")

        assert "Introduction to Python" in result
        assert "Lesson 1" in result
        assert "Python is a programming language." in result
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name=None,
            lesson_number=None
        )

    def test_execute_search_with_course_filter(self, search_tool, mock_vector_store):
        """Execute should pass course_name filter to vector store"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["MCP content here"],
            metadata=[{"course_title": "MCP Course", "lesson_number": None}],
            distances=[0.2]
        )
        mock_vector_store.get_course_link.return_value = "https://example.com/mcp"

        result = search_tool.execute(query="What is MCP?", course_name="MCP")

        assert "MCP Course" in result
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_search_with_lesson_filter(self, search_tool, mock_vector_store):
        """Execute should pass lesson_number filter to vector store"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/3"

        result = search_tool.execute(query="content", lesson_number=3)

        assert "Lesson 3" in result
        mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name=None,
            lesson_number=3
        )

    def test_execute_search_with_both_filters(self, search_tool, mock_vector_store):
        """Execute should handle both course_name and lesson_number filters"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/2"

        result = search_tool.execute(
            query="test",
            course_name="MCP",
            lesson_number=2
        )

        assert "MCP Course" in result
        assert "Lesson 2" in result
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="MCP",
            lesson_number=2
        )

    def test_execute_multiple_results(self, search_tool, mock_vector_store):
        """Execute should format multiple search results correctly"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Result 1 content", "Result 2 content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        result = search_tool.execute(query="test")

        assert "Course A" in result
        assert "Course B" in result
        assert "Result 1 content" in result
        assert "Result 2 content" in result
        # Results should be separated
        assert "\n\n" in result

    # ==================== Empty Results Tests ====================

    def test_execute_empty_results_basic(self, search_tool, mock_vector_store):
        """Execute should return informative message for empty results"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )

        result = search_tool.execute(query="nonexistent")

        assert "No relevant content found" in result
        assert "nonexistent" not in result  # Should not echo query back with error

    def test_execute_empty_results_with_course_filter(self, search_tool, mock_vector_store):
        """Execute should include course name in empty result message"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )

        result = search_tool.execute(query="test", course_name="Unknown Course")

        assert "No relevant content found" in result
        assert "Unknown Course" in result

    def test_execute_empty_results_with_lesson_filter(self, search_tool, mock_vector_store):
        """Execute should include lesson number in empty result message"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )

        result = search_tool.execute(query="test", lesson_number=99)

        assert "No relevant content found" in result
        assert "lesson 99" in result.lower()

    # ==================== Error Handling Tests ====================

    def test_execute_error_from_vector_store(self, search_tool, mock_vector_store):
        """Execute should return error message from vector store"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )

        result = search_tool.execute(query="test")

        assert result == "Database connection failed"

    def test_execute_course_not_found_error(self, search_tool, mock_vector_store):
        """Execute should return error when course name not resolved"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'Nonexistent'"
        )

        result = search_tool.execute(query="test", course_name="Nonexistent")

        assert "No course found" in result

    def test_execute_exception_handling(self, search_tool, mock_vector_store):
        """Execute should handle exceptions gracefully"""
        mock_vector_store.search.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception):
            search_tool.execute(query="test")

    # ==================== Sources Tracking Tests ====================

    def test_sources_tracking_single_result(self, search_tool, mock_vector_store):
        """Tool should track sources from search results"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        search_tool.execute(query="test")
        sources = search_tool.last_sources

        assert len(sources) == 1
        assert sources[0]["title"] == "Test Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson/1"

    def test_sources_tracking_multiple_results(self, search_tool, mock_vector_store):
        """Tool should track all sources from multiple results"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        search_tool.execute(query="test")
        sources = search_tool.last_sources

        assert len(sources) == 2
        assert sources[0]["title"] == "Course A - Lesson 1"
        assert sources[1]["title"] == "Course B"

    def test_sources_reset_on_new_search(self, search_tool, mock_vector_store):
        """Sources should be reset on each new search"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        search_tool.execute(query="first")
        first_sources = search_tool.last_sources

        # Second search
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 2"],
            metadata=[{"course_title": "Course B", "lesson_number": 2}],
            distances=[0.1]
        )
        search_tool.execute(query="second")
        second_sources = search_tool.last_sources

        assert len(first_sources) == 1
        assert first_sources[0]["title"] == "Course A - Lesson 1"
        assert len(second_sources) == 1
        assert second_sources[0]["title"] == "Course B - Lesson 2"

    # ==================== Output Format Tests ====================

    def test_output_format_header_structure(self, search_tool, mock_vector_store):
        """Output should have proper header format [Course Title - Lesson N]"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Some content"],
            metadata=[{"course_title": "Python 101", "lesson_number": 5}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com"

        result = search_tool.execute(query="test")

        assert "[Python 101 - Lesson 5]" in result

    def test_output_format_without_lesson_number(self, search_tool, mock_vector_store):
        """Output should handle missing lesson number gracefully"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Course overview"],
            metadata=[{"course_title": "General Course", "lesson_number": None}],
            distances=[0.1]
        )
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        result = search_tool.execute(query="overview")

        assert "[General Course]" in result
        assert "Lesson" not in result.split("[General Course]")[1].split("\n")[0]


class TestToolManager:
    """Test suite for ToolManager class"""

    @pytest.fixture
    def tool_manager(self):
        """Create a fresh ToolManager"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool"""
        tool = Mock()
        tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object", "properties": {}}
        }
        tool.execute.return_value = "Tool executed"
        return tool

    def test_register_tool(self, tool_manager, mock_tool):
        """ToolManager should register tools by name"""
        tool_manager.register_tool(mock_tool)

        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] == mock_tool

    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """ToolManager should return all tool definitions"""
        tool_manager.register_tool(mock_tool)
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_execute_tool(self, tool_manager, mock_tool):
        """ToolManager should execute tools by name"""
        tool_manager.register_tool(mock_tool)
        result = tool_manager.execute_tool("test_tool", param1="value1")

        assert result == "Tool executed"
        mock_tool.execute.assert_called_once_with(param1="value1")

    def test_execute_nonexistent_tool(self, tool_manager):
        """ToolManager should return error for nonexistent tool"""
        result = tool_manager.execute_tool("nonexistent")

        assert "not found" in result

    def test_get_last_sources(self, tool_manager, mock_tool):
        """ToolManager should get sources from tools with last_sources"""
        mock_tool.last_sources = [{"title": "Source 1"}]
        tool_manager.register_tool(mock_tool)

        sources = tool_manager.get_last_sources()

        assert sources == [{"title": "Source 1"}]

    def test_get_last_sources_empty(self, tool_manager):
        """ToolManager should return empty list when no sources"""
        sources = tool_manager.get_last_sources()

        assert sources == []

    def test_reset_sources(self, tool_manager, mock_tool):
        """ToolManager should reset sources on all tools"""
        mock_tool.last_sources = [{"title": "Source"}]
        tool_manager.register_tool(mock_tool)

        tool_manager.reset_sources()

        assert mock_tool.last_sources == []


class TestSearchResultsDataclass:
    """Test SearchResults dataclass behavior"""

    def test_is_empty_true(self):
        """SearchResults.is_empty should return True for empty documents"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """SearchResults.is_empty should return False with documents"""
        results = SearchResults(
            documents=["doc1"],
            metadata=[{}],
            distances=[0.1]
        )
        assert results.is_empty() is False

    def test_empty_factory(self):
        """SearchResults.empty should create results with error"""
        results = SearchResults.empty("Test error")

        assert results.documents == []
        assert results.error == "Test error"
        assert results.is_empty() is True
