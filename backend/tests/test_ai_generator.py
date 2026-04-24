"""
Tests for AIGenerator tool calling behavior.

Run with: uv run pytest backend/tests/test_ai_generator.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from ai_generator import AIGenerator


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling behavior"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        return MagicMock()

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test_key", model="test-model")
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def tool_manager(self):
        """Create a mock tool manager"""
        manager = Mock()
        manager.get_tool_definitions.return_value = [{
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }]
        manager.execute_tool.return_value = "Search results here"
        return manager

    # ==================== Basic Response Tests ====================

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """AIGenerator should return direct response when no tools used"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Direct answer")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response(query="What is Python?")

        assert result == "Direct answer"
        mock_anthropic_client.messages.create.assert_called_once()

    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """AIGenerator should include conversation history in system prompt"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Answer")]
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response(
            query="Follow up question",
            conversation_history="User: What is Python?\nAssistant: A language."
        )

        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert "What is Python?" in system_content

    # ==================== Tool Calling Tests ====================

    def test_tool_use_triggers_tool_execution(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should execute tool when stop_reason is tool_use"""
        # First response: tool use
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_123"
        tool_use_content.input = {"query": "What is MCP?"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        # Second response: final answer
        second_response = MagicMock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MagicMock(text="MCP is Model Context Protocol.")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        result = ai_generator.generate_response(
            query="What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is MCP?"
        )
        assert result == "MCP is Model Context Protocol."

    def test_tool_use_with_all_parameters(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should pass all parameters to tool"""
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_456"
        tool_use_content.input = {
            "query": "API design",
            "course_name": "MCP Course",
            "lesson_number": 3
        }

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        second_response = MagicMock()
        second_response.content = [MagicMock(text="Answer")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        ai_generator.generate_response(
            query="Tell me about API design",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="API design",
            course_name="MCP Course",
            lesson_number=3
        )

    def test_multiple_tool_calls_sequential(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should handle multiple tool calls in response"""
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"query": "first search"}

        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "second search"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_1, tool_use_2]

        second_response = MagicMock()
        second_response.content = [MagicMock(text="Combined answer")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        ai_generator.generate_response(
            query="Compare two topics",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert tool_manager.execute_tool.call_count == 2

    # ==================== Tool Result Handling Tests ====================

    def test_tool_results_included_in_follow_up(self, ai_generator, mock_anthropic_client, tool_manager):
        """Tool results should be properly formatted in follow-up API call"""
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_789"
        tool_use_content.input = {"query": "test"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        second_response = MagicMock()
        second_response.content = [MagicMock(text="Final answer")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        ai_generator.generate_response(
            query="test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Check second API call has tool results
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Should have: user query, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Tool result should have correct structure
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_789"

    def test_follow_up_call_without_tools(self, ai_generator, mock_anthropic_client, tool_manager):
        """Follow-up call after tool execution should not include tools"""
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_xyz"
        tool_use_content.input = {"query": "test"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        second_response = MagicMock()
        second_response.content = [MagicMock(text="Answer")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        ai_generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Check second call doesn't have tools
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        assert "tools" not in second_call.kwargs

    # ==================== Error Handling Tests ====================

    def test_tool_execution_error_propagates(self, ai_generator, mock_anthropic_client, tool_manager):
        """Tool execution errors should be included in tool results"""
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_err"
        tool_use_content.input = {"query": "test"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        second_response = MagicMock()
        second_response.content = [MagicMock(text="No relevant content found.")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]
        tool_manager.execute_tool.return_value = "No course found matching 'test'"

        result = ai_generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Tool error should be passed to AI
        tool_manager.execute_tool.assert_called_once()

    def test_api_error_handling(self, ai_generator, mock_anthropic_client):
        """API errors should propagate appropriately"""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            ai_generator.generate_response(query="test")

    # ==================== Configuration Tests ====================

    def test_model_configuration(self, mock_anthropic_client):
        """AIGenerator should use configured model"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="key", model="custom-model")

            assert generator.model == "custom-model"

    def test_base_url_configuration(self):
        """AIGenerator should support custom base URL"""
        with patch('ai_generator.anthropic.Anthropic') as mock_client_class:
            AIGenerator(api_key="key", model="model", base_url="https://custom.api")

            mock_client_class.assert_called_once_with(
                api_key="key",
                base_url="https://custom.api"
            )

    def test_temperature_and_max_tokens(self, ai_generator, mock_anthropic_client):
        """AIGenerator should use configured temperature and max_tokens"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Answer")]
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response(query="test")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["temperature"] == 0
        assert call_args.kwargs["max_tokens"] == 800


class TestAIGeneratorSystemPrompt:
    """Test AIGenerator system prompt behavior"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic'):
            return AIGenerator(api_key="key", model="model")

    def test_system_prompt_exists(self, ai_generator):
        """AIGenerator should have a non-empty system prompt"""
        assert hasattr(ai_generator, 'SYSTEM_PROMPT')
        assert len(ai_generator.SYSTEM_PROMPT) > 0

    def test_system_prompt_includes_tool_usage(self, ai_generator):
        """System prompt should mention tool usage guidelines"""
        assert "tool" in ai_generator.SYSTEM_PROMPT.lower()
        assert "search" in ai_generator.SYSTEM_PROMPT.lower()

    def test_system_prompt_includes_response_guidelines(self, ai_generator):
        """System prompt should include response guidelines"""
        prompt_lower = ai_generator.SYSTEM_PROMPT.lower()
        assert "concise" in prompt_lower or "brief" in prompt_lower
        assert "educational" in prompt_lower


class TestAIGeneratorEdgeCases:
    """Test edge cases and special scenarios"""

    @pytest.fixture
    def ai_generator(self):
        with patch('ai_generator.anthropic.Anthropic'):
            return AIGenerator(api_key="key", model="model")

    @pytest.fixture
    def mock_anthropic_client(self):
        return MagicMock()

    def test_empty_query(self, ai_generator, mock_anthropic_client):
        """AIGenerator should handle empty query"""
        ai_generator.client = mock_anthropic_client
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Please provide a question.")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response(query="")

        assert result is not None

    def test_tool_use_without_tool_manager(self, ai_generator, mock_anthropic_client):
        """Should not execute tools if tool_manager not provided"""
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.input = {"query": "test"}

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [tool_use_content]

        mock_anthropic_client.messages.create.return_value = mock_response
        ai_generator.client = mock_anthropic_client

        # Should not raise error, just return raw response content
        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "test_tool"}]
            # No tool_manager provided
        )

        # Should only make one API call
        assert mock_anthropic_client.messages.create.call_count == 1
