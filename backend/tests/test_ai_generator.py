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

    # ==================== Single Tool Call Tests ====================

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

        # Second response: final answer (no tool use in round 1)
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
        second_response.stop_reason = "end_turn"
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

    # ==================== Sequential Tool Calling Tests ====================

    def test_sequential_tool_calls(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should support up to 2 sequential tool calling rounds"""
        # Round 0: First tool call
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"query": "Lesson 4 topic"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_1]

        # Round 1: Second tool call based on first results
        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "API design patterns"}

        second_response = MagicMock()
        second_response.stop_reason = "tool_use"
        second_response.content = [tool_use_2]

        # Final response after max rounds
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(text="Combined answer from both searches.")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response, final_response]

        result = ai_generator.generate_response(
            query="Find courses discussing the same topic as Lesson 4",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        assert result == "Combined answer from both searches."
        # Verify 3 API calls: round 0, round 1, final
        assert mock_anthropic_client.messages.create.call_count == 3

    def test_early_termination_no_tool_use(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should terminate early if Claude decides no more tools needed"""
        # Round 0: Tool call
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_1"
        tool_use_content.input = {"query": "test"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        # Round 1: Claude decides to answer directly (no tool use)
        second_response = MagicMock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MagicMock(text="Direct answer after one search.")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        result = ai_generator.generate_response(
            query="test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Only one tool executed
        assert tool_manager.execute_tool.call_count == 1
        assert result == "Direct answer after one search."
        # Only 2 API calls: round 0 (tool), round 1 (direct answer)
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_max_rounds_enforcement(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should stop at MAX_TOOL_ROUNDS even if Claude wants more tools"""
        # Round 0: Tool call
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"query": "search 1"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_1]

        # Round 1: Tool call
        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "search 2"}

        second_response = MagicMock()
        second_response.stop_reason = "tool_use"
        second_response.content = [tool_use_2]

        # Round 2 (max reached): Claude wants third tool but gets forced final
        third_response = MagicMock()
        third_response.stop_reason = "end_turn"
        third_response.content = [MagicMock(text="Final answer after 2 searches.")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response, third_response]

        result = ai_generator.generate_response(
            query="complex query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Only 2 tools executed (MAX_TOOL_ROUNDS)
        assert tool_manager.execute_tool.call_count == 2
        assert result == "Final answer after 2 searches."
        # 3 API calls: round 0, round 1, final (forced)
        assert mock_anthropic_client.messages.create.call_count == 3

    def test_message_history_preserved_across_rounds(self, ai_generator, mock_anthropic_client, tool_manager):
        """Message history should be preserved across sequential tool calls"""
        # Round 0: Tool call
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"query": "first"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_1]

        # Round 1: Tool call
        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "second"}

        second_response = MagicMock()
        second_response.stop_reason = "tool_use"
        second_response.content = [tool_use_2]

        # Final
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(text="Answer")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response, final_response]

        ai_generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify 3 API calls were made
        assert mock_anthropic_client.messages.create.call_count == 3

        # Verify tool execution order
        assert tool_manager.execute_tool.call_count == 2

        # Verify the sequence of API calls includes tools in first two calls
        call_0 = mock_anthropic_client.messages.create.call_args_list[0]
        call_1 = mock_anthropic_client.messages.create.call_args_list[1]
        call_2 = mock_anthropic_client.messages.create.call_args_list[2]

        # First two calls should have tools
        assert "tools" in call_0.kwargs
        assert "tools" in call_1.kwargs
        # Final call should NOT have tools
        assert "tools" not in call_2.kwargs

    # ==================== Multiple Tool Calls in Single Response Tests ====================

    def test_multiple_tool_calls_in_single_response(self, ai_generator, mock_anthropic_client, tool_manager):
        """AIGenerator should handle multiple tool calls in a single response"""
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
        second_response.stop_reason = "end_turn"
        second_response.content = [MagicMock(text="Combined answer")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        ai_generator.generate_response(
            query="Compare two topics",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert tool_manager.execute_tool.call_count == 2

    # ==================== Error Handling Tests ====================

    def test_tool_execution_error_included_in_results(self, ai_generator, mock_anthropic_client, tool_manager):
        """Tool execution errors should be included in tool results, not raised"""
        tool_use_content = MagicMock()
        tool_use_content.type = "tool_use"
        tool_use_content.name = "search_course_content"
        tool_use_content.id = "tool_err"
        tool_use_content.input = {"query": "test"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_content]

        second_response = MagicMock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MagicMock(text="I encountered an error but here's what I know.")]

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]
        tool_manager.execute_tool.side_effect = Exception("Search failed")

        result = ai_generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Tool was called
        tool_manager.execute_tool.assert_called_once()
        # Result includes error handling response
        assert result == "I encountered an error but here's what I know."

        # Verify error was included in tool results
        round_1_call = mock_anthropic_client.messages.create.call_args_list[1]
        tool_result = round_1_call.kwargs["messages"][2]["content"][0]
        assert tool_result["is_error"] is True
        assert "Error:" in tool_result["content"]

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

    def test_max_tool_rounds_constant(self, ai_generator):
        """AIGenerator should have MAX_TOOL_ROUNDS constant"""
        assert hasattr(ai_generator, 'MAX_TOOL_ROUNDS')
        assert ai_generator.MAX_TOOL_ROUNDS == 2


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

    def test_system_prompt_allows_multiple_searches(self, ai_generator):
        """System prompt should mention tools for course materials"""
        assert "tool" in ai_generator.SYSTEM_PROMPT.lower()

    def test_system_prompt_includes_response_guidelines(self, ai_generator):
        """System prompt should include response guidelines"""
        prompt_lower = ai_generator.SYSTEM_PROMPT.lower()
        assert "concise" in prompt_lower or "brief" in prompt_lower


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
        """Should make direct call when tools provided but no tool_manager"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Direct answer")]
        mock_anthropic_client.messages.create.return_value = mock_response
        ai_generator.client = mock_anthropic_client

        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "test_tool"}]
            # No tool_manager provided
        )

        # Should make direct call without tool execution
        assert mock_anthropic_client.messages.create.call_count == 1
        assert result == "Direct answer"

    def test_tools_without_tool_manager_skips_loop(self, ai_generator, mock_anthropic_client):
        """When tools provided but no tool_manager, should skip tool loop"""
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"  # Would trigger tool use if manager existed
        mock_response.content = [MagicMock(text="Response")]
        mock_anthropic_client.messages.create.return_value = mock_response
        ai_generator.client = mock_anthropic_client

        result = ai_generator.generate_response(
            query="test",
            tools=[{"name": "test_tool"}]
            # No tool_manager
        )

        # Should still work, just no tool execution
        assert result == "Response"
