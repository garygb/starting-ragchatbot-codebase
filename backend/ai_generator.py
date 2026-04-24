import anthropic
from typing import List, Optional, Dict, Any, Tuple

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Configurable limits
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials with access to two tools.

TOOL SELECTION - READ CAREFULLY:

**Tool 1: list_all_lessons_in_course**
- Use when user asks: "what lessons", "list lessons", "show syllabus", "course structure", "how many lessons"
- Example: "What lessons are in the MCP course?" -> use list_all_lessons_in_course(course_title="MCP")

**Tool 2: search_content_within_lessons**
- Use when user asks about a TOPIC or CONCEPT: "what is X", "explain Y", "how to do Z"
- Parameters: query (required), course_name (optional), lesson_number (optional)
- IMPORTANT: If course_name is NOT specified, it searches ALL courses
- Use lesson_number to narrow search to a specific lesson

DECISION RULE:
- User asks for LESSON LIST -> list_all_lessons_in_course
- User asks about a TOPIC -> search_content_within_lessons
- User asks to compare across courses -> search WITHOUT course_name parameter

CROSS-COURSE SEARCH EXAMPLES:
- "Are there other courses covering the same topic as MCP lesson 5?"
  Step 1: search_content_within_lessons(query="topic", course_name="MCP", lesson_number=5) to find the topic
  Step 2: search_content_within_lessons(query="same topic") WITHOUT course_name to find in other courses
- "Which courses mention tools?" -> search_content_within_lessons(query="tools") without course_name

Examples:
- "What lessons are in MCP course?" -> list_all_lessons_in_course(course_title="MCP")
- "What is MCP?" -> search_content_within_lessons(query="what is MCP")
- "Find MCP content about clients" -> search_content_within_lessons(query="clients", course_name="MCP")
- "What do other courses say about API?" -> search_content_within_lessons(query="API") without course_name

Response Guidelines:
- Provide direct, concise answers
- No meta-commentary about which tool was used
"""
    
    def __init__(self, api_key: str, model: str, base_url: str = ""):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with sequential tool calling support.

        Supports up to MAX_TOOL_ROUNDS of tool execution, allowing Claude
        to reason about results and make additional tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history
        messages = [{"role": "user", "content": query}]

        # Execute tool calling loop if tools are available
        if tools and tool_manager:
            messages, exit_reason, direct_text = self._execute_tool_loop(
                messages, tools, tool_manager, system_content
            )
            # If Claude returned direct text (no tool use), return it
            if direct_text is not None:
                return direct_text
            # Otherwise, make final call without tools
            return self._make_final_call(messages, system_content)

        # No tools - direct API call
        return self._make_direct_call(messages, system_content)

    def _execute_tool_loop(self,
                          messages: List[Dict],
                          tools: List[Dict],
                          tool_manager,
                          system_content: str) -> Tuple[List[Dict], str, Optional[str]]:
        """
        Execute the tool calling loop for up to MAX_TOOL_ROUNDS.

        Args:
            messages: Current message history
            tools: Available tool definitions
            tool_manager: Manager to execute tools
            system_content: System prompt content

        Returns:
            Tuple of (updated_messages, exit_reason, direct_text)
            exit_reason is one of: "complete", "max_rounds"
            direct_text is set if Claude returns text without tool use
        """
        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Make API call with tools available
            response = self.client.messages.create(
                **self.base_params,
                messages=messages,
                system=system_content,
                tools=tools,
                tool_choice={"type": "auto"}
            )

            # Termination: Claude finished without tool use
            if response.stop_reason != "tool_use":
                # Direct text response - return it immediately
                return messages, "complete", self._extract_text(response)

            # Process tool calls
            tool_results = self._execute_tools(response, tool_manager)

            # Append to message history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # Max rounds reached
        return messages, "max_rounds", None

    def _execute_tools(self, response, tool_manager) -> List[Dict]:
        """
        Execute all tool_use blocks in the response.

        Args:
            response: API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result blocks
        """
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                except Exception as e:
                    # Graceful error handling - include error in result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {str(e)}",
                        "is_error": True
                    })

        return tool_results

    def _make_final_call(self, messages: List[Dict], system_content: str) -> str:
        """
        Make final API call without tools to get text response.

        Args:
            messages: Full conversation history
            system_content: System prompt content

        Returns:
            Final text response
        """
        response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system_content
        )
        return self._extract_text(response)

    def _make_direct_call(self, messages: List[Dict], system_content: str) -> str:
        """
        Make direct API call without tools.

        Args:
            messages: Message history (typically just user query)
            system_content: System prompt content

        Returns:
            Text response
        """
        response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system_content
        )
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """
        Extract text from response, skipping ThinkingBlocks.

        Some models (like Baidu Qianfan) return ThinkingBlock + TextBlock.
        This method finds and returns the text from TextBlock.

        Args:
            response: API response object

        Returns:
            Extracted text string
        """
        for block in response.content:
            if hasattr(block, 'text') and block.type == "text":
                return block.text
        # Fallback: return first block's text if it exists
        if response.content and hasattr(response.content[0], 'text'):
            return response.content[0].text
        return ""