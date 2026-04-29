from abc import ABC, abstractmethod
from typing import Any

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_content_within_lessons",
            "description": "Search for specific information or concepts within course lessons. Use when user asks about a TOPIC (e.g., 'what is X', 'explain Y', 'how to do Z'). DO NOT use when user wants to see all lessons.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or concept to search for (e.g., 'what is MCP', 'how to use tools')",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata, strict=False):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI with link
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
            link = (
                self.store.get_lesson_link(course_title, lesson_num)
                if lesson_num is not None
                else None
            )
            if not link:
                link = self.store.get_course_link(course_title)
            sources.append({"title": source, "url": link})

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outline with lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources for the UI

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "list_all_lessons_in_course",
            "description": "Get a numbered list of ALL LESSONS in a course. MUST use this tool when user asks: 'what lessons', 'list lessons', 'show syllabus', 'course structure', 'how many lessons'. Returns course title, link, and numbered lesson list.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title or partial name (e.g., 'MCP', 'Introduction to Python')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """
        Retrieve and format course outline.

        Args:
            course_title: Course title or partial name (fuzzy matched)

        Returns:
            Formatted course outline with title, link, and lesson list
        """
        outline = self.store.get_course_outline(course_title)

        if not outline:
            return f"No course found matching '{course_title}'."

        return self._format_outline(outline)

    def _format_outline(self, outline: dict[str, Any]) -> str:
        """Format course outline as markdown-style text"""
        lines = []

        title = outline.get("title", "Unknown Course")
        course_link = outline.get("course_link")
        instructor = outline.get("instructor")

        lines.append(f"## {title}")

        if instructor:
            lines.append(f"**Instructor:** {instructor}")

        # Track source for UI
        self.last_sources = [{"title": title, "url": course_link}]

        if course_link:
            lines.append(f"**Course Link:** {course_link}")

        lessons = outline.get("lessons", [])
        lesson_count = outline.get("lesson_count", len(lessons))

        if lessons:
            lines.append(f"\n**Lessons ({lesson_count} total):**")
            for lesson in lessons:
                lesson_num = lesson.get("lesson_number", "?")
                lesson_title = lesson.get("lesson_title", "Untitled")
                lines.append(f"  {lesson_num}. {lesson_title}")
        else:
            lines.append("\nNo lessons found for this course.")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
