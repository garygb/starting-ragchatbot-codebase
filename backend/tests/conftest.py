"""
Shared fixtures and configuration for all tests.

Run tests with: uv run pytest backend/tests/ -v
"""

import pytest
import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


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
