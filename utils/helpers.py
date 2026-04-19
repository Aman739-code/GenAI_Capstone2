"""
Utility helper functions for the Solar Grid Optimization Agent.
"""

from typing import Any, Optional, Union


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a datetime as ISO string."""
    return (dt or datetime.now()).isoformat()


def safe_json_loads(text: str, default: Optional[Union[dict, list]] = None) -> Union[dict, list]:
    """Safely parse JSON with fallback."""
    try:
        # Try direct parse
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Try extracting JSON from markdown
        if "```json" in str(text):
            text = str(text).split("```json", 1)[1].split("```", 1)[0]
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass
        return default if default is not None else {}


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_report_section(title: str, content: Union[str, dict, list]) -> str:
    """Format a report section for display."""
    if isinstance(content, dict):
        content = json.dumps(content, indent=2, default=str)
    elif isinstance(content, list):
        content = "\n".join(f"  • {item}" if isinstance(item, str) else json.dumps(item, indent=2, default=str) for item in content)
    return f"### {title}\n{content}\n"
