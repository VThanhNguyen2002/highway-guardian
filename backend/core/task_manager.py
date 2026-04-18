"""
backend/core/task_manager.py

In-memory task registry for managing background inference tasks.
"""

import enum
from typing import Any, Dict, Optional

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# In-memory dictionary to store task statuses. 
# Format: { task_id: {"status": TaskStatus, "result": Any, "error": str} }
_tasks_registry: Dict[str, Dict[str, Any]] = {}


def create_task(task_id: str) -> None:
    """Initialize a new task in the registry."""
    _tasks_registry[task_id] = {
        "status": TaskStatus.PENDING,
        "result": None,
        "error": None
    }

def update_task_status(task_id: str, status: TaskStatus, result: Optional[Any] = None, error: Optional[str] = None) -> None:
    """Update an existing task in the registry."""
    if task_id in _tasks_registry:
        _tasks_registry[task_id]["status"] = status
        if result is not None:
            _tasks_registry[task_id]["result"] = result
        if error is not None:
            _tasks_registry[task_id]["error"] = error

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the current status of a task."""
    return _tasks_registry.get(task_id)
