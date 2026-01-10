from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    PERSISTING = "persisting"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TaskState:
    """작업 상태 관리"""
    task_id: str
    status: TaskStatus
    progress: int  # 0-100
    result: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# 임시 인메모리 스토리지 (나중에 Redis/DB로 교체 가능)
_task_storage: Dict[str, TaskState] = {}


def create_task(task_id: str) -> TaskState:
    """새 작업 생성"""
    task = TaskState(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0,
    )
    _task_storage[task_id] = task
    return task


def get_task(task_id: str) -> Optional[TaskState]:
    """작업 상태 조회"""
    return _task_storage.get(task_id)


def update_task_progress(
    task_id: str, 
    progress: int, 
    status: Optional[TaskStatus] = None
) -> TaskState:
    """작업 진행률 업데이트"""
    task = _task_storage.get(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    task.progress = progress
    if status:
        task.status = status
    task.updated_at = datetime.utcnow()
    
    return task


def complete_task(task_id: str, result: dict) -> TaskState:
    """작업 완료 처리"""
    task = _task_storage.get(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    task.status = TaskStatus.DONE
    task.progress = 100
    task.result = result
    task.updated_at = datetime.utcnow()
    
    return task


def fail_task(task_id: str, error: str) -> TaskState:
    """작업 실패 처리"""
    task = _task_storage.get(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    task.status = TaskStatus.FAILED
    task.error = error
    task.updated_at = datetime.utcnow()
    
    return task


def delete_task(task_id: str) -> bool:
    """작업 삭제 (정리용)"""
    if task_id in _task_storage:
        del _task_storage[task_id]
        return True
    return False