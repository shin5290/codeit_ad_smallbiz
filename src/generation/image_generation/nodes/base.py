"""
Base Node Classes for Image Generation Pipeline
노드 기반 이미지 생성 파이프라인의 기반 클래스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NodeMetadata:
    """노드 실행 메타데이터"""

    node_name: str
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """메타데이터를 딕셔너리로 변환"""
        return {
            "node_name": self.node_name,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "status": self.status,
            "error_message": self.error_message,
        }


class BaseNode(ABC):
    """
    모든 노드의 기본 추상 클래스

    각 노드는 다음을 구현해야 함:
    - process(): 실제 처리 로직
    - get_required_inputs(): 필수 입력 키 목록
    - get_output_keys(): 출력 키 목록

    Example:
        class MyNode(BaseNode):
            def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                image = inputs["image"]
                processed = self._do_something(image)
                return {"processed_image": processed}

            def get_required_inputs(self) -> list:
                return ["image"]

            def get_output_keys(self) -> list:
                return ["processed_image"]
    """

    def __init__(self, node_name: Optional[str] = None):
        """
        Args:
            node_name: 노드 이름 (None일 경우 클래스 이름 사용)
        """
        self.node_name = node_name or self.__class__.__name__
        self.metadata = NodeMetadata(node_name=self.node_name)

    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        노드의 핵심 처리 로직 (서브클래스에서 반드시 구현)

        Args:
            inputs: 입력 데이터 딕셔너리
                예: {"image": PIL.Image, "prompt": str}

        Returns:
            출력 데이터 딕셔너리
                예: {"generated_image": PIL.Image, "seed": int}
        """
        pass

    @abstractmethod
    def get_required_inputs(self) -> list:
        """
        이 노드가 필요로 하는 입력 키 목록 반환

        Returns:
            필수 입력 키 리스트
                예: ["image", "prompt"]
        """
        pass

    @abstractmethod
    def get_output_keys(self) -> list:
        """
        이 노드가 출력하는 키 목록 반환

        Returns:
            출력 키 리스트
                예: ["generated_image", "seed"]
        """
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        입력 검증 (필수 키가 모두 있는지 확인)

        Args:
            inputs: 검증할 입력 딕셔너리

        Raises:
            ValueError: 필수 입력이 누락된 경우
        """
        required = self.get_required_inputs()
        missing = [key for key in required if key not in inputs]

        if missing:
            raise ValueError(
                f"Node '{self.node_name}' missing required inputs: {missing}"
            )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        노드 실행 (검증 + 처리 + 메타데이터 업데이트)

        Args:
            inputs: 입력 데이터

        Returns:
            출력 데이터

        Raises:
            Exception: 노드 실행 중 발생한 모든 예외
        """
        # 입력 검증
        self.validate_inputs(inputs)

        # 메타데이터 초기화
        self.metadata.status = "running"
        self.metadata.timestamp = datetime.now().isoformat()
        start_time = time.time()

        try:
            # 실제 처리
            outputs = self.process(inputs)

            # 성공 메타데이터 업데이트
            self.metadata.status = "completed"
            self.metadata.execution_time = time.time() - start_time

            return outputs

        except Exception as e:
            # 실패 메타데이터 업데이트
            self.metadata.status = "failed"
            self.metadata.execution_time = time.time() - start_time
            self.metadata.error_message = str(e)

            # 예외 재발생 (상위에서 처리하도록)
            raise

    def get_metadata(self) -> Dict[str, Any]:
        """현재 노드의 메타데이터 반환"""
        return self.metadata.to_dict()

    def reset_metadata(self) -> None:
        """메타데이터 초기화 (재실행 시 사용)"""
        self.metadata = NodeMetadata(node_name=self.node_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.node_name}')"


class PassthroughNode(BaseNode):
    """
    입력을 그대로 출력하는 더미 노드 (테스트용)

    Example:
        node = PassthroughNode()
        output = node.execute({"data": "hello"})
        # output == {"data": "hello"}
    """

    def __init__(self, pass_keys: Optional[list] = None):
        """
        Args:
            pass_keys: 전달할 키 목록 (None이면 모든 입력을 전달)
        """
        super().__init__()
        self.pass_keys = pass_keys

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.pass_keys:
            return {k: inputs[k] for k in self.pass_keys if k in inputs}
        return inputs.copy()

    def get_required_inputs(self) -> list:
        return self.pass_keys or []

    def get_output_keys(self) -> list:
        return self.pass_keys or []
