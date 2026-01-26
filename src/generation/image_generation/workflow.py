"""
Image Generation Workflow

여러 노드를 연결하여 이미지 생성 파이프라인을 구성하는 워크플로우 정의
"""

from typing import List, Dict, Any, Optional, Callable
from .nodes.base import BaseNode


class ImageGenerationWorkflow:
    """
    노드 기반 이미지 생성 워크플로우

    여러 BaseNode를 순차적으로 실행하여 복잡한 이미지 생성 파이프라인 구성

    Example:
        # 워크플로우 생성
        workflow = ImageGenerationWorkflow()

        # 노드 추가 (메서드 체이닝 가능)
        workflow.add_node(Text2ImageNode())
        workflow.add_node(BackgroundRemovalNode())
        workflow.add_node(ResizeNode())

        # 실행
        result = workflow.run({"prompt": "cozy cafe interior"})
        image = result["image"]

        # 실행 리포트 확인
        report = workflow.get_execution_report()
        print(f"Total time: {report['total_time']}s")
    """

    def __init__(self, name: str = "ImageGenerationWorkflow"):
        """
        워크플로우 초기화

        Args:
            name: 워크플로우 이름 (디버깅/로깅용)
        """
        # 워크플로우에 포함된 노드들을 순서대로 저장
        self.nodes: List[BaseNode] = []

        # 워크플로우 이름
        self.name = name

        # 각 노드의 실행 메타데이터를 저장 (실행 시간, 상태 등)
        self.execution_metadata: List[Dict[str, Any]] = []

    def add_node(self, node: BaseNode) -> "ImageGenerationWorkflow":
        """
        워크플로우에 노드 추가

        Args:
            node: 추가할 BaseNode 인스턴스

        Returns:
            self (메서드 체이닝을 위해 자기 자신 반환)

        Example:
            # 한 줄로 여러 노드 추가 가능
            workflow.add_node(node1).add_node(node2).add_node(node3)
        """
        # 노드 리스트에 추가
        self.nodes.append(node)

        # 메서드 체이닝 패턴: self를 반환하여 .add_node().add_node() 가능
        return self

    def run(
        self,
        initial_inputs: Dict[str, Any],
        *,
        on_node_start: Optional[Callable[[BaseNode, Dict[str, Any]], None]] = None,
        on_node_end: Optional[Callable[[BaseNode, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        워크플로우 실행 (모든 노드를 순차적으로 실행)

        동작 방식:
        1. 첫 번째 노드에 initial_inputs 전달
        2. 각 노드의 출력을 다음 노드의 입력에 추가
        3. 마지막 노드의 출력 반환

        Args:
            initial_inputs: 첫 번째 노드에 전달할 초기 입력 데이터
                예: {"prompt": "cafe interior", "aspect_ratio": "16:9"}

        Returns:
            마지막 노드의 출력 포함한 전체 데이터
            예: {"prompt": "cafe", "image": <PIL.Image>, "seed": 42}

        Raises:
            Exception: 노드 실행 중 에러 발생 시

        Example:
            result = workflow.run({"prompt": "modern cafe"})
            # result = {"prompt": "modern cafe", "image": <PIL.Image>, ...}
        """
        # 실행 메타데이터 초기화 (이전 실행 기록 삭제)
        self.execution_metadata = []

        # 현재 데이터 컨테이너 (노드 간 데이터 전달용)
        # 초기값은 initial_inputs의 복사본
        current_data = initial_inputs.copy()

        # 모든 노드를 순서대로 실행
        for i, node in enumerate(self.nodes):
            # 현재 실행 중인 노드 정보 출력
            print(f"[{self.name}] Executing node {i+1}/{len(self.nodes)}: {node.node_name}")

            try:
                if on_node_start:
                    try:
                        on_node_start(node, current_data)
                    except Exception:
                        pass

                # 노드 실행
                # execute()는 base.py의 BaseNode에 정의됨
                # 내부에서 validate_inputs() -> process() -> 메타데이터 업데이트 수행
                output = node.execute(current_data)

                # 노드의 출력을 현재 데이터에 병합
                # 예시:
                #   current_data = {"prompt": "cafe"}
                #   output = {"image": <PIL.Image>, "seed": 42}
                #   -> current_data = {"prompt": "cafe", "image": <PIL.Image>, "seed": 42}
                current_data.update(output)

                if on_node_end:
                    try:
                        on_node_end(node, current_data)
                    except Exception:
                        pass

                # 이 노드의 실행 메타데이터 저장
                self.execution_metadata.append(node.get_metadata())

            except Exception as e:
                # 에러 발생 시에도 메타데이터 저장 (실패 정보 포함)
                self.execution_metadata.append(node.get_metadata())

                # 어느 노드에서 실패했는지 명확히 표시하여 예외 재발생
                raise Exception(
                    f"Workflow '{self.name}' failed at node {i+1} "
                    f"({node.node_name}): {str(e)}"
                ) from e

        # 마지막 노드까지 실행 완료 후 전체 데이터 반환
        return current_data

    def get_execution_report(self) -> Dict[str, Any]:
        """
        워크플로우 실행 리포트 반환

        실행 후 각 노드의 실행 시간, 상태 등을 확인할 수 있음
        병목 구간 분석에 유용

        Returns:
            실행 통계 및 각 노드 메타데이터

        Example:
            report = workflow.get_execution_report()
            # {
            #     "workflow_name": "ImageGenerationWorkflow",
            #     "total_nodes": 3,
            #     "total_time": 12.5,  # 초
            #     "nodes": [
            #         {
            #             "node_name": "Text2ImageNode",
            #             "execution_time": 10.2,
            #             "status": "completed",
            #             ...
            #         },
            #         {
            #             "node_name": "ResizeNode",
            #             "execution_time": 2.3,
            #             "status": "completed",
            #             ...
            #         }
            #     ]
            # }
        """
        # 총 실행 시간 계산 (모든 노드의 실행 시간 합산)
        total_time = sum(
            meta["execution_time"]
            for meta in self.execution_metadata
        )

        return {
            "workflow_name": self.name,
            "total_nodes": len(self.nodes),
            "total_time": total_time,
            "nodes": self.execution_metadata
        }

    def clear_nodes(self) -> None:
        """
        모든 노드 제거 (워크플로우 재구성 시 사용)

        Example:
            workflow.clear_nodes()
            workflow.add_node(NewNode())  # 새로 구성
        """
        self.nodes = []
        self.execution_metadata = []

    def reset_metadata(self) -> None:
        """
        모든 노드의 메타데이터 초기화 (재실행 전)

        노드 구성은 그대로 두고 실행 기록만 초기화

        Example:
            workflow.run(inputs1)
            workflow.reset_metadata()  # 메타데이터만 초기화
            workflow.run(inputs2)      # 같은 워크플로우 재실행
        """
        for node in self.nodes:
            node.reset_metadata()
        self.execution_metadata = []

    def __repr__(self) -> str:
        """
        워크플로우 문자열 표현

        Example:
            print(workflow)
            # ImageGenerationWorkflow(nodes=['Text2ImageNode', 'ResizeNode'])
        """
        node_names = [node.node_name for node in self.nodes]
        return f"ImageGenerationWorkflow(nodes={node_names})"
