"""
Image Generation Wrorkflow

여러 노드를 연결하여 이미지 생성 pipline을 구성하는 워크플로우 정의
"""

from typing import List, Dict, Any, Optional
from .nodes.base import BaseNode

class ImageGenerationWorkflow:
    """
    노드 기반 이미지 생성 워크플로우

    여러 BaseNode를 순차적으로 실행하여 복잡한 이미지 생성 pipeline을 구성
    
    Example:
        workflow = ImageGnerationWorkflow()
        workflow.add_node(Text2ImageNode(...))
        workflow.add_node(BackgroundRemovalNode(...))

        result = workflow.run({'prompt': 'cafe interior'})
        image = result['image']
    """
    def __init__(self, name: str = "ImageGenerationWorkflow"):
        """
        Args:
            name: 워크플로우 이름(디버깅/로깅용)
        """
        # 워크플로우에 포함된 노드들을 저장할 리스트
        self.nodes: List[BaseNode] = []

        # 워크플로우 이름
        self.name = name

        # 전체 실행 메타데이터 (각 노드의 실행 정보)
        self.execution_metadat: List[Dict[str, Any]] = []


    def add_node(self, node: BaseNode) -> "ImageGenerationWorkflow":
        """
        워크플로우에 노드 추가
        
        Args:
            node: 추가할 BaseNode 인스턴스
            
        Returns:
            self (메서드 체이닝을 위해)
            
        Example:
            workflow.add_node(node1).add_node(node2).add_node(node3)
        """
        # 노드 리스트에 추가
        self.nodes.append(node)
        
        # 메서드 체이닝을 위해 self 반환
        # 이렇게 하면 workflow.add_node(a).add_node(b) 가능
        return self
    
    def run(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        워크플로우 실행 (모든 노드를 순차적으로 실행)
        
        Args:
            initial_inputs: 첫 번째 노드에 전달할 초기 입력
                예: {"prompt": "cafe", "aspect_ratio": "16:9"}
        
        Returns:
            마지막 노드의 출력
            
        Raises:
            Exception: 노드 실행 중 에러 발생 시
        """
        # 실행 메타데이터 초기화
        self.execution_metadata = []
        
        # 현재 데이터 컨테이너 (노드 간 데이터 전달용)
        # 첫 번째 노드는 initial_inputs를 받음
        current_data = initial_inputs.copy()
        
        # 모든 노드를 순서대로 실행
        for i, node in enumerate(self.nodes):
            print(f"[{self.name}] Executing node {i+1}/{len(self.nodes)}: {node.node_name}")
            
            try:
                # 노드 실행 (execute는 base.py에서 정의한 메서드)
                # execute 내부에서 validate_inputs -> process -> 메타데이터 업데이트
                output = node.execute(current_data)
                
                # 이전 데이터에 새 출력을 병합
                # 예: current_data = {"prompt": "cafe"}
                #     output = {"image": PIL.Image}
                #     -> current_data = {"prompt": "cafe", "image": PIL.Image}
                current_data.update(output)
                
                # 이 노드의 실행 메타데이터 저장
                self.execution_metadata.append(node.get_metadata())
                
            except Exception as e:
                # 에러 발생 시 메타데이터 저장 후 예외 재발생
                self.execution_metadata.append(node.get_metadata())
                
                # 에러 메시지에 워크플로우 정보 추가
                raise Exception(
                    f"Workflow '{self.name}' failed at node {i+1} "
                    f"({node.node_name}): {str(e)}"
                ) from e
        
        # 마지막 노드의 출력 반환
        return current_data
    
    def get_execution_report(self) -> Dict[str, Any]:
        """
        워크플로우 실행 리포트 반환
        
        Returns:
            실행 통계 및 각 노드 메타데이터
            
        Example:
            {
                "workflow_name": "ImageGenerationWorkflow",
                "total_nodes": 3,
                "total_time": 12.5,
                "nodes": [
                    {"node_name": "Text2ImageNode", "execution_time": 10.2, ...},
                    {"node_name": "ResizeNode", "execution_time": 2.3, ...}
                ]
            }
        """
        # 총 실행 시간 계산 (모든 노드의 실행 시간 합)
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
        """모든 노드 제거 (워크플로우 재구성 시 사용)"""
        self.nodes = []
        self.execution_metadata = []
    
    def reset_metadata(self) -> None:
        """모든 노드의 메타데이터 초기화 (재실행 전)"""
        for node in self.nodes:
            node.reset_metadata()
        self.execution_metadata = []
    
    def __repr__(self) -> str:
        """워크플로우 문자열 표현"""
        node_names = [node.node_name for node in self.nodes]
        return f"ImageGenerationWorkflow(nodes={node_names})"