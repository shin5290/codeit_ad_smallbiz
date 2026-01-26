"""
SDXL 일러스트 공지 포스터 프롬프트 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (import 가능하게)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.image_generation.workflow import ImageGenerationWorkflow
from src.generation.image_generation.nodes.text2image import Text2ImageNode

# 출력 디렉토리 생성
OUTPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_poster_prompt():
    """
    SDXL 프롬프트 테스트: 일러스트 공지 포스터용
    """
    print("\n" + "=" * 60)
    print("TEST: SDXL Illustrated Announcement Poster Prompt")
    print("=" * 60)

    workflow = ImageGenerationWorkflow(name="PosterPromptTest")
    workflow.add_node(Text2ImageNode())

    prompt = (
        "cute illustrated announcement poster design, "
        "hand drawn flat illustration style, "
        "soft yellow lined paper background, "
        "notebook paper texture, "
        "rounded paper edges at the top, "
        "cute white cat illustration waving at the bottom center, "
        "simple black cat silhouette icon on the right, "
        "small flower doodle and leaf illustrations, "
        "soft green bushes at the bottom, "
        "pastel green and yellow color palette, "
        "friendly and warm mood, "
        "large bold Korean headline typography space in the center, "
        "clear text hierarchy for title and date, "
        "simple layout with plenty of empty space, "
        "children book illustration style, "
        "korean stationery design 느낌, "
        "clean vector illustration, "
        "flat design poster, "
        "designed graphic poster, "
        "not photography, not realistic, "
        "high quality illustration"
    )

    negative_prompt = (
        "photo realistic, "
        "realistic lighting, "
        "3d render, "
        "depth of field, "
        "cinematic, "
        "complex background, "
        "busy composition, "
        "shadow heavy, "
        "text artifacts, "
        "distorted text, "
        "watermark, "
        "logo"
    )

    inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": "3:4",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seed": 4242
    }

    print(f"\nInput: {inputs}\n")

    try:
        result = workflow.run(inputs)

        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Image size: {result['width']}x{result['height']}")
        print(f"Seed used: {result['seed']}")
        print(f"Image object: {result['image']}")

        output_path = OUTPUT_DIR / "test_poster_prompt.png"
        result["image"].save(output_path)
        print(f"\n✅ Image saved to: {output_path}")

        report = workflow.get_execution_report()
        print("\n" + "=" * 60)
        print("EXECUTION REPORT:")
        print("=" * 60)
        print(f"Workflow: {report['workflow_name']}")
        print(f"Total nodes: {report['total_nodes']}")
        print(f"Total time: {report['total_time']:.2f}s")
        for node_meta in report["nodes"]:
            print(f"  - {node_meta['node_name']}: {node_meta['execution_time']:.2f}s ({node_meta['status']})")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🚀 Starting Poster Prompt Test\n")
    passed = test_poster_prompt()
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}: Poster Prompt")
