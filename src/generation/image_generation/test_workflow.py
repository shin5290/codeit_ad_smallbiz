"""
Workflow Test Script
노드 기반 워크플로우 시스템 테스트

간단한 Text2Image 워크플로우를 실행하여 시스템이 정상 작동하는지 확인
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가 (import 가능하게)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.image_generation.workflow import ImageGenerationWorkflow
from generation.image_generation.nodes.text2image import Text2ImageNode

# 출력 디렉토리 생성
OUTPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_basic_workflow():
    """
    기본 워크플로우 테스트

    Text2ImageNode만 사용하여 간단한 이미지 생성
    """
    print("=" * 60)
    print("TEST 1: Basic Text2Image Workflow")
    print("=" * 60)

    # 1. 워크플로우 생성
    workflow = ImageGenerationWorkflow(name="BasicTest")

    # 2. Text2ImageNode 추가
    workflow.add_node(Text2ImageNode())

    # 3. 입력 데이터 준비
    inputs = {
        "prompt": "cozy cafe interior with wooden furniture",
        "aspect_ratio": "1:1",  # 정사각형
        "industry": "cafe",  # 업종별 스타일 자동 적용
        "num_inference_steps": 30,  # 빠른 테스트용
        "guidance_scale": 7.5,
        "seed": 42  # 재현 가능하게
    }

    print(f"\nInput: {inputs}\n")

    # 4. 워크플로우 실행
    try:
        result = workflow.run(inputs)

        # 5. 결과 확인
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Image size: {result['width']}x{result['height']}")
        print(f"Seed used: {result['seed']}")
        print(f"Image object: {result['image']}")

        # 6. 이미지 저장
        output_path = OUTPUT_DIR / "test_workflow_output_basic.png"
        result['image'].save(output_path)
        print(f"\n✅ Image saved to: {output_path}")

        # 7. 실행 리포트 출력
        print("\n" + "=" * 60)
        print("EXECUTION REPORT:")
        print("=" * 60)
        report = workflow.get_execution_report()
        print(f"Workflow: {report['workflow_name']}")
        print(f"Total nodes: {report['total_nodes']}")
        print(f"Total time: {report['total_time']:.2f}s")
        print("\nNode details:")
        for node_meta in report['nodes']:
            print(f"  - {node_meta['node_name']}: {node_meta['execution_time']:.2f}s ({node_meta['status']})")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False




def test_workflow_with_metadata():
    """
    워크플로우 메타데이터 기능 테스트

    메타데이터 추적, 초기화 등의 기능이 정상 작동하는지 확인
    """
    print("\n\n" + "=" * 60)
    print("TEST 3: Workflow Metadata Features")
    print("=" * 60)

    workflow = ImageGenerationWorkflow(name="MetadataTest")
    node = Text2ImageNode()
    workflow.add_node(node)

    # 첫 번째 실행
    print("\n--- First Run ---")
    result1 = workflow.run({
        "prompt": "simple test image",
        "num_inference_steps": 20,
        "seed": 1
    })

    report1 = workflow.get_execution_report()
    print(f"First run took: {report1['total_time']:.2f}s")
    print(f"Node status: {report1['nodes'][0]['status']}")

    # 메타데이터 초기화
    print("\n--- Resetting Metadata ---")
    workflow.reset_metadata()

    # 두 번째 실행
    print("\n--- Second Run ---")
    result2 = workflow.run({
        "prompt": "another test image",
        "num_inference_steps": 20,
        "seed": 2
    })

    report2 = workflow.get_execution_report()
    print(f"Second run took: {report2['total_time']:.2f}s")
    print(f"Node status: {report2['nodes'][0]['status']}")

    # 노드 언로드 테스트
    print("\n--- Unloading Pipeline ---")
    node.unload_pipeline()
    print("Pipeline unloaded successfully")

    print("\n✅ Metadata test completed")
    return True


def test_style_experiments():
    """
    스타일 실험: Ultra Realistic / Semi Realistic / Anime

    3가지 스타일 × 다양한 주제(인테리어 + 인물)
    각 스타일에 맞는 체크포인트 모델 사용:
    - Ultra Realistic: SG161222/RealVisXL_V4.0
    - Semi Realistic: John6666/bss-equinox-il-semi-realistic-model-v25-sdxl
    - Anime: cagliostrolab/animagine-xl-3.1

    모델은 image_generation/models/ 폴더에 캐싱되며,
    각 생성 완료 후 자동으로 언로드되어 메모리 절약
    """
    print("\n\n" + "=" * 70)
    print("TEST 4: Style Experiments (Ultra Realistic / Semi Realistic / Anime)")
    print("=" * 70)

    # 테스트 케이스 정의
    test_cases = [
        # ===== ULTRA REALISTIC =====
        {
            "style": "ultra_realistic",
            "subject": "bakery_interior",
            "prompt": "ultra realistic photograph of artisan bakery interior, fresh croissants and bread on wooden shelves, glass display case, warm ambient lighting, rustic brick wall, professional food photography, 8K, highly detailed",
            "negative": "cartoon, anime, illustration, people, faces, low quality, blurry",
            "aspect_ratio": "4:3",
            "steps": 40,
            "cfg": 8.0,
            "seed": 1000
        },
        {
            "style": "ultra_realistic",
            "subject": "barista_portrait",
            "prompt": "ultra realistic portrait photograph of professional barista, smiling friendly expression, wearing apron, cafe background with coffee equipment, natural lighting, shallow depth of field, professional portrait photography, 8K",
            "negative": "cartoon, anime, illustration, multiple people, low quality, blurry, deformed face",
            "aspect_ratio": "3:4",
            "steps": 40,
            "cfg": 8.0,
            "seed": 1001
        },
        {
            "style": "ultra_realistic",
            "subject": "hair_salon",
            "prompt": "ultra realistic photograph of modern hair salon interior with styling chairs, large mirrors with LED lights, minimalist design, clean professional atmosphere, bright natural light, 8K",
            "negative": "cartoon, anime, people, low quality, blurry",
            "aspect_ratio": "16:9",
            "steps": 40,
            "cfg": 8.0,
            "seed": 1002
        },

        # ===== SEMI REALISTIC =====
        {
            "style": "semi_realistic",
            "subject": "flower_shop",
            "prompt": "semi realistic flower shop interior, colorful fresh bouquets, wooden displays, natural light, artistic composition, clean painterly style, warm colors",
            "negative": "full anime, cartoon, people, low quality",
            "aspect_ratio": "1:1",
            "steps": 35,
            "cfg": 7.0,
            "seed": 2000
        },
        {
            "style": "semi_realistic",
            "subject": "florist_portrait",
            "prompt": "semi realistic portrait of smiling florist, holding beautiful bouquet, flower shop background, soft painterly style, clean features, pleasant warm atmosphere",
            "negative": "full anime, chibi, cartoon, extreme stylization, low quality",
            "aspect_ratio": "3:4",
            "steps": 35,
            "cfg": 7.0,
            "seed": 2001
        },
        {
            "style": "semi_realistic",
            "subject": "bookstore",
            "prompt": "semi realistic cozy bookstore interior, wooden bookshelves filled with books, reading corner with comfortable armchair, warm ambient lighting, artistic rendering",
            "negative": "full anime, cartoon, people, low quality",
            "aspect_ratio": "4:3",
            "steps": 35,
            "cfg": 7.0,
            "seed": 2002
        },

        # ===== ANIME STYLE =====
        {
            "style": "anime",
            "subject": "cafe_interior",
            "prompt": "anime style illustration of cozy cafe interior, vibrant colors, clean linework, wooden furniture, coffee cups and pastries, warm atmosphere, detailed background art, studio quality",
            "negative": "realistic, photograph, 3D render, low quality, messy",
            "aspect_ratio": "16:9",
            "steps": 30,
            "cfg": 7.5,
            "seed": 3000
        },
        {
            "style": "anime",
            "subject": "barista_character",
            "prompt": "anime style character illustration of cheerful barista, bright expressive eyes, friendly smile, wearing apron, cafe background, clean linework, vibrant colors, professional anime art",
            "negative": "realistic, photograph, western cartoon, low quality",
            "aspect_ratio": "3:4",
            "steps": 30,
            "cfg": 7.5,
            "seed": 3001
        },
        {
            "style": "anime",
            "subject": "baker_character",
            "prompt": "anime style character illustration of friendly baker, chef hat, holding fresh bread, warm smile, bakery background, clean anime art style, vibrant colors, detailed",
            "negative": "realistic, photograph, western style, low quality",
            "aspect_ratio": "3:4",
            "steps": 30,
            "cfg": 7.5,
            "seed": 3002
        },
    ]

    results = []
    all_metadata = []

    # 각 테스트 실행
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_cases)}: {tc['style']} - {tc['subject']}")
        print(f"{'='*70}")
        print(f"Prompt: {tc['prompt'][:80]}...")
        print(f"Settings: {tc['aspect_ratio']}, Steps={tc['steps']}, CFG={tc['cfg']}, Seed={tc['seed']}")

        try:
            # 스타일에 맞는 워크플로우 생성 (LoRA 적용)
            workflow = ImageGenerationWorkflow(name=f"StyleExperiment_{tc['style']}")

            if tc['style'] == 'ultra_realistic':
                # Ultra Realistic: RealVisXL 모델 사용
                print(f"   Using Model: SG161222/RealVisXL_V4.0")
                workflow.add_node(Text2ImageNode(
                    model_id="SG161222/RealVisXL_V4.0",
                    auto_unload=True  # 생성 후 자동 언로드
                ))
            elif tc['style'] == 'anime':
                # Anime: Animagine XL 모델 사용
                print(f"   Using Model: cagliostrolab/animagine-xl-3.1")
                workflow.add_node(Text2ImageNode(
                    model_id="cagliostrolab/animagine-xl-3.1",
                    auto_unload=True
                ))
            else:  # semi_realistic
                # Semi Realistic: John6666/bss-equinox-il-semi-realistic-model-v25-sdxl 사용
                print(f"   Using Model: John6666/bss-equinox-il-semi-realistic-model-v25-sdxl")
                workflow.add_node(Text2ImageNode(
                    model_id="John6666/bss-equinox-il-semi-realistic-model-v25-sdxl",
                    auto_unload=True
                ))

            result = workflow.run({
                "prompt": tc["prompt"],
                "negative_prompt": tc["negative"],
                "style": tc["style"],  # 스타일별 네거티브 프롬프트 자동 적용
                "aspect_ratio": tc["aspect_ratio"],
                "num_inference_steps": tc["steps"],
                "guidance_scale": tc["cfg"],
                "seed": tc["seed"]
            })

            filename = f"{tc['style']}_{tc['subject']}_seed{tc['seed']}.png"
            output_path = OUTPUT_DIR / filename
            result['image'].save(output_path)

            report = workflow.get_execution_report()

            print(f"✅ Success! {result['width']}x{result['height']}")
            print(f"   Time: {report['total_time']:.2f}s")
            print(f"   Saved: {output_path}")

            results.append({
                "style": tc['style'],
                "subject": tc['subject'],
                "filename": filename,
                "success": True,
                "time": report['total_time']
            })

            all_metadata.append({
                "style": tc['style'],
                "subject": tc['subject'],
                "report": report
            })

        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "style": tc['style'],
                "subject": tc['subject'],
                "success": False,
                "error": str(e)
            })

    # 결과 요약
    print(f"\n{'='*70}")
    print("STYLE EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    for style in ["ultra_realistic", "semi_realistic", "anime"]:
        style_results = [r for r in results if r.get("style") == style]
        success = sum(1 for r in style_results if r["success"])
        print(f"\n{style.upper()}:")
        print(f"  Total: {len(style_results)}, Success: {success}")
        for r in style_results:
            status = "✅" if r["success"] else "❌"
            if r["success"]:
                print(f"  {status} {r['subject']} - {r['time']:.2f}s")
            else:
                print(f"  {status} {r['subject']} - {r.get('error', 'Unknown')[:50]}...")

    total_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(results)} tests, {total_success} success, {len(results)-total_success} failed")

    # 메타데이터를 JSON으로 저장
    report_path = OUTPUT_DIR / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "success": total_success,
                "failed": len(results) - total_success,
                "timestamp": datetime.now().isoformat()
            },
            "results": results,
            "metadata": all_metadata
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Report saved: {report_path}")
    print(f"🖼️  Images saved: {OUTPUT_DIR}")

    return total_success == len(results)


if __name__ == "__main__":
    print("\n🚀 Starting Workflow System Tests\n")

    # 테스트 실행
    test_results = []

    # Test 1: 기본 워크플로우
    test_results.append(("Basic Workflow", test_basic_workflow()))

    # Test 2: 메타데이터 기능
    test_results.append(("Metadata Features", test_workflow_with_metadata()))

    # Test 3: 스타일 실험 with LoRA (Ultra Realistic / Semi Realistic / Anime)
    test_results.append(("Style Experiments with LoRA", test_style_experiments()))

    # 전체 결과
    print("\n\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, passed in test_results if passed)
    print(f"\nTotal: {len(test_results)}, Passed: {total_passed}, Failed: {len(test_results) - total_passed}")

    if total_passed == len(test_results):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")
