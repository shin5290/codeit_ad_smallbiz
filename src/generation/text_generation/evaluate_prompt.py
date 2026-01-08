"""
Civitai 25개 프롬프트 심화 분석 기반 평가 시스템 v2.0

새로 추가된 평가 항목:
- 요소 순서 일관성 (Medium → Lighting → Technical)
- 렌즈 중복 체크
- 자연어 비율 (15% 최적)
"""

import re
from typing import Dict, Tuple, List
from dataclasses import dataclass


# ============================================
# 평가 결과 데이터 클래스
# ============================================

@dataclass
class EvaluationResult:
    civitai_compliance: Dict[str, bool]
    compliance_rate: float
    quality_metrics: Dict[str, float]
    avg_quality: float
    details: Dict
    warnings: List[str]


# ============================================
# Evaluator 본체
# ============================================

class CivitaiEnhancedEvaluator:
    """
    Civitai 25개 프롬프트 분석 기반 평가 시스템
    """

    BENCHMARK_STATS = {
        "word_count": {"optimal_range": (26, 41)},
        "comma_count": {"optimal_range": (6, 9)},
        "natural_language_ratio": {"optimal_range": (10, 20)},
        "keyword_repetition": {"optimal_range": (0, 2)},
        "photo_terms": {
            "lighting": 0.96,
            "lens": 0.80,
            "composition": 0.68,
            "aperture": 0.52,
        }
    }

    NATURAL_LANGUAGE_PATTERNS = [
        r'(Professional|Commercial|Editorial)\s+\w+\s+photography\s+of',
        r'\w+\s+(on|in|at|with|during|featuring)\s+',
        r'\w+ing\s+\w+',
    ]

    QUALITY_SPAM = [
        'masterpiece', 'best quality', 'high quality',
        'ultra detailed', 'highly detailed',
        '8k', '4k', '16k', 'award-winning'
    ]

    # ============================================
    # 1단계: Civitai 권장사항
    # ============================================

    def check_civitai_compliance(self, prompt: str, negative: str) -> Dict[str, bool]:
        checks = {}

        # 자연어 구문
        nl_count = sum(1 for p in self.NATURAL_LANGUAGE_PATTERNS if re.search(p, prompt))
        checks['자연어_구문'] = nl_count >= 1

        # Quality spam
        prompt_lower = prompt.lower()
        checks['quality_spam_제거'] = not any(spam in prompt_lower for spam in self.QUALITY_SPAM)

        # Negative 최소화
        neg_keywords = [k.strip() for k in negative.split(',') if k.strip()]
        checks['negative_최소화'] = 5 <= len(neg_keywords) <= 7

        # 사진 용어
        has_lighting = any(t in prompt_lower for t in ['light', 'lighting'])
        has_lens = bool(re.search(r'\d+mm', prompt_lower)) or 'lens' in prompt_lower
        checks['사진_용어'] = has_lighting or has_lens

        return checks

    # ============================================
    # 2단계: 품질 지표
    # ============================================

    def calculate_quality_metrics(self, prompt: str) -> Tuple[Dict[str, float], List[str]]:
        metrics = {}
        warnings = []

        # 길이
        wc = len(prompt.split())
        lo, hi = self.BENCHMARK_STATS['word_count']['optimal_range']
        metrics['길이'] = 100.0 if lo <= wc <= hi else max(0, 100 - abs(wc - (lo if wc < lo else hi)) * 5)
        if wc < lo:
            warnings.append(f"프롬프트가 너무 짧음 ({wc} < {lo})")
        elif wc > hi:
            warnings.append(f"프롬프트가 너무 김 ({wc} > {hi})")

        # 키워드 밸런스
        commas = prompt.count(',')
        lo, hi = self.BENCHMARK_STATS['comma_count']['optimal_range']
        metrics['키워드_밸런스'] = 100.0 if lo <= commas <= hi else max(0, 100 - abs(commas - (lo if commas < lo else hi)) * 10)

        # 자연어 비율
        nl_ratio = self._calculate_nl_ratio(prompt)
        lo, hi = self.BENCHMARK_STATS['natural_language_ratio']['optimal_range']
        if lo <= nl_ratio <= hi:
            metrics['자연어_비율'] = 100.0
        elif nl_ratio < lo:
            metrics['자연어_비율'] = (nl_ratio / lo) * 100
            warnings.append(f"자연어 구문 부족 ({nl_ratio:.0f}% < {lo}%)")
        else:
            metrics['자연어_비율'] = max(0, 100 - (nl_ratio - hi) * 5)
            warnings.append(f"자연어 비율 과다 ({nl_ratio:.0f}% > {hi}%)")

        # 요소 순서
        score, w = self._check_element_order(prompt)
        metrics['요소_순서'] = score
        warnings.extend(w)

        # 렌즈 중복
        score, w = self._check_lens_duplication(prompt)
        metrics['렌즈_사양'] = score
        warnings.extend(w)

        # 사진 용어 완성도
        metrics['사진_용어_완성도'] = self._calculate_photo_completeness(prompt)

        # 키워드 중복
        metrics['키워드_중복_최소화'] = self._check_keyword_repetition(prompt)

        return metrics, warnings

    # ============================================
    # 세부 계산 함수들
    # ============================================

    def _calculate_nl_ratio(self, prompt: str) -> float:
        preps = ['on', 'in', 'at', 'with', 'of', 'from', 'during', 'by']
        verbs = re.findall(r'\w+ing\b', prompt.lower())
        count = sum(1 for p in preps if f' {p} ' in f' {prompt.lower()} ') + len(verbs)
        return (count / len(prompt.split())) * 100 if prompt.split() else 0

    def _check_element_order(self, prompt: str) -> Tuple[float, List[str]]:
        warnings = []
        words = prompt.lower().split()
        total = len(words)

        def pos(keys):
            for i, w in enumerate(words):
                if any(k in w for k in keys):
                    return (i / total) * 100
            return None

        medium = pos(['photograph'])
        lighting = pos(['light', 'lighting', 'lit'])
        technical = pos(['mm', 'lens', 'f/'])

        score = 100.0
        if medium and lighting and medium > lighting:
            score -= 20
            warnings.append("Medium이 Lighting보다 뒤에 위치")
        if lighting and technical and lighting > technical:
            score -= 20
            warnings.append("Lighting이 Technical보다 뒤에 위치")

        return max(0, score), warnings

    def _check_lens_duplication(self, prompt: str) -> Tuple[float, List[str]]:
        lenses = re.findall(r'\d+mm', prompt)
        if len(lenses) > 1:
            return 0.0, [f"여러 렌즈 명시됨: {', '.join(lenses)}"]
        return 100.0, []

    def _calculate_photo_completeness(self, prompt: str) -> float:
        p = prompt.lower()
        terms = {
            'lens': bool(re.search(r'\d+mm', p) or 'lens' in p),
            'aperture': bool(re.search(r'f/\d', p)),
            'lighting': any(t in p for t in ['light', 'lighting']),
            'composition': any(t in p for t in ['shot', 'angle', 'focus', 'depth']),
        }
        score = 0
        for k, v in terms.items():
            if v and self.BENCHMARK_STATS['photo_terms'].get(k, 0) >= 0.7:
                score += 25
        return score

    def _check_keyword_repetition(self, prompt: str) -> float:
        words = [w.lower().strip(',.') for w in prompt.split() if len(w) > 3]
        repeated = sum(1 for w in set(words) if words.count(w) > 1)
        lo, hi = self.BENCHMARK_STATS['keyword_repetition']['optimal_range']
        return 100.0 if repeated <= hi else max(0, 100 - (repeated - hi) * 20)

    # ============================================
    # 종합 평가
    # ============================================

    def evaluate(self, prompt: str, negative: str, industry=None, config=None) -> EvaluationResult:
        config = config or {}

        compliance = self.check_civitai_compliance(prompt, negative)
        compliance_rate = sum(compliance.values()) / len(compliance)

        metrics, warnings = self.calculate_quality_metrics(prompt)
        avg_quality = sum(metrics.values()) / len(metrics)

        details = {
            "word_count": len(prompt.split()),
            "comma_count": prompt.count(','),
            "negative_count": len([k for k in negative.split(',') if k.strip()]),
            "nl_ratio": self._calculate_nl_ratio(prompt)
        }

        return EvaluationResult(
            civitai_compliance=compliance,
            compliance_rate=compliance_rate,
            quality_metrics=metrics,
            avg_quality=avg_quality,
            details=details,
            warnings=warnings
        )

from datetime import datetime
from pathlib import Path


def save_md_report(
    result,
    prompt: str,
    negative: str,
    industry: str | None = None,
    output_dir: str = "reports"
):
    """
    EvaluationResult → Markdown 리포트 저장
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_evaluation_{timestamp}.md"
    filepath = Path(output_dir) / filename

    lines = []

    # =========================
    # Header
    # =========================
    lines.append("# Prompt Evaluation Report")
    lines.append("")
    lines.append(f"- **Generated at**: {timestamp}")
    if industry:
        lines.append(f"- **Industry**: {industry}")
    lines.append("")

    # =========================
    # Prompt
    # =========================
    lines.append("## Prompt")
    lines.append("```")
    lines.append(prompt.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Negative Prompt")
    lines.append("```")
    lines.append(negative.strip())
    lines.append("```")
    lines.append("")

    # =========================
    # Compliance
    # =========================
    lines.append("## Civitai Compliance")
    lines.append("")
    lines.append(f"- **Compliance Rate**: {result.compliance_rate * 100:.0f}%")
    lines.append("")
    for k, v in result.civitai_compliance.items():
        status = "✅" if v else "❌"
        lines.append(f"- {status} {k}")
    lines.append("")

    # =========================
    # Quality Metrics
    # =========================
    lines.append("## Quality Metrics (Benchmark-based)")
    lines.append("")
    lines.append(f"- **Average Quality Score**: {result.avg_quality:.1f}%")
    lines.append("")

    for metric, score in result.quality_metrics.items():
        lines.append(f"- **{metric}**: {score:.1f}%")
    lines.append("")

    # =========================
    # Details
    # =========================
    lines.append("## Details")
    lines.append("")
    for k, v in result.details.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.2f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    # =========================
    # Warnings
    # =========================
    if result.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in result.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")
    else:
        lines.append("## Warnings")
        lines.append("")
        lines.append("- None 🎉")
        lines.append("")

    # =========================
    # Save
    # =========================
    filepath.write_text("\n".join(lines), encoding="utf-8")

    return filepath



# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    evaluator = CivitaiEnhancedEvaluator()

    prompt = """Professional food photography of strawberry latte on marble table,
    minimalist cafe interior with natural light,
    soft window lighting from left,
    overhead shot, warm pastel tones,
    85mm lens, f/1.8 aperture"""

    negative = "cartoon, illustration, painting, low quality, blurry, artificial"

    result = evaluator.evaluate(prompt, negative)

    print("준수율:", f"{result.compliance_rate*100:.0f}%")
    print("평균 품질:", f"{result.avg_quality:.1f}%")
    print("경고:", result.warnings)
