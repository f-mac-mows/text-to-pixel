"""
Dataset Diversity & Redundancy Audit
======================================
"Output 토큰 문자열 중복 + Input 문장 구조 편향으로 데이터가 섞여서 문제가 생겼다"는
가설을 정량적으로 검증하기 위한 도구. (input, output) 쌍 데이터셋(jsonl)을 받아
중복도/다양성 지표를 계산한다.

지표 6종:
  1. Output 완전 중복률
  2. Output 근사 중복률 (n-gram Jaccard, 샘플링 추정)
  3. Output gzip 압축률 (반복성 프록시)
  4. Output n-gram 엔트로피
  5. Input 템플릿 다양성 (색상/숫자를 자리표시자로 치환 후 고유 템플릿 비율)
  6. 조건부 Output 다양성 (같은 input 템플릿에 매핑되는 output이 얼마나 다양한가) - 진짜 원인 판별용
"""

import json
import gzip
import math
import re
import random
from collections import Counter, defaultdict


def load_pairs(path):
    """jsonl: {"input": "...", "output": "..."} 형식을 가정. 컬럼명 다르면 여기만 수정."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((obj["input"], obj["output"]))
    return pairs


def split_by_task(pairs):
    """<TASK_GEN>(자연어->DSL)과 <TASK_DESC>(DSL->자연어) prefix로 분리.
    두 태스크는 output 분포가 완전히 다르므로(DSL 토큰 vs 자연어 문장) 섞어서
    감사하면 지표가 왜곡된다."""
    buckets = defaultdict(list)
    for inp, out in pairs:
        m = re.match(r"^<([A-Z_]+)>\s*(.*)", inp)
        if m:
            task, rest = m.group(1), m.group(2)
        else:
            task, rest = "UNKNOWN", inp
        buckets[task].append((rest, out))
    return buckets


def strip_border_boilerplate(s, prefix_pattern=r"^(p16_(000|255)\s+1X\s*)+"):
    """16x16 스프라이트의 상하 여백 행("p16_000 1X" 반복)은 도메인 특성상 당연한 반복이라
    편향 지표를 왜곡시킬 수 있다. 이 함수로 제거한 버전도 같이 비교해볼 수 있게 남겨둔다."""
    return re.sub(prefix_pattern, "", s)


def exact_duplicate_rate(strings):
    total = len(strings)
    unique = len(set(strings))
    return (1 - unique / total if total else 0.0), unique, total


def ngram_set(s, n=3):
    return set(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def near_duplicate_rate(strings, n=3, sample_size=1500, threshold=0.8, seed=42):
    """전수 비교는 O(n^2)라 비싸니 샘플링으로 근사."""
    random.seed(seed)
    sample = random.sample(strings, min(sample_size, len(strings)))
    ngrams = [ngram_set(s, n) for s in sample]
    dup_pairs = 0
    total_pairs = 0
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            total_pairs += 1
            if jaccard(ngrams[i], ngrams[j]) >= threshold:
                dup_pairs += 1
    return dup_pairs / total_pairs if total_pairs else 0.0


def compression_ratio(strings):
    """gzip 압축 후 크기 / 원본 크기. 1.0에 가까울수록 안 눌림(다양함), 낮을수록 반복 심함."""
    joined = "\n".join(strings).encode("utf-8")
    if not joined:
        return 1.0
    compressed = gzip.compress(joined)
    return len(compressed) / len(joined)


def shannon_entropy_of_ngrams(strings, n=2):
    counter = Counter()
    total = 0
    for s in strings:
        for i in range(len(s) - n + 1):
            counter[s[i:i + n]] += 1
            total += 1
    if total == 0:
        return 0.0, 0.0, 0
    ent = -sum((c / total) * math.log2(c / total) for c in counter.values())
    max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1.0
    return ent, ent / max_entropy, len(counter)


def input_template(s):
    """입력 문장을 구조 템플릿으로 정규화 - 색상/숫자를 자리표시자로 치환."""
    s = re.sub(r"\b\d+\b", "<NUM>", s)
    colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
    for c in colors:
        s = re.sub(rf"\b{c}\b", "<COLOR>", s, flags=re.IGNORECASE)
    return s.strip().lower()


def template_diversity(inputs):
    templates = [input_template(s) for s in inputs]
    counter = Counter(templates)
    return (len(counter) / len(templates) if templates else 0.0), counter.most_common(10)


def conditional_output_diversity(pairs, top_n=10):
    """가장 흔한 input 템플릿들 각각에 대해, 매핑되는 output이 얼마나 다양한지 측정.
    이게 낮으면 '입력 구조가 좁아서 출력도 똑같이 찍혀나온다'는 가설의 직접 증거가 된다."""
    template_to_outputs = defaultdict(list)
    for inp, out in pairs:
        template_to_outputs[input_template(inp)].append(out)

    ranked = sorted(template_to_outputs.items(), key=lambda kv: -len(kv[1]))[:top_n]
    results = []
    for tmpl, outs in ranked:
        unique_ratio = len(set(outs)) / len(outs)
        results.append((tmpl, len(outs), unique_ratio))
    return results


def run_audit(pairs):
    inputs = [p[0] for p in pairs]
    outputs = [p[1] for p in pairs]

    print("=" * 70)
    print(f"전체 샘플 수: {len(pairs)}")
    print("=" * 70)

    dup_rate, uniq, total = exact_duplicate_rate(outputs)
    print(f"\n[1] Output 완전 중복률: {dup_rate*100:.1f}%  (고유 {uniq} / 전체 {total})")

    near_dup = near_duplicate_rate(outputs)
    print(f"[2] Output 근사 중복률 (3-gram Jaccard>=0.8, 샘플 추정): {near_dup*100:.1f}%")

    comp_ratio = compression_ratio(outputs)
    print(f"[3] Output gzip 압축률: {comp_ratio:.3f}  (1.0=안 눌림/다양함, 낮을수록 반복 심함)")

    ent, norm_ent, vocab_size = shannon_entropy_of_ngrams(outputs, n=2)
    print(f"[4] Output 2-gram 엔트로피: {ent:.2f} bits (정규화 {norm_ent:.3f}, 고유 2-gram {vocab_size}개)")

    tmpl_ratio, top_templates = template_diversity(inputs)
    print(f"\n[5] Input 템플릿 다양성: {tmpl_ratio*100:.1f}%  (고유 템플릿 수 / 전체 샘플 수)")
    print("    상위 템플릿(빈도순):")
    for t, c in top_templates:
        print(f"      '{t}'  x{c}")

    print(f"\n[6] 조건부 Output 다양성 (흔한 input 템플릿별로 output이 얼마나 다양한가):")
    cond = conditional_output_diversity(pairs)
    for tmpl, count, uniq_ratio in cond:
        flag = "  <- ⚠ 편향 의심 (같은 입력 구조에 output이 몇 종류로만 찍힘)" if uniq_ratio < 0.3 else ""
        print(f"      '{tmpl}' (n={count}): output 고유비율={uniq_ratio*100:.1f}%{flag}")

    print("\n" + "=" * 70)
    print("해석 가이드:")
    print("  - [5]가 낮은데 [6]도 낮다  -> '입력 구조가 좁아서 출력도 같이 좁아졌다'는 가설 지지")
    print("  - [5]는 낮은데 [6]은 높다  -> 입력은 단순해도 출력은 다양함, 문제는 다른 곳")
    print("  - [5]가 높은데 [1]/[2]가 높다 -> 입력은 다양한데 출력만 중복, DSL/증강 설계 문제")


def symmetry_collapse_rate(pairs):
    """generate_all_variants()가 만든 (원본, flipped, rotated) 3종 세트에서,
    실제로 도형이 대칭이라 output이 원본과 똑같이 찍힌 비율을 측정한다.
    prompt가 '...', '... flipped', '... rotated' 세 쌍으로 존재한다고 가정."""
    by_base = defaultdict(dict)
    for inp, out in pairs:
        m = re.match(r"^(.*?)(\s+(flipped|rotated))?$", inp)
        base = m.group(1)
        variant = m.group(3) or "original"
        by_base[base][variant] = out

    total_pairs_checked = 0
    collapsed = 0
    for base, variants in by_base.items():
        if "original" not in variants:
            continue
        orig = variants["original"]
        for variant_name in ("flipped", "rotated"):
            if variant_name in variants:
                total_pairs_checked += 1
                if variants[variant_name] == orig:
                    collapsed += 1

    rate = collapsed / total_pairs_checked if total_pairs_checked else 0.0
    return rate, collapsed, total_pairs_checked


def run_symmetry_audit(pairs):
    print("\n" + "=" * 70)
    print("[증강 대칭 붕괴 감사] flip/rotate variant가 원본과 output이 동일한 비율")
    print("=" * 70)
    rate, collapsed, total = symmetry_collapse_rate(pairs)
    print(f"대칭으로 인해 output이 원본과 동일해진 증강 비율: {rate*100:.1f}%  ({collapsed}/{total})")
    if rate > 0.15:
        print("⚠ generate_all_variants()가 도형 대칭 여부를 검사하지 않고 무조건")
        print("  flip/rotate를 적용하고 있어, 서로 다른 input이 동일한 output에")
        print("  매핑되는 구조적 중복이 의심됩니다 (pixel_data_generator.py 확인).")


def run_audit_by_task(pairs):
    buckets = split_by_task(pairs)
    print(f"발견된 태스크: {[(k, len(v)) for k, v in buckets.items()]}\n")

    for task, sub_pairs in buckets.items():
        print("#" * 70)
        print(f"# TASK = {task}  (n={len(sub_pairs)})")
        print("#" * 70)
        run_audit(sub_pairs)

        if task == "TASK_GEN":
            print("\n[보너스] 상하 여백 boilerplate(p16_000/255 1X 반복) 제거 후 재측정:")
            stripped_outputs = [strip_border_boilerplate(o) for _, o in sub_pairs]
            dup_rate, uniq, total = exact_duplicate_rate(stripped_outputs)
            print(f"  완전 중복률(boilerplate 제거 후): {dup_rate*100:.1f}%  (고유 {uniq}/{total})")
            near_dup = near_duplicate_rate(stripped_outputs)
            print(f"  근사 중복률(boilerplate 제거 후): {near_dup*100:.1f}%")
            run_symmetry_audit(sub_pairs)
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python dataset_audit.py <데이터셋.jsonl>")
        sys.exit(1)
    pairs = load_pairs(sys.argv[1])
    run_audit_by_task(pairs)