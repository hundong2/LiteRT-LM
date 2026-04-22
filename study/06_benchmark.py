"""
LiteRT-LM 학습 예제 06: 성능 벤치마크
=======================================

[이론]
LiteRT-LM은 BenchmarkInfo를 통해 다음 지표를 측정합니다:

  init_time_in_second           : 엔진 초기화 및 모델 로딩 시간
  time_to_first_token_in_second : TTFT(Time To First Token) — 첫 토큰까지 걸리는 시간
  last_prefill_token_count      : 마지막 Prefill 단계의 토큰 수
  last_prefill_tokens_per_second: Prefill 처리량 (tokens/sec)
  last_decode_token_count       : 마지막 Decode 단계의 토큰 수
  last_decode_tokens_per_second : Decode 처리량 (tokens/sec)

[핵심 지표]
  - TTFT: 사용자 체감 응답속도에 직결. 짧을수록 좋음.
  - Decode TPS (Tokens Per Second): 실시간 스트리밍 품질. 30 TPS 이상이면 자연스럽게 느껴짐.
  - Prefill TPS: 긴 문서 처리 성능.

[실전 활용]
  - 배포 전 하드웨어별 성능 측정
  - CPU vs GPU 백엔드 비교
  - Speculative Decoding 효과 측정
  - 다양한 prefill/decode 토큰 수로 병목 구간 파악
"""

import time

import litert_lm

MODEL_PATH = "path/to/your/model.litertlm"


def example_basic_benchmark():
    """기본 벤치마크 실행 예제."""
    print("=" * 60)
    print("예제 1: 기본 벤치마크")
    print("=" * 60)

    benchmark = litert_lm.Benchmark(
        MODEL_PATH,
        backend=litert_lm.Backend.CPU,
        prefill_tokens=256,
        decode_tokens=256,
    )

    result = benchmark.run()
    _print_benchmark_result(result)


def example_benchmark_comparison():
    """다양한 설정으로 벤치마크를 비교하는 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 설정별 벤치마크 비교")
    print("=" * 60)

    configs = [
        {"prefill_tokens": 64,  "decode_tokens": 64,  "label": "짧은 시퀀스"},
        {"prefill_tokens": 256, "decode_tokens": 256, "label": "중간 시퀀스"},
        {"prefill_tokens": 512, "decode_tokens": 512, "label": "긴 시퀀스"},
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        benchmark = litert_lm.Benchmark(
            MODEL_PATH,
            backend=litert_lm.Backend.CPU,
            **cfg,
        )
        result = benchmark.run()
        results.append((label, cfg, result))
        print(f"\n[{label}] prefill={cfg['prefill_tokens']}, decode={cfg['decode_tokens']}")
        _print_benchmark_result(result)

    # 요약 테이블
    print("\n" + "─" * 70)
    print(f"{'설정':<15} {'TTFT(s)':<12} {'Prefill TPS':<15} {'Decode TPS':<15}")
    print("─" * 70)
    for label, _, r in results:
        print(
            f"{label:<15} "
            f"{r.time_to_first_token_in_second:<12.3f} "
            f"{r.last_prefill_tokens_per_second:<15.1f} "
            f"{r.last_decode_tokens_per_second:<15.1f}"
        )


def example_speculative_decoding_benchmark():
    """Speculative Decoding 활성화/비활성화 벤치마크 비교."""
    print("\n" + "=" * 60)
    print("예제 3: Speculative Decoding 성능 비교")
    print("=" * 60)

    # Speculative Decoding: Draft 모델이 여러 토큰을 미리 예측하고
    # 메인 모델이 병렬로 검증. 품질 유지하면서 속도 향상.
    for sd_enabled, label in [(False, "비활성화"), (True, "활성화")]:
        print(f"\nSpeculative Decoding {label}:")
        try:
            benchmark = litert_lm.Benchmark(
                MODEL_PATH,
                backend=litert_lm.Backend.CPU,
                prefill_tokens=128,
                decode_tokens=256,
                enable_speculative_decoding=sd_enabled,
            )
            result = benchmark.run()
            _print_benchmark_result(result)
        except Exception as e:
            print(f"  오류: {e}")


def example_manual_timing():
    """Conversation API를 사용한 수동 타이밍 측정 예제."""
    print("\n" + "=" * 60)
    print("예제 4: 수동 타이밍 측정 (Conversation API)")
    print("=" * 60)

    with litert_lm.Engine(MODEL_PATH) as engine:
        # 엔진 초기화 타이밍
        init_start = time.perf_counter()
        with engine.create_conversation() as conversation:
            init_time = time.perf_counter() - init_start

            # 추론 타이밍
            infer_start = time.perf_counter()
            first_token_time = None
            tokens = []

            for chunk in conversation.send_message_async(
                "인공지능의 미래에 대해 설명해주세요."
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                content_list = chunk.get("content", [])
                for item in content_list:
                    if item.get("type") == "text":
                        tokens.append(item["text"])

            total_time = time.perf_counter() - infer_start
            ttft = (first_token_time - infer_start) if first_token_time else None

            print(f"대화 초기화 시간: {init_time:.3f}s")
            print(f"첫 토큰 시간(TTFT): {ttft:.3f}s" if ttft else "TTFT: N/A")
            print(f"총 추론 시간: {total_time:.3f}s")
            print(f"총 토큰 청크: {len(tokens)}")
            if total_time > 0 and len(tokens) > 0:
                print(f"평균 TPS(청크 기준): {len(tokens)/total_time:.1f}")


def example_memory_efficiency():
    """max_num_tokens 설정이 성능에 미치는 영향 측정."""
    print("\n" + "=" * 60)
    print("예제 5: KV Cache 크기별 성능 측정")
    print("=" * 60)

    # KV Cache가 크면 긴 대화 가능하지만 메모리와 초기화 시간 증가
    token_sizes = [512, 1024, 2048]

    for max_tokens in token_sizes:
        print(f"\n[max_num_tokens={max_tokens}]")
        start = time.perf_counter()
        benchmark = litert_lm.Benchmark(
            MODEL_PATH,
            backend=litert_lm.Backend.CPU,
            prefill_tokens=64,
            decode_tokens=64,
        )
        result = benchmark.run()
        elapsed = time.perf_counter() - start
        print(f"  초기화 시간: {result.init_time_in_second:.3f}s")
        print(f"  Decode TPS: {result.last_decode_tokens_per_second:.1f}")
        print(f"  전체 측정 시간: {elapsed:.3f}s")


def _print_benchmark_result(result: litert_lm.BenchmarkInfo):
    """BenchmarkInfo를 보기 좋게 출력하는 헬퍼 함수."""
    print(f"  초기화 시간         : {result.init_time_in_second:.3f}s")
    print(f"  첫 토큰 시간 (TTFT) : {result.time_to_first_token_in_second:.3f}s")
    print(f"  Prefill 토큰 수     : {result.last_prefill_token_count}")
    print(f"  Prefill 속도        : {result.last_prefill_tokens_per_second:.1f} tokens/sec")
    print(f"  Decode 토큰 수      : {result.last_decode_token_count}")
    print(f"  Decode 속도         : {result.last_decode_tokens_per_second:.1f} tokens/sec")


if __name__ == "__main__":
    print("LiteRT-LM 벤치마크 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # MODEL_PATH 설정 후 주석 해제
    # example_basic_benchmark()
    # example_benchmark_comparison()
    # example_speculative_decoding_benchmark()
    # example_manual_timing()
    # example_memory_efficiency()

    print("예제 코드 구조 확인 완료.")
