"""
LiteRT-LM 학습 예제 — 토크나이저 API와 Tokenizer 활용
======================================================

[이론]
Engine은 토크나이저 API를 제공하여 텍스트를 토큰 ID로 변환하거나
토큰 ID를 텍스트로 복원할 수 있습니다.

주요 용도:
  - 입력 길이 사전 계산 (KV Cache 오버플로 방지)
  - 특수 토큰(BOS/EOS) 확인
  - 커스텀 토큰 처리 파이프라인 구축
"""

import litert_lm

MODEL_PATH = "path/to/your/model.litertlm"


def example_tokenize():
    """텍스트를 토큰 ID로 변환하는 예제."""
    print("=" * 60)
    print("예제 1: 텍스트 → 토큰 ID")
    print("=" * 60)

    with litert_lm.Engine(MODEL_PATH) as engine:
        texts = [
            "Hello, World!",
            "안녕하세요, LiteRT-LM!",
            "The quick brown fox jumps over the lazy dog.",
        ]

        for text in texts:
            token_ids = engine.tokenize(text)
            decoded = engine.detokenize(token_ids)
            print(f"입력: {text!r}")
            print(f"토큰 수: {len(token_ids)}")
            print(f"토큰 IDs: {token_ids}")
            print(f"복원: {decoded!r}")
            print()


def example_special_tokens():
    """BOS/EOS 특수 토큰 정보 확인 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 특수 토큰 확인 (BOS/EOS)")
    print("=" * 60)

    with litert_lm.Engine(MODEL_PATH) as engine:
        bos_id = engine.bos_token_id
        eos_ids = engine.eos_token_ids

        print(f"BOS 토큰 ID: {bos_id}")
        print(f"EOS 토큰 시퀀스: {eos_ids}")


def example_token_length_check():
    """입력이 KV Cache를 초과하는지 사전에 확인하는 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 입력 길이 사전 검증")
    print("=" * 60)

    MAX_TOKENS = 2048

    with litert_lm.Engine(MODEL_PATH, max_num_tokens=MAX_TOKENS) as engine:
        prompts = [
            "짧은 입력",
            "A" * 500,   # 중간 길이
            "B" * 5000,  # 너무 긴 입력
        ]

        for prompt in prompts:
            token_ids = engine.tokenize(prompt)
            token_count = len(token_ids)
            status = "✓ OK" if token_count < MAX_TOKENS * 0.9 else "⚠ 경고: 길이 초과 위험"
            print(f"'{prompt[:20]}...' → {token_count} 토큰 [{status}]")


if __name__ == "__main__":
    print("LiteRT-LM 토크나이저 API 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # example_tokenize()
    # example_special_tokens()
    # example_token_length_check()

    print("예제 코드 구조 확인 완료.")
