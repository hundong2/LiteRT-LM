"""
LiteRT-LM 학습 예제 05: SamplerConfig — 샘플링 파라미터 제어
=============================================================

[이론]
Language Model의 출력은 다음 토큰 확률 분포에서 샘플링합니다.
SamplerConfig로 샘플링 전략을 세밀하게 제어할 수 있습니다.

파라미터 설명:
  temperature (float ≥ 0):
    - 높을수록(예: 1.5) → 분포가 평탄해져 다양하고 창의적인 출력
    - 낮을수록(예: 0.1) → 분포가 뾰족해져 결정론적 출력
    - 0.0 = greedy decoding (항상 가장 높은 확률 토큰 선택)

  top_k (int > 0):
    - 상위 K개 토큰만 후보군에 포함
    - top_k=1이면 greedy와 동일
    - 일반적으로 top_k=40~100 사용

  top_p (float ∈ [0, 1]):
    - 누적 확률이 p에 도달할 때까지의 토큰만 후보군에 포함
    - top_p=0.9 → 상위 90% 누적 확률 토큰만 사용
    - top_k보다 더 동적으로 후보 수를 조절

  seed (int):
    - 난수 시드. 동일한 seed → 동일한 출력 (재현 가능)

[조합 가이드]
  창의적 글쓰기  : temperature=1.0~1.2, top_p=0.95, top_k=50
  코드 생성      : temperature=0.2~0.4, top_p=0.9,  top_k=40
  사실 기반 Q&A  : temperature=0.1~0.3, top_p=0.85, top_k=20
  재현 가능 테스트: temperature=0.0, seed=42

[실전 활용]
- A/B 테스트로 최적의 파라미터 탐색
- 사용자별 개인화된 톤 설정
- 동일 입력에 대해 다양한 응답 생성
"""

import litert_lm

MODEL_PATH = "path/to/your/model.litertlm"

PROMPT = "짧은 시 한 편을 써주세요. (자연과 관련된 내용으로)"


def example_default_sampling():
    """기본 샘플링 (SamplerConfig 없음)."""
    print("=" * 60)
    print("예제 1: 기본 샘플링 (모델 기본값 사용)")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        response = conversation.send_message(PROMPT)
        print(response["content"][0]["text"])


def example_deterministic():
    """결정론적 출력 — temperature 낮게, seed 고정."""
    print("\n" + "=" * 60)
    print("예제 2: 결정론적 출력 (temperature=0.1, seed=42)")
    print("=" * 60)

    config = litert_lm.SamplerConfig(
        top_k=1,
        temperature=0.1,
        seed=42,
    )

    print("-- 첫 번째 실행 --")
    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(sampler_config=config) as conversation,
    ):
        r1 = conversation.send_message(PROMPT)
        print(r1["content"][0]["text"])

    print("\n-- 두 번째 실행 (같은 결과여야 함) --")
    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(sampler_config=config) as conversation,
    ):
        r2 = conversation.send_message(PROMPT)
        print(r2["content"][0]["text"])

    same = r1["content"][0]["text"] == r2["content"][0]["text"]
    print(f"\n두 결과 동일: {same}")


def example_creative():
    """창의적 출력 — temperature 높게."""
    print("\n" + "=" * 60)
    print("예제 3: 창의적 출력 (temperature=1.2, top_p=0.95)")
    print("=" * 60)

    config = litert_lm.SamplerConfig(
        top_k=50,
        top_p=0.95,
        temperature=1.2,
    )

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(sampler_config=config) as conversation,
    ):
        response = conversation.send_message(PROMPT)
        print(response["content"][0]["text"])


def example_conservative():
    """보수적 출력 — 사실 기반 Q&A에 적합."""
    print("\n" + "=" * 60)
    print("예제 4: 보수적 출력 (temperature=0.3, top_k=20)")
    print("=" * 60)

    config = litert_lm.SamplerConfig(
        top_k=20,
        top_p=0.85,
        temperature=0.3,
    )

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(sampler_config=config) as conversation,
    ):
        response = conversation.send_message("한국의 수도와 면적을 알려주세요.")
        print(response["content"][0]["text"])


def example_compare_temperatures():
    """다양한 temperature 값의 효과를 비교하는 예제."""
    print("\n" + "=" * 60)
    print("예제 5: Temperature 비교")
    print("=" * 60)

    temperatures = [0.1, 0.5, 1.0, 1.5]
    prompt = "하늘은 왜 파란색인가요? 한 문장으로 답해주세요."

    for temp in temperatures:
        config = litert_lm.SamplerConfig(temperature=temp, top_k=40, seed=0)
        with (
            litert_lm.Engine(MODEL_PATH) as engine,
            engine.create_conversation(sampler_config=config) as conversation,
        ):
            response = conversation.send_message(prompt)
            text = response["content"][0]["text"]
            print(f"[temperature={temp}] {text}")


def example_session_sampler():
    """Session API에서도 SamplerConfig를 사용하는 예제."""
    print("\n" + "=" * 60)
    print("예제 6: Session API + SamplerConfig")
    print("=" * 60)

    config = litert_lm.SamplerConfig(
        top_k=40,
        top_p=0.9,
        temperature=0.7,
        seed=123,
    )

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_session(sampler_config=config) as session,
    ):
        session.run_prefill(["봄, 여름, 가을, 겨울 중 가장 좋은 계절은?"])
        responses = session.run_decode()
        print("응답:", responses.texts[0] if responses.texts else "(없음)")


def example_validation_errors():
    """잘못된 파라미터 입력 시 발생하는 에러 예제."""
    print("\n" + "=" * 60)
    print("예제 7: 유효성 검사 에러")
    print("=" * 60)

    test_cases = [
        {"top_k": 0, "desc": "top_k=0 (0 이하는 불가)"},
        {"top_p": 1.5, "desc": "top_p=1.5 (0~1 범위 초과)"},
        {"top_p": -0.1, "desc": "top_p=-0.1 (음수 불가)"},
        {"temperature": -1.0, "desc": "temperature=-1.0 (음수 불가)"},
    ]

    for case in test_cases:
        desc = case.pop("desc")
        try:
            litert_lm.SamplerConfig(**case)
            print(f"  {desc}: 오류 없음 (예상치 못한 결과)")
        except ValueError as e:
            print(f"  {desc}: ValueError — {e}")


if __name__ == "__main__":
    print("LiteRT-LM SamplerConfig 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # 유효성 검사는 모델 없이 실행 가능
    example_validation_errors()

    # 모델이 있을 때 실행 (주석 해제)
    # example_default_sampling()
    # example_deterministic()
    # example_creative()
    # example_conservative()
    # example_compare_temperatures()
    # example_session_sampler()
