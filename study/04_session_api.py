"""
LiteRT-LM 학습 예제 04: Session API (저수준 추론 제어)
======================================================

[이론]
Session API는 Conversation API보다 한 단계 낮은 수준의 인터페이스입니다.
Prefill과 Decode 단계를 직접 제어할 수 있어 고급 사용 시나리오에 활용됩니다.

주요 메서드:
  run_prefill(contents)     : 입력 텍스트를 KV Cache에 적재 (여러 번 호출 가능)
  run_decode()              : 동기 방식으로 전체 응답 생성 → Responses 반환
  run_decode_async()        : 스트리밍 방식으로 응답 생성 → Iterator[Responses] 반환
  run_text_scoring(target)  : 주어진 텍스트의 로그우도(log-likelihood) 점수 계산
  cancel_process()          : 현재 추론 중단

Responses 객체:
  .texts         : 생성된 텍스트 목록 (batch_size만큼)
  .scores        : 각 텍스트의 로그우도 점수
  .token_lengths : 각 텍스트의 토큰 수 (scoring 시 사용)

[실전 활용]
- apply_prompt_template=False: 커스텀 토크나이저 출력을 그대로 전달
- Text Scoring: RAG 문서 선택, 분류, 다중 선택 평가에 활용
- 배치 처리: 여러 프롬프트를 순서대로 Prefill 후 일괄 Decode
"""

import litert_lm

MODEL_PATH = "path/to/your/model.litertlm"


def example_basic_session():
    """기본 Session API 사용 예제."""
    print("=" * 60)
    print("예제 1: 기본 Session API (Prefill + Decode)")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_session() as session,
    ):
        # Prefill: 프롬프트를 KV Cache에 적재
        session.run_prefill(["안녕하세요! 오늘 기분이 어떠세요?"])

        # Decode: KV Cache를 바탕으로 응답 생성
        responses = session.run_decode()

        print("생성된 텍스트:", responses.texts)
        print("로그우도 점수:", responses.scores)


def example_chunked_prefill():
    """Prefill을 여러 청크로 나눠서 입력하는 예제.

    긴 컨텍스트를 메모리 효율적으로 처리하거나,
    시스템 프롬프트와 사용자 프롬프트를 분리해서 처리할 때 유용합니다.
    """
    print("\n" + "=" * 60)
    print("예제 2: 청크별 Prefill")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_session() as session,
    ):
        # 시스템 프롬프트와 사용자 입력을 분리해서 Prefill
        session.run_prefill(["시스템: 당신은 친절한 AI 어시스턴트입니다.\n"])
        session.run_prefill(["사용자: Python에서 리스트와 튜플의 차이점은?\n"])
        session.run_prefill(["어시스턴트: "])

        # Decode
        responses = session.run_decode()
        print("응답:", responses.texts[0] if responses.texts else "(없음)")


def example_async_decode():
    """비동기(스트리밍) Decode 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 비동기 Decode (스트리밍)")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_session() as session,
    ):
        session.run_prefill(["딥러닝의 역사를 간략히 설명해주세요."])

        print("스트리밍 응답: ", end="", flush=True)
        full_text = []

        # run_decode_async()는 Iterator[Responses]를 반환
        for chunk in session.run_decode_async():
            for text in chunk.texts:
                print(text, end="", flush=True)
                full_text.append(text)

        print(f"\n총 청크 수: {len(full_text)}")


def example_text_scoring():
    """텍스트 스코어링 예제 — 모델이 각 응답에 얼마나 동의하는지 측정."""
    print("\n" + "=" * 60)
    print("예제 4: 텍스트 스코어링 (Log-Likelihood)")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_session() as session,
    ):
        # 컨텍스트 Prefill
        session.run_prefill(["다음 중 파이썬의 특징은 무엇인가요?"])

        # 여러 후보 텍스트에 대한 점수 계산
        candidates = [
            "동적 타입 언어입니다.",
            "정적 타입 언어입니다.",
            "컴파일 언어입니다.",
        ]

        # store_token_lengths=True이면 토큰 수도 함께 반환
        scoring = session.run_text_scoring(candidates, store_token_lengths=True)

        print("각 후보 텍스트의 로그우도 점수:")
        for text, score, tlen in zip(candidates, scoring.scores, scoring.token_lengths):
            print(f"  [{score:+.4f}] ({tlen} tokens) {text}")

        best_idx = scoring.scores.index(max(scoring.scores))
        print(f"\n가장 높은 점수의 후보: '{candidates[best_idx]}'")


def example_text_scoring_mcq():
    """다중 선택 질문(MCQ) 평가 예제 — LLM 평가에 자주 사용되는 패턴."""
    print("\n" + "=" * 60)
    print("예제 5: MCQ 평가 (Text Scoring 응용)")
    print("=" * 60)

    questions = [
        {
            "question": "지구에서 가장 큰 대륙은?",
            "choices": ["아프리카", "아시아", "북아메리카", "유럽"],
            "answer_idx": 1,
        },
        {
            "question": "물의 화학식은?",
            "choices": ["CO2", "H2O", "NaCl", "O2"],
            "answer_idx": 1,
        },
    ]

    with litert_lm.Engine(MODEL_PATH) as engine:
        correct = 0
        for q in questions:
            with engine.create_session() as session:
                # 질문을 Prefill
                session.run_prefill([f"질문: {q['question']}\n정답: "])

                # 각 선택지 점수 계산
                scoring = session.run_text_scoring(q["choices"])
                predicted_idx = scoring.scores.index(max(scoring.scores))

            is_correct = predicted_idx == q["answer_idx"]
            correct += int(is_correct)
            status = "✓" if is_correct else "✗"
            print(
                f"  {status} Q: {q['question']}\n"
                f"     예측: {q['choices'][predicted_idx]} | "
                f"정답: {q['choices'][q['answer_idx']]}"
            )

    print(f"\n정확도: {correct}/{len(questions)}")


def example_no_template_session():
    """apply_prompt_template=False — 프롬프트 템플릿 없이 raw 입력 처리."""
    print("\n" + "=" * 60)
    print("예제 6: 템플릿 없는 Session (raw 입력)")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        # 모델의 chat template을 적용하지 않고 입력을 그대로 전달
        engine.create_session(apply_prompt_template=False) as session,
    ):
        # 직접 모델이 기대하는 형식으로 프롬프트 구성
        raw_prompt = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        session.run_prefill([raw_prompt])
        responses = session.run_decode()
        print("Raw 응답:", responses.texts[0] if responses.texts else "(없음)")


def example_cancel_decode():
    """Decode 도중 취소하는 예제."""
    print("\n" + "=" * 60)
    print("예제 7: Decode 취소")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_session() as session,
    ):
        session.run_prefill(["우주의 역사에 대해 상세히 설명해주세요."])

        collected = []
        for chunk in session.run_decode_async():
            collected.extend(chunk.texts)
            # 3개 청크 후 취소
            if len(collected) >= 3:
                session.cancel_process()
                break

        print(f"취소 전 수집된 텍스트: {''.join(collected)}")


if __name__ == "__main__":
    print("LiteRT-LM Session API 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # MODEL_PATH 설정 후 주석 해제
    # example_basic_session()
    # example_chunked_prefill()
    # example_async_decode()
    # example_text_scoring()
    # example_text_scoring_mcq()
    # example_no_template_session()
    # example_cancel_decode()

    print("예제 코드 구조 확인 완료.")
