"""
LiteRT-LM 학습 예제 01: Engine과 Conversation 기본 사용법
============================================================

[이론]
- Engine: 모델 파일(.litertlm)을 로드하고 추론을 담당하는 핵심 객체
- Conversation: 멀티턴 대화 컨텍스트를 관리. send_message()로 메시지를 보내고 응답을 받음
- with 문(컨텍스트 매니저)으로 리소스를 자동 해제함

[응답 형식]
  {
    "role": "assistant",
    "content": [{"type": "text", "text": "모델 응답 텍스트"}]
  }

[실전 활용]
- 챗봇, Q&A 시스템, 요약 도구 등 기본 텍스트 생성 태스크에 사용
"""

import litert_lm

# ── 모델 경로 설정 ──────────────────────────────────────────────────────────────
# 실제 사용 시 HuggingFace에서 다운로드한 .litertlm 파일 경로로 교체하세요.
# 예: "~/.cache/litert-lm/gemma-3n-E2B-it-int4/gemma-3n-E2B-it-int4.litertlm"
MODEL_PATH = "path/to/your/model.litertlm"


def example_basic_chat():
    """가장 단순한 단발성 대화 예제."""
    print("=" * 60)
    print("예제 1: 기본 단발성 대화")
    print("=" * 60)

    # Engine과 Conversation을 with 문으로 관리 (리소스 자동 해제)
    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        # 문자열로 직접 메시지 전송 (편의 기능)
        response = conversation.send_message("한국의 수도는 어디인가요?")

        # 응답 구조: {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        print("역할:", response["role"])
        print("응답:", response["content"][0]["text"])


def example_multiturn_chat():
    """멀티턴(multi-turn) 대화 예제 — 대화 히스토리가 유지됩니다."""
    print("\n" + "=" * 60)
    print("예제 2: 멀티턴 대화")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        turns = [
            "내 이름은 김철수야.",
            "내 이름이 뭐라고 했지?",          # 이전 컨텍스트를 기억하는지 확인
            "그럼 한 문장으로 날 소개해줘.",
        ]

        for user_text in turns:
            print(f"\n[사용자] {user_text}")
            response = conversation.send_message(user_text)
            print(f"[어시스턴트] {response['content'][0]['text']}")


def example_system_prompt():
    """시스템 프롬프트로 모델의 역할/페르소나를 설정하는 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 시스템 프롬프트 설정")
    print("=" * 60)

    system_message = {
        "role": "system",
        "content": (
            "당신은 친절한 한국어 요리 전문가입니다. "
            "항상 재료와 조리 단계를 명확하게 설명해주세요."
        ),
    }

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        # messages 파라미터로 대화 맥락(시스템 메시지 포함)을 미리 설정
        engine.create_conversation(messages=[system_message]) as conversation,
    ):
        response = conversation.send_message("간단한 김치찌개 레시피를 알려주세요.")
        print("[어시스턴트]", response["content"][0]["text"])


def example_dict_message():
    """딕셔너리 형식의 메시지를 직접 전달하는 예제."""
    print("\n" + "=" * 60)
    print("예제 4: 딕셔너리 형식 메시지")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        # role과 content를 명시적으로 지정
        user_message = {
            "role": "user",
            "content": "Python의 리스트 컴프리헨션을 예시와 함께 설명해주세요.",
        }
        response = conversation.send_message(user_message)
        print("[어시스턴트]", response["content"][0]["text"])


def example_backend_selection():
    """CPU/GPU 백엔드 선택 예제."""
    print("\n" + "=" * 60)
    print("예제 5: 백엔드 선택 (CPU vs GPU)")
    print("=" * 60)

    # CPU 백엔드 (기본값, 모든 디바이스 지원)
    with (
        litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU) as engine,
        engine.create_conversation() as conversation,
    ):
        response = conversation.send_message("CPU 백엔드로 실행 중입니다.")
        print("[CPU 응답]", response["content"][0]["text"])

    # GPU 백엔드 (GPU가 있는 디바이스에서 더 빠름)
    # with (
    #     litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.GPU) as engine,
    #     engine.create_conversation() as conversation,
    # ):
    #     response = conversation.send_message("GPU 백엔드로 실행 중입니다.")
    #     print("[GPU 응답]", response["content"][0]["text"])


def example_max_tokens():
    """KV Cache 최대 토큰 수를 설정하는 예제."""
    print("\n" + "=" * 60)
    print("예제 6: KV Cache 토큰 수 제한")
    print("=" * 60)

    # max_num_tokens: KV Cache 크기. 긴 대화를 지원하려면 늘려야 하지만 메모리 사용량 증가
    # 엣지 디바이스에서 메모리 부족 시 낮춰야 할 수 있음
    with (
        litert_lm.Engine(
            MODEL_PATH,
            backend=litert_lm.Backend.CPU,
            max_num_tokens=2048,    # 기본값보다 늘리거나 줄일 수 있음
        ) as engine,
        engine.create_conversation() as conversation,
    ):
        response = conversation.send_message("KV Cache 크기를 2048로 설정했습니다.")
        print("[응답]", response["content"][0]["text"])


if __name__ == "__main__":
    print("LiteRT-LM 기본 사용법 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # 아래 주석을 해제하고 MODEL_PATH를 설정하면 실제 실행됩니다.
    # example_basic_chat()
    # example_multiturn_chat()
    # example_system_prompt()
    # example_dict_message()
    # example_backend_selection()
    # example_max_tokens()

    print("예제 코드 구조 확인 완료. MODEL_PATH 설정 후 각 함수를 호출하세요.")
