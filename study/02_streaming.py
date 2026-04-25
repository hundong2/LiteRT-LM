"""
LiteRT-LM 학습 예제 02: 스트리밍(Streaming) 응답
==================================================

[이론]
- send_message_async()는 이터레이터를 반환하며, 토큰이 생성될 때마다 청크를 yield합니다.
- 각 청크는 send_message()와 동일한 형식이지만 부분 텍스트를 포함합니다.
- 스트리밍은 체감 응답속도(latency)를 크게 개선합니다 — 첫 토큰이 나오자마자 표시 가능.
- cancel_process()로 진행 중인 추론을 중단할 수 있습니다.

[실전 활용]
- 실시간 채팅 UI (ChatGPT 스타일의 타이핑 효과)
- 긴 텍스트 생성 시 사용자 경험 향상
- 특정 조건에서 생성을 조기 종료해야 하는 경우
"""

import sys
import time

import litert_lm

MODEL_PATH = "path/to/your/model.litertlm"


def example_basic_streaming():
    """기본 스트리밍 출력 예제 — 타이핑 효과."""
    print("=" * 60)
    print("예제 1: 기본 스트리밍")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        print("[사용자] 한국의 4계절에 대해 설명해주세요.\n")
        print("[어시스턴트] ", end="", flush=True)

        # send_message_async()는 Iterator[Mapping[str, Any]]를 반환
        for chunk in conversation.send_message_async("한국의 4계절에 대해 설명해주세요."):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    # 청크 텍스트를 즉시 출력 (flush=True로 버퍼링 방지)
                    print(item["text"], end="", flush=True)

        print()  # 줄바꿈


def example_streaming_with_timing():
    """스트리밍 중 각 청크의 도착 시간을 측정하는 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 스트리밍 타이밍 측정")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        start_time = time.time()
        first_token_time = None
        token_count = 0
        full_text = []

        for chunk in conversation.send_message_async("머신러닝이란 무엇인가요?"):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    text = item["text"]
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        print(f"[첫 토큰 도착 시간: {ttft:.3f}초]")
                    full_text.append(text)
                    token_count += 1

        total_time = time.time() - start_time
        print("\n" + "─" * 40)
        print(f"전체 응답: {''.join(full_text)}")
        print(f"총 청크 수: {token_count}")
        print(f"총 소요 시간: {total_time:.3f}초")


def example_streaming_cancel():
    """스트리밍 도중 취소(cancel)하는 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 스트리밍 취소")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        print("[사용자] 1부터 100까지 세어보세요.\n")
        print("[어시스턴트] ", end="", flush=True)

        collected = []
        for chunk in conversation.send_message_async("1부터 100까지 세어보세요."):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    text = item["text"]
                    print(text, end="", flush=True)
                    collected.append(text)

            # 5개 청크 후 취소
            if len(collected) >= 5:
                print("\n[취소 요청!]")
                conversation.cancel_process()
                break

        print(f"\n[수집된 청크 수: {len(collected)}]")


def example_colored_streaming():
    """색상 코드를 사용한 스트리밍 출력 — 터미널 UI 예제."""
    print("\n" + "=" * 60)
    print("예제 4: 색상 터미널 출력")
    print("=" * 60)

    # ANSI 색상 코드
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        user_input = "Python과 JavaScript의 차이점은?"
        print(f"{CYAN}[사용자]{RESET} {user_input}\n")
        print(f"{YELLOW}[어시스턴트]{RESET} ", end="", flush=True)

        for chunk in conversation.send_message_async(user_input):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    print(f"{YELLOW}{item['text']}{RESET}", end="", flush=True)

        print()


def example_collect_full_response():
    """스트리밍으로 받은 청크를 모아서 전체 응답을 완성하는 패턴."""
    print("\n" + "=" * 60)
    print("예제 5: 청크 수집 후 전체 응답 사용")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation() as conversation,
    ):
        chunks = []
        for chunk in conversation.send_message_async("JSON이란 무엇인가요?"):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    chunks.append(item["text"])

        full_response = "".join(chunks)
        print(f"전체 응답 ({len(full_response)}자):")
        print(full_response)


def example_interactive_streaming_chat():
    """대화형 스트리밍 채팅 루프 예제 — 실제 챗봇 구현 패턴."""
    print("\n" + "=" * 60)
    print("예제 6: 대화형 스트리밍 챗봇 (Ctrl+C로 종료)")
    print("=" * 60)

    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(
            messages=[{"role": "system", "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다."}]
        ) as conversation,
    ):
        print("대화를 시작하세요. 'quit' 또는 'exit'를 입력하면 종료됩니다.\n")
        while True:
            try:
                user_input = input(f"{GREEN}나: {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n대화를 종료합니다.")
                break

            if user_input.lower() in ("quit", "exit", "종료"):
                print("대화를 종료합니다.")
                break

            if not user_input:
                continue

            print(f"{YELLOW}AI: {RESET}", end="", flush=True)
            for chunk in conversation.send_message_async(user_input):
                content_list = chunk.get("content", [])
                for item in content_list:
                    if item.get("type") == "text":
                        print(f"{YELLOW}{item['text']}{RESET}", end="", flush=True)
            print()


if __name__ == "__main__":
    print("LiteRT-LM 스트리밍 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # 실행할 예제를 선택하세요 (MODEL_PATH 설정 후 주석 해제)
    # example_basic_streaming()
    # example_streaming_with_timing()
    # example_streaming_cancel()
    # example_colored_streaming()
    # example_collect_full_response()
    # example_interactive_streaming_chat()

    print("예제 코드 구조 확인 완료. MODEL_PATH 설정 후 각 함수를 호출하세요.")
