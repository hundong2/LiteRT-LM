"""
LiteRT-LM 학습 예제 08: Gemini-Compatible HTTP API 서버/클라이언트
=================================================================

[이론]
LiteRT-LM은 Gemini API와 호환되는 로컬 HTTP 서버를 제공합니다.
이를 통해 Gemini SDK로 작성된 기존 코드를 수정 없이 로컬 모델로 전환할 수 있습니다.

서버 엔드포인트:
  POST /v1beta/models/{model_id}:generateContent       → 동기 응답
  POST /v1beta/models/{model_id}:streamGenerateContent → 스트리밍 응답

요청 형식 (Gemini API 호환):
  {
    "contents": [{"role": "user", "parts": [{"text": "..."}]}],
    "systemInstruction": {"parts": [{"text": "시스템 프롬프트"}]},
    "tools": [{"functionDeclarations": [...]}]
  }

[CLI로 서버 시작]
  litert-lm serve --host localhost --port 9379

[실전 활용]
- 기존 Gemini API 코드를 로컬 추론으로 교체
- OpenAI 호환 레이어와 조합하여 다양한 클라이언트 지원
- 엣지 서버에서 여러 클라이언트에 모델 서비스 제공
"""

import json
import threading
import urllib.request
from typing import Any
from urllib.error import URLError

# 서버 설정
SERVER_HOST = "localhost"
SERVER_PORT = 9379
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


# ── HTTP 클라이언트 유틸리티 ────────────────────────────────────────────────────

def post_json(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """JSON POST 요청을 보내고 응답을 반환하는 헬퍼 함수."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as e:
        raise ConnectionError(f"서버 연결 실패: {e}. 서버가 실행 중인지 확인하세요.") from e


def stream_post_json(path: str, body: dict[str, Any]):
    """SSE(Server-Sent Events) 스트리밍 응답을 처리하는 헬퍼 함수."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    yield chunk
    except URLError as e:
        raise ConnectionError(f"서버 연결 실패: {e}. 서버가 실행 중인지 확인하세요.") from e


def check_server():
    """서버가 실행 중인지 확인."""
    try:
        urllib.request.urlopen(f"{BASE_URL}/", timeout=2)
        return True
    except Exception:
        return False


# ── 예제 함수들 ──────────────────────────────────────────────────────────────────

def example_basic_generate(model_id: str = "my-model"):
    """기본 generateContent 요청 예제."""
    print("=" * 60)
    print(f"예제 1: generateContent (모델: {model_id})")
    print("=" * 60)

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "한국의 수도는 어디인가요?"}],
            }
        ]
    }

    response = post_json(f"/v1beta/models/{model_id}:generateContent", body)
    print("응답:")
    print(json.dumps(response, ensure_ascii=False, indent=2))

    # 텍스트 추출
    candidates = response.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        print(f"\n텍스트: {text}")


def example_with_system_instruction(model_id: str = "my-model"):
    """시스템 지시문(system instruction) 포함 요청 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 시스템 지시문 포함")
    print("=" * 60)

    body = {
        "systemInstruction": {
            "parts": [{"text": "당신은 짧고 명확하게 답하는 AI입니다. 항상 한 문장으로만 답하세요."}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "지구의 나이는?"}],
            }
        ],
    }

    response = post_json(f"/v1beta/models/{model_id}:generateContent", body)
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    print("응답:", "".join(p.get("text", "") for p in parts))


def example_stream_generate(model_id: str = "my-model"):
    """스트리밍 응답 예제."""
    print("\n" + "=" * 60)
    print("예제 3: streamGenerateContent (스트리밍)")
    print("=" * 60)

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "인공지능의 역사를 간략히 설명해주세요."}],
            }
        ]
    }

    print("스트리밍 응답: ", end="", flush=True)
    for chunk in stream_post_json(
        f"/v1beta/models/{model_id}:streamGenerateContent", body
    ):
        candidates = chunk.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    print(part["text"], end="", flush=True)
    print()


def example_multiturn_chat(model_id: str = "my-model"):
    """멀티턴 대화 예제 — contents에 이전 대화 기록 포함."""
    print("\n" + "=" * 60)
    print("예제 4: 멀티턴 대화")
    print("=" * 60)

    # 대화 기록
    conversation_history = []

    turns = [
        "내 이름은 박민준이야.",
        "내 이름이 뭐라고 했지?",
    ]

    for user_text in turns:
        conversation_history.append({
            "role": "user",
            "parts": [{"text": user_text}],
        })

        body = {"contents": conversation_history}
        response = post_json(f"/v1beta/models/{model_id}:generateContent", body)

        parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        assistant_text = "".join(p.get("text", "") for p in parts)

        print(f"[사용자] {user_text}")
        print(f"[AI] {assistant_text}\n")

        # 어시스턴트 응답을 히스토리에 추가
        conversation_history.append({
            "role": "model",
            "parts": [{"text": assistant_text}],
        })


def example_function_calling_api(model_id: str = "my-model"):
    """HTTP API를 통한 Function Calling 예제."""
    print("\n" + "=" * 60)
    print("예제 5: Function Calling via HTTP API")
    print("=" * 60)

    # 도구 정의 (Gemini API 형식)
    body = {
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "도시의 현재 날씨를 조회합니다.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "날씨를 조회할 도시 이름",
                                }
                            },
                            "required": ["city"],
                        },
                    }
                ]
            }
        ],
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "서울의 날씨는 어때요?"}],
            }
        ],
    }

    response = post_json(f"/v1beta/models/{model_id}:generateContent", body)
    print("응답 (도구 호출 포함):")
    print(json.dumps(response, ensure_ascii=False, indent=2))

    # functionCall이 있으면 클라이언트에서 실행 후 결과 전달
    candidates = response.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                print(f"\n[함수 호출 감지] {fc['name']}({fc.get('args', {})})")
                print("클라이언트에서 실행 후 functionResponse로 결과를 전달해야 합니다.")


def example_start_server_in_background(model_path: str, model_id: str = "study-model"):
    """서버를 백그라운드 스레드에서 시작하는 예제 — 테스트용.

    실제 배포에서는 litert-lm serve CLI 명령을 사용하세요.
    """
    print("=" * 60)
    print("예제 6: 백그라운드 서버 시작 (테스트용)")
    print("=" * 60)

    # litert-lm 모델을 로컬에 등록하고 서버를 실행
    import subprocess
    import time

    # 모델 import (이미 등록되어 있다면 생략)
    subprocess.run(
        ["litert-lm", "import", model_path, model_id],
        check=False,
        capture_output=True,
    )

    print(f"모델 '{model_id}' 등록 완료")
    print(f"서버 시작: litert-lm serve --port {SERVER_PORT}")
    print(f"요청 URL: {BASE_URL}/v1beta/models/{model_id}:generateContent")
    print("\n실제 실행 방법:")
    print("  1. 터미널 1: litert-lm serve --port 9379")
    print("  2. 터미널 2: python 08_serve_client.py")


if __name__ == "__main__":
    print("LiteRT-LM HTTP API 서버/클라이언트 예제\n")
    print("사전 준비:")
    print("  1. litert-lm 설치: pip install litert-lm")
    print("  2. 모델 import: litert-lm import ./model.litertlm my-model")
    print(f"  3. 서버 시작: litert-lm serve --port {SERVER_PORT}")
    print(f"  4. 서버 URL: {BASE_URL}\n")

    # 서버 연결 확인
    MODEL_ID = "my-model"  # 등록한 모델 ID로 변경

    # 서버가 실행 중일 때 (주석 해제)
    # try:
    #     example_basic_generate(MODEL_ID)
    #     example_with_system_instruction(MODEL_ID)
    #     example_stream_generate(MODEL_ID)
    #     example_multiturn_chat(MODEL_ID)
    #     example_function_calling_api(MODEL_ID)
    # except ConnectionError as e:
    #     print(f"오류: {e}")
    #     print(f"먼저 'litert-lm serve --port {SERVER_PORT}' 를 실행하세요.")

    print("예제 코드 구조 확인 완료.")
