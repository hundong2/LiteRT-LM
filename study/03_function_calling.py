"""
LiteRT-LM 학습 예제 03: Function Calling (Tool Use)
=====================================================

[이론]
Function Calling은 LLM이 외부 함수를 자율적으로 호출하는 기능입니다.

동작 원리:
  1. 사용자가 tools 목록을 Engine에 등록
  2. 사용자가 질문 전송
  3. 모델이 필요한 함수와 인자를 JSON으로 결정
  4. LiteRT-LM 런타임이 실제 함수를 실행
  5. 결과를 모델에 돌려주어 최종 자연어 응답 생성

두 가지 등록 방식:
  A) Python 함수 직접 전달 — docstring에서 자동으로 스키마 생성
  B) Tool 클래스 상속 — 더 정밀한 스키마 제어

automatic_tool_calling=True  → 런타임이 자동 실행 (기본값)
automatic_tool_calling=False → 모델의 tool_call을 그대로 클라이언트에 반환

[실전 활용]
- 날씨/검색/계산 등 실시간 데이터 조회
- 데이터베이스 CRUD 작업
- 외부 API 호출 에이전트
- 복잡한 다단계 추론(ReAct 패턴)
"""

import json
import math
from collections.abc import Sequence
from typing import Any

import litert_lm
from litert_lm.tools import tool_from_function

MODEL_PATH = "path/to/your/model.litertlm"


# ── 예제 도구 함수들 ─────────────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    """현재 날씨 정보를 조회합니다.

    Args:
        city: 날씨를 조회할 도시 이름 (한국어 또는 영어).
    """
    # 실제 환경에서는 날씨 API(OpenWeatherMap 등)를 호출
    weather_data = {
        "서울": "맑음, 기온 18°C, 습도 45%",
        "부산": "흐림, 기온 22°C, 습도 70%",
        "제주": "비, 기온 20°C, 습도 85%",
        "Seoul": "Clear, 18°C, humidity 45%",
    }
    print(f"  [Tool 실행] get_weather(city='{city}')")
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


def calculate(expression: str) -> str:
    """수학 식을 계산합니다.

    Args:
        expression: 계산할 수식 문자열. 예: '2 + 3 * 4', 'sqrt(16)'.
    """
    print(f"  [Tool 실행] calculate(expression='{expression}')")
    try:
        # 실제 환경에서는 더 안전한 파서 사용 권장
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
        return f"결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"


def search_web(query: str, max_results: int = 3) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색할 키워드 또는 질문.
        max_results: 반환할 최대 결과 수.
    """
    print(f"  [Tool 실행] search_web(query='{query}', max_results={max_results})")
    # 실제 환경에서는 SerpAPI, Tavily 등 사용
    return f"'{query}'에 대한 검색 결과 {max_results}개: [시뮬레이션 결과]"


def get_product(numbers: Sequence[float]) -> float:
    """숫자들의 곱을 계산합니다.

    Args:
        numbers: 곱할 숫자 목록.
    """
    print(f"  [Tool 실행] get_product(numbers={list(numbers)})")
    result = 1.0
    for n in numbers:
        result *= n
    return result


# ── 예제 함수들 ──────────────────────────────────────────────────────────────────

def example_single_tool():
    """단일 함수 도구 등록 예제."""
    print("=" * 60)
    print("예제 1: 단일 도구 — 날씨 조회")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(tools=[get_weather]) as conversation,
    ):
        response = conversation.send_message("서울 날씨가 어때요?")
        print("[최종 응답]", response["content"][0]["text"])


def example_multiple_tools():
    """여러 도구를 동시에 등록하는 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 복수 도구 등록")
    print("=" * 60)

    tools = [get_weather, calculate, search_web]

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(tools=tools) as conversation,
    ):
        # 어떤 도구를 쓸지는 모델이 자동으로 결정
        response = conversation.send_message("서울 날씨와 2의 10승을 알려주세요.")
        print("[최종 응답]", response["content"][0]["text"])


def example_streaming_with_tools():
    """스트리밍 모드에서 도구 호출 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 스트리밍 + 도구 호출")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(tools=[get_product]) as conversation,
    ):
        print("[사용자] 1.5, 2.5, 3.5의 곱은 얼마인가요?\n")
        print("[응답] ", end="", flush=True)

        for chunk in conversation.send_message_async("1.5, 2.5, 3.5의 곱은 얼마인가요?"):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    print(item["text"], end="", flush=True)
        print()


def example_manual_tool_calling():
    """automatic_tool_calling=False — 도구 호출을 수동으로 처리하는 예제.

    서버 API나 원격 실행이 필요한 경우 유용합니다.
    """
    print("\n" + "=" * 60)
    print("예제 4: 수동 도구 호출 (automatic_tool_calling=False)")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(
            tools=[get_weather],
            automatic_tool_calling=False,  # 모델이 tool_call만 반환, 실행은 직접
        ) as conversation,
    ):
        response = conversation.send_message("부산 날씨를 알려줘.")
        print("[원시 응답]", json.dumps(response, ensure_ascii=False, indent=2))

        # tool_calls가 있으면 직접 실행
        tool_calls = response.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name")
            args = func.get("arguments", {})
            print(f"\n[수동 실행] {name}({args})")

            if name == "get_weather":
                result = get_weather(**args)
                print(f"[실행 결과] {result}")


def example_tool_event_handler():
    """ToolEventHandler를 사용하여 도구 호출 전/후를 가로채는 예제."""
    print("\n" + "=" * 60)
    print("예제 5: ToolEventHandler — 도구 승인/응답 후처리")
    print("=" * 60)

    class MyToolHandler(litert_lm.ToolEventHandler):
        """도구 호출을 승인하고, 응답을 로깅하는 핸들러."""

        def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
            func_name = tool_call.get("function", {}).get("name", "unknown")
            print(f"  [Handler] 도구 승인 요청: {func_name}")
            # 특정 도구를 차단하려면 False 반환
            if func_name == "search_web":
                print("  [Handler] search_web 차단!")
                return False
            return True

        def process_tool_response(
            self, tool_response: dict[str, Any]
        ) -> dict[str, Any]:
            print(f"  [Handler] 도구 응답 수신: {tool_response}")
            # 응답을 수정하거나 민감 정보를 제거할 수 있음
            return tool_response

    handler = MyToolHandler()

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(
            tools=[get_weather, search_web],
            tool_event_handler=handler,
        ) as conversation,
    ):
        response = conversation.send_message("서울 날씨 검색해줘.")
        print("[최종 응답]", response["content"][0]["text"])


def example_tool_schema_inspection():
    """tool_from_function()으로 생성된 도구 스키마를 확인하는 예제."""
    print("\n" + "=" * 60)
    print("예제 6: 도구 스키마 자동 생성 확인")
    print("=" * 60)

    tool = tool_from_function(get_weather)
    schema = tool.get_tool_description()
    print("자동 생성된 OpenAPI 스키마:")
    print(json.dumps(schema, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print("LiteRT-LM Function Calling 예제")
    print("주의: 실제 모델 파일이 필요합니다. MODEL_PATH를 수정하세요.\n")

    # 스키마 확인은 모델 없이 실행 가능
    example_tool_schema_inspection()

    # 모델이 있을 때 실행 (주석 해제)
    # example_single_tool()
    # example_multiple_tools()
    # example_streaming_with_tools()
    # example_manual_tool_calling()
    # example_tool_event_handler()
