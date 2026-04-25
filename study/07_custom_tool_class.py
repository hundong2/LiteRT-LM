"""
LiteRT-LM 학습 예제 07: 커스텀 Tool 클래스
==========================================

[이론]
LiteRT-LM의 Tool 인터페이스는 두 가지 방식으로 구현할 수 있습니다:

  방식 A: tool_from_function(func) — Python 함수를 자동으로 Tool로 변환
    - 장점: 간단하고 빠름
    - 단점: 스키마 제어가 제한적

  방식 B: Tool 추상 클래스 상속 — 완전한 제어
    - get_tool_description() : OpenAPI 스키마를 직접 정의
    - execute(param)         : 실행 로직 구현
    - 장점: 복잡한 타입, enum, nested 객체 등 지원
    - 단점: 더 많은 코드 필요

OpenAPI 함수 스키마 형식:
  {
    "type": "function",
    "function": {
      "name": "함수명",
      "description": "함수 설명",
      "parameters": {
        "type": "object",
        "properties": {
          "param1": {"type": "string", "description": "파라미터 설명"},
          "param2": {"type": "integer", "enum": [1, 2, 3]}
        },
        "required": ["param1"]
      }
    }
  }

[실전 활용]
- 외부 API 클라이언트 래핑
- 데이터베이스 커넥터
- 보안 검증이 필요한 도구
- enum/nested 스키마가 필요한 도구
"""

import json
import math
import random
import sqlite3
import tempfile
from collections.abc import Mapping
from typing import Any

import litert_lm
from litert_lm import Tool

MODEL_PATH = "path/to/your/model.litertlm"


# ── 커스텀 Tool 구현 예시들 ──────────────────────────────────────────────────────

class TemperatureConverterTool(Tool):
    """섭씨/화씨/켈빈 온도 변환 도구 — enum 파라미터 사용 예시."""

    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "convert_temperature",
                "description": "온도 단위를 변환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "변환할 온도 값",
                        },
                        "from_unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit", "kelvin"],
                            "description": "변환 원본 단위",
                        },
                        "to_unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit", "kelvin"],
                            "description": "변환 목표 단위",
                        },
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
            },
        }

    def execute(self, param: Mapping[str, Any]) -> Any:
        value = float(param["value"])
        from_unit = param["from_unit"]
        to_unit = param["to_unit"]

        print(f"  [Tool] convert_temperature({value} {from_unit} → {to_unit})")

        # 모두 켈빈으로 변환
        if from_unit == "celsius":
            kelvin = value + 273.15
        elif from_unit == "fahrenheit":
            kelvin = (value - 32) * 5 / 9 + 273.15
        else:
            kelvin = value

        # 켈빈에서 목표 단위로 변환
        if to_unit == "celsius":
            result = kelvin - 273.15
        elif to_unit == "fahrenheit":
            result = (kelvin - 273.15) * 9 / 5 + 32
        else:
            result = kelvin

        return f"{value}{from_unit[0].upper()} = {result:.2f}{to_unit[0].upper()}"


class DatabaseQueryTool(Tool):
    """SQLite 데이터베이스 조회 도구 — 실제 DB 연동 예시."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "SQL SELECT 쿼리를 실행하고 결과를 반환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "실행할 SQL SELECT 쿼리",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, param: Mapping[str, Any]) -> Any:
        query = param["query"]
        print(f"  [Tool] query_database('{query}')")

        # 보안: SELECT만 허용
        if not query.strip().upper().startswith("SELECT"):
            return "오류: SELECT 쿼리만 허용됩니다."

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return {"columns": columns, "rows": rows[:10]}  # 최대 10행
        except sqlite3.Error as e:
            return f"DB 오류: {e}"


class StockPriceTool(Tool):
    """주가 조회 도구 — 외부 API 시뮬레이션."""

    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "주식 종목의 현재가와 변동률을 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "주식 티커 심볼 (예: AAPL, 005930.KS)",
                        },
                        "currency": {
                            "type": "string",
                            "enum": ["KRW", "USD"],
                            "description": "표시 통화",
                        },
                    },
                    "required": ["ticker"],
                },
            },
        }

    def execute(self, param: Mapping[str, Any]) -> Any:
        ticker = param["ticker"]
        currency = param.get("currency", "USD")
        print(f"  [Tool] get_stock_price(ticker='{ticker}', currency='{currency}')")

        # 실제 환경에서는 yfinance, Alpha Vantage 등 사용
        random.seed(hash(ticker))
        price = round(random.uniform(10, 1000), 2)
        change = round(random.uniform(-5, 5), 2)
        return {
            "ticker": ticker,
            "price": price,
            "change_percent": change,
            "currency": currency,
        }


# ── 예제 함수들 ──────────────────────────────────────────────────────────────────

def example_temperature_converter():
    """온도 변환 도구 사용 예제."""
    print("=" * 60)
    print("예제 1: 커스텀 Tool 클래스 — 온도 변환")
    print("=" * 60)

    tool = TemperatureConverterTool()
    print("스키마:")
    print(json.dumps(tool.get_tool_description(), ensure_ascii=False, indent=2))

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(tools=[tool]) as conversation,
    ):
        response = conversation.send_message("100°C는 몇 화씨인가요?")
        print("\n[응답]", response["content"][0]["text"])


def example_database_tool():
    """SQLite 데이터베이스 조회 도구 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 커스텀 Tool 클래스 — DB 조회")
    print("=" * 60)

    # 임시 DB 생성
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE products (id INTEGER, name TEXT, price REAL, category TEXT)")
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [
            (1, "노트북", 1200000, "전자제품"),
            (2, "스마트폰", 800000, "전자제품"),
            (3, "책상", 250000, "가구"),
            (4, "의자", 150000, "가구"),
            (5, "모니터", 400000, "전자제품"),
        ],
    )
    conn.commit()
    conn.close()

    db_tool = DatabaseQueryTool(db_path)
    print("스키마:", json.dumps(db_tool.get_tool_description(), ensure_ascii=False, indent=2))

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(tools=[db_tool]) as conversation,
    ):
        response = conversation.send_message("전자제품 카테고리의 제품 목록을 조회해주세요.")
        print("\n[응답]", response["content"][0]["text"])


def example_mixed_tools():
    """함수 기반 도구와 클래스 기반 도구를 혼합하는 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 함수 + 클래스 도구 혼합")
    print("=" * 60)

    def get_time() -> str:
        """현재 시간을 반환합니다."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools = [
        get_time,                  # 함수 기반
        TemperatureConverterTool(),  # 클래스 기반
        StockPriceTool(),            # 클래스 기반
    ]

    with (
        litert_lm.Engine(MODEL_PATH) as engine,
        engine.create_conversation(tools=tools) as conversation,
    ):
        response = conversation.send_message("현재 시간과 삼성전자(005930.KS) 주가를 알려주세요.")
        print("[응답]", response["content"][0]["text"])


def example_tool_schema_manual_test():
    """모델 없이 Tool 스키마와 execute를 테스트하는 예제."""
    print("\n" + "=" * 60)
    print("예제 4: Tool 단독 테스트 (모델 불필요)")
    print("=" * 60)

    tool = TemperatureConverterTool()

    # 스키마 출력
    print("스키마:")
    print(json.dumps(tool.get_tool_description(), ensure_ascii=False, indent=2))

    # 직접 실행 테스트
    test_cases = [
        {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"},
        {"value": 32, "from_unit": "fahrenheit", "to_unit": "celsius"},
        {"value": 300, "from_unit": "kelvin", "to_unit": "celsius"},
    ]

    print("\n직접 실행 테스트:")
    for params in test_cases:
        result = tool.execute(params)
        print(f"  {params} → {result}")


if __name__ == "__main__":
    print("LiteRT-LM 커스텀 Tool 클래스 예제\n")

    # 모델 없이 실행 가능
    example_tool_schema_manual_test()

    # 모델이 있을 때 실행 (주석 해제)
    # example_temperature_converter()
    # example_database_tool()
    # example_mixed_tools()
