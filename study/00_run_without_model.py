"""
LiteRT-LM 학습 예제 — 모델 없이 실행 가능한 테스트
====================================================

이 파일은 실제 .litertlm 모델 파일 없이도 실행할 수 있는 예제들을 포함합니다.
LiteRT-LM의 API 구조, Tool 스키마 생성, SamplerConfig 검증 등을 확인합니다.
"""

import json
from collections.abc import Mapping
from typing import Any

import litert_lm
from litert_lm import Tool
from litert_lm.tools import tool_from_function


def test_sampler_config():
    """SamplerConfig 생성 및 유효성 검사 테스트."""
    print("=" * 60)
    print("테스트 1: SamplerConfig")
    print("=" * 60)

    # 정상 케이스
    configs = [
        litert_lm.SamplerConfig(),
        litert_lm.SamplerConfig(top_k=40, top_p=0.9, temperature=0.8, seed=42),
        litert_lm.SamplerConfig(top_k=1, temperature=0.0),
        litert_lm.SamplerConfig(top_p=1.0),
    ]
    for i, cfg in enumerate(configs):
        print(f"  [OK] 정상 케이스 {i+1}: {cfg}")

    # 에러 케이스
    error_cases = [
        ({"top_k": 0},         "top_k=0"),
        ({"top_k": -1},        "top_k=-1"),
        ({"top_p": 1.1},       "top_p=1.1"),
        ({"top_p": -0.1},      "top_p=-0.1"),
        ({"temperature": -0.1}, "temperature=-0.1"),
    ]
    for kwargs, label in error_cases:
        try:
            litert_lm.SamplerConfig(**kwargs)
            print(f"  [FAIL] {label}: 예외가 발생했어야 함")
        except ValueError as e:
            print(f"  [OK] {label}: ValueError — {e}")


def test_tool_from_function():
    """tool_from_function()으로 Python 함수를 Tool로 변환하는 테스트."""
    print("\n" + "=" * 60)
    print("테스트 2: tool_from_function()")
    print("=" * 60)

    def search_news(query: str, max_results: int = 5) -> str:
        """뉴스를 검색합니다.

        Args:
            query: 검색 키워드.
            max_results: 반환할 최대 기사 수.
        """
        return f"'{query}' 검색 결과 {max_results}개"

    tool = tool_from_function(search_news)
    schema = tool.get_tool_description()

    print("자동 생성된 스키마:")
    print(json.dumps(schema, ensure_ascii=False, indent=2))

    # 직접 실행 테스트
    result = tool.execute({"query": "AI 뉴스", "max_results": 3})
    print(f"\n실행 결과: {result}")


def test_custom_tool_class():
    """커스텀 Tool 클래스 구현 및 테스트."""
    print("\n" + "=" * 60)
    print("테스트 3: 커스텀 Tool 클래스")
    print("=" * 60)

    class UnitConverterTool(Tool):
        """단위 변환 도구."""

        def get_tool_description(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": "convert_unit",
                    "description": "단위를 변환합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number", "description": "변환할 값"},
                            "from_unit": {
                                "type": "string",
                                "enum": ["km", "miles", "kg", "lbs"],
                                "description": "원본 단위",
                            },
                            "to_unit": {
                                "type": "string",
                                "enum": ["km", "miles", "kg", "lbs"],
                                "description": "목표 단위",
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

            conversions = {
                ("km", "miles"): 0.621371,
                ("miles", "km"): 1.60934,
                ("kg", "lbs"): 2.20462,
                ("lbs", "kg"): 0.453592,
            }

            key = (from_unit, to_unit)
            if key in conversions:
                result = value * conversions[key]
                return f"{value} {from_unit} = {result:.4f} {to_unit}"
            elif from_unit == to_unit:
                return f"{value} {from_unit} (변환 불필요)"
            else:
                return f"지원하지 않는 변환: {from_unit} → {to_unit}"

    tool = UnitConverterTool()

    print("스키마:")
    print(json.dumps(tool.get_tool_description(), ensure_ascii=False, indent=2))

    print("\n변환 테스트:")
    test_cases = [
        {"value": 10, "from_unit": "km", "to_unit": "miles"},
        {"value": 100, "from_unit": "lbs", "to_unit": "kg"},
        {"value": 5, "from_unit": "km", "to_unit": "km"},
    ]
    for params in test_cases:
        result = tool.execute(params)
        print(f"  {params['value']} {params['from_unit']} → {result}")


def test_interface_inheritance():
    """LiteRT-LM 추상 클래스 상속 구조 확인 테스트."""
    print("\n" + "=" * 60)
    print("테스트 4: 인터페이스 상속 구조")
    print("=" * 60)

    from litert_lm.interfaces import (
        AbstractBenchmark,
        AbstractConversation,
        AbstractEngine,
        AbstractSession,
        Backend,
        BenchmarkInfo,
        Responses,
        SamplerConfig,
        Tool,
        ToolEventHandler,
    )

    classes = [
        ("Backend", Backend),
        ("SamplerConfig", SamplerConfig),
        ("AbstractEngine", AbstractEngine),
        ("AbstractConversation", AbstractConversation),
        ("AbstractSession", AbstractSession),
        ("AbstractBenchmark", AbstractBenchmark),
        ("BenchmarkInfo", BenchmarkInfo),
        ("Responses", Responses),
        ("Tool", Tool),
        ("ToolEventHandler", ToolEventHandler),
    ]

    print("핵심 클래스 목록:")
    for name, cls in classes:
        import abc
        is_abstract = hasattr(cls, "__abstractmethods__") and bool(cls.__abstractmethods__)
        kind = "추상 클래스" if is_abstract else "구체 클래스"
        print(f"  {name:<25} ({kind})")

    print("\nBackend 열거형:")
    for member in Backend:
        print(f"  Backend.{member.name} = {member.value}")


def test_responses_dataclass():
    """Responses 데이터클래스 사용 테스트."""
    print("\n" + "=" * 60)
    print("테스트 5: Responses 데이터클래스")
    print("=" * 60)

    from litert_lm.interfaces import Responses

    r = Responses(texts=["안녕하세요"], scores=[-1.23], token_lengths=[5])
    print(f"texts: {r.texts}")
    print(f"scores: {r.scores}")
    print(f"token_lengths: {r.token_lengths}")

    empty_r = Responses()
    print(f"\n빈 Responses: texts={empty_r.texts}, scores={empty_r.scores}")


if __name__ == "__main__":
    print("LiteRT-LM 모델 없이 실행 가능한 테스트\n")
    print("이 파일은 litert-lm이 설치된 환경에서 모델 없이 실행됩니다.")
    print("설치: pip install litert-lm\n")

    try:
        test_sampler_config()
        test_tool_from_function()
        test_custom_tool_class()
        test_interface_inheritance()
        test_responses_dataclass()
        print("\n" + "=" * 60)
        print("✓ 모든 테스트 완료!")
    except ImportError as e:
        print(f"오류: litert-lm이 설치되지 않았습니다. pip install litert-lm\n{e}")
