# LiteRT-LM 학습 가이드

## 📌 LiteRT-LM이란?

**LiteRT-LM**은 Google이 개발한 **엣지 디바이스용 LLM 추론 프레임워크**입니다.  
스마트폰, Chromebook, Raspberry Pi 등 자원이 제한된 환경에서도 Gemma, Llama, Phi, Qwen 같은 대형 언어 모델을 **로컬에서 직접 실행**할 수 있게 해줍니다.

> Chrome, Chromebook Plus, Pixel Watch 등 Google 제품의 온디바이스 AI를 구동하는 핵심 인프라입니다.

---

## 🧠 핵심 이론 개념

### 1. LLM 추론(Inference) 파이프라인

LLM 추론은 크게 두 단계로 나뉩니다:

| 단계 | 이름 | 설명 |
|------|------|------|
| 1단계 | **Prefill** | 입력 프롬프트 전체를 한 번에 처리하여 KV Cache를 채움 |
| 2단계 | **Decode** | KV Cache를 이용해 토큰을 하나씩 자기회귀(autoregressive) 방식으로 생성 |

- **KV Cache**: Attention 연산에서 Key/Value 행렬을 저장해두고 재사용하는 최적화 기법. 매번 전체 시퀀스를 재계산하지 않아도 됩니다.
- **Speculative Decoding**: Draft 모델이 여러 토큰을 미리 예측하고 주 모델이 병렬로 검증하는 기법. 출력 품질은 유지하면서 속도를 높입니다.

### 2. 샘플링(Sampling) 전략

모델이 다음 토큰을 선택하는 방식:

| 파라미터 | 역할 |
|----------|------|
| **Temperature** | 높을수록 다양하고 창의적, 낮을수록 결정론적. `0`이면 greedy |
| **Top-K** | 확률 상위 K개 토큰만 후보로 사용 |
| **Top-P (Nucleus)** | 누적 확률이 P에 도달할 때까지의 토큰만 후보로 사용 |
| **Seed** | 재현 가능한 출력을 위한 난수 시드 |

### 3. 하드웨어 백엔드

| 백엔드 | 설명 |
|--------|------|
| `CPU` | 범용. 모든 디바이스에서 동작 |
| `GPU` | 모바일/데스크탑 GPU 가속. 높은 처리량 |
| `NPU` | 뉴럴 프로세싱 유닛. 최저 전력 소비 |

### 4. 멀티모달(Multimodal)

- **Vision**: 이미지를 입력으로 받아 텍스트 답변 생성 (e.g., Gemma 3)
- **Audio**: 오디오 파일을 입력으로 받아 텍스트 답변 생성 (e.g., Gemma 3n)

### 5. Function Calling (Tool Use)

LLM이 외부 함수/API를 자율적으로 호출하는 기능:
1. 모델이 사용자 의도를 파악하고 어떤 함수를 호출할지 결정
2. 함수 이름과 인자를 JSON 형식으로 출력
3. 런타임이 실제 함수를 실행하고 결과를 모델에 다시 전달
4. 모델이 결과를 바탕으로 최종 답변 생성

---

## 🏗️ 아키텍처 개요

```
litert_lm (Python API)
├── Engine           ← 모델 로딩 & 추론 엔진 (C++ 백엔드 래퍼)
│   ├── create_conversation()  ← 고수준 채팅 API
│   └── create_session()       ← 저수준 Prefill/Decode API
├── Conversation     ← 멀티턴 대화 관리
│   ├── send_message()         ← 동기 응답
│   └── send_message_async()   ← 스트리밍 응답
├── Session          ← 세밀한 추론 제어
│   ├── run_prefill()
│   ├── run_decode()
│   └── run_text_scoring()
├── Tool / tool_from_function() ← Function Calling
├── SamplerConfig    ← 샘플링 파라미터
└── Benchmark        ← 성능 측정
```

---

## 📂 학습 파일 목록

| 파일 | 내용 |
|------|------|
| `01_basics.py` | Engine + Conversation 기본 사용법 |
| `02_streaming.py` | 스트리밍(async) 토큰 출력 |
| `03_function_calling.py` | 함수 호출(Tool Use) |
| `04_session_api.py` | 저수준 Session API (Prefill/Decode) |
| `05_sampler_config.py` | 샘플링 파라미터 제어 |
| `06_benchmark.py` | 성능 벤치마크 |
| `07_custom_tool_class.py` | 커스텀 Tool 클래스 구현 |
| `08_serve_client.py` | Gemini-compatible HTTP API 서버/클라이언트 |
| `09_multimodal.py` | 멀티모달 (이미지/오디오) 사용 |

---

## 🚀 빠른 시작

### 설치
```bash
pip install litert-lm
# 또는 uv 사용
uv tool install litert-lm
```

### 모델 다운로드 (CLI)
```bash
# HuggingFace에서 직접 실행
litert-lm run \
  --from-huggingface-repo=google/gemma-3n-E2B-it-litert-lm \
  gemma-3n-E2B-it-int4 \
  --prompt="한국의 수도는 어디인가요?"
```

### Python에서 기본 사용
```python
import litert_lm

with (
    litert_lm.Engine("path/to/model.litertlm") as engine,
    engine.create_conversation() as conv,
):
    response = conv.send_message("안녕하세요!")
    print(response["content"][0]["text"])
```

---

## 📚 참고 링크

- [공식 문서](https://ai.google.dev/edge/litert-lm)
- [GitHub 저장소](https://github.com/google-ai-edge/LiteRT-LM)
- [HuggingFace 모델](https://huggingface.co/litert-community)
- [Python API 가이드](https://ai.google.dev/edge/litert-lm/python)
