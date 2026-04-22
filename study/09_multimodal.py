"""
LiteRT-LM 학습 예제 09: 멀티모달 (이미지/오디오)
===================================================

[이론]
LiteRT-LM은 텍스트 외에 이미지와 오디오를 입력으로 받을 수 있는 멀티모달 모델을 지원합니다.

지원 입력 유형:
  - 텍스트  : {"type": "text", "text": "..."}
  - 이미지  : {"type": "image", "path": "/path/to/image.jpg"}
  - 오디오  : {"type": "audio", "path": "/path/to/audio.wav"}

백엔드 설정:
  vision_backend : 이미지 인코더에 사용할 하드웨어 (CPU/GPU)
  audio_backend  : 오디오 인코더에 사용할 하드웨어 (CPU/GPU)

지원 모델 예시:
  - 이미지: Gemma 3 multimodal 버전
  - 오디오: Gemma 3n (audio 지원)

지원 이미지 형식: JPEG, PNG, WebP 등
지원 오디오 형식: WAV, MP3, FLAC 등

[실전 활용]
- 이미지 설명/캡션 생성
- 문서 OCR 및 내용 추출
- 오디오 전사(transcription)
- 시각적 Q&A
- 이미지 기반 코드 생성 (스크린샷 → 코드)
"""

import os

import litert_lm

MODEL_PATH = "path/to/your/multimodal-model.litertlm"
IMAGE_PATH = "path/to/your/image.jpg"
AUDIO_PATH = "path/to/your/audio.wav"


def example_image_description():
    """이미지 설명 생성 예제."""
    print("=" * 60)
    print("예제 1: 이미지 설명 생성")
    print("=" * 60)

    with (
        # vision_backend 설정 필수
        litert_lm.Engine(MODEL_PATH, vision_backend=litert_lm.Backend.CPU) as engine,
        engine.create_conversation() as conversation,
    ):
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": os.path.abspath(IMAGE_PATH),
                },
                {
                    "type": "text",
                    "text": "이 이미지를 한국어로 설명해주세요.",
                },
            ],
        }

        response = conversation.send_message(user_message)
        print("[이미지 설명]", response["content"][0]["text"])


def example_image_qa():
    """이미지 기반 질문 답변 예제."""
    print("\n" + "=" * 60)
    print("예제 2: 이미지 Q&A")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH, vision_backend=litert_lm.Backend.CPU) as engine,
        engine.create_conversation() as conversation,
    ):
        # 이미지와 함께 여러 질문
        questions = [
            "이미지에서 가장 눈에 띄는 색깔은 무엇인가요?",
            "이미지에 사람이 있나요?",
            "이미지의 분위기를 한 단어로 표현한다면?",
        ]

        for question in questions:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "path": os.path.abspath(IMAGE_PATH)},
                    {"type": "text", "text": question},
                ],
            }
            response = conversation.send_message(user_message)
            print(f"Q: {question}")
            print(f"A: {response['content'][0]['text']}\n")


def example_audio_transcription():
    """오디오 전사(Transcription) 예제."""
    print("\n" + "=" * 60)
    print("예제 3: 오디오 전사")
    print("=" * 60)

    with (
        # audio_backend 설정 필수
        litert_lm.Engine(MODEL_PATH, audio_backend=litert_lm.Backend.CPU) as engine,
        engine.create_conversation() as conversation,
    ):
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": os.path.abspath(AUDIO_PATH),
                },
                {
                    "type": "text",
                    "text": "이 오디오의 내용을 텍스트로 전사해주세요.",
                },
            ],
        }

        response = conversation.send_message(user_message)
        print("[전사 결과]", response["content"][0]["text"])


def example_audio_analysis():
    """오디오 분석 예제 — 내용 요약 및 감정 분석."""
    print("\n" + "=" * 60)
    print("예제 4: 오디오 분석")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH, audio_backend=litert_lm.Backend.CPU) as engine,
        engine.create_conversation() as conversation,
    ):
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": os.path.abspath(AUDIO_PATH),
                },
                {
                    "type": "text",
                    "text": "이 오디오를 분석해주세요: 1) 주제, 2) 화자의 감정, 3) 핵심 내용",
                },
            ],
        }

        response = conversation.send_message(user_message)
        print("[오디오 분석]", response["content"][0]["text"])


def example_image_and_audio_combined():
    """이미지와 오디오를 함께 사용하는 예제."""
    print("\n" + "=" * 60)
    print("예제 5: 이미지 + 오디오 복합 입력")
    print("=" * 60)

    with (
        litert_lm.Engine(
            MODEL_PATH,
            vision_backend=litert_lm.Backend.CPU,
            audio_backend=litert_lm.Backend.CPU,
        ) as engine,
        engine.create_conversation() as conversation,
    ):
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "path": os.path.abspath(IMAGE_PATH)},
                {"type": "audio", "path": os.path.abspath(AUDIO_PATH)},
                {
                    "type": "text",
                    "text": "이미지와 오디오의 내용이 서로 관련이 있나요?",
                },
            ],
        }

        response = conversation.send_message(user_message)
        print("[복합 분석]", response["content"][0]["text"])


def example_image_streaming():
    """이미지 입력과 스트리밍 응답 조합 예제."""
    print("\n" + "=" * 60)
    print("예제 6: 이미지 + 스트리밍 응답")
    print("=" * 60)

    with (
        litert_lm.Engine(MODEL_PATH, vision_backend=litert_lm.Backend.CPU) as engine,
        engine.create_conversation() as conversation,
    ):
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "path": os.path.abspath(IMAGE_PATH)},
                {"type": "text", "text": "이 이미지를 상세히 설명해주세요."},
            ],
        }

        print("[스트리밍 설명] ", end="", flush=True)
        for chunk in conversation.send_message_async(user_message):
            content_list = chunk.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    print(item["text"], end="", flush=True)
        print()


def example_cli_multimodal():
    """CLI를 사용한 멀티모달 실행 방법 안내."""
    print("\n" + "=" * 60)
    print("예제 7: CLI로 멀티모달 실행 방법")
    print("=" * 60)

    print("이미지 첨부:")
    print("  litert-lm run my-model \\")
    print("    --vision-backend=cpu \\")
    print("    --attachment=photo.jpg \\")
    print("    --prompt='이 사진을 설명해주세요'")
    print()
    print("오디오 첨부:")
    print("  litert-lm run my-model \\")
    print("    --audio-backend=cpu \\")
    print("    --attachment=recording.wav \\")
    print("    --prompt='이 오디오를 전사해주세요'")
    print()
    print("여러 파일 첨부:")
    print("  litert-lm run my-model \\")
    print("    --vision-backend=cpu \\")
    print("    --attachment=image1.jpg \\")
    print("    --attachment=image2.jpg \\")
    print("    --prompt='두 이미지의 차이점은?'")


if __name__ == "__main__":
    print("LiteRT-LM 멀티모달 예제")
    print("주의: 멀티모달을 지원하는 모델 파일이 필요합니다.")
    print(f"모델 경로: {MODEL_PATH}")
    print(f"이미지 경로: {IMAGE_PATH}")
    print(f"오디오 경로: {AUDIO_PATH}\n")

    # CLI 사용법은 모델 없이 확인 가능
    example_cli_multimodal()

    # 모델이 있을 때 실행 (주석 해제, 경로 수정 필수)
    # example_image_description()
    # example_image_qa()
    # example_audio_transcription()
    # example_audio_analysis()
    # example_image_and_audio_combined()
    # example_image_streaming()

    print("\n예제 코드 구조 확인 완료.")
