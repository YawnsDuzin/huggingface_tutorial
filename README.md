# Hugging Face 튜토리얼

## 📚 Hugging Face란?

Hugging Face는 자연어 처리(NLP) 및 머신러닝 분야의 선도적인 오픈소스 플랫폼입니다. 2016년에 설립되어 현재는 AI/ML 커뮤니티에서 가장 중요한 허브 중 하나로 자리잡았습니다.

### 핵심 특징

1. **사전 훈련된 모델 허브**
   - 100,000개 이상의 사전 훈련된 모델 제공
   - BERT, GPT, T5, LLAMA 등 최신 모델 지원
   - 다양한 언어와 태스크를 위한 모델

2. **Transformers 라이브러리**
   - PyTorch, TensorFlow, JAX 지원
   - 통일된 API로 다양한 모델 사용 가능
   - 간단한 코드로 복잡한 모델 활용

3. **Datasets 라이브러리**
   - 10,000개 이상의 공개 데이터셋
   - 효율적인 데이터 로딩 및 전처리
   - Apache Arrow 기반 빠른 처리

4. **Model Hub**
   - 모델 공유 및 협업 플랫폼
   - 버전 관리 및 문서화
   - 원클릭 배포 및 추론 API

5. **Spaces**
   - ML 데모 및 애플리케이션 호스팅
   - Gradio/Streamlit 통합
   - 무료 GPU 지원

## 🎯 주요 사용 사례

### 1. 자연어 처리 (NLP)
- **텍스트 분류**: 감성 분석, 주제 분류, 스팸 탐지
- **개체명 인식 (NER)**: 인명, 지명, 조직명 추출
- **질의응답**: 문서 기반 질문 답변
- **텍스트 생성**: 자동 완성, 요약, 번역
- **문장 임베딩**: 의미적 유사도 계산

### 2. 컴퓨터 비전
- 이미지 분류
- 객체 탐지
- 이미지 세그멘테이션
- 이미지 생성 (Stable Diffusion 등)

### 3. 오디오 처리
- 음성 인식 (ASR)
- 음성 합성 (TTS)
- 오디오 분류

### 4. 멀티모달
- 이미지 캡셔닝
- 비전-언어 모델 (CLIP, BLIP)
- 문서 이해

## 🔧 주요 라이브러리

### Transformers
사전 훈련된 모델을 쉽게 사용할 수 있는 핵심 라이브러리

```python
from transformers import pipeline

# 간단한 파이프라인 사용
classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")
```

### Datasets
대규모 데이터셋을 효율적으로 처리

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

### Tokenizers
빠르고 효율적인 토크나이저

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### Accelerate
분산 학습을 간단하게

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

## 🌟 Hugging Face의 장점

1. **사용 편의성**
   - 몇 줄의 코드로 최신 모델 활용
   - 직관적인 API 설계
   - 풍부한 문서와 예제

2. **커뮤니티**
   - 활발한 오픈소스 커뮤니티
   - 지속적인 업데이트와 개선
   - 다양한 튜토리얼과 블로그

3. **호환성**
   - 여러 프레임워크 지원 (PyTorch, TensorFlow)
   - 다양한 플랫폼에서 실행 가능
   - 클라우드 서비스와 통합

4. **확장성**
   - 소규모 실험부터 프로덕션까지
   - 분산 학습 지원
   - 효율적인 추론 최적화

5. **비용 효율성**
   - 무료로 사용 가능한 대부분의 기능
   - 사전 훈련된 모델로 학습 비용 절감
   - 무료 호스팅 옵션 (Spaces, Inference API)

## 📖 튜토리얼 구성

이 저장소는 다음과 같은 단계별 튜토리얼을 제공합니다:

1. **기본 설정** - 환경 설정 및 라이브러리 설치
2. **텍스트 분류** - 감성 분석 실습
3. **텍스트 생성** - GPT 모델을 활용한 텍스트 생성
4. **파인튜닝** - 커스텀 데이터로 모델 학습
5. **한국어 모델** - 한국어 특화 모델 활용
6. **라즈베리파이 가이드** - 라즈베리파이에서 모델 실행하기

각 튜토리얼은 `tutorials/` 디렉토리에 있으며, 실행 가능한 예제 코드를 포함합니다.

## 🚀 빠른 시작

```bash
# 라이브러리 설치
pip install transformers datasets torch

# 첫 번째 예제 실행
python examples/01_text_classification.py
```

## 📚 참고 자료

- [Hugging Face 공식 웹사이트](https://huggingface.co/)
- [Transformers 문서](https://huggingface.co/docs/transformers)
- [Datasets 문서](https://huggingface.co/docs/datasets)
- [Hugging Face 코스](https://huggingface.co/course)
- [Hugging Face 블로그](https://huggingface.co/blog)

## 🤝 기여

이 튜토리얼에 기여하고 싶으시다면 Pull Request를 보내주세요!

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
