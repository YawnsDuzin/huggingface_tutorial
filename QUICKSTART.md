# 빠른 시작 가이드

## 5분 안에 Hugging Face 시작하기

### 1. 설치 (2분)

```bash
# 가상 환경 생성 (권장)
python -m venv hf_env
source hf_env/bin/activate  # Linux/Mac
# 또는
hf_env\Scripts\activate  # Windows

# 필수 패키지 설치
pip install transformers torch datasets
```

### 2. 첫 번째 모델 실행 (1분)

**감성 분석:**
```python
from transformers import pipeline

# 파이프라인 생성
classifier = pipeline("sentiment-analysis")

# 텍스트 분석
result = classifier("I love Hugging Face!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 3. 텍스트 생성 시도 (1분)

```python
# 텍스트 생성 파이프라인
generator = pipeline("text-generation", model="gpt2")

# 텍스트 생성
text = generator("Once upon a time", max_length=50)
print(text[0]['generated_text'])
```

### 4. 다른 태스크 시도 (1분)

**질의응답:**
```python
qa = pipeline("question-answering")

context = "Paris is the capital of France."
question = "What is the capital of France?"

answer = qa(question=question, context=context)
print(answer['answer'])  # Paris
```

**번역:**
```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

result = translator("Hello, how are you?")
print(result[0]['translation_text'])
```

**요약:**
```python
summarizer = pipeline("summarization")

text = """Your long text here..."""
summary = summarizer(text, max_length=100, min_length=30)
print(summary[0]['summary_text'])
```

## 다음 단계

1. **기본 학습**: `tutorials/01_installation_and_setup.md` 읽기
2. **예제 실행**: `examples/` 폴더의 Python 파일 실행
3. **실습**: 각 튜토리얼의 연습 문제 풀기

## 자주 사용하는 파이프라인

| 태스크 | 파이프라인 이름 | 설명 |
|--------|----------------|------|
| 감성 분석 | `sentiment-analysis` | 텍스트의 감성 판단 |
| 텍스트 생성 | `text-generation` | 자동 텍스트 완성 |
| 질의응답 | `question-answering` | 문서에서 답변 찾기 |
| 요약 | `summarization` | 긴 텍스트 요약 |
| 번역 | `translation` | 언어 번역 |
| 개체명 인식 | `ner` | 인명, 지명 등 추출 |
| Zero-Shot 분류 | `zero-shot-classification` | 레이블 없이 분류 |

## 문제 해결

**모델 다운로드 느림:**
```bash
# 캐시 디렉토리 확인
echo $HF_HOME
# 또는
python -c "from transformers import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)"
```

**GPU 사용:**
```python
# GPU 사용 가능 확인
import torch
print(torch.cuda.is_available())

# GPU에서 실행
classifier = pipeline("sentiment-analysis", device=0)
```

**메모리 부족:**
```python
# 작은 모델 사용
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 또는 배치 크기 줄이기
results = classifier(texts, batch_size=8)
```

## 유용한 링크

- 공식 문서: https://huggingface.co/docs
- 모델 허브: https://huggingface.co/models
- 데이터셋: https://huggingface.co/datasets
- 커뮤니티: https://discuss.huggingface.co/

## 도움이 필요하신가요?

- 튜토리얼 문서를 읽어보세요: `tutorials/` 폴더
- 예제 코드를 실행해보세요: `examples/` 폴더
- GitHub Issues에서 질문하세요
