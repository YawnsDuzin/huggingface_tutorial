# 튜토리얼 2: 텍스트 분류 (Text Classification)

## 목표
이 튜토리얼에서는 Hugging Face를 사용하여 텍스트 분류를 수행하는 방법을 배웁니다.

## 텍스트 분류란?

텍스트 분류는 주어진 텍스트를 미리 정의된 카테고리로 분류하는 작업입니다.

### 주요 응용 분야
- **감성 분석**: 긍정/부정/중립 판단
- **스팸 탐지**: 스팸/정상 메일 구분
- **주제 분류**: 뉴스 기사의 카테고리 분류
- **의도 분류**: 챗봇에서 사용자 의도 파악
- **언어 감지**: 텍스트의 언어 식별

## 1. Pipeline을 사용한 간단한 분류

Pipeline은 모델 사용을 가장 간단하게 만들어주는 고수준 API입니다.

### 기본 감성 분석

```python
from transformers import pipeline

# 기본 감성 분석 파이프라인 (영어)
classifier = pipeline("sentiment-analysis")

# 단일 텍스트 분류
result = classifier("I absolutely love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 여러 텍스트 분류
texts = [
    "This movie was fantastic!",
    "I wasted my money on this.",
    "It was okay, nothing special."
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})\n")
```

### 특정 모델 지정

```python
# 한국어 감성 분석 모델
ko_classifier = pipeline(
    "sentiment-analysis",
    model="snunlp/KR-FinBert-SC"
)

result = ko_classifier("이 제품 정말 좋아요!")
print(result)
```

## 2. 다양한 분류 태스크

### Zero-Shot Classification

레이블을 학습하지 않고도 분류 가능:

```python
classifier = pipeline("zero-shot-classification")

text = "This is a tutorial about natural language processing."
candidate_labels = ["technology", "sports", "politics", "education"]

result = classifier(text, candidate_labels)
print(f"Text: {text}")
print(f"Labels: {result['labels']}")
print(f"Scores: {result['scores']}")
```

### 다중 레이블 분류

```python
classifier = pipeline("zero-shot-classification")

text = "This smartphone has an excellent camera and long battery life."
candidate_labels = ["camera quality", "battery life", "price", "design"]

result = classifier(
    text,
    candidate_labels,
    multi_label=True  # 여러 레이블 가능
)

for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score:.4f}")
```

## 3. 모델과 토크나이저 직접 사용

더 많은 제어가 필요한 경우:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델과 토크나이저 로드
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 텍스트 토크나이징
text = "This is a wonderful experience!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# 결과 해석
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
predicted_class = torch.argmax(predictions).item()
confidence = predictions[0][predicted_class].item()

print(f"Prediction: {label_map[predicted_class]}")
print(f"Confidence: {confidence:.4f}")
print(f"All scores: {predictions}")
```

## 4. 배치 처리

효율적인 대량 텍스트 처리:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)  # GPU 사용 (가능한 경우)

# 큰 데이터셋
texts = [
    "Great product!",
    "Terrible experience.",
    "Average quality.",
    # ... 수천 개의 텍스트
] * 100

# 배치 처리
results = classifier(texts, batch_size=32)

# 통계 계산
positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')

print(f"Positive: {positive_count}, Negative: {negative_count}")
```

## 5. 실전 예제: 영화 리뷰 분석

### 데이터 준비

```python
import pandas as pd

# 샘플 영화 리뷰 데이터
reviews = [
    {"text": "This movie was absolutely brilliant!", "rating": 5},
    {"text": "Waste of time and money.", "rating": 1},
    {"text": "Pretty good, would recommend.", "rating": 4},
    {"text": "Boring and predictable plot.", "rating": 2},
    {"text": "Masterpiece! Best film of the year.", "rating": 5},
    {"text": "Not bad, but could be better.", "rating": 3},
]

df = pd.DataFrame(reviews)
print(df)
```

### 감성 분석 수행

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# 모든 리뷰 분석
df['sentiment'] = df['text'].apply(
    lambda x: classifier(x)[0]['label']
)
df['confidence'] = df['text'].apply(
    lambda x: classifier(x)[0]['score']
)

print(df[['text', 'rating', 'sentiment', 'confidence']])
```

### 결과 시각화

```python
import matplotlib.pyplot as plt

# 감성별 개수
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.close()

# 평점과 감성의 관계
accuracy = (
    (df['rating'] >= 4) & (df['sentiment'] == 'POSITIVE') |
    (df['rating'] <= 2) & (df['sentiment'] == 'NEGATIVE')
).mean()

print(f"Sentiment-Rating Agreement: {accuracy:.2%}")
```

## 6. 여러 모델 비교

```python
from transformers import pipeline

models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment",
    "nlptown/bert-base-multilingual-uncased-sentiment"
]

text = "This product exceeded my expectations!"

print(f"Text: {text}\n")
for model_name in models:
    classifier = pipeline("sentiment-analysis", model=model_name)
    result = classifier(text)[0]
    print(f"Model: {model_name}")
    print(f"Result: {result}\n")
```

## 7. 성능 최적화

### GPU 사용

```python
from transformers import pipeline

# GPU 사용 (CUDA 사용 가능 시)
classifier = pipeline(
    "sentiment-analysis",
    device=0  # GPU 0번 사용, CPU는 -1
)
```

### 모델 양자화

```python
from transformers import pipeline

# 8비트 양자화로 메모리 사용량 감소
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    model_kwargs={"load_in_8bit": True}
)
```

### 배치 크기 조정

```python
# 큰 배치 크기로 처리 속도 향상
results = classifier(texts, batch_size=64)
```

## 8. 한국어 텍스트 분류

### 한국어 감성 분석

```python
from transformers import pipeline

# 한국어 BERT 모델
ko_classifier = pipeline(
    "sentiment-analysis",
    model="beomi/kcbert-base"
)

korean_texts = [
    "이 영화 정말 재미있어요!",
    "시간 낭비였습니다.",
    "그냥 그래요."
]

for text in korean_texts:
    result = ko_classifier(text)[0]
    print(f"{text} -> {result}")
```

## 9. 오류 처리 및 예외 상황

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def safe_classify(text):
    try:
        # 빈 텍스트 처리
        if not text or not text.strip():
            return {"label": "UNKNOWN", "score": 0.0}

        # 너무 긴 텍스트 자르기
        max_length = 512
        text = text[:max_length]

        result = classifier(text)[0]
        return result

    except Exception as e:
        print(f"Error processing text: {e}")
        return {"label": "ERROR", "score": 0.0}

# 테스트
texts = [
    "Normal text",
    "",  # 빈 텍스트
    "A" * 10000,  # 매우 긴 텍스트
]

for text in texts:
    result = safe_classify(text)
    print(f"Result: {result}")
```

## 10. 실습 과제

### 과제 1: 감성 분석기 만들기
사용자 입력을 받아 감성을 분석하는 간단한 프로그램을 작성하세요.

### 과제 2: CSV 파일 처리
CSV 파일에서 텍스트를 읽어 감성 분석 결과를 저장하세요.

### 과제 3: 다국어 분류
영어, 한국어, 중국어 텍스트를 자동으로 감지하고 적절한 모델로 분류하세요.

### 과제 4: 실시간 분류
웹 스크래핑으로 실시간 뉴스 헤드라인을 수집하고 분류하세요.

## 11. 성능 평가

```python
from sklearn.metrics import accuracy_score, classification_report

# 실제 레이블
true_labels = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]

# 예측 레이블
texts = [
    "Great!",
    "Terrible!",
    "Amazing experience",
    "Worst product ever"
]

classifier = pipeline("sentiment-analysis")
predictions = [classifier(text)[0]['label'] for text in texts]

# 정확도 계산
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.2%}")

# 상세 리포트
print(classification_report(true_labels, predictions))
```

## 다음 단계

다음 튜토리얼에서는 텍스트 생성을 다룹니다:
- **튜토리얼 3**: 텍스트 생성 (GPT, BART 등)

## 참고 자료

- [Hugging Face Text Classification](https://huggingface.co/tasks/text-classification)
- [Pipeline 문서](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [모델 허브 - Text Classification](https://huggingface.co/models?pipeline_tag=text-classification)
