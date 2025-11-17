# 보너스 튜토리얼: 한국어 모델 활용하기

## 목표
이 튜토리얼에서는 한국어 특화 모델을 사용하는 방법을 배웁니다.

## 한국어 모델 소개

### 주요 한국어 BERT 모델

1. **KoBERT** (SKT)
   - 한국어 위키백과로 학습
   - 감성 분석, 개체명 인식 등에 활용

2. **KcBERT** (Beomi)
   - 댓글 데이터로 학습
   - 실제 사용되는 한국어 표현에 강함

3. **KoELECTRA** (Monologg)
   - ELECTRA 아키텍처
   - 효율적인 학습

4. **KR-FinBERT** (SNUNLP)
   - 금융 도메인 특화
   - 뉴스, 리포트 분석에 적합

## 1. 한국어 감성 분석

### KoBERT 사용

```python
from transformers import pipeline

# KoBERT 기반 감성 분석
classifier = pipeline(
    "sentiment-analysis",
    model="beomi/kcbert-base"
)

# 한국어 텍스트 분석
texts = [
    "이 영화 정말 재미있어요!",
    "시간 낭비였습니다.",
    "그냥 그래요.",
    "강력 추천합니다!"
]

for text in texts:
    result = classifier(text)[0]
    print(f"{text}")
    print(f"  -> {result['label']} ({result['score']:.4f})\n")
```

### 다양한 한국어 모델 비교

```python
models = [
    "beomi/kcbert-base",
    "snunlp/KR-FinBert-SC",
    "klue/roberta-base"
]

text = "이 제품 정말 마음에 들어요!"

for model_name in models:
    try:
        classifier = pipeline("sentiment-analysis", model=model_name)
        result = classifier(text)[0]
        print(f"Model: {model_name}")
        print(f"Result: {result}\n")
    except Exception as e:
        print(f"Model: {model_name}")
        print(f"Error: {e}\n")
```

## 2. 한국어 텍스트 생성

### KoGPT 사용

```python
from transformers import pipeline

# 한국어 GPT 모델
generator = pipeline(
    "text-generation",
    model="skt/kogpt2-base-v2"
)

# 텍스트 생성
prompts = [
    "인공지능의 미래는",
    "오늘 날씨가 좋아서",
    "한국의 전통 음식 중에서"
]

for prompt in prompts:
    result = generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}\n")
```

## 3. 한국어 개체명 인식 (NER)

```python
from transformers import pipeline

# 한국어 NER 모델
ner = pipeline(
    "ner",
    model="klue/roberta-base",
    aggregation_strategy="simple"
)

# 텍스트에서 개체 추출
text = "삼성전자는 서울에 본사를 두고 있으며, 이재용 부회장이 경영하고 있습니다."
entities = ner(text)

print(f"Text: {text}\n")
print("추출된 개체:")
for entity in entities:
    print(f"  {entity['word']}: {entity['entity_group']} ({entity['score']:.4f})")
```

## 4. 한국어 질의응답

### KorQuAD 데이터셋으로 학습된 모델

```python
from transformers import pipeline

# 한국어 QA 모델
qa = pipeline(
    "question-answering",
    model="monologg/koelectra-base-v3-discriminator"
)

# 컨텍스트와 질문
context = """
한국은 동아시아에 위치한 나라입니다. 수도는 서울이며,
인구는 약 5천만 명입니다. 한국은 IT 산업과 K-pop,
한국 드라마 등으로 세계적으로 유명합니다.
"""

questions = [
    "한국의 수도는 어디인가요?",
    "한국의 인구는 몇 명인가요?",
    "한국이 유명한 것은 무엇인가요?"
]

for question in questions:
    result = qa(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (confidence: {result['score']:.4f})\n")
```

## 5. 한국어 형태소 분석

### KoNLPy 통합

```python
# KoNLPy 설치 필요
# pip install konlpy

from konlpy.tag import Okt

okt = Okt()

text = "한국어 자연어 처리는 매우 흥미롭습니다."

# 형태소 분석
morphs = okt.morphs(text)
print(f"형태소: {morphs}")

# 품사 태깅
pos = okt.pos(text)
print(f"품사: {pos}")

# 명사 추출
nouns = okt.nouns(text)
print(f"명사: {nouns}")
```

## 6. 한국어 번역

### 한영/영한 번역

```python
from transformers import pipeline

# 한국어 -> 영어
ko_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

korean_text = "안녕하세요. 만나서 반갑습니다."
result = ko_to_en(korean_text)
print(f"한국어: {korean_text}")
print(f"English: {result[0]['translation_text']}\n")

# 영어 -> 한국어
en_to_ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

english_text = "Hello, nice to meet you."
result = en_to_ko(english_text)
print(f"English: {english_text}")
print(f"한국어: {result[0]['translation_text']}")
```

## 7. 한국어 문장 임베딩

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 한국어 문장 임베딩 모델
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 문장들
sentences = [
    "오늘 날씨가 정말 좋네요.",
    "오늘 기분이 매우 좋습니다.",
    "내일 비가 온다고 합니다.",
    "저는 커피를 좋아합니다."
]

# 임베딩 생성
embeddings = model.encode(sentences)

# 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print("문장 간 유사도:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"{sentences[i]}")
        print(f"{sentences[j]}")
        print(f"유사도: {similarities[i][j]:.4f}\n")
```

## 8. 한국어 텍스트 분류 파인튜닝

### NSMC(네이버 영화 리뷰) 데이터셋

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# NSMC 데이터셋 로드
dataset = load_dataset("nsmc")

# 토크나이저 및 모델
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['document'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# 토크나이징
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 모델
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./korean_sentiment_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

# 학습 (실제로는 시간이 오래 걸립니다)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['test'],
# )
# trainer.train()
```

## 9. 한국어 특수 처리

### 자모 분리 처리

```python
def decompose_hangul(text):
    """한글 자모 분리"""
    # 한글 유니코드 범위
    HANGUL_BASE = 0xAC00
    CHOSUNG_BASE = 0x1100
    JUNGSUNG_BASE = 0x1161
    JONGSUNG_BASE = 0x11A7

    result = []
    for char in text:
        if '가' <= char <= '힣':
            code = ord(char) - HANGUL_BASE
            jong = code % 28
            jung = ((code - jong) // 28) % 21
            cho = ((code - jong) // 28) // 21

            result.append(chr(CHOSUNG_BASE + cho))
            result.append(chr(JUNGSUNG_BASE + jung))
            if jong:
                result.append(chr(JONGSUNG_BASE + jong))
        else:
            result.append(char)

    return ''.join(result)

text = "한글"
decomposed = decompose_hangul(text)
print(f"원본: {text}")
print(f"분해: {decomposed}")
```

## 10. 추천 한국어 모델 목록

### 범용 모델
- `beomi/kcbert-base` - 댓글 데이터 학습, 일상 언어에 강함
- `klue/roberta-base` - KLUE 벤치마크로 평가된 모델
- `monologg/koelectra-base-v3-discriminator` - 효율적인 ELECTRA

### 특화 모델
- `snunlp/KR-FinBert-SC` - 금융 도메인
- `beomi/KcELECTRA-base-v2022` - 최신 댓글 데이터
- `skt/kogpt2-base-v2` - 텍스트 생성

### 문장 임베딩
- `jhgan/ko-sroberta-multitask` - 문장 유사도
- `BM-K/KoSimCSE-roberta` - SimCSE 한국어 버전

## 11. 한국어 데이터셋

### Hugging Face에서 사용 가능한 한국어 데이터셋

```python
from datasets import load_dataset

# NSMC (네이버 영화 리뷰)
nsmc = load_dataset("nsmc")

# KorQuAD (한국어 QA)
korquad = load_dataset("squad_kor_v1")

# KLUE (Korean Language Understanding Evaluation)
klue_tc = load_dataset("klue", "tc")  # 주제 분류
klue_sts = load_dataset("klue", "sts")  # 문장 유사도
klue_nli = load_dataset("klue", "nli")  # 자연어 추론
```

## 12. 실습 과제

### 과제 1: 한국어 감성 분석기
NSMC 데이터셋으로 영화 리뷰 감성 분석 모델을 만드세요.

### 과제 2: 한국어 텍스트 생성
KoGPT를 사용하여 한국어 시를 생성하세요.

### 과제 3: 한국어 챗봇
간단한 한국어 대화형 챗봇을 구현하세요.

### 과제 4: 한국어 문서 분류
뉴스 기사를 카테고리별로 분류하는 모델을 만드세요.

## 13. 주의사항

### 인코딩 문제
```python
# UTF-8 인코딩 명시
with open('korean_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

### 토크나이저 선택
- 한국어 특화 토크나이저 사용 권장
- Byte-level BPE보다 형태소 기반이 더 효과적일 수 있음

### 데이터 전처리
- 이모지, 특수문자 처리
- 띄어쓰기 정규화
- 반복 문자 처리

## 참고 자료

- [KcBERT GitHub](https://github.com/Beomi/KcBERT)
- [KoELECTRA GitHub](https://github.com/monologg/KoELECTRA)
- [KLUE Benchmark](https://klue-benchmark.com/)
- [KoNLPy Documentation](https://konlpy.org/)
- [한국어 Hugging Face 모델](https://huggingface.co/models?language=ko)
