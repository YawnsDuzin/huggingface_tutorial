# 튜토리얼 3: 텍스트 생성 (Text Generation)

## 목표
이 튜토리얼에서는 Hugging Face를 사용하여 텍스트를 생성하는 다양한 방법을 배웁니다.

## 텍스트 생성이란?

텍스트 생성은 주어진 프롬프트나 컨텍스트를 기반으로 새로운 텍스트를 자동으로 만드는 작업입니다.

### 주요 응용 분야
- **자동 완성**: 이메일, 코드, 문서 작성 지원
- **대화 생성**: 챗봇, 가상 비서
- **창작**: 스토리, 시, 가사 작성
- **요약**: 긴 문서를 짧게 요약
- **번역**: 다국어 번역
- **코드 생성**: 자연어를 코드로 변환

## 1. Pipeline을 사용한 기본 텍스트 생성

### GPT-2를 이용한 간단한 생성

```python
from transformers import pipeline

# 텍스트 생성 파이프라인
generator = pipeline("text-generation", model="gpt2")

# 프롬프트 제공
prompt = "Once upon a time"
result = generator(prompt, max_length=50, num_return_sequences=1)

print(result[0]['generated_text'])
```

### 여러 개의 결과 생성

```python
# 여러 버전 생성
prompt = "The future of artificial intelligence is"
results = generator(
    prompt,
    max_length=50,
    num_return_sequences=3,
    temperature=0.7  # 창의성 조절 (0.0-1.0)
)

for i, result in enumerate(results, 1):
    print(f"\n버전 {i}:")
    print(result['generated_text'])
```

## 2. 생성 파라미터 이해하기

### 주요 파라미터

```python
generator = pipeline("text-generation", model="gpt2")

prompt = "In the year 2050,"

result = generator(
    prompt,
    max_length=100,           # 생성할 최대 토큰 수
    min_length=50,            # 생성할 최소 토큰 수
    num_return_sequences=2,   # 생성할 결과 개수
    temperature=0.8,          # 무작위성 (낮음: 보수적, 높음: 창의적)
    top_k=50,                 # 상위 k개 토큰만 고려
    top_p=0.95,               # 누적 확률 p까지의 토큰만 고려
    do_sample=True,           # 샘플링 사용 여부
    no_repeat_ngram_size=2,   # n-gram 반복 방지
)

print(result)
```

### 파라미터 상세 설명

- **temperature**: 0에 가까울수록 결정적, 1 이상이면 더 창의적
- **top_k**: 확률이 높은 상위 k개 토큰만 선택
- **top_p (nucleus sampling)**: 누적 확률이 p가 될 때까지의 토큰 선택
- **do_sample**: False면 항상 가장 확률 높은 토큰 선택 (greedy)
- **no_repeat_ngram_size**: 반복되는 n-gram 방지

## 3. 다양한 텍스트 생성 모델

### GPT-2 (영어)

```python
from transformers import pipeline

# 작은 모델
generator = pipeline("text-generation", model="gpt2")

# 큰 모델 (더 좋은 품질, 더 느림)
# generator = pipeline("text-generation", model="gpt2-large")
```

### GPT-Neo/GPT-J (더 큰 모델)

```python
# 1.3B 파라미터 모델
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

prompt = "The meaning of life is"
result = generator(prompt, max_length=100)
print(result[0]['generated_text'])
```

### 한국어 모델 (KoGPT)

```python
generator = pipeline("text-generation", model="skt/kogpt2-base-v2")

prompt = "인공지능의 미래는"
result = generator(prompt, max_length=100)
print(result[0]['generated_text'])
```

## 4. 모델과 토크나이저 직접 사용

더 세밀한 제어가 필요한 경우:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 프롬프트 인코딩
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 생성
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# 디코딩
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## 5. 대화형 생성 (챗봇)

### DialoGPT를 사용한 대화

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# 대화 히스토리
chat_history_ids = None

# 대화 시뮬레이션
user_inputs = [
    "Hello, how are you?",
    "What can you do?",
    "Tell me a joke."
]

for user_input in user_inputs:
    print(f"User: {user_input}")

    # 사용자 입력 인코딩
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors='pt'
    )

    # 대화 히스토리와 결합
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # 응답 생성
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # 응답 디코딩
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    print(f"Bot: {response}\n")
```

## 6. 조건부 생성

### 감정을 조절한 생성

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# 긍정적인 톤
positive_prompt = "This wonderful day"
result = generator(positive_prompt, max_length=50)
print("Positive:", result[0]['generated_text'])

# 부정적인 톤
negative_prompt = "This terrible situation"
result = generator(negative_prompt, max_length=50)
print("Negative:", result[0]['generated_text'])
```

### 스타일 제어

```python
# 형식적인 스타일
formal_prompt = "Dear Sir or Madam, I am writing to"
result = generator(formal_prompt, max_length=80)
print("Formal:", result[0]['generated_text'])

# 비형식적인 스타일
casual_prompt = "Hey, guess what happened today?"
result = generator(casual_prompt, max_length=80)
print("Casual:", result[0]['generated_text'])
```

## 7. 실전 예제: 스토리 생성기

```python
from transformers import pipeline

def generate_story(genre, character, setting, max_length=200):
    """장르, 캐릭터, 배경을 기반으로 스토리 생성"""

    generator = pipeline("text-generation", model="gpt2")

    # 프롬프트 구성
    prompt = f"Genre: {genre}\nCharacter: {character}\nSetting: {setting}\n\nStory: "

    # 생성
    result = generator(
        prompt,
        max_length=max_length,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1
    )

    return result[0]['generated_text']

# 예제 실행
story = generate_story(
    genre="Science Fiction",
    character="a brave astronaut",
    setting="a distant planet"
)

print(story)
```

## 8. 요약 생성 (Summarization)

```python
from transformers import pipeline

# 요약 파이프라인
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 긴 텍스트
article = """
Artificial intelligence (AI) is intelligence demonstrated by machines,
as opposed to natural intelligence displayed by animals including humans.
AI research has been defined as the field of study of intelligent agents,
which refers to any system that perceives its environment and takes actions
that maximize its chance of achieving its goals. The term "artificial intelligence"
had previously been used to describe machines that mimic and display "human"
cognitive skills that are associated with the human mind, such as "learning"
and "problem-solving". This definition has since been rejected by major AI
researchers who now describe AI in terms of rationality and acting rationally,
which does not limit how intelligence can be articulated.
"""

# 요약 생성
summary = summarizer(
    article,
    max_length=60,
    min_length=30,
    do_sample=False
)

print("Original length:", len(article.split()))
print("Summary length:", len(summary[0]['summary_text'].split()))
print("\nSummary:", summary[0]['summary_text'])
```

## 9. 번역 (Translation)

```python
from transformers import pipeline

# 영어 -> 한국어
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

text = "Hello, how are you today?"
result = translator(text)
print(f"English: {text}")
print(f"Korean: {result[0]['translation_text']}")

# 한국어 -> 영어
translator_ko_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

korean_text = "안녕하세요, 오늘 기분이 어떠세요?"
result = translator_ko_en(korean_text)
print(f"\nKorean: {korean_text}")
print(f"English: {result[0]['translation_text']}")
```

## 10. 질의응답 생성

```python
from transformers import pipeline

# 질의응답 파이프라인
qa_pipeline = pipeline("question-answering")

# 컨텍스트와 질문
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists
and intellectuals for its design, but it has become a global cultural icon of France and one
of the most recognizable structures in the world. The tower is 330 metres tall.
"""

question = "How tall is the Eiffel Tower?"

result = qa_pipeline(question=question, context=context)

print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.4f}")
```

## 11. 성능 최적화

### GPU 사용

```python
from transformers import pipeline

# GPU 사용 (CUDA 사용 가능 시)
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0  # GPU 0번 사용
)
```

### 반정밀도 (FP16) 사용

```python
import torch

generator = pipeline(
    "text-generation",
    model="gpt2",
    torch_dtype=torch.float16,
    device=0
)
```

### 배치 생성

```python
# 여러 프롬프트 동시 처리
prompts = [
    "Once upon a time",
    "In a galaxy far away",
    "The future of technology"
]

results = generator(prompts, max_length=50, batch_size=3)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Result: {result[0]['generated_text']}\n")
```

## 12. 생성 품질 개선

### Beam Search 사용

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The best way to learn programming is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Beam Search
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,  # 빔 개수
    early_stopping=True,
    no_repeat_ngram_size=2
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 반복 방지

```python
# n-gram 반복 방지
output = model.generate(
    input_ids,
    max_length=100,
    no_repeat_ngram_size=3,  # 3-gram 반복 방지
    do_sample=True,
    top_k=50,
    top_p=0.95
)
```

## 13. 실습 과제

### 과제 1: 시 생성기
주제를 입력받아 시를 생성하는 프로그램을 작성하세요.

### 과제 2: 이메일 자동 완성
이메일 시작 부분을 입력하면 나머지를 자동으로 완성하세요.

### 과제 3: 대화형 챗봇
사용자와 여러 턴 대화할 수 있는 챗봇을 만드세요.

### 과제 4: 다국어 번역기
여러 언어 간 번역을 지원하는 번역기를 구현하세요.

## 14. 주의사항

### 생성된 텍스트의 신뢰성
- AI가 생성한 텍스트는 항상 사실이 아닐 수 있습니다
- 중요한 정보는 반드시 검증하세요
- 편향이나 부적절한 내용이 포함될 수 있습니다

### 윤리적 고려사항
- 허위 정보 생성에 사용하지 마세요
- 저작권을 존중하세요
- 개인정보를 포함하지 마세요
- 혐오 발언이나 유해 콘텐츠 생성을 피하세요

## 다음 단계

다음 튜토리얼에서는 모델 파인튜닝을 다룹니다:
- **튜토리얼 4**: 커스텀 데이터로 모델 파인튜닝

## 참고 자료

- [Hugging Face Text Generation](https://huggingface.co/tasks/text-generation)
- [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [GPT-2 Documentation](https://huggingface.co/gpt2)
- [Model Hub - Text Generation](https://huggingface.co/models?pipeline_tag=text-generation)
