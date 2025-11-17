# 튜토리얼 4: 모델 파인튜닝 (Fine-tuning)

## 목표
이 튜토리얼에서는 사전 훈련된 모델을 커스텀 데이터셋으로 파인튜닝하는 방법을 배웁니다.

## 파인튜닝이란?

파인튜닝은 사전 훈련된 모델을 특정 작업이나 도메인에 맞게 추가로 학습시키는 과정입니다.

### 파인튜닝의 장점
- **빠른 학습**: 처음부터 학습하는 것보다 훨씬 빠름
- **적은 데이터**: 소량의 데이터로도 좋은 성능
- **비용 절감**: 계산 자원과 시간 절약
- **높은 성능**: 사전 학습된 지식 활용

## 1. 환경 준비

### 필수 라이브러리 설치

```bash
pip install transformers datasets
pip install torch torchvision
pip install accelerate -U
pip install evaluate
pip install scikit-learn
```

## 2. 데이터셋 준비

### Hugging Face Datasets 사용

```python
from datasets import load_dataset

# 공개 데이터셋 로드
dataset = load_dataset("imdb")

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

# 샘플 확인
print(dataset['train'][0])
```

### 커스텀 데이터셋 로드

```python
from datasets import Dataset
import pandas as pd

# CSV에서 로드
data = {
    'text': [
        "I love this product!",
        "This is terrible.",
        "Amazing quality!"
    ],
    'label': [1, 0, 1]
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Train/Test 분할
dataset = dataset.train_test_split(test_size=0.2)
```

### 로컬 파일에서 로드

```python
from datasets import load_dataset

# CSV 파일
dataset = load_dataset('csv', data_files='data.csv')

# JSON 파일
dataset = load_dataset('json', data_files='data.json')

# 텍스트 파일
dataset = load_dataset('text', data_files='data.txt')
```

## 3. 토크나이저 준비

```python
from transformers import AutoTokenizer

# 토크나이저 로드
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 토크나이징 함수
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# 데이터셋에 적용
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

## 4. 기본 파인튜닝 (Trainer API)

### 모델 로드

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 이진 분류
)
```

### Training Arguments 설정

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)
```

### Trainer 생성 및 학습

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# 학습 시작
trainer.train()

# 평가
results = trainer.evaluate()
print(results)
```

## 5. 평가 메트릭 추가

```python
import numpy as np
from datasets import load_metric

# 메트릭 로드
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric.compute(predictions=predictions, references=labels)

    return accuracy

# Trainer에 메트릭 추가
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

## 6. 다양한 메트릭 사용

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
```

## 7. 데이터 증강 (Data Augmentation)

```python
# 간단한 텍스트 증강
def augment_text(text):
    """동의어 치환, 단어 순서 변경 등"""
    # 예시: 간단한 증강
    return text

# 데이터셋에 적용
def augment_dataset(examples):
    examples['text'] = [augment_text(t) for t in examples['text']]
    return examples

augmented_dataset = dataset.map(augment_dataset, batched=True)
```

## 8. Early Stopping

```python
from transformers import TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

## 9. 학습률 스케줄링

```python
from transformers import get_linear_schedule_with_warmup

# Warmup steps 계산
num_training_steps = len(tokenized_datasets['train']) * training_args.num_train_epochs // training_args.per_device_train_batch_size
num_warmup_steps = num_training_steps // 10

# 스케줄러
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

## 10. 모델 저장 및 로드

### 모델 저장

```python
# 모델 저장
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 또는 Trainer를 통해
trainer.save_model("./my_model")
```

### 모델 로드 및 사용

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 모델과 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")

# 파이프라인 생성
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 추론
result = classifier("This is a great product!")
print(result)
```

## 11. GPU 메모리 최적화

### Gradient Accumulation

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # 작은 배치 사이즈
    gradient_accumulation_steps=4,  # 4번 누적 후 업데이트
    # 실제 배치 사이즈 = 4 * 4 = 16
)
```

### Mixed Precision Training

```python
training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,  # FP16 사용
    # 또는
    bf16=True,  # BF16 사용 (Ampere GPU 이상)
)
```

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./results",
    gradient_checkpointing=True,
)
```

## 12. 분산 학습

### 단일 노드, 다중 GPU

```bash
# torchrun 사용
torchrun --nproc_per_node=2 train.py

# accelerate 사용
accelerate launch train.py
```

### Accelerate 설정

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 학습 루프
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## 13. 하이퍼파라미터 튜닝

### Optuna를 사용한 자동 튜닝

```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
    }

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
)
```

## 14. 실전 예제: 감성 분석 모델 파인튜닝

### 완전한 파인튜닝 파이프라인

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. 데이터 로드
dataset = load_dataset("imdb")
small_train = dataset['train'].shuffle(seed=42).select(range(1000))
small_test = dataset['test'].shuffle(seed=42).select(range(200))

# 2. 토크나이저
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train = small_train.map(tokenize_function, batched=True)
tokenized_test = small_test.map(tokenize_function, batched=True)

# 3. 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 4. 메트릭
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. 학습
trainer.train()

# 8. 평가
results = trainer.evaluate()
print(results)

# 9. 저장
trainer.save_model("./sentiment_model_final")
```

## 15. LoRA를 사용한 효율적 파인튜닝

Parameter-Efficient Fine-Tuning (PEFT):

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
)

# PEFT 모델 생성
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()

# 일반적인 방식으로 학습
trainer = Trainer(...)
trainer.train()
```

## 16. 모니터링 및 시각화

### TensorBoard

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir='./logs',
    logging_steps=10,
)

# TensorBoard 실행
# tensorboard --logdir ./logs
```

### Weights & Biases

```python
# wandb 설치
# pip install wandb

import wandb

wandb.init(project="my-finetuning")

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
)
```

## 17. 주의사항

### Overfitting 방지
- 적절한 dropout 사용
- Early stopping
- 데이터 증강
- 정규화 (weight decay)

### 학습 팁
- 작은 learning rate 사용 (2e-5 ~ 5e-5)
- Warmup 사용
- 배치 사이즈 조정
- Gradient clipping

## 18. 실습 과제

### 과제 1: 감성 분석 모델
IMDB 데이터셋으로 영화 리뷰 감성 분석 모델을 파인튜닝하세요.

### 과제 2: 주제 분류
뉴스 기사를 카테고리별로 분류하는 모델을 만드세요.

### 과제 3: 다국어 모델
한국어 텍스트 분류 모델을 파인튜닝하세요.

### 과제 4: 최적화
LoRA를 사용하여 메모리 효율적인 파인튜닝을 구현하세요.

## 다음 단계

다음 튜토리얼에서는 고급 주제를 다룹니다:
- **튜토리얼 5**: 모델 배포 및 최적화

## 참고 자료

- [Hugging Face Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)
- [Trainer API Documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
