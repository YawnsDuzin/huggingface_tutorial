"""
모델 파인튜닝 예제
Hugging Face Transformers를 사용한 모델 파인튜닝
"""

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch


def prepare_custom_dataset():
    """커스텀 데이터셋 준비"""
    print("=" * 60)
    print("1. 커스텀 데이터셋 준비")
    print("=" * 60)

    # 샘플 데이터 생성
    data = {
        'text': [
            "I love this product! It's amazing.",
            "This is the worst thing I ever bought.",
            "Pretty good, would recommend to friends.",
            "Terrible quality, waste of money.",
            "Excellent service and fast delivery!",
            "Not bad, but could be better.",
            "Absolutely fantastic experience!",
            "Disappointing and overpriced.",
            "Great value for money!",
            "Poor customer service.",
            "Best purchase I've made this year!",
            "Low quality materials.",
            "Exceeded my expectations!",
            "Complete waste of time.",
            "Highly recommended product!",
            "Defective item received.",
        ] * 10,  # 160 샘플
        'label': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
    }

    df = pd.DataFrame(data)

    # Dataset 생성
    dataset = Dataset.from_pandas(df)

    # Train/Test 분할
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"\nTrain samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"\n샘플 데이터:")
    print(dataset['train'][0])

    return dataset


def tokenize_dataset(dataset, model_name="distilbert-base-uncased"):
    """데이터셋 토크나이징"""
    print("\n" + "=" * 60)
    print("2. 데이터셋 토크나이징")
    print("=" * 60)

    # 토크나이저 로드
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

    print(f"\n토크나이저: {model_name}")
    print(f"최대 길이: 128")
    print(f"\n토크나이징 완료!")

    # 샘플 확인
    print(f"\n샘플 토큰 IDs: {tokenized_datasets['train'][0]['input_ids'][:20]}...")

    return tokenized_datasets, tokenizer


def create_compute_metrics():
    """평가 메트릭 함수 생성"""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # 정확도
        accuracy = accuracy_score(labels, predictions)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='binary'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    return compute_metrics


def basic_fine_tuning(tokenized_datasets, tokenizer):
    """기본 파인튜닝"""
    print("\n" + "=" * 60)
    print("3. 기본 파인튜닝")
    print("=" * 60)

    # 모델 로드
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    print(f"\n모델: {model_name}")
    print(f"레이블 수: 2 (긍정/부정)")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,  # 최대 2개 체크포인트만 저장
    )

    print(f"\n학습 설정:")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Epochs: {training_args.num_train_epochs}")

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=create_compute_metrics(),
    )

    print("\n학습 시작...")
    print("-" * 60)

    # 학습
    trainer.train()

    print("\n학습 완료!")

    # 평가
    print("\n최종 평가:")
    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return trainer, model


def save_and_load_model(trainer, tokenizer):
    """모델 저장 및 로드"""
    print("\n" + "=" * 60)
    print("4. 모델 저장 및 로드")
    print("=" * 60)

    # 모델 저장
    save_path = "./my_sentiment_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n모델 저장 완료: {save_path}")

    # 모델 로드
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=save_path,
        tokenizer=save_path
    )

    print(f"모델 로드 완료!")

    # 테스트
    test_texts = [
        "This product is absolutely amazing!",
        "Worst purchase ever, very disappointed.",
        "Pretty decent for the price."
    ]

    print("\n모델 테스트:")
    for text in test_texts:
        result = classifier(text)[0]
        label = "긍정" if result['label'] == 'LABEL_1' else "부정"
        print(f"\n  Text: {text}")
        print(f"  Prediction: {label} (confidence: {result['score']:.4f})")

    return classifier


def demonstrate_overfitting():
    """오버피팅 예제 및 방지법"""
    print("\n" + "=" * 60)
    print("5. 오버피팅 방지")
    print("=" * 60)

    print("\n오버피팅 방지 기법:")
    print("  1. Early Stopping - 검증 성능이 개선되지 않으면 조기 종료")
    print("  2. Dropout - 랜덤하게 뉴런 비활성화")
    print("  3. Weight Decay - 가중치 정규화")
    print("  4. Data Augmentation - 데이터 증강")
    print("  5. Learning Rate Scheduling - 학습률 조정")

    print("\nTrainingArguments 예제:")
    print("""
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,  # L2 정규화
        load_best_model_at_end=True,  # 최고 성능 모델 로드
        metric_for_best_model="accuracy",
    )
    """)


def memory_optimization_tips():
    """메모리 최적화 팁"""
    print("\n" + "=" * 60)
    print("6. 메모리 최적화")
    print("=" * 60)

    print("\n메모리 최적화 기법:")

    print("\n1. Gradient Accumulation:")
    print("   - 작은 배치를 여러 번 누적하여 큰 배치 효과")
    print("""
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # 실제 배치 = 16
    )
    """)

    print("\n2. Mixed Precision Training:")
    print("   - FP16 또는 BF16 사용으로 메모리 절약")
    print("""
    training_args = TrainingArguments(
        fp16=True,  # NVIDIA GPU
        # 또는
        bf16=True,  # Ampere 이상 GPU
    )
    """)

    print("\n3. Gradient Checkpointing:")
    print("   - 메모리와 계산 시간 트레이드오프")
    print("""
    model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        gradient_checkpointing=True,
    )
    """)


def hyperparameter_search_example():
    """하이퍼파라미터 탐색 예제"""
    print("\n" + "=" * 60)
    print("7. 하이퍼파라미터 탐색")
    print("=" * 60)

    print("\n하이퍼파라미터 탐색 방법:")
    print("  1. Grid Search - 모든 조합 시도")
    print("  2. Random Search - 랜덤 샘플링")
    print("  3. Bayesian Optimization - 베이지안 최적화")
    print("  4. Optuna - 자동 하이퍼파라미터 튜닝")

    print("\nOptuna 사용 예제:")
    print("""
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
        }

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=10,
    )
    """)


def practical_tips():
    """실전 팁"""
    print("\n" + "=" * 60)
    print("8. 실전 파인튜닝 팁")
    print("=" * 60)

    print("\n✓ 데이터 준비:")
    print("  - 최소 수백~수천 개의 레이블된 샘플")
    print("  - 균형잡힌 클래스 분포")
    print("  - 고품질 레이블")

    print("\n✓ 모델 선택:")
    print("  - 작은 데이터: DistilBERT, ALBERT")
    print("  - 큰 데이터: BERT, RoBERTa")
    print("  - 다국어: mBERT, XLM-RoBERTa")

    print("\n✓ 학습 설정:")
    print("  - Learning rate: 2e-5 ~ 5e-5")
    print("  - Epochs: 3 ~ 5")
    print("  - Warmup: 전체 스텝의 10%")
    print("  - Weight decay: 0.01")

    print("\n✓ 모니터링:")
    print("  - Train/Validation loss 그래프")
    print("  - 메트릭 추이 확인")
    print("  - 오버피팅 감지")

    print("\n✓ 검증:")
    print("  - 별도의 테스트 셋 사용")
    print("  - Cross-validation 고려")
    print("  - 실제 데이터로 테스트")


def check_device():
    """사용 가능한 디바이스 확인"""
    print("=" * 60)
    print("시스템 정보")
    print("=" * 60)

    import transformers

    print(f"\nTransformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA is not available. Using CPU.")
        print("  (파인튜닝은 GPU 사용을 권장합니다)")

    print()


def main():
    """메인 함수"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        Hugging Face 모델 파인튜닝 예제                      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # 시스템 정보
    check_device()

    try:
        # 1. 데이터셋 준비
        dataset = prepare_custom_dataset()

        # 2. 토크나이징
        tokenized_datasets, tokenizer = tokenize_dataset(dataset)

        # 3. 파인튜닝 (간단한 예제로 epoch 수를 줄임)
        print("\n" + "=" * 60)
        print("주의: 이 예제는 데모용으로 작은 데이터셋을 사용합니다.")
        print("실제 프로젝트에서는 더 많은 데이터와 에폭이 필요합니다.")
        print("=" * 60)

        # 사용자에게 학습 여부 확인
        print("\n파인튜닝을 시작하시겠습니까?")
        print("(학습에는 몇 분이 걸릴 수 있습니다)")
        print("\n계속하려면 이 코드를 주석 해제하고 실행하세요:")
        print("# trainer, model = basic_fine_tuning(tokenized_datasets, tokenizer)")
        print("# classifier = save_and_load_model(trainer, tokenizer)")

        # 실제 학습은 주석 처리 (데모 목적)
        # trainer, model = basic_fine_tuning(tokenized_datasets, tokenizer)
        # classifier = save_and_load_model(trainer, tokenizer)

        # 4. 이론 및 팁
        demonstrate_overfitting()
        memory_optimization_tips()
        hyperparameter_search_example()
        practical_tips()

        print("\n" + "=" * 60)
        print("파인튜닝 가이드 완료!")
        print("=" * 60)

        print("\n다음 단계:")
        print("  1. 실제 데이터로 파인튜닝 시도")
        print("  2. 하이퍼파라미터 튜닝")
        print("  3. 모델 평가 및 분석")
        print("  4. 프로덕션 배포")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n필요한 라이브러리를 설치했는지 확인하세요:")
        print("  pip install transformers datasets torch scikit-learn pandas")


if __name__ == "__main__":
    main()
