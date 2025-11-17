"""
텍스트 분류 예제
Hugging Face Transformers를 사용한 감성 분석
"""

from transformers import pipeline
import torch

def basic_sentiment_analysis():
    """기본 감성 분석"""
    print("=" * 60)
    print("1. 기본 감성 분석")
    print("=" * 60)

    # 파이프라인 생성
    classifier = pipeline("sentiment-analysis")

    # 단일 텍스트 분류
    text = "I absolutely love this product!"
    result = classifier(text)[0]
    print(f"\nText: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")

    # 여러 텍스트 분류
    texts = [
        "This movie was fantastic!",
        "I wasted my money on this.",
        "It was okay, nothing special."
    ]

    print("\n여러 텍스트 분류:")
    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"  {text}")
        print(f"  -> {result['label']} ({result['score']:.4f})\n")


def zero_shot_classification():
    """Zero-Shot 분류"""
    print("=" * 60)
    print("2. Zero-Shot 분류")
    print("=" * 60)

    classifier = pipeline("zero-shot-classification")

    text = "This is a tutorial about natural language processing."
    candidate_labels = ["technology", "sports", "politics", "education"]

    result = classifier(text, candidate_labels)

    print(f"\nText: {text}")
    print("\n분류 결과:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.4f}")


def multi_label_classification():
    """다중 레이블 분류"""
    print("=" * 60)
    print("3. 다중 레이블 분류")
    print("=" * 60)

    classifier = pipeline("zero-shot-classification")

    text = "This smartphone has an excellent camera and long battery life."
    candidate_labels = ["camera quality", "battery life", "price", "design"]

    result = classifier(
        text,
        candidate_labels,
        multi_label=True
    )

    print(f"\nText: {text}")
    print("\n관련 레이블들:")
    for label, score in zip(result['labels'], result['scores']):
        if score > 0.5:  # 임계값 이상만 출력
            print(f"  ✓ {label}: {score:.4f}")


def detailed_classification():
    """상세한 분류 (모델과 토크나이저 직접 사용)"""
    print("=" * 60)
    print("4. 상세한 분류")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # 모델과 토크나이저 로드
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 텍스트 토크나이징
    text = "This is a wonderful experience!"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    print(f"\nText: {text}")
    print(f"Tokenized input IDs: {inputs['input_ids'][0][:10]}...")  # 처음 10개만

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # 결과 해석
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    predicted_class = torch.argmax(predictions).item()
    confidence = predictions[0][predicted_class].item()

    print(f"\nPrediction: {label_map[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Negative: {predictions[0][0]:.4f}, Positive: {predictions[0][1]:.4f}")


def batch_processing():
    """배치 처리"""
    print("=" * 60)
    print("5. 배치 처리")
    print("=" * 60)

    classifier = pipeline("sentiment-analysis")

    # 여러 텍스트
    texts = [
        "Great product!",
        "Terrible experience.",
        "Average quality.",
        "Absolutely amazing!",
        "Worst purchase ever.",
        "Pretty good overall."
    ]

    # 배치 처리
    results = classifier(texts, batch_size=3)

    # 통계 계산
    positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
    negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')

    print(f"\n총 {len(texts)}개 텍스트 분석:")
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")

    print("\n상세 결과:")
    for text, result in zip(texts, results):
        emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
        print(f"  {emoji} {text[:30]:30s} -> {result['label']:8s} ({result['score']:.4f})")


def movie_review_analysis():
    """영화 리뷰 분석 예제"""
    print("=" * 60)
    print("6. 영화 리뷰 분석")
    print("=" * 60)

    import pandas as pd

    # 샘플 데이터
    reviews = [
        {"text": "This movie was absolutely brilliant!", "rating": 5},
        {"text": "Waste of time and money.", "rating": 1},
        {"text": "Pretty good, would recommend.", "rating": 4},
        {"text": "Boring and predictable plot.", "rating": 2},
        {"text": "Masterpiece! Best film of the year.", "rating": 5},
        {"text": "Not bad, but could be better.", "rating": 3},
        {"text": "Incredible cinematography and acting.", "rating": 5},
        {"text": "Disappointing ending ruined it.", "rating": 2},
    ]

    df = pd.DataFrame(reviews)

    # 감성 분석
    classifier = pipeline("sentiment-analysis")

    sentiments = classifier(df['text'].tolist())
    df['sentiment'] = [s['label'] for s in sentiments]
    df['confidence'] = [s['score'] for s in sentiments]

    print("\n분석 결과:")
    print(df.to_string(index=False))

    # 정확도 계산 (평점 4-5: POSITIVE, 1-2: NEGATIVE로 가정)
    df['expected_sentiment'] = df['rating'].apply(
        lambda x: 'POSITIVE' if x >= 4 else ('NEGATIVE' if x <= 2 else 'NEUTRAL')
    )

    # NEUTRAL 제외하고 정확도 계산
    non_neutral = df[df['expected_sentiment'] != 'NEUTRAL']
    if len(non_neutral) > 0:
        accuracy = (non_neutral['sentiment'] == non_neutral['expected_sentiment']).mean()
        print(f"\n감성 분석 정확도: {accuracy:.2%}")


def error_handling():
    """오류 처리"""
    print("=" * 60)
    print("7. 오류 처리")
    print("=" * 60)

    classifier = pipeline("sentiment-analysis")

    def safe_classify(text):
        try:
            # 빈 텍스트 처리
            if not text or not text.strip():
                return {"label": "UNKNOWN", "score": 0.0, "error": "Empty text"}

            # 너무 긴 텍스트 자르기
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
                truncated = True
            else:
                truncated = False

            result = classifier(text)[0]
            if truncated:
                result['warning'] = 'Text was truncated'
            return result

        except Exception as e:
            return {"label": "ERROR", "score": 0.0, "error": str(e)}

    # 테스트 케이스
    test_cases = [
        ("Normal text for analysis", "정상 텍스트"),
        ("", "빈 텍스트"),
        ("   ", "공백만 있는 텍스트"),
        ("A" * 1000, "매우 긴 텍스트"),
    ]

    print("\n다양한 입력 처리:")
    for text, description in test_cases:
        result = safe_classify(text)
        print(f"\n{description}:")
        if 'error' in result:
            print(f"  ⚠️  Error: {result['error']}")
        elif 'warning' in result:
            print(f"  ⚠️  {result['warning']}")
            print(f"  Result: {result['label']} ({result['score']:.4f})")
        else:
            print(f"  ✓ Result: {result['label']} ({result['score']:.4f})")


def compare_models():
    """여러 모델 비교"""
    print("=" * 60)
    print("8. 여러 모델 비교")
    print("=" * 60)

    # 비교할 모델들
    models = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]

    text = "This product exceeded my expectations!"

    print(f"\nText: {text}\n")

    for model_name in models:
        try:
            print(f"Model: {model_name}")
            classifier = pipeline("sentiment-analysis", model=model_name)
            result = classifier(text)[0]
            print(f"  Result: {result['label']} (confidence: {result['score']:.4f})\n")
        except Exception as e:
            print(f"  Error: {e}\n")


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
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ CUDA is not available. Using CPU.")

    print()


def main():
    """메인 함수"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        Hugging Face 텍스트 분류 예제                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # 시스템 정보
    check_device()

    # 예제 실행
    try:
        basic_sentiment_analysis()
        print("\n")

        zero_shot_classification()
        print("\n")

        multi_label_classification()
        print("\n")

        detailed_classification()
        print("\n")

        batch_processing()
        print("\n")

        movie_review_analysis()
        print("\n")

        error_handling()
        print("\n")

        compare_models()
        print("\n")

        print("=" * 60)
        print("모든 예제가 성공적으로 완료되었습니다!")
        print("=" * 60)

    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("필요한 라이브러리를 설치했는지 확인하세요:")
        print("  pip install transformers torch pandas")


if __name__ == "__main__":
    main()
