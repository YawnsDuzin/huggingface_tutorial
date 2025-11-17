"""
텍스트 생성 예제
Hugging Face Transformers를 사용한 다양한 텍스트 생성
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def basic_text_generation():
    """기본 텍스트 생성"""
    print("=" * 60)
    print("1. 기본 텍스트 생성 (GPT-2)")
    print("=" * 60)

    # 텍스트 생성 파이프라인
    generator = pipeline("text-generation", model="gpt2")

    # 단일 결과 생성
    prompt = "Once upon a time"
    result = generator(prompt, max_length=50, num_return_sequences=1)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}")


def multiple_generations():
    """여러 버전 생성"""
    print("\n" + "=" * 60)
    print("2. 여러 버전 생성")
    print("=" * 60)

    generator = pipeline("text-generation", model="gpt2")

    prompt = "The future of artificial intelligence is"
    results = generator(
        prompt,
        max_length=60,
        num_return_sequences=3,
        temperature=0.8,
        do_sample=True
    )

    print(f"\nPrompt: {prompt}\n")
    for i, result in enumerate(results, 1):
        print(f"Version {i}:")
        print(f"  {result['generated_text']}\n")


def generation_parameters():
    """생성 파라미터 비교"""
    print("=" * 60)
    print("3. 생성 파라미터 비교")
    print("=" * 60)

    generator = pipeline("text-generation", model="gpt2")
    prompt = "In the year 2050,"

    # 낮은 temperature (보수적)
    print("\n[Low Temperature - 보수적 생성]")
    result = generator(
        prompt,
        max_length=60,
        temperature=0.3,
        do_sample=True,
        num_return_sequences=1
    )
    print(result[0]['generated_text'])

    # 높은 temperature (창의적)
    print("\n[High Temperature - 창의적 생성]")
    result = generator(
        prompt,
        max_length=60,
        temperature=1.2,
        do_sample=True,
        num_return_sequences=1
    )
    print(result[0]['generated_text'])

    # Greedy decoding (결정적)
    print("\n[Greedy Decoding - 결정적 생성]")
    result = generator(
        prompt,
        max_length=60,
        do_sample=False,
        num_return_sequences=1
    )
    print(result[0]['generated_text'])


def detailed_generation():
    """상세한 생성 제어"""
    print("\n" + "=" * 60)
    print("4. 상세한 생성 제어")
    print("=" * 60)

    # 모델과 토크나이저 로드
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 프롬프트 인코딩
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\nPrompt: {prompt}")
    print(f"Input IDs: {input_ids[0][:10].tolist()}...")

    # 생성
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=2
        )

    # 디코딩
    print("\n생성된 텍스트:")
    for i, seq in enumerate(output, 1):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"{i}. {text}\n")


def story_generator():
    """스토리 생성기"""
    print("=" * 60)
    print("5. 인터랙티브 스토리 생성기")
    print("=" * 60)

    generator = pipeline("text-generation", model="gpt2")

    # 스토리 설정
    scenarios = [
        {
            "genre": "Science Fiction",
            "character": "a brave astronaut",
            "setting": "a distant planet"
        },
        {
            "genre": "Mystery",
            "character": "a detective",
            "setting": "an old mansion"
        },
        {
            "genre": "Fantasy",
            "character": "a young wizard",
            "setting": "a magical forest"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        # 프롬프트 구성
        prompt = (
            f"{scenario['genre']} story about {scenario['character']} "
            f"in {scenario['setting']}. "
        )

        # 생성
        result = generator(
            prompt,
            max_length=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )

        print(f"\n[Story {i}]")
        print(f"Genre: {scenario['genre']}")
        print(f"Character: {scenario['character']}")
        print(f"Setting: {scenario['setting']}\n")
        print(result[0]['generated_text'])
        print("-" * 60)


def summarization_example():
    """텍스트 요약"""
    print("\n" + "=" * 60)
    print("6. 텍스트 요약")
    print("=" * 60)

    try:
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
        which does not limit how intelligence can be articulated. AI applications include
        advanced web search engines, recommendation systems, understanding human speech,
        self-driving cars, automated decision-making and competing at the highest level
        in strategic game systems.
        """

        # 요약 생성
        summary = summarizer(
            article,
            max_length=80,
            min_length=40,
            do_sample=False
        )

        print(f"\nOriginal length: {len(article.split())} words")
        print(f"Summary length: {len(summary[0]['summary_text'].split())} words")
        print(f"\nOriginal:\n{article}")
        print(f"\nSummary:\n{summary[0]['summary_text']}")

    except Exception as e:
        print(f"요약 예제를 건너뜁니다: {e}")
        print("(BART 모델이 크므로 다운로드에 시간이 걸릴 수 있습니다)")


def question_answering():
    """질의응답"""
    print("\n" + "=" * 60)
    print("7. 질의응답")
    print("=" * 60)

    qa_pipeline = pipeline("question-answering")

    # 컨텍스트와 질문들
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading
    artists and intellectuals for its design, but it has become a global cultural icon of
    France and one of the most recognizable structures in the world. The tower is 330 metres
    tall, about the same height as an 81-story building. It was the tallest man-made structure
    in the world until the Chrysler Building in New York was finished in 1930.
    """

    questions = [
        "How tall is the Eiffel Tower?",
        "When was the Eiffel Tower built?",
        "Who designed the Eiffel Tower?",
        "Where is the Eiffel Tower located?"
    ]

    print(f"\nContext: {context[:100]}...\n")

    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['score']:.4f})\n")


def creative_writing():
    """창의적 글쓰기"""
    print("=" * 60)
    print("8. 창의적 글쓰기")
    print("=" * 60)

    generator = pipeline("text-generation", model="gpt2")

    # 다양한 스타일의 프롬프트
    prompts = [
        ("Formal Email", "Dear Sir or Madam, I am writing to express my"),
        ("Casual Chat", "Hey! Guess what happened to me today?"),
        ("News Article", "Breaking News: Scientists have discovered that"),
        ("Recipe", "Here's how to make the perfect chocolate cake:")
    ]

    print("\n다양한 스타일의 텍스트 생성:\n")
    for style, prompt in prompts:
        result = generator(
            prompt,
            max_length=80,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )

        print(f"[{style}]")
        print(result[0]['generated_text'])
        print("-" * 60)


def beam_search_example():
    """Beam Search를 사용한 생성"""
    print("\n" + "=" * 60)
    print("9. Beam Search vs Sampling")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    prompt = "The best way to learn programming is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Beam Search
    print("\n[Beam Search - 더 일관성 있는 결과]")
    output = model.generate(
        input_ids,
        max_length=60,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # Sampling
    print("\n[Sampling - 더 다양한 결과]")
    output = model.generate(
        input_ids,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        no_repeat_ngram_size=2
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def repetition_control():
    """반복 제어"""
    print("\n" + "=" * 60)
    print("10. 반복 제어")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    prompt = "I really love"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # 반복 제어 없음
    print("\n[반복 제어 없음]")
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # 반복 제어 있음
    print("\n[반복 제어 있음 (no_repeat_ngram_size=3)]")
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        no_repeat_ngram_size=3
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def batch_generation():
    """배치 생성"""
    print("\n" + "=" * 60)
    print("11. 배치 생성 (여러 프롬프트 동시 처리)")
    print("=" * 60)

    generator = pipeline("text-generation", model="gpt2")

    prompts = [
        "Once upon a time",
        "In a galaxy far away",
        "The future of technology"
    ]

    print("\n여러 프롬프트 동시 처리:\n")

    # 배치 처리
    results = generator(
        prompts,
        max_length=50,
        num_return_sequences=1,
        batch_size=len(prompts)
    )

    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Generated: {result[0]['generated_text']}\n")


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
    else:
        print("✗ CUDA is not available. Using CPU.")

    print()


def main():
    """메인 함수"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        Hugging Face 텍스트 생성 예제                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # 시스템 정보
    check_device()

    try:
        # 예제 실행
        basic_text_generation()
        print("\n")

        multiple_generations()
        print("\n")

        generation_parameters()
        print("\n")

        detailed_generation()
        print("\n")

        story_generator()
        print("\n")

        summarization_example()
        print("\n")

        question_answering()
        print("\n")

        creative_writing()
        print("\n")

        beam_search_example()
        print("\n")

        repetition_control()
        print("\n")

        batch_generation()
        print("\n")

        print("=" * 60)
        print("모든 예제가 성공적으로 완료되었습니다!")
        print("=" * 60)

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n필요한 라이브러리를 설치했는지 확인하세요:")
        print("  pip install transformers torch")


if __name__ == "__main__":
    main()
