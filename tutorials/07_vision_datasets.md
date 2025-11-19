# Vision 데이터셋 가이드

이 문서는 컴퓨터 비전 분야에서 가장 널리 사용되는 데이터셋들에 대한 상세한 정보를 제공합니다. Hugging Face를 통해 이러한 데이터셋들을 쉽게 활용할 수 있는 방법도 함께 소개합니다.

## 목차
1. [ImageNet](#imagenet)
2. [COCO (Common Objects in Context)](#coco)
3. [MNIST](#mnist)
4. [CIFAR-10/100](#cifar-10100)
5. [Pascal VOC](#pascal-voc)
6. [Fashion-MNIST](#fashion-mnist)
7. [CelebA](#celeba)
8. [Places365](#places365)
9. [ADE20K](#ade20k)
10. [Cityscapes](#cityscapes)
11. [Open Images](#open-images)
12. [Hugging Face에서 Vision 데이터셋 사용하기](#hugging-face에서-vision-데이터셋-사용하기)

---

## ImageNet

### 개요
ImageNet은 컴퓨터 비전 연구에서 가장 영향력 있는 대규모 이미지 데이터셋입니다. 2009년에 시작된 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)를 통해 딥러닝 혁명의 촉매제 역할을 했습니다.

### 주요 통계
- **이미지 수**: 1,400만 개 이상의 이미지
- **클래스 수**: 22,000개 이상의 카테고리
- **ILSVRC 버전**: 1,000개 클래스, 약 120만 개의 학습 이미지
- **용량**: 약 150GB (ILSVRC2012 기준)

### 특징
- WordNet 계층 구조를 기반으로 한 체계적인 분류
- 각 이미지는 수동으로 레이블링
- 다양한 해상도와 종횡비
- 실제 환경의 다양한 조건 반영

### 주요 활용 분야
- 이미지 분류 (Image Classification)
- 전이 학습의 사전 학습 데이터셋
- 모델 벤치마킹
- Feature Extraction

### 접근 방법
```python
# ImageNet은 라이선스 제약으로 직접 다운로드가 필요합니다
# 대안으로 ImageNet-1K의 일부를 사용할 수 있습니다
from datasets import load_dataset

# ImageNet Sketch (테스트용 변형 버전)
dataset = load_dataset("imagenet_sketch")
```

### 중요 이정표
- 2012년: AlexNet이 ILSVRC에서 획기적인 성능 달성
- 2015년: ResNet이 인간 수준의 성능 달성
- 현재: 대부분의 비전 모델의 표준 벤치마크

---

## COCO

### 개요
COCO (Common Objects in Context)는 객체 탐지, 세그멘테이션, 캡셔닝을 위한 대규모 데이터셋입니다. 일상적인 장면에서 객체들의 맥락적 관계를 이해하는 것을 목표로 합니다.

### 주요 통계
- **이미지 수**: 330,000개 이상
- **객체 인스턴스**: 250만 개 이상의 레이블링된 인스턴스
- **객체 카테고리**: 80개
- **사람 키포인트**: 25만 명 이상
- **이미지 캡션**: 5개의 캡션/이미지

### 특징
- 복잡한 일상 장면
- 객체별 인스턴스 세그멘테이션
- 키포인트 탐지 (사람의 관절 등)
- 풍부한 맥락 정보
- Panoptic 세그멘테이션 지원

### 주요 활용 분야
- 객체 탐지 (Object Detection)
- 인스턴스 세그멘테이션 (Instance Segmentation)
- 키포인트 탐지 (Keypoint Detection)
- 이미지 캡셔닝 (Image Captioning)
- Panoptic 세그멘테이션

### 데이터셋 구조
```
80개 객체 카테고리:
- 사람: person
- 차량: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- 동물: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- 가구: chair, couch, potted plant, bed, dining table, toilet
- 음식: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza...
등
```

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# COCO 2017 데이터셋 로드
dataset = load_dataset("detection-datasets/coco")

# 첫 번째 샘플 확인
print(dataset['train'][0])

# 이미지와 어노테이션 접근
example = dataset['train'][0]
image = example['image']
annotations = example['objects']
```

### 평가 메트릭
- mAP (mean Average Precision)
- AP50, AP75 (IoU 임계값별 정확도)
- 크기별 성능 (small, medium, large objects)

---

## MNIST

### 개요
MNIST (Modified National Institute of Standards and Technology)는 손글씨 숫자 인식을 위한 고전적인 데이터셋입니다. 머신러닝 입문자들이 가장 먼저 접하는 데이터셋 중 하나입니다.

### 주요 통계
- **학습 이미지**: 60,000개
- **테스트 이미지**: 10,000개
- **클래스**: 10개 (숫자 0-9)
- **이미지 크기**: 28x28 픽셀 (그레이스케일)
- **용량**: 약 11MB

### 특징
- 중앙 정렬된 숫자
- 정규화된 크기
- 노이즈가 적은 깨끗한 이미지
- 균형잡힌 클래스 분포

### 주요 활용 분야
- 딥러닝 입문 교육
- 새로운 모델 아키텍처 테스트
- 분류 알고리즘 프로토타이핑
- 오토인코더, GAN 실험

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# MNIST 데이터셋 로드
mnist = load_dataset("mnist")

# 데이터 확인
print(f"학습 샘플 수: {len(mnist['train'])}")
print(f"테스트 샘플 수: {len(mnist['test'])}")

# 첫 번째 이미지 보기
image = mnist['train'][0]['image']
label = mnist['train'][0]['label']
print(f"레이블: {label}")
```

### 간단한 예제 코드
```python
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch

# 데이터셋 로드
dataset = load_dataset("mnist")

# 전처리
feature_extractor = AutoFeatureExtractor.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")

def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

# 데이터 변환
prepared_dataset = dataset['train'].with_transform(transform)
```

### 성능 벤치마크
- 전통적인 ML: ~95-98% 정확도
- CNN: ~99%+ 정확도
- 최신 모델: ~99.8% 정확도

---

## CIFAR-10/100

### 개요
CIFAR (Canadian Institute For Advanced Research)는 작은 크기의 컬러 이미지 분류 데이터셋입니다. CIFAR-10과 CIFAR-100 두 버전이 있습니다.

### CIFAR-10 통계
- **학습 이미지**: 50,000개
- **테스트 이미지**: 10,000개
- **클래스**: 10개
- **이미지 크기**: 32x32 픽셀 (RGB)
- **용량**: 약 170MB

### CIFAR-100 통계
- **학습 이미지**: 50,000개
- **테스트 이미지**: 10,000개
- **클래스**: 100개 (20개의 상위 클래스)
- **이미지 크기**: 32x32 픽셀 (RGB)

### CIFAR-10 클래스
```
10개 클래스:
1. airplane (비행기)
2. automobile (자동차)
3. bird (새)
4. cat (고양이)
5. deer (사슴)
6. dog (개)
7. frog (개구리)
8. horse (말)
9. ship (배)
10. truck (트럭)
```

### CIFAR-100 구조
- 20개의 상위 클래스 (superclass)
- 각 상위 클래스당 5개의 하위 클래스
- 예: 상위 클래스 "aquatic mammals"에는 beaver, dolphin, otter, seal, whale 포함

### 특징
- 작은 이미지 크기로 빠른 실험 가능
- 다양한 관점과 조명 조건
- 실제 사진에서 추출
- 클래스당 균등한 샘플 수

### 주요 활용 분야
- 이미지 분류 모델 개발
- 데이터 증강 기법 테스트
- 정규화 기법 실험
- 경량 모델 벤치마킹

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# CIFAR-10 로드
cifar10 = load_dataset("cifar10")

# CIFAR-100 로드
cifar100 = load_dataset("cifar100")

# 데이터 확인
print(f"CIFAR-10 클래스: {cifar10['train'].features['label'].names}")
print(f"CIFAR-100 fine labels: {cifar100['train'].features['fine_label'].num_classes}")
print(f"CIFAR-100 coarse labels: {cifar100['train'].features['coarse_label'].num_classes}")

# 샘플 확인
example = cifar10['train'][0]
print(f"이미지 크기: {example['image'].size}")
print(f"레이블: {example['label']}")
```

### 도전 과제
- 낮은 해상도 (32x32)
- 클래스 간 유사성 (특히 CIFAR-100)
- 제한된 학습 데이터

---

## Pascal VOC

### 개요
Pascal VOC (Visual Object Classes)는 2005년부터 2012년까지 진행된 객체 탐지 및 세그멘테이션 챌린지의 데이터셋입니다. 객체 탐지 연구의 표준 벤치마크로 사용되어 왔습니다.

### 주요 통계
- **이미지 수**: 약 11,500개 (VOC2012 기준)
- **객체 클래스**: 20개
- **주석된 객체**: 27,000개 이상
- **세그멘테이션**: 픽셀 단위 세그멘테이션 마스크 제공

### 20개 객체 클래스
```
4개 카테고리로 분류:

Person: person

Animals: bird, cat, cow, dog, horse, sheep

Vehicles: aeroplane, bicycle, boat, bus, car, motorbike, train

Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
```

### 특징
- Bounding box 어노테이션
- 객체별 세그멘테이션 마스크
- Occlusion 및 truncation 플래그
- 포즈 정보 (일부 클래스)
- 다양한 난이도 레벨

### 주요 활용 분야
- 객체 탐지 (Object Detection)
- 의미론적 세그멘테이션 (Semantic Segmentation)
- 인스턴스 세그멘테이션
- 분류 (Classification)
- Action Recognition

### 데이터셋 버전
- **VOC2007**: 9,963개 이미지
- **VOC2012**: 11,530개 이미지 (가장 널리 사용)

### XML 어노테이션 구조
```xml
<annotation>
    <folder>VOC2012</folder>
    <filename>2007_000027.jpg</filename>
    <size>
        <width>486</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>174</xmin>
            <ymin>101</ymin>
            <xmax>349</xmax>
            <ymax>351</ymax>
        </bndbox>
        <difficult>0</difficult>
        <truncated>0</truncated>
    </object>
</annotation>
```

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# Pascal VOC 2012 로드
voc2012 = load_dataset("detection-datasets/pascal_voc", "2012")

# 데이터 구조 확인
example = voc2012['train'][0]
print(f"이미지 ID: {example['image_id']}")
print(f"객체 수: {len(example['objects']['bbox'])}")
print(f"클래스: {example['objects']['label']}")
```

### 평가 메트릭
- **mAP (mean Average Precision)**: IoU ≥ 0.5
- 클래스별 AP (Average Precision)
- Precision-Recall 곡선

---

## Fashion-MNIST

### 개요
Fashion-MNIST는 MNIST를 대체하기 위해 Zalando Research에서 만든 데이터셋입니다. MNIST와 동일한 구조를 가지고 있지만 패션 아이템 이미지로 구성되어 있어 더 도전적입니다.

### 주요 통계
- **학습 이미지**: 60,000개
- **테스트 이미지**: 10,000개
- **클래스**: 10개
- **이미지 크기**: 28x28 픽셀 (그레이스케일)
- **용량**: 약 30MB

### 10개 클래스
```
0. T-shirt/top (티셔츠/상의)
1. Trouser (바지)
2. Pullover (풀오버)
3. Dress (드레스)
4. Coat (코트)
5. Sandal (샌들)
6. Shirt (셔츠)
7. Sneaker (스니커즈)
8. Bag (가방)
9. Ankle boot (앵클부츠)
```

### 특징
- MNIST와 동일한 형식 (드롭인 대체 가능)
- 더 복잡한 패턴과 텍스처
- 실제 Zalando 제품 이미지
- 클래스 간 더 큰 다양성

### MNIST 대비 장점
- 더 어려운 분류 문제
- 실용적인 응용 사례
- 더 현실적인 벤치마크
- 과적합 감지에 유용

### 주요 활용 분야
- 의류 분류
- 패션 추천 시스템
- 딥러닝 교육 및 프로토타이핑
- AutoML 벤치마킹

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# Fashion-MNIST 로드
fashion_mnist = load_dataset("fashion_mnist")

# 클래스 이름 매핑
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 샘플 확인
example = fashion_mnist['train'][0]
label = example['label']
print(f"클래스: {class_names[label]}")

# 시각화
import matplotlib.pyplot as plt
plt.imshow(example['image'], cmap='gray')
plt.title(f"Label: {class_names[label]}")
plt.show()
```

### 성능 벤치마크
- 기본 선형 모델: ~84% 정확도
- 간단한 CNN: ~90-93% 정확도
- 최신 모델: ~96% 정확도

### 실용적 응용
```python
# 간단한 분류 파이프라인
from transformers import pipeline

classifier = pipeline("image-classification",
                     model="farleyknight/fashion-mnist-mlp-2022-09-04")

# 예측
result = classifier(image)
print(result)
```

---

## CelebA

### 개요
CelebFaces Attributes (CelebA)는 대규모 얼굴 속성 데이터셋으로, 200,000개 이상의 유명인 이미지와 40개의 이진 속성 레이블을 포함합니다.

### 주요 통계
- **이미지 수**: 202,599개
- **인물 수**: 10,177명
- **속성**: 40개의 이진 속성
- **랜드마크**: 5개의 얼굴 랜드마크
- **Bounding Box**: 얼굴 위치 정보

### 40개 속성 예시
```
외모 속성:
- Bald (대머리)
- Bangs (앞머리)
- Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair
- Eyeglasses (안경)
- Male (남성)
- Mustache (콧수염)
- No_Beard (수염 없음)
- Pale_Skin (창백한 피부)
- Young (젊음)

액세서리:
- Earrings (귀걸이)
- Hat (모자)
- Lipstick (립스틱)
- Necklace (목걸이)
- Necktie (넥타이)
- Wearing_Earrings

표정:
- Smiling (웃음)
- Mouth_Slightly_Open (입 살짝 벌림)

기타:
- Attractive (매력적)
- Chubby (통통함)
- Double_Chin (이중턱)
- High_Cheekbones (높은 광대뼈)
등...
```

### 특징
- 대량의 레이블링된 얼굴 이미지
- 다양한 포즈, 표정, 배경
- 고해상도 이미지 (일부)
- 정렬된 버전과 비정렬 버전 제공

### 주요 활용 분야
- 얼굴 속성 인식
- 얼굴 생성 (GAN 학습)
- 얼굴 편집 및 합성
- 얼굴 인식 및 검증
- 다중 레이블 분류
- 전이 학습

### 데이터셋 분할
- **Train**: 162,770 이미지
- **Val**: 19,867 이미지
- **Test**: 19,962 이미지

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# CelebA 로드 (기본 버전)
celeba = load_dataset("celeba_hq")

# 또는 전체 CelebA
celeba = load_dataset("nielsr/CelebA-faces")

# 샘플 확인
example = celeba['train'][0]
image = example['image']
attributes = example['attributes']

# 특정 속성 확인
print(f"Smiling: {attributes['Smiling']}")
print(f"Male: {attributes['Male']}")
print(f"Young: {attributes['Young']}")
```

### 인기있는 변형 버전
- **CelebA-HQ**: 고해상도 버전 (1024x1024)
- **CelebA-Mask-HQ**: 세그멘테이션 마스크 포함
- **CelebA-Spoof**: Deepfake 탐지용

### 응용 예제
```python
# 다중 레이블 분류 예제
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

model_name = "nateraw/vit-base-patch16-224-celeba"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# 이미지 전처리
inputs = feature_extractor(images=image, return_tensors="pt")

# 예측
outputs = model(**inputs)
predictions = outputs.logits.sigmoid()

# 속성 출력
threshold = 0.5
attributes = model.config.id2label
for idx, prob in enumerate(predictions[0]):
    if prob > threshold:
        print(f"{attributes[idx]}: {prob:.2f}")
```

---

## Places365

### 개요
Places365는 장면 인식을 위한 대규모 데이터셋으로, 365개의 다양한 장면 카테고리를 포함합니다. MIT에서 개발하였으며, 장소와 환경 이해에 특화되어 있습니다.

### 주요 통계
- **이미지 수**: 180만 개 이상 (Places365-Standard)
- **카테고리**: 365개의 장면 클래스
- **고해상도**: 최소 256x256 픽셀
- **Places365-Challenge**: 800만 개 이상의 이미지

### 장면 카테고리 예시
```
실내 장면:
- bedroom (침실)
- kitchen (부엌)
- living_room (거실)
- office (사무실)
- restaurant (레스토랑)
- classroom (교실)
- library (도서관)
- hospital (병원)

실외 장면:
- beach (해변)
- forest (숲)
- mountain (산)
- street (거리)
- bridge (다리)
- park (공원)
- stadium (경기장)

자연 장면:
- waterfall (폭포)
- canyon (협곡)
- desert (사막)
- ocean (바다)

도시 장면:
- skyscraper (고층 빌딩)
- downtown (시내)
- industrial_area (산업 지역)

등 365개 카테고리...
```

### 특징
- 장면 중심의 분류 (객체가 아닌 환경)
- 다양한 관점과 조명 조건
- 세밀한 장면 구분
- 계층적 카테고리 구조

### 주요 활용 분야
- 장면 인식 (Scene Recognition)
- 장소 분류 (Place Classification)
- 환경 이해 (Environment Understanding)
- 로봇 네비게이션
- 이미지 검색
- 전이 학습의 사전 학습

### ImageNet vs Places365
```
ImageNet:
- 객체 중심 ("무엇"에 집중)
- 예: "개", "자동차", "의자"

Places365:
- 장면 중심 ("어디"에 집중)
- 예: "공원", "주방", "사무실"
```

### 데이터셋 버전
- **Places365-Standard**: 180만 이미지 (학습용)
- **Places365-Challenge**: 800만+ 이미지
- **Places205**: 이전 버전 (205개 카테고리)

### Hugging Face에서 사용하기
```python
# Places365 사전 학습 모델 사용
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests

# 모델 로드
model_name = "Siddharth11/Places365"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# 이미지 로드
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 예측
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# 결과 출력
predicted_class_idx = logits.argmax(-1).item()
print(f"예측된 장면: {model.config.id2label[predicted_class_idx]}")
```

### 전이 학습 활용
```python
# Places365로 사전 학습된 모델을 커스텀 장면 분류에 활용
from transformers import AutoModelForImageClassification

# 사전 학습 모델 로드
model = AutoModelForImageClassification.from_pretrained(
    "Siddharth11/Places365",
    num_labels=10,  # 커스텀 클래스 수
    ignore_mismatched_sizes=True
)

# Fine-tuning 진행
# ...
```

### 응용 사례
- 사진 자동 정리 및 분류
- 위치 기반 서비스
- AR/VR 환경 인식
- 스마트 홈 시스템
- 자율 주행 (환경 인식)

---

## ADE20K

### 개요
ADE20K (MIT Scene Parsing Benchmark)는 장면 파싱과 의미론적 세그멘테이션을 위한 포괄적인 데이터셋입니다. 복잡한 실내외 장면에서 150개의 객체 및 물질 카테고리를 픽셀 수준으로 레이블링합니다.

### 주요 통계
- **이미지 수**: 25,000개 이상
- **학습 이미지**: 20,210개
- **검증 이미지**: 2,000개
- **테스트 이미지**: 3,352개
- **객체 카테고리**: 150개 (+ 1개 배경)
- **어노테이션**: 픽셀 단위 세그멘테이션

### 150개 카테고리 주요 예시
```
구조물:
- wall, building, floor, ceiling, road, grass, tree
- fence, sidewalk, sky, earth, mountain, rock

가구:
- bed, windowpane, cabinet, person, door, table
- chair, sofa, bookshelf, desk, cushion

차량:
- car, van, truck, bus, boat, airplane

자연물:
- tree, plant, flower, grass, dirt, sand, water

조명 및 장식:
- lamp, light, chandelier, painting, poster, mirror

등 150개 클래스...
```

### 특징
- 상세한 픽셀 수준 어노테이션
- 복잡한 장면 구성
- 다양한 카테고리 (객체 + 재질 + 배경)
- 계층적 레이블 구조
- 부분-전체 관계 어노테이션

### 주요 활용 분야
- 의미론적 세그멘테이션 (Semantic Segmentation)
- 장면 파싱 (Scene Parsing)
- 인스턴스 세그멘테이션
- Panoptic 세그멘테이션
- 실내 장면 이해

### 세그멘테이션 레벨
```
1. 의미론적 세그멘테이션: 픽셀을 150개 클래스로 분류
2. 인스턴스 세그멘테이션: 동일 클래스의 개별 인스턴스 구분
3. 부분 세그멘테이션: 객체의 부분 요소 레이블링
```

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# ADE20K 데이터셋 로드
ade20k = load_dataset("scene_parse_150")

# 샘플 확인
example = ade20k['train'][0]
image = example['image']
segmentation_map = example['annotation']

print(f"이미지 크기: {image.size}")
print(f"세그멘테이션 맵 크기: {segmentation_map.size}")

# 고유 클래스 확인
import numpy as np
unique_labels = np.unique(np.array(segmentation_map))
print(f"이미지 내 클래스 수: {len(unique_labels)}")
```

### 세그멘테이션 모델 사용
```python
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

# 모델 로드
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)

# 이미지 전처리
inputs = feature_extractor(images=image, return_tensors="pt")

# 예측
outputs = model(**inputs)
logits = outputs.logits

# 세그멘테이션 맵 생성
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0]

# 결과 시각화
import matplotlib.pyplot as plt
plt.imshow(pred_seg.cpu().numpy())
plt.show()
```

### 평가 메트릭
- **Pixel Accuracy**: 올바르게 분류된 픽셀 비율
- **Mean IoU**: 클래스별 IoU의 평균
- **Mean Accuracy**: 클래스별 정확도의 평균
- **Frequency Weighted IoU**: 빈도 가중 IoU

### 벤치마크 모델
- FCN (Fully Convolutional Networks)
- PSPNet (Pyramid Scene Parsing Network)
- DeepLab v3+
- SegFormer
- Mask2Former

---

## Cityscapes

### 개요
Cityscapes는 도시 환경의 자율 주행을 위한 대규모 데이터셋입니다. 50개 도시의 거리 장면에서 수집된 고해상도 이미지와 정밀한 픽셀 단위 어노테이션을 제공합니다.

### 주요 통계
- **세밀한 어노테이션**: 5,000개 이미지
- **거친 어노테이션**: 20,000개 추가 이미지
- **이미지 해상도**: 2048x1024 픽셀
- **클래스**: 30개 (평가용 19개 주요 클래스)
- **도시**: 50개 유럽 도시
- **프레임**: 27개 도시의 비디오 시퀀스

### 19개 평가 클래스
```
Flat (평평한 표면):
- road (도로)
- sidewalk (보도)

Human (사람):
- person (사람)
- rider (탑승자)

Vehicle (차량):
- car (자동차)
- truck (트럭)
- bus (버스)
- train (기차)
- motorcycle (오토바이)
- bicycle (자전거)

Construction (구조물):
- building (건물)
- wall (벽)
- fence (울타리)

Object (객체):
- pole (기둥)
- traffic sign (교통 표지)
- traffic light (신호등)

Nature (자연):
- vegetation (초목)
- terrain (지형)

Sky (하늘):
- sky (하늘)
```

### 특징
- 고해상도 이미지 (2048x1024)
- 세밀한 픽셀 단위 어노테이션
- 인스턴스 레벨 세그멘테이션
- 스테레오 이미지 쌍
- 다양한 날씨와 조명 조건
- GPS 좌표 및 차량 속도 정보
- 연속된 비디오 프레임

### 주요 활용 분야
- 자율 주행
- 도시 장면 이해
- 의미론적 세그멘테이션
- 인스턴스 세그멘테이션
- Panoptic 세그멘테이션
- 깊이 추정
- 광학 흐름 예측

### 어노테이션 종류
```
1. Fine annotations (세밀한 어노테이션):
   - 5,000개 이미지
   - 완벽한 픽셀 단위 레이블링

2. Coarse annotations (거친 어노테이션):
   - 20,000개 추가 이미지
   - 대략적인 폴리곤 어노테이션

3. Instance IDs:
   - 동일 클래스의 개별 인스턴스 구분
```

### Hugging Face에서 사용하기
```python
from datasets import load_dataset

# Cityscapes 데이터셋 로드
# 참고: Cityscapes는 라이선스 등록이 필요할 수 있습니다
cityscapes = load_dataset("cityscapes", trust_remote_code=True)

# 샘플 확인
example = cityscapes['train'][0]
image = example['image']
semantic_map = example['semantic']
instance_map = example['instance']

print(f"이미지 크기: {image.size}")
```

### 세그멘테이션 예제
```python
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch

# Cityscapes로 학습된 모델 로드
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)

# 이미지 전처리
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)

# 세그멘테이션 맵 생성
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]

# 시각화
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_semantic_map.cpu().numpy())
plt.title("Semantic Segmentation")
plt.axis('off')
plt.show()
```

### 평가 메트릭
- **IoU (Intersection over Union)**: 클래스별 IoU
- **iIoU**: 인스턴스 레벨 IoU
- **AP (Average Precision)**: 인스턴스 세그멘테이션용

### 도전 과제
```
1. 클래스 불균형 (도로와 건물이 대부분의 픽셀 차지)
2. 작은 객체 탐지 (멀리 있는 사람, 표지판)
3. Occlusion 처리
4. 실시간 처리 요구사항 (자율 주행)
5. 다양한 날씨와 조명 조건
```

### 관련 데이터셋
- **Mapillary Vistas**: 더 다양한 지역과 조건
- **BDD100K**: Berkeley DeepDrive, 10만 개 이미지
- **KITTI**: 자율 주행 종합 벤치마크

---

## Open Images

### 개요
Open Images는 Google에서 공개한 대규모 이미지 데이터셋으로, 9백만 개 이상의 이미지와 600개 이상의 객체 카테고리를 포함합니다. 다양한 컴퓨터 비전 태스크를 지원하는 가장 큰 공개 데이터셋 중 하나입니다.

### 주요 통계 (Open Images V7)
- **이미지 수**: 약 900만 개
- **객체 클래스**: 600개
- **Bounding Box**: 1,600만 개
- **세그멘테이션**: 280만 개 인스턴스
- **시각적 관계**: 37만 개 관계 어노테이션
- **이미지 레벨 레이블**: 7,800만 개

### 특징
- 대규모 다양성
- 다중 태스크 지원
- 실제 환경의 복잡한 장면
- 계층적 레이블 구조
- Creative Commons 라이선스

### 지원 태스크
```
1. 이미지 분류 (Image Classification):
   - 이미지 레벨 레이블
   - 19,957개 클래스

2. 객체 탐지 (Object Detection):
   - Bounding box 어노테이션
   - 600개 객체 클래스

3. 인스턴스 세그멘테이션:
   - 픽셀 단위 마스크
   - 350개 클래스

4. 시각적 관계 탐지:
   - 객체 간 관계
   - <subject, predicate, object> 형태

5. 이미지 캡셔닝:
   - 자연어 설명
```

### 600개 객체 카테고리 예시
```
동물:
- Dog, Cat, Horse, Elephant, Bear, Zebra, Giraffe...

차량:
- Car, Truck, Bus, Bicycle, Motorcycle, Airplane, Boat...

음식:
- Pizza, Hamburger, Sandwich, Fruit, Vegetable, Dessert...

가구:
- Chair, Table, Sofa, Bed, Desk, Cabinet...

전자제품:
- Computer, Phone, Television, Camera, Laptop...

의류:
- Shirt, Pants, Dress, Shoes, Hat, Jacket...

스포츠:
- Football, Basketball, Tennis racket, Surfboard...

등 600개 클래스...
```

### Hugging Face에서 사용하기
```python
# Open Images 데이터셋은 크기가 매우 크므로 스트리밍 모드 권장
from datasets import load_dataset

# 스트리밍 모드로 로드
dataset = load_dataset("google/open-images-dataset", streaming=True)

# 또는 특정 subset만 로드
detection_dataset = load_dataset(
    "Francesco/open-images-detection",
    split="train[:1000]"  # 첫 1000개만
)

# 샘플 확인
for example in dataset['train'].take(1):
    print(f"이미지 ID: {example['image_id']}")
    print(f"레이블: {example['labels']}")
```

### 객체 탐지 예제
```python
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import torch

# YOLO 모델 로드 (Open Images로 학습 가능)
model_name = "hustvl/yolos-tiny"
processor = YolosImageProcessor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)

# 이미지 로드
image = Image.open("image.jpg")

# 전처리 및 예측
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 후처리
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

# 결과 출력
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} "
        f"with confidence {round(score.item(), 3)} at location {box}"
    )
```

### 시각적 관계 예제
```
이미지 내 관계:
- Person riding Horse (사람이 말을 타고 있음)
- Dog playing with Ball (개가 공을 가지고 놀고 있음)
- Car parked on Street (자동차가 거리에 주차되어 있음)
- Person holding Umbrella (사람이 우산을 들고 있음)
```

### COCO와의 비교
```
Open Images:
✓ 더 많은 이미지 (9M vs 330K)
✓ 더 많은 클래스 (600 vs 80)
✓ 시각적 관계 어노테이션
✓ 더 다양한 장면
- 어노테이션 밀도가 낮음

COCO:
✓ 더 정밀한 어노테이션
✓ 모든 객체가 레이블링됨
✓ 더 많은 벤치마크 모델
- 상대적으로 작은 규모
```

### 다운로드 및 접근
```python
# 전체 데이터셋 다운로드 (권장하지 않음 - 매우 큼)
# 대신 필요한 부분만 선택적으로 다운로드

# OIDv6 다운로드 도구 사용
# pip install openimages

from openimages.download import download_dataset

download_dataset(
    dest_dir='./data',
    class_labels=['Dog', 'Cat'],  # 특정 클래스만
    annotation_format='pascal',
    limit=1000  # 이미지 수 제한
)
```

### 활용 사례
- 대규모 모델 사전 학습
- Zero-shot 학습
- 다중 태스크 학습
- 관계 인식
- 객체 탐지 벤치마킹

---

## Hugging Face에서 Vision 데이터셋 사용하기

### 기본 사용법

#### 데이터셋 로드하기
```python
from datasets import load_dataset

# 기본 로드
dataset = load_dataset("dataset_name")

# 특정 설정으로 로드
dataset = load_dataset("dataset_name", "config_name")

# 특정 split만 로드
train_dataset = load_dataset("dataset_name", split="train")

# 일부만 로드 (빠른 테스트용)
small_dataset = load_dataset("dataset_name", split="train[:100]")

# 스트리밍 모드 (대용량 데이터셋)
dataset = load_dataset("dataset_name", streaming=True)
```

#### 데이터 확인하기
```python
# 데이터셋 정보
print(dataset)
print(dataset['train'].features)
print(dataset['train'][0])

# 이미지 확인
image = dataset['train'][0]['image']
image.show()

# 통계
print(f"학습 샘플 수: {len(dataset['train'])}")
print(f"테스트 샘플 수: {len(dataset['test'])}")
```

### 데이터 전처리

#### 이미지 변환
```python
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

# 데이터셋 로드
dataset = load_dataset("cifar10")

# 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 변환 함수
def transform_images(examples):
    examples['pixel_values'] = [
        transform(image.convert("RGB"))
        for image in examples['image']
    ]
    return examples

# 데이터셋에 적용
transformed_dataset = dataset.with_transform(transform_images)
```

#### Transformers 라이브러리와 함께 사용
```python
from transformers import AutoFeatureExtractor
from datasets import load_dataset

# 데이터셋과 Feature Extractor 로드
dataset = load_dataset("cifar10")
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/resnet-50"
)

# 전처리 함수
def preprocess_images(examples):
    examples['pixel_values'] = feature_extractor(
        [image.convert("RGB") for image in examples['image']],
        return_tensors="pt"
    )['pixel_values']
    return examples

# 배치로 적용
dataset = dataset.with_transform(preprocess_images)
```

### 데이터 증강

```python
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# 증강 파이프라인
transform = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def augment_images(examples):
    examples['pixel_values'] = [
        transform(image=np.array(image))['image']
        for image in examples['image']
    ]
    return examples

# 데이터셋에 적용
dataset = load_dataset("cifar10")
augmented_dataset = dataset['train'].map(
    augment_images,
    batched=True,
    batch_size=32
)
```

### DataLoader 생성

```python
from torch.utils.data import DataLoader
from datasets import load_dataset

# 데이터셋 준비
dataset = load_dataset("cifar10")
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# DataLoader 생성
train_dataloader = DataLoader(
    dataset['train'],
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 사용 예제
for batch in train_dataloader:
    images = batch['pixel_values']
    labels = batch['label']
    # 학습 코드...
```

### 커스텀 데이터셋 업로드

```python
from datasets import Dataset, DatasetDict, Features, Image, ClassLabel
from PIL import Image as PILImage
import os

# 이미지 경로와 레이블 수집
image_paths = []
labels = []

for class_name in os.listdir("./my_images"):
    class_dir = os.path.join("./my_images", class_name)
    for img_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_name))
        labels.append(class_name)

# 데이터셋 생성
dataset = Dataset.from_dict({
    "image": image_paths,
    "label": labels
}).cast_column("image", Image())

# Features 정의
features = Features({
    'image': Image(),
    'label': ClassLabel(names=['cat', 'dog', 'bird'])
})

# Hub에 업로드
dataset.push_to_hub("username/my-custom-dataset")
```

### 객체 탐지 데이터셋 다루기

```python
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO 데이터셋 로드
dataset = load_dataset("detection-datasets/coco")

# 샘플 가져오기
example = dataset['train'][0]
image = example['image']
objects = example['objects']

# 시각화
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# Bounding box 그리기
for bbox, category in zip(objects['bbox'], objects['category']):
    x, y, width, height = bbox
    rect = patches.Rectangle(
        (x, y), width, height,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x, y, category, fontsize=12,
            bbox=dict(facecolor='yellow', alpha=0.5))

plt.axis('off')
plt.show()
```

### 세그멘테이션 데이터셋 다루기

```python
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# ADE20K 데이터셋 로드
dataset = load_dataset("scene_parse_150")

# 샘플 가져오기
example = dataset['train'][0]
image = example['image']
segmentation = example['annotation']

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(segmentation)
axes[1].set_title('Segmentation Map')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 캐싱 및 성능 최적화

```python
from datasets import load_dataset

# 캐시 디렉토리 지정
dataset = load_dataset(
    "cifar10",
    cache_dir="./my_cache"
)

# 멀티프로세싱으로 빠른 전처리
dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=100,
    num_proc=4  # 4개 프로세스 사용
)

# 필요한 컬럼만 유지
dataset = dataset.remove_columns(['unnecessary_column'])

# 데이터셋 저장 및 로드
dataset.save_to_disk("./processed_dataset")
loaded_dataset = Dataset.load_from_disk("./processed_dataset")
```

### 유용한 팁

#### 1. 메모리 효율적인 로딩
```python
# 스트리밍 모드 사용
dataset = load_dataset("large_dataset", streaming=True)

# 필요한 만큼만 가져오기
for i, example in enumerate(dataset['train']):
    if i >= 1000:
        break
    # 처리...
```

#### 2. 데이터셋 필터링
```python
# 조건에 맞는 샘플만 선택
filtered_dataset = dataset['train'].filter(
    lambda example: example['label'] in [0, 1, 2]
)

# 특정 이미지 크기만 선택
def filter_by_size(example):
    return example['image'].size[0] >= 224

large_images = dataset.filter(filter_by_size)
```

#### 3. 데이터셋 분할
```python
# 학습/검증 분할
dataset = load_dataset("cifar10")
train_test = dataset['train'].train_test_split(test_size=0.1)

train_dataset = train_test['train']
val_dataset = train_test['test']
```

#### 4. 클래스 불균형 처리
```python
from collections import Counter

# 클래스 분포 확인
labels = dataset['train']['label']
class_counts = Counter(labels)
print(class_counts)

# 오버샘플링
from datasets import concatenate_datasets

minority_class = dataset['train'].filter(lambda x: x['label'] == 0)
oversampled = concatenate_datasets([dataset['train'], minority_class, minority_class])
```

### 일반적인 워크플로우

```python
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import Trainer, TrainingArguments
import torch

# 1. 데이터셋 로드
dataset = load_dataset("cifar10")

# 2. Feature Extractor 로드
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")

# 3. 전처리
def preprocess(examples):
    examples['pixel_values'] = feature_extractor(
        [image.convert("RGB") for image in examples['image']],
        return_tensors="pt"
    )['pixel_values']
    return examples

dataset = dataset.map(preprocess, batched=True)

# 4. 데이터셋 포맷 설정
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# 5. 모델 로드
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    num_labels=10,
    ignore_mismatched_sizes=True
)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# 7. Trainer 생성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

trainer.train()
```

---

## 마치며

이 문서에서는 컴퓨터 비전 분야의 주요 데이터셋들을 다루었습니다:

- **ImageNet**: 대규모 이미지 분류의 표준
- **COCO**: 다목적 객체 탐지 및 세그멘테이션
- **MNIST & Fashion-MNIST**: 입문자용 데이터셋
- **CIFAR-10/100**: 빠른 프로토타이핑용
- **Pascal VOC**: 객체 탐지의 클래식
- **CelebA**: 얼굴 속성 인식
- **Places365**: 장면 인식
- **ADE20K**: 세밀한 장면 파싱
- **Cityscapes**: 자율 주행용 도시 장면
- **Open Images**: 대규모 다목적 데이터셋

각 데이터셋은 특정 태스크와 응용 분야에 최적화되어 있으며, Hugging Face를 통해 쉽게 접근하고 사용할 수 있습니다.

### 다음 단계

1. **프로젝트에 맞는 데이터셋 선택**
2. **Hugging Face에서 데이터셋 로드 및 탐색**
3. **사전 학습된 모델로 전이 학습 시도**
4. **커스텀 데이터로 Fine-tuning**

### 추가 리소스

- [Hugging Face Datasets 문서](https://huggingface.co/docs/datasets)
- [Hugging Face Vision 모델](https://huggingface.co/models?pipeline_tag=image-classification)
- [Transformers Vision 가이드](https://huggingface.co/docs/transformers/tasks/image_classification)

행운을 빕니다! 🚀
