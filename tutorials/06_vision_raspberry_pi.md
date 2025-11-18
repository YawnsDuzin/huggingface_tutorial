# 라즈베리파이용 Hugging Face Vision 모델 + GStreamer 통합 가이드

## 📋 목차
1. [소개](#소개)
2. [시스템 요구사항](#시스템-요구사항)
3. [환경 설정](#환경-설정)
4. [라즈베리파이에서 사용 가능한 Vision 모델](#라즈베리파이에서-사용-가능한-vision-모델)
5. [GStreamer와의 통합](#gstreamer와의-통합)
6. [실용적인 예제들](#실용적인-예제들)
7. [성능 최적화](#성능-최적화)
8. [추가 가능한 기능들](#추가-가능한-기능들)
9. [트러블슈팅](#트러블슈팅)

## 소개

이 튜토리얼은 라즈베리파이에서 Hugging Face의 Vision 모델을 GStreamer와 통합하여 실시간 영상 처리 애플리케이션을 개발하는 방법을 다룹니다. 라즈베리파이의 제한된 리소스 환경에서도 효과적으로 동작할 수 있는 경량 모델들과 최적화 기법을 소개합니다.

### 왜 Hugging Face + GStreamer인가?

- **Hugging Face**: 최신 사전 훈련된 Vision 모델에 쉽게 접근
- **GStreamer**: 강력하고 유연한 멀티미디어 프레임워크
- **라즈베리파이**: 저렴하고 접근성 높은 엣지 컴퓨팅 플랫폼

## 시스템 요구사항

### 하드웨어
- **라즈베리파이 4 Model B (4GB/8GB RAM 권장)**
  - 라즈베리파이 3도 가능하지만 성능 제한
  - 라즈베리파이 5는 더 나은 성능 제공
- **카메라 모듈** (Raspberry Pi Camera Module V2/V3 또는 USB 웹캠)
- **마이크로SD 카드** (32GB 이상, Class 10 권장)
- **충분한 전원 공급** (5V 3A 이상)

### 소프트웨어
- **OS**: Raspberry Pi OS (64-bit 권장)
- **Python**: 3.8 이상
- **GStreamer**: 1.18 이상

## 환경 설정

### 1. 시스템 업데이트 및 기본 패키지 설치

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y libatlas-base-dev libopenblas-dev libjpeg-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install -y libqt5gui5 libqt5core5a libqt5widgets5
```

### 2. GStreamer 설치

```bash
# GStreamer 코어 및 플러그인 설치
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Python GStreamer 바인딩
sudo apt install -y python3-gi python3-gst-1.0 \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0

# 카메라 지원
sudo apt install -y gstreamer1.0-rpicamsrc
```

### 3. Python 가상환경 설정

```bash
# 가상환경 생성
python3 -m venv ~/venv/hf-vision
source ~/venv/hf-vision/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

### 4. Hugging Face 라이브러리 설치

```bash
# 기본 라이브러리
pip install transformers pillow numpy

# PyTorch (라즈베리파이용 경량 버전)
# ARM 아키텍처를 위한 사전 컴파일된 휠 사용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 추가 유틸리티
pip install opencv-python-headless
pip install timm  # PyTorch Image Models (추가 모델 지원)
```

### 5. ONNX Runtime (선택사항, 성능 향상)

```bash
# ONNX Runtime으로 추론 속도 향상
pip install onnxruntime optimum
```

## 라즈베리파이에서 사용 가능한 Vision 모델

### 1. 이미지 분류 (Image Classification)

#### MobileNet V2
- **모델 크기**: 약 14MB
- **추론 속도**: ~100-200ms (라즈베리파이 4)
- **사용 사례**: 실시간 객체 인식, 장면 분류

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 모델 로드
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
```

#### EfficientNet
- **모델 크기**: 약 20-30MB (B0-B2 버전)
- **추론 속도**: ~150-300ms
- **사용 사례**: 높은 정확도가 필요한 이미지 분류

```python
processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0")
```

#### Vision Transformer (ViT) - 경량 버전
- **모델 크기**: 약 86MB (base-patch16-224)
- **추론 속도**: ~400-600ms
- **사용 사례**: 더 나은 정확도가 필요하지만 속도가 덜 중요한 경우

```python
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

### 2. 객체 탐지 (Object Detection)

#### YOLOS (You Only Look at One Sequence)
- **모델 크기**: 약 115MB (tiny 버전)
- **추론 속도**: ~500-800ms
- **사용 사례**: 실시간에 가까운 객체 탐지
- **탐지 가능**: COCO 데이터셋의 80개 클래스

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
```

#### DETR (DEtection TRansformer) - ResNet-50 백본
- **모델 크기**: 약 160MB
- **추론 속도**: ~800-1200ms
- **사용 사례**: 정확한 객체 탐지 및 위치 지정

```python
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

#### Conditional DETR
- **모델 크기**: 약 180MB
- **추론 속도**: ~700-1000ms
- **사용 사례**: DETR보다 빠른 수렴, 더 나은 소규모 객체 탐지

```python
processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")
```

### 3. 이미지 세그멘테이션 (Image Segmentation)

#### SegFormer
- **모델 크기**: 약 14-60MB (b0-b2 버전)
- **추론 속도**: ~300-600ms (b0)
- **사용 사례**: 시맨틱 세그멘테이션, 장면 이해

```python
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
```

#### MobileNetV2 기반 DeepLabV3
- **모델 크기**: 약 20MB
- **추론 속도**: ~200-400ms
- **사용 사례**: 빠른 세그멘테이션

```python
# torchvision 모델 사용
import torchvision.models.segmentation as segmentation
model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
```

### 4. 특수 용도 모델

#### Zero-Shot 이미지 분류 (CLIP)
- **모델 크기**: 약 150MB (base 버전)
- **추론 속도**: ~300-500ms
- **사용 사례**: 사전 정의된 클래스 없이 이미지 분류

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

#### 이미지 캡셔닝
- **모델 크기**: 약 500MB
- **추론 속도**: ~1000-2000ms
- **사용 사례**: 이미지 설명 생성

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
```

#### 깊이 추정 (Depth Estimation)
- **모델 크기**: 약 350MB
- **추론 속도**: ~500-800ms
- **사용 사례**: 3D 재구성, 장애물 감지

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
```

## GStreamer와의 통합

### GStreamer 기본 개념

GStreamer는 파이프라인 기반의 멀티미디어 프레임워크입니다. 각 요소(element)는 특정 작업을 수행하고, 이들이 연결되어 파이프라인을 형성합니다.

**기본 파이프라인 구조**:
```
소스 -> 디코더 -> 변환 -> AI 처리 -> 변환 -> 인코더 -> 싱크
```

### Python에서 GStreamer 사용하기

#### 방법 1: PyGObject를 사용한 GStreamer 파이프라인

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# GStreamer 초기화
Gst.init(None)

class GStreamerPipeline:
    def __init__(self):
        # 파이프라인 생성
        self.pipeline = Gst.Pipeline.new("vision-pipeline")

        # 카메라 소스 (라즈베리파이 카메라)
        self.source = Gst.ElementFactory.make("v4l2src", "camera")

        # 비디오 변환
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "convert")

        # 비디오 스케일
        self.videoscale = Gst.ElementFactory.make("videoscale", "scale")

        # 캡스 필터 (해상도 설정)
        self.caps_filter = Gst.ElementFactory.make("capsfilter", "filter")
        caps = Gst.Caps.from_string("video/x-raw,width=640,height=480,framerate=30/1")
        self.caps_filter.set_property("caps", caps)

        # appsink (프레임 추출용)
        self.appsink = Gst.ElementFactory.make("appsink", "sink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)

        # 요소들을 파이프라인에 추가
        self.pipeline.add(self.source)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.videoscale)
        self.pipeline.add(self.caps_filter)
        self.pipeline.add(self.appsink)

        # 요소들 연결
        self.source.link(self.videoconvert)
        self.videoconvert.link(self.videoscale)
        self.videoscale.link(self.caps_filter)
        self.caps_filter.link(self.appsink)

        # 시그널 연결
        self.appsink.connect("new-sample", self.on_new_sample)

    def on_new_sample(self, appsink):
        """새 프레임이 도착했을 때 호출"""
        sample = appsink.emit("pull-sample")
        if sample:
            # 프레임 처리
            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # 여기서 Hugging Face 모델로 처리
            # process_with_model(buffer, caps)

        return Gst.FlowReturn.OK

    def start(self):
        """파이프라인 시작"""
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        """파이프라인 중지"""
        self.pipeline.set_state(Gst.State.NULL)
```

#### 방법 2: OpenCV를 중간 레이어로 사용

```python
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class GStreamerOpenCVBridge:
    def __init__(self, model_name="google/mobilenet_v2_1.0_224"):
        # GStreamer 파이프라인 문자열
        # 라즈베리파이 카메라 사용 예시
        self.gst_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "appsink"
        )

        # OpenCV VideoCapture 생성
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)

        # Hugging Face 모델 로드
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def process_frame(self, frame):
        """프레임을 모델로 처리"""
        # OpenCV BGR을 PIL RGB로 변환
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")

        # 추론
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        return self.model.config.id2label[predicted_class_idx]

    def run(self):
        """메인 루프"""
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        frame_count = 0
        process_every_n_frames = 10  # 10프레임마다 처리 (성능 최적화)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1

                # N 프레임마다 AI 처리
                if frame_count % process_every_n_frames == 0:
                    prediction = self.process_frame(frame)
                    print(f"예측: {prediction}")

                    # 프레임에 텍스트 추가
                    cv2.putText(frame, prediction, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 디스플레이 (X11 포워딩이나 HDMI 연결 시)
                cv2.imshow('Hugging Face Vision', frame)

                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
```

### 카메라 소스 설정

#### 라즈베리파이 카메라 모듈 (CSI)

```python
# libcamera 사용 (Raspberry Pi OS Bullseye 이상)
gst_pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)

# 구형 시스템에서 rpicamsrc 사용
gst_pipeline = (
    "rpicamsrc ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)
```

#### USB 웹캠

```python
gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)
```

#### 비디오 파일

```python
gst_pipeline = (
    "filesrc location=/path/to/video.mp4 ! "
    "qtdemux ! "
    "h264parse ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "videoscale ! "
    "video/x-raw,width=640,height=480 ! "
    "appsink"
)
```

#### RTSP 스트림

```python
gst_pipeline = (
    "rtspsrc location=rtsp://192.168.1.100:8554/stream ! "
    "rtph264depay ! "
    "h264parse ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink"
)
```

## 실용적인 예제들

### 예제 1: 실시간 이미지 분류

```python
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

class RealtimeImageClassifier:
    def __init__(self, model_name="google/mobilenet_v2_1.0_224"):
        # 모델 로드
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()

        # GStreamer 파이프라인
        gst_str = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        # 성능 측정용
        self.fps_counter = []

    def classify(self, frame):
        """프레임 분류"""
        # BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")

        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        inference_time = time.time() - start_time

        # Top-3 예측
        top3_prob, top3_idx = torch.topk(probs, 3)
        results = []
        for prob, idx in zip(top3_prob[0], top3_idx[0]):
            label = self.model.config.id2label[idx.item()]
            results.append((label, prob.item()))

        return results, inference_time

    def draw_predictions(self, frame, predictions, inference_time):
        """프레임에 예측 결과 그리기"""
        y_offset = 30

        # 배경 그리기 (가독성 향상)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 예측 결과
        for i, (label, prob) in enumerate(predictions):
            text = f"{i+1}. {label}: {prob*100:.1f}%"
            cv2.putText(frame, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 추론 시간 표시
        fps_text = f"Inference: {inference_time*1000:.0f}ms"
        cv2.putText(frame, fps_text, (10, y_offset + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def run(self):
        """메인 루프"""
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        frame_count = 0
        process_every = 5  # 5프레임마다 처리
        current_predictions = []
        current_inference_time = 0

        print("실시간 이미지 분류 시작... 'q'를 눌러 종료")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

                frame_count += 1

                # 주기적으로 분류 수행
                if frame_count % process_every == 0:
                    current_predictions, current_inference_time = self.classify(frame)

                # 예측 결과 그리기
                if current_predictions:
                    frame = self.draw_predictions(frame, current_predictions,
                                                 current_inference_time)

                # 화면 출력
                cv2.imshow('Real-time Image Classification', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    classifier = RealtimeImageClassifier()
    classifier.run()
```

### 예제 2: 실시간 객체 탐지

```python
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import time

class RealtimeObjectDetector:
    def __init__(self, model_name="hustvl/yolos-tiny", confidence_threshold=0.5):
        # 모델 로드
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.eval()

        self.confidence_threshold = confidence_threshold

        # GStreamer 파이프라인
        gst_str = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        # 색상 맵 (각 클래스별로 다른 색상)
        np.random.seed(42)
        self.colors = {}

    def detect_objects(self, frame):
        """객체 탐지 수행"""
        # BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")

        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        inference_time = time.time() - start_time

        # 후처리
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()]
            detections.append({
                'box': box,
                'label': label_name,
                'score': score.item()
            })

        return detections, inference_time

    def get_color(self, label):
        """라벨별 고유 색상 반환"""
        if label not in self.colors:
            self.colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[label]

    def draw_detections(self, frame, detections, inference_time):
        """탐지 결과를 프레임에 그리기"""
        for det in detections:
            box = det['box']
            label = det['label']
            score = det['score']
            color = self.get_color(label)

            # 바운딩 박스
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 라벨과 신뢰도
            text = f"{label}: {score*100:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # 텍스트 배경
            cv2.rectangle(frame,
                         (box[0], box[1] - text_size[1] - 10),
                         (box[0] + text_size[0], box[1]),
                         color, -1)

            # 텍스트
            cv2.putText(frame, text, (box[0], box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 통계 정보
        info_text = f"Objects: {len(detections)} | Inference: {inference_time*1000:.0f}ms"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def run(self):
        """메인 루프"""
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        frame_count = 0
        process_every = 5  # 5프레임마다 처리
        current_detections = []
        current_inference_time = 0

        print("실시간 객체 탐지 시작... 'q'를 눌러 종료")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

                frame_count += 1

                # 주기적으로 탐지 수행
                if frame_count % process_every == 0:
                    current_detections, current_inference_time = self.detect_objects(frame)

                # 탐지 결과 그리기
                frame = self.draw_detections(frame, current_detections, current_inference_time)

                # 화면 출력
                cv2.imshow('Real-time Object Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    detector = RealtimeObjectDetector(confidence_threshold=0.3)
    detector.run()
```

### 예제 3: 세그멘테이션

```python
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import time

class RealtimeSegmentation:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        # 모델 로드
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()

        # GStreamer 파이프라인
        gst_str = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=512,height=512,framerate=15/1 ! "  # 해상도 낮춤
            "videoconvert ! "
            "appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        # 색상 맵 생성
        self.create_colormap()

    def create_colormap(self):
        """세그멘테이션 클래스별 색상 맵"""
        # ADE20K 데이터셋은 150개 클래스
        np.random.seed(42)
        self.colormap = np.random.randint(0, 255, (150, 3), dtype=np.uint8)

    def segment(self, frame):
        """세그멘테이션 수행"""
        # BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")

        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        inference_time = time.time() - start_time

        # 세그멘테이션 맵 생성
        # (batch_size, num_classes, height, width) -> (height, width)
        seg_map = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        return seg_map, inference_time

    def visualize_segmentation(self, frame, seg_map):
        """세그멘테이션 결과 시각화"""
        # 세그멘테이션 맵을 컬러로 변환
        color_seg = self.colormap[seg_map]

        # 원본 프레임 크기로 리사이즈
        color_seg = cv2.resize(color_seg, (frame.shape[1], frame.shape[0]))

        # 원본 프레임과 블렌딩
        alpha = 0.6
        blended = cv2.addWeighted(frame, alpha, color_seg, 1-alpha, 0)

        return blended

    def run(self, show_original=True):
        """메인 루프"""
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        frame_count = 0
        process_every = 3  # 3프레임마다 처리
        current_seg_map = None
        current_inference_time = 0

        print("실시간 세그멘테이션 시작... 'q'를 눌러 종료")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

                frame_count += 1

                # 주기적으로 세그멘테이션 수행
                if frame_count % process_every == 0:
                    current_seg_map, current_inference_time = self.segment(frame)

                # 시각화
                if current_seg_map is not None:
                    result_frame = self.visualize_segmentation(frame, current_seg_map)

                    # 추론 시간 표시
                    cv2.putText(result_frame,
                               f"Inference: {current_inference_time*1000:.0f}ms",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 화면 출력
                    if show_original:
                        # 원본과 결과를 나란히 표시
                        combined = np.hstack([frame, result_frame])
                        cv2.imshow('Segmentation (Original | Result)', combined)
                    else:
                        cv2.imshow('Segmentation', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    segmenter = RealtimeSegmentation()
    segmenter.run()
```

### 예제 4: Zero-Shot 분류 (CLIP)

```python
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time

class ZeroShotClassifier:
    def __init__(self, candidate_labels, model_name="openai/clip-vit-base-patch32"):
        # 모델 로드
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.candidate_labels = candidate_labels

        # GStreamer 파이프라인
        gst_str = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def classify(self, frame):
        """Zero-shot 분류"""
        # BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 전처리
        inputs = self.processor(
            text=self.candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        inference_time = time.time() - start_time

        # 결과 정리
        results = []
        for label, prob in zip(self.candidate_labels, probs[0]):
            results.append((label, prob.item()))

        # 확률순 정렬
        results.sort(key=lambda x: x[1], reverse=True)

        return results, inference_time

    def draw_predictions(self, frame, predictions, inference_time):
        """예측 결과 그리기"""
        y_offset = 30

        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, len(predictions)*30 + 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 예측 결과
        for i, (label, prob) in enumerate(predictions[:5]):  # Top 5만 표시
            text = f"{label}: {prob*100:.1f}%"
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 추론 시간
        cv2.putText(frame, f"Inference: {inference_time*1000:.0f}ms",
                   (10, y_offset + len(predictions[:5])*30 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def run(self):
        """메인 루프"""
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        frame_count = 0
        process_every = 10  # 10프레임마다 처리
        current_predictions = []
        current_inference_time = 0

        print("Zero-shot 분류 시작...")
        print(f"후보 라벨: {', '.join(self.candidate_labels)}")
        print("'q'를 눌러 종료")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % process_every == 0:
                    current_predictions, current_inference_time = self.classify(frame)

                if current_predictions:
                    frame = self.draw_predictions(frame, current_predictions,
                                                 current_inference_time)

                cv2.imshow('Zero-Shot Classification', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# 실행 예시
if __name__ == "__main__":
    # 원하는 라벨을 자유롭게 정의
    labels = [
        "a person",
        "a dog",
        "a cat",
        "a car",
        "a bicycle",
        "a chair",
        "a laptop",
        "a phone",
        "a book",
        "food"
    ]

    classifier = ZeroShotClassifier(candidate_labels=labels)
    classifier.run()
```

## 성능 최적화

### 1. 모델 최적화

#### ONNX 변환

ONNX Runtime은 CPU에서도 훨씬 빠른 추론 속도를 제공합니다.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from optimum.onnxruntime import ORTModelForImageClassification
import torch

# PyTorch 모델 로드
model_name = "google/mobilenet_v2_1.0_224"
processor = AutoImageProcessor.from_pretrained(model_name)

# ONNX로 변환 및 저장
from optimum.onnxruntime import ORTModelForImageClassification

ort_model = ORTModelForImageClassification.from_pretrained(
    model_name,
    export=True  # 자동으로 ONNX 변환
)

# 저장
save_path = "./mobilenet_v2_onnx"
ort_model.save_pretrained(save_path)
processor.save_pretrained(save_path)

# 나중에 로드
ort_model = ORTModelForImageClassification.from_pretrained(save_path)
```

#### 양자화 (Quantization)

양자화를 통해 모델 크기를 줄이고 추론 속도를 높일 수 있습니다.

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 양자화 설정
quantizer = ORTQuantizer.from_pretrained(ort_model)

# 동적 양자화 (Dynamic Quantization)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# 양자화 수행
quantizer.quantize(
    save_dir="./mobilenet_v2_quantized",
    quantization_config=dqconfig
)
```

#### 모델 프루닝

불필요한 가중치를 제거하여 모델 크기를 줄입니다.

```python
import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """모델 프루닝"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    return model

# 사용 예시
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
pruned_model = prune_model(model, amount=0.3)  # 30% 프루닝
```

### 2. 프레임 처리 최적화

#### 멀티스레딩

```python
import cv2
import threading
import queue
from collections import deque

class ThreadedVideoCapture:
    def __init__(self, gst_pipeline, queue_size=128):
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        """캡처 스레드 시작"""
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        return self

    def _reader(self):
        """백그라운드에서 프레임 읽기"""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                self.q.put(frame)
            else:
                # 큐가 가득 차면 잠시 대기
                import time
                time.sleep(0.01)

    def read(self):
        """큐에서 프레임 가져오기"""
        return self.q.get()

    def stop(self):
        """스레드 중지"""
        self.stopped = True
        self.cap.release()

# 사용 예시
gst_str = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
video_capture = ThreadedVideoCapture(gst_str).start()

while True:
    frame = video_capture.read()
    # 프레임 처리...
```

#### 프레임 건너뛰기

```python
class AdaptiveFrameProcessor:
    def __init__(self, target_fps=10):
        self.target_fps = target_fps
        self.last_process_time = 0
        self.frame_interval = 1.0 / target_fps

    def should_process(self):
        """현재 프레임을 처리해야 하는지 결정"""
        import time
        current_time = time.time()
        if current_time - self.last_process_time >= self.frame_interval:
            self.last_process_time = current_time
            return True
        return False

# 사용 예시
processor = AdaptiveFrameProcessor(target_fps=5)  # 초당 5프레임만 처리

while True:
    ret, frame = cap.read()
    if processor.should_process():
        # AI 처리 수행
        result = model(frame)
```

### 3. 메모리 최적화

```python
import gc
import torch

# 주기적으로 가비지 컬렉션
gc.collect()

# PyTorch 캐시 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 입력 이미지 크기 줄이기
def resize_for_inference(image, max_size=384):
    """추론을 위한 이미지 리사이즈"""
    from PIL import Image

    # 비율 유지하면서 크기 조정
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image
```

### 4. 시스템 레벨 최적화

#### CPU 성능 모드 설정

```bash
# CPU 거버너를 performance로 설정 (최대 성능)
sudo apt install cpufrequtils
sudo cpufreq-set -g performance

# 또는 모든 코어에 대해
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" | sudo tee $cpu
done
```

#### 스왑 메모리 증가

```bash
# 스왑 파일 생성 (2GB)
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 부팅 시 자동 마운트
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### GPU 가속 (라즈베리파이 4의 VideoCore VI)

```bash
# GPU 메모리 증가
sudo raspi-config
# Performance Options -> GPU Memory -> 256
```

## 추가 가능한 기능들

### 1. 객체 추적 (Object Tracking)

객체 탐지에 추적 기능을 추가하여 프레임 간 객체를 연속적으로 따라갑니다.

```python
from collections import defaultdict
import numpy as np

class ObjectTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """새 객체 등록"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """객체 제거"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """탐지된 객체로 추적 업데이트"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([self._get_centroid(det) for det in detections])

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # 거리 계산
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids,
                axis=2
            )

            # 최소 거리 매칭
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # 매칭되지 않은 객체 처리
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # 새로운 객체 등록
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

    def _get_centroid(self, detection):
        """바운딩 박스에서 중심점 계산"""
        box = detection['box']
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
```

### 2. 이벤트 기반 알림

특정 객체가 탐지되면 알림을 보냅니다.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2

class EventNotifier:
    def __init__(self, email_config=None):
        self.email_config = email_config
        self.last_alert_time = {}
        self.cooldown = 60  # 60초 쿨다운

    def should_alert(self, event_type):
        """알림을 보낼지 결정 (쿨다운 체크)"""
        import time
        current_time = time.time()

        if event_type not in self.last_alert_time:
            self.last_alert_time[event_type] = current_time
            return True

        if current_time - self.last_alert_time[event_type] > self.cooldown:
            self.last_alert_time[event_type] = current_time
            return True

        return False

    def send_email_alert(self, subject, body, image=None):
        """이메일 알림 전송"""
        if not self.email_config:
            return

        msg = MIMEMultipart()
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if image is not None:
            # OpenCV 이미지를 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', image)
            img_data = buffer.tobytes()

            image_attachment = MIMEImage(img_data, name='detection.jpg')
            msg.attach(image_attachment)

        try:
            server = smtplib.SMTP(self.email_config['smtp_server'],
                                 self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'],
                        self.email_config['password'])
            server.send_message(msg)
            server.quit()
            print(f"알림 전송 완료: {subject}")
        except Exception as e:
            print(f"알림 전송 실패: {e}")

    def trigger_alert(self, event_type, message, frame=None):
        """알림 트리거"""
        if self.should_alert(event_type):
            print(f"🚨 알림: {message}")
            if self.email_config:
                self.send_email_alert(f"Vision Alert: {event_type}",
                                     message, frame)

# 사용 예시
email_config = {
    'from': 'your-email@gmail.com',
    'to': 'recipient@gmail.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your-email@gmail.com',
    'password': 'your-app-password'
}

notifier = EventNotifier(email_config)

# 객체 탐지 루프에서
for detection in detections:
    if detection['label'] == 'person':
        notifier.trigger_alert('person_detected',
                              f"사람이 감지되었습니다 (신뢰도: {detection['score']:.2f})",
                              frame)
```

### 3. 비디오 녹화

이벤트 발생 시 자동으로 비디오를 녹화합니다.

```python
import cv2
from datetime import datetime
import os

class EventRecorder:
    def __init__(self, output_dir='recordings', fps=20, duration=10):
        self.output_dir = output_dir
        self.fps = fps
        self.duration = duration
        self.is_recording = False
        self.video_writer = None
        self.frames_recorded = 0
        self.max_frames = fps * duration

        os.makedirs(output_dir, exist_ok=True)

    def start_recording(self, frame_size):
        """녹화 시작"""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"event_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, frame_size)

        self.is_recording = True
        self.frames_recorded = 0
        print(f"📹 녹화 시작: {filename}")

    def record_frame(self, frame):
        """프레임 녹화"""
        if not self.is_recording:
            return

        self.video_writer.write(frame)
        self.frames_recorded += 1

        if self.frames_recorded >= self.max_frames:
            self.stop_recording()

    def stop_recording(self):
        """녹화 중지"""
        if not self.is_recording:
            return

        self.video_writer.release()
        self.is_recording = False
        print("⏹️ 녹화 종료")

# 사용 예시
recorder = EventRecorder(duration=15)  # 15초 녹화

while True:
    ret, frame = cap.read()

    # 이벤트 감지
    if person_detected and not recorder.is_recording:
        recorder.start_recording((frame.shape[1], frame.shape[0]))

    # 녹화 중이면 프레임 저장
    if recorder.is_recording:
        recorder.record_frame(frame)
```

### 4. 웹 스트리밍

Flask를 사용하여 웹 브라우저에서 실시간 스트림을 볼 수 있습니다.

```python
from flask import Flask, Response, render_template_string
import cv2
import threading

class VideoStreamer:
    def __init__(self, detector):
        self.detector = detector
        self.output_frame = None
        self.lock = threading.Lock()

        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Flask 라우트 설정"""
        @self.app.route('/')
        def index():
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Vision Stream</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            text-align: center;
                            background-color: #1a1a1a;
                            color: white;
                        }
                        img {
                            max-width: 90%;
                            border: 2px solid #333;
                            margin-top: 20px;
                        }
                    </style>
                </head>
                <body>
                    <h1>🎥 Real-time Vision Stream</h1>
                    <img src="{{ url_for('video_feed') }}" />
                </body>
                </html>
            ''')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames(self):
        """프레임 생성기"""
        while True:
            with self.lock:
                if self.output_frame is None:
                    continue

                # JPEG 인코딩
                _, buffer = cv2.imencode('.jpg', self.output_frame)
                frame_bytes = buffer.tobytes()

            # 멀티파트 응답 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def update_frame(self, frame):
        """프레임 업데이트"""
        with self.lock:
            self.output_frame = frame.copy()

    def run(self, host='0.0.0.0', port=5000):
        """웹 서버 실행"""
        self.app.run(host=host, port=port, threaded=True)

# 사용 예시
detector = RealtimeObjectDetector()
streamer = VideoStreamer(detector)

# 별도 스레드에서 웹 서버 실행
threading.Thread(target=streamer.run, daemon=True).start()

# 메인 루프
while True:
    ret, frame = cap.read()
    # 객체 탐지
    detections, _ = detector.detect_objects(frame)
    frame = detector.draw_detections(frame, detections, 0)

    # 프레임 업데이트
    streamer.update_frame(frame)
```

### 5. 데이터 로깅 및 분석

탐지 결과를 데이터베이스에 저장하고 분석합니다.

```python
import sqlite3
from datetime import datetime
import json

class DetectionLogger:
    def __init__(self, db_path='detections.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT NOT NULL,
                frame_path TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                label TEXT NOT NULL,
                count INTEGER NOT NULL,
                avg_confidence REAL NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def log_detection(self, detection, frame_path=None):
        """탐지 결과 로깅"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        bbox = json.dumps(detection['box'])

        cursor.execute('''
            INSERT INTO detections (timestamp, label, confidence, bbox, frame_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, detection['label'], detection['score'], bbox, frame_path))

        conn.commit()
        conn.close()

    def get_statistics(self, start_date=None, end_date=None):
        """통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT label, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM detections
        '''

        params = []
        if start_date:
            query += " WHERE timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?" if start_date else " WHERE timestamp <= ?"
            params.append(end_date)

        query += " GROUP BY label ORDER BY count DESC"

        cursor.execute(query, params)
        results = cursor.fetchall()

        conn.close()
        return results

    def get_hourly_distribution(self, label=None):
        """시간대별 탐지 분포"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM detections
        '''

        if label:
            query += " WHERE label = ?"
            cursor.execute(query + " GROUP BY hour ORDER BY hour", (label,))
        else:
            cursor.execute(query + " GROUP BY hour ORDER BY hour")

        results = cursor.fetchall()
        conn.close()

        return results

# 사용 예시
logger = DetectionLogger()

# 탐지 루프에서
for detection in detections:
    logger.log_detection(detection)

# 통계 조회
stats = logger.get_statistics()
for label, count, avg_conf in stats:
    print(f"{label}: {count}회 탐지, 평균 신뢰도: {avg_conf:.2f}")
```

### 6. 다중 카메라 지원

여러 카메라를 동시에 처리합니다.

```python
import cv2
import threading
import queue

class MultiCameraSystem:
    def __init__(self, camera_configs, detector):
        self.cameras = {}
        self.detector = detector
        self.running = False

        for cam_id, config in camera_configs.items():
            self.cameras[cam_id] = {
                'pipeline': config['pipeline'],
                'capture': None,
                'thread': None,
                'queue': queue.Queue(maxsize=10),
                'name': config.get('name', f'Camera {cam_id}')
            }

    def start_camera(self, cam_id):
        """카메라 시작"""
        cam = self.cameras[cam_id]
        cam['capture'] = cv2.VideoCapture(cam['pipeline'], cv2.CAP_GSTREAMER)

        def capture_loop():
            while self.running:
                ret, frame = cam['capture'].read()
                if ret and not cam['queue'].full():
                    cam['queue'].put(frame)

        cam['thread'] = threading.Thread(target=capture_loop, daemon=True)
        cam['thread'].start()

    def process_cameras(self):
        """모든 카메라 처리"""
        while self.running:
            for cam_id, cam in self.cameras.items():
                try:
                    frame = cam['queue'].get(timeout=1)

                    # 객체 탐지
                    detections, _ = self.detector.detect_objects(frame)

                    # 결과 표시
                    result_frame = self.detector.draw_detections(frame, detections, 0)
                    cv2.imshow(f"{cam['name']}", result_frame)

                except queue.Empty:
                    continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

    def start(self):
        """시스템 시작"""
        self.running = True

        # 모든 카메라 시작
        for cam_id in self.cameras:
            self.start_camera(cam_id)

        # 처리 루프
        self.process_cameras()

    def stop(self):
        """시스템 중지"""
        self.running = False

        for cam in self.cameras.values():
            if cam['capture']:
                cam['capture'].release()

        cv2.destroyAllWindows()

# 사용 예시
camera_configs = {
    'front': {
        'pipeline': 'v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink',
        'name': 'Front Camera'
    },
    'back': {
        'pipeline': 'v4l2src device=/dev/video1 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink',
        'name': 'Back Camera'
    }
}

detector = RealtimeObjectDetector()
multi_cam = MultiCameraSystem(camera_configs, detector)
multi_cam.start()
```

### 7. 모션 감지 + AI

모션이 감지된 영역만 AI로 처리하여 리소스를 절약합니다.

```python
import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.previous_frame = None

    def detect(self, frame):
        """모션 감지"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 첫 프레임 초기화
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, []

        # 프레임 차이 계산
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # 노이즈 제거
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 모션 영역 필터링
        motion_areas = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            motion_areas.append((x, y, w, h))

        # 이전 프레임 업데이트
        self.previous_frame = gray

        has_motion = len(motion_areas) > 0
        return has_motion, motion_areas

class MotionTriggeredDetector:
    def __init__(self, detector):
        self.motion_detector = MotionDetector()
        self.ai_detector = detector

    def process_frame(self, frame):
        """모션 기반 AI 처리"""
        has_motion, motion_areas = self.motion_detector.detect(frame)

        detections = []
        if has_motion:
            # 모션이 있을 때만 AI 처리
            detections, inference_time = self.ai_detector.detect_objects(frame)

            # 모션 영역 표시 (디버깅용)
            for (x, y, w, h) in motion_areas:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

        return frame, detections, has_motion

# 사용 예시
ai_detector = RealtimeObjectDetector()
motion_triggered = MotionTriggeredDetector(ai_detector)

while True:
    ret, frame = cap.read()
    result_frame, detections, has_motion = motion_triggered.process_frame(frame)

    if has_motion:
        print(f"모션 감지! {len(detections)}개 객체 탐지됨")

    cv2.imshow('Motion Triggered Detection', result_frame)
```

## 트러블슈팅

### 일반적인 문제들

#### 1. 카메라를 열 수 없음

```bash
# 카메라 장치 확인
ls -l /dev/video*

# 권한 확인
sudo usermod -a -G video $USER
# 재로그인 필요

# V4L2 유틸리티로 테스트
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

#### 2. GStreamer 파이프라인 오류

```python
# 디버깅을 위한 환경 변수
import os
os.environ['GST_DEBUG'] = '3'  # 로그 레벨 (1-5)

# 파이프라인 테스트
import subprocess
result = subprocess.run([
    'gst-launch-1.0',
    'v4l2src', 'device=/dev/video0', '!',
    'video/x-raw,width=640,height=480', '!',
    'videoconvert', '!',
    'autovideosink'
], capture_output=True, text=True)
print(result.stderr)
```

#### 3. 메모리 부족

```python
# 메모리 모니터링
import psutil

def print_memory_usage():
    mem = psutil.virtual_memory()
    print(f"메모리 사용률: {mem.percent}%")
    print(f"사용 가능: {mem.available / 1024**3:.2f} GB")

# 주기적으로 호출
print_memory_usage()
```

#### 4. 낮은 FPS

```python
import time

class FPSCounter:
    def __init__(self, avg_over=30):
        self.avg_over = avg_over
        self.frame_times = []

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time)

        if len(self.frame_times) > self.avg_over:
            self.frame_times.pop(0)

    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0

        elapsed = self.frame_times[-1] - self.frame_times[0]
        return len(self.frame_times) / elapsed if elapsed > 0 else 0

# 사용
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()
    # 처리...

    fps_counter.update()
    fps = fps_counter.get_fps()
    print(f"FPS: {fps:.1f}")
```

### 성능 벤치마킹

```python
import time
import numpy as np

class PerformanceBenchmark:
    def __init__(self):
        self.metrics = {
            'frame_read': [],
            'preprocessing': [],
            'inference': [],
            'postprocessing': [],
            'visualization': []
        }

    def measure(self, name, func, *args, **kwargs):
        """함수 실행 시간 측정"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        self.metrics[name].append(elapsed)
        return result

    def report(self):
        """성능 리포트 출력"""
        print("\n=== Performance Report ===")
        total_avg = 0

        for name, times in self.metrics.items():
            if not times:
                continue

            avg = np.mean(times) * 1000  # ms
            std = np.std(times) * 1000
            min_t = np.min(times) * 1000
            max_t = np.max(times) * 1000

            print(f"{name:20s}: {avg:6.1f}ms ±{std:5.1f}ms "
                  f"(min: {min_t:6.1f}ms, max: {max_t:6.1f}ms)")

            total_avg += avg

        print(f"{'Total':20s}: {total_avg:6.1f}ms")
        print(f"{'Theoretical FPS':20s}: {1000/total_avg:.1f}")

# 사용 예시
benchmark = PerformanceBenchmark()

# 메인 루프
for i in range(100):  # 100 프레임 벤치마크
    frame = benchmark.measure('frame_read', cap.read)
    # ... 다른 처리들

benchmark.report()
```

## 참고 자료

### 공식 문서
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GStreamer 문서](https://gstreamer.freedesktop.org/documentation/)
- [Raspberry Pi 문서](https://www.raspberrypi.com/documentation/)

### 모델 허브
- [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=image-classification)
- [Vision Models](https://huggingface.co/models?pipeline_tag=object-detection)

### 커뮤니티
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Raspberry Pi Forums](https://forums.raspberrypi.com/)

## 결론

이 튜토리얼에서는 라즈베리파이에서 Hugging Face Vision 모델과 GStreamer를 통합하는 방법을 상세히 다뤘습니다. 제한된 리소스 환경에서도 최신 AI 모델을 활용할 수 있으며, 다양한 실용적인 애플리케이션을 개발할 수 있습니다.

핵심 포인트:
- ✅ 경량 모델 선택 (MobileNet, YOLOS, SegFormer 등)
- ✅ GStreamer를 통한 유연한 비디오 처리
- ✅ 성능 최적화 (ONNX, 양자화, 프레임 건너뛰기)
- ✅ 실용적인 기능 추가 (추적, 알림, 녹화, 스트리밍)

행복한 코딩 되세요! 🚀
