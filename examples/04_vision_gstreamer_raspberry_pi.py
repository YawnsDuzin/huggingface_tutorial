#!/usr/bin/env python3
"""
Hugging Face Vision Models + GStreamer Integration for Raspberry Pi
라즈베리파이용 Hugging Face Vision 모델 + GStreamer 통합 예제

이 스크립트는 라즈베리파이에서 GStreamer와 Hugging Face Vision 모델을
통합하여 실시간 비전 처리를 수행합니다.

지원 기능:
- 실시간 이미지 분류
- 실시간 객체 탐지
- 실시간 세그멘테이션
- Zero-shot 분류
- 성능 최적화

사용법:
    python 04_vision_gstreamer_raspberry_pi.py --task classification
    python 04_vision_gstreamer_raspberry_pi.py --task detection --model hustvl/yolos-tiny
    python 04_vision_gstreamer_raspberry_pi.py --task segmentation
    python 04_vision_gstreamer_raspberry_pi.py --task zero-shot --labels "person,dog,cat,car"
"""

import argparse
import cv2
import torch
import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path

# Hugging Face imports
try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        AutoModelForObjectDetection,
        AutoModelForSemanticSegmentation,
        CLIPProcessor,
        CLIPModel
    )
except ImportError:
    print("❌ transformers 라이브러리가 설치되어 있지 않습니다.")
    print("다음 명령어로 설치하세요:")
    print("  pip install transformers pillow torch")
    sys.exit(1)


class PerformanceMonitor:
    """성능 모니터링 클래스"""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.inference_times = []

    def update(self, inference_time):
        """프레임 처리 시간 업데이트"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.inference_times.append(inference_time)

        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            self.inference_times.pop(0)

    def get_fps(self):
        """평균 FPS 계산"""
        if len(self.frame_times) < 2:
            return 0.0

        elapsed = self.frame_times[-1] - self.frame_times[0]
        return len(self.frame_times) / elapsed if elapsed > 0 else 0.0

    def get_avg_inference_time(self):
        """평균 추론 시간 계산 (ms)"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000

    def get_stats(self):
        """통계 정보 반환"""
        return {
            'fps': self.get_fps(),
            'avg_inference_ms': self.get_avg_inference_time(),
            'min_inference_ms': min(self.inference_times) * 1000 if self.inference_times else 0,
            'max_inference_ms': max(self.inference_times) * 1000 if self.inference_times else 0
        }


class ImageClassificationPipeline:
    """이미지 분류 파이프라인"""

    def __init__(self, model_name="google/mobilenet_v2_1.0_224"):
        print(f"📦 모델 로딩 중: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        print("✅ 모델 로딩 완료!")

    def process(self, frame):
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

    def visualize(self, frame, results):
        """결과 시각화"""
        y_offset = 30
        overlay = frame.copy()

        # 반투명 배경
        cv2.rectangle(overlay, (0, 0), (500, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 예측 결과 표시
        for i, (label, prob) in enumerate(results):
            text = f"{i+1}. {label}: {prob*100:.1f}%"
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


class ObjectDetectionPipeline:
    """객체 탐지 파이프라인"""

    def __init__(self, model_name="hustvl/yolos-tiny", confidence_threshold=0.3):
        print(f"📦 모델 로딩 중: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.eval()
        self.confidence_threshold = confidence_threshold

        # 색상 맵
        np.random.seed(42)
        self.colors = {}

        print("✅ 모델 로딩 완료!")

    def process(self, frame):
        """객체 탐지"""
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
        """라벨별 색상"""
        if label not in self.colors:
            self.colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[label]

    def visualize(self, frame, detections):
        """탐지 결과 시각화"""
        for det in detections:
            box = det['box']
            label = det['label']
            score = det['score']
            color = self.get_color(label)

            # 바운딩 박스
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 라벨
            text = f"{label}: {score*100:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # 텍스트 배경
            cv2.rectangle(frame,
                         (box[0], box[1] - text_size[1] - 10),
                         (box[0] + text_size[0] + 10, box[1]),
                         color, -1)

            # 텍스트
            cv2.putText(frame, text, (box[0] + 5, box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 탐지 개수
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame


class SemanticSegmentationPipeline:
    """시맨틱 세그멘테이션 파이프라인"""

    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        print(f"📦 모델 로딩 중: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()

        # 색상 맵
        np.random.seed(42)
        self.colormap = np.random.randint(0, 255, (150, 3), dtype=np.uint8)

        print("✅ 모델 로딩 완료!")

    def process(self, frame):
        """세그멘테이션"""
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

        # 세그멘테이션 맵
        seg_map = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        return seg_map, inference_time

    def visualize(self, frame, seg_map):
        """세그멘테이션 시각화"""
        # 세그멘테이션 맵을 컬러로 변환
        color_seg = self.colormap[seg_map]

        # 원본 크기로 리사이즈
        color_seg = cv2.resize(color_seg, (frame.shape[1], frame.shape[0]))

        # 블렌딩
        alpha = 0.6
        blended = cv2.addWeighted(frame, alpha, color_seg, 1-alpha, 0)

        return blended


class ZeroShotClassificationPipeline:
    """Zero-shot 분류 파이프라인"""

    def __init__(self, candidate_labels, model_name="openai/clip-vit-base-patch32"):
        print(f"📦 모델 로딩 중: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.candidate_labels = candidate_labels

        print(f"✅ 모델 로딩 완료!")
        print(f"🏷️  후보 라벨: {', '.join(candidate_labels)}")

    def process(self, frame):
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

    def visualize(self, frame, results):
        """결과 시각화"""
        y_offset = 30
        overlay = frame.copy()

        # 반투명 배경
        height = min(len(results), 5) * 30 + 40
        cv2.rectangle(overlay, (0, 0), (400, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 예측 결과 (Top 5)
        for i, (label, prob) in enumerate(results[:5]):
            text = f"{label}: {prob*100:.1f}%"
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


class GStreamerVisionApp:
    """GStreamer + Vision 통합 애플리케이션"""

    def __init__(self, pipeline, camera_source='default', camera_width=640,
                 camera_height=480, camera_fps=30, process_every_n_frames=5):
        self.pipeline = pipeline
        self.process_every_n_frames = process_every_n_frames

        # GStreamer 파이프라인 구성
        self.gst_pipeline = self._build_gst_pipeline(
            camera_source, camera_width, camera_height, camera_fps
        )

        # 비디오 캡처
        print(f"📹 카메라 초기화 중...")
        print(f"   파이프라인: {self.gst_pipeline}")
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            print("\n디버깅 팁:")
            print("1. 카메라 장치 확인: ls -l /dev/video*")
            print("2. 권한 확인: sudo usermod -a -G video $USER")
            print("3. GStreamer 테스트:")
            print("   gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink")
            sys.exit(1)

        print("✅ 카메라 초기화 완료!")

        # 성능 모니터
        self.perf_monitor = PerformanceMonitor()

        # 상태 변수
        self.frame_count = 0
        self.current_result = None
        self.current_inference_time = 0

    def _build_gst_pipeline(self, source, width, height, fps):
        """GStreamer 파이프라인 구성"""
        if source == 'default':
            # 기본 V4L2 소스
            pipeline = (
                f"v4l2src device=/dev/video0 ! "
                f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
                f"videoconvert ! "
                f"appsink"
            )
        elif source == 'libcamera':
            # 라즈베리파이 카메라 (libcamera)
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
                f"videoconvert ! "
                f"appsink"
            )
        elif source == 'rpicamsrc':
            # 라즈베리파이 카메라 (구형)
            pipeline = (
                f"rpicamsrc ! "
                f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
                f"videoconvert ! "
                f"appsink"
            )
        elif source.startswith('rtsp://'):
            # RTSP 스트림
            pipeline = (
                f"rtspsrc location={source} ! "
                f"rtph264depay ! "
                f"h264parse ! "
                f"avdec_h264 ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw,width={width},height={height} ! "
                f"appsink"
            )
        elif source.endswith(('.mp4', '.avi', '.mkv')):
            # 비디오 파일
            pipeline = (
                f"filesrc location={source} ! "
                f"qtdemux ! "
                f"h264parse ! "
                f"avdec_h264 ! "
                f"videoconvert ! "
                f"videoscale ! "
                f"video/x-raw,width={width},height={height} ! "
                f"appsink"
            )
        else:
            # 커스텀 장치
            pipeline = (
                f"v4l2src device={source} ! "
                f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
                f"videoconvert ! "
                f"appsink"
            )

        return pipeline

    def run(self):
        """메인 루프"""
        print("\n🚀 애플리케이션 시작!")
        print("   'q' 키를 눌러 종료")
        print("   's' 키를 눌러 통계 출력")
        print("-" * 50)

        try:
            while True:
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  프레임을 읽을 수 없습니다.")
                    break

                self.frame_count += 1

                # 주기적으로 AI 처리
                if self.frame_count % self.process_every_n_frames == 0:
                    self.current_result, self.current_inference_time = self.pipeline.process(frame)
                    self.perf_monitor.update(self.current_inference_time)

                # 시각화
                if self.current_result is not None:
                    frame = self.pipeline.visualize(frame, self.current_result)

                # 성능 정보 표시
                fps = self.perf_monitor.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f} | Inference: {self.current_inference_time*1000:.0f}ms",
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 화면 출력
                cv2.imshow('Hugging Face Vision + GStreamer', frame)

                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_statistics()

        except KeyboardInterrupt:
            print("\n⚠️  사용자에 의해 중단됨")
        finally:
            self._cleanup()

    def _print_statistics(self):
        """통계 정보 출력"""
        stats = self.perf_monitor.get_stats()
        print("\n" + "="*50)
        print("📊 성능 통계")
        print("="*50)
        print(f"평균 FPS:          {stats['fps']:.2f}")
        print(f"평균 추론 시간:    {stats['avg_inference_ms']:.2f}ms")
        print(f"최소 추론 시간:    {stats['min_inference_ms']:.2f}ms")
        print(f"최대 추론 시간:    {stats['max_inference_ms']:.2f}ms")
        print(f"처리된 프레임:     {self.frame_count}")
        print("="*50 + "\n")

    def _cleanup(self):
        """정리"""
        print("\n🧹 정리 중...")
        self._print_statistics()
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 종료 완료!")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Hugging Face Vision + GStreamer for Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  이미지 분류:
    python %(prog)s --task classification

  객체 탐지:
    python %(prog)s --task detection --confidence 0.3

  세그멘테이션:
    python %(prog)s --task segmentation --camera libcamera

  Zero-shot 분류:
    python %(prog)s --task zero-shot --labels "person,dog,cat,car,bike"

  비디오 파일:
    python %(prog)s --task detection --camera /path/to/video.mp4

  RTSP 스트림:
    python %(prog)s --task detection --camera rtsp://192.168.1.100:8554/stream
        """
    )

    # 작업 선택
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'detection', 'segmentation', 'zero-shot'],
        default='classification',
        help='수행할 작업 (기본값: classification)'
    )

    # 모델 선택
    parser.add_argument(
        '--model',
        type=str,
        help='사용할 모델 (기본값: 각 작업별 기본 모델)'
    )

    # 카메라 설정
    parser.add_argument(
        '--camera',
        type=str,
        default='default',
        help='카메라 소스 (default, libcamera, rpicamsrc, /dev/videoN, rtsp://..., video.mp4)'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='프레임 너비 (기본값: 640)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='프레임 높이 (기본값: 480)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='프레임레이트 (기본값: 30)'
    )

    # 처리 설정
    parser.add_argument(
        '--process-every',
        type=int,
        default=5,
        help='N 프레임마다 AI 처리 (기본값: 5, 성능 향상용)'
    )

    # 작업별 설정
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='객체 탐지 신뢰도 임계값 (기본값: 0.3)'
    )

    parser.add_argument(
        '--labels',
        type=str,
        help='Zero-shot 분류용 라벨 (쉼표로 구분, 예: "person,dog,cat")'
    )

    args = parser.parse_args()

    # 작업별 기본 모델
    default_models = {
        'classification': 'google/mobilenet_v2_1.0_224',
        'detection': 'hustvl/yolos-tiny',
        'segmentation': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'zero-shot': 'openai/clip-vit-base-patch32'
    }

    model_name = args.model or default_models[args.task]

    print("="*60)
    print("🤗 Hugging Face Vision + GStreamer for Raspberry Pi")
    print("="*60)
    print(f"작업:        {args.task}")
    print(f"모델:        {model_name}")
    print(f"카메라:      {args.camera}")
    print(f"해상도:      {args.width}x{args.height}")
    print(f"FPS:         {args.fps}")
    print(f"처리 주기:   {args.process_every} 프레임마다")
    print("="*60 + "\n")

    # 파이프라인 생성
    try:
        if args.task == 'classification':
            pipeline = ImageClassificationPipeline(model_name)

        elif args.task == 'detection':
            pipeline = ObjectDetectionPipeline(model_name, args.confidence)

        elif args.task == 'segmentation':
            pipeline = SemanticSegmentationPipeline(model_name)

        elif args.task == 'zero-shot':
            if not args.labels:
                print("❌ Zero-shot 분류에는 --labels 인자가 필요합니다.")
                print("예: --labels \"person,dog,cat,car\"")
                sys.exit(1)

            labels = [label.strip() for label in args.labels.split(',')]
            pipeline = ZeroShotClassificationPipeline(labels, model_name)

    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        sys.exit(1)

    # 애플리케이션 실행
    try:
        app = GStreamerVisionApp(
            pipeline=pipeline,
            camera_source=args.camera,
            camera_width=args.width,
            camera_height=args.height,
            camera_fps=args.fps,
            process_every_n_frames=args.process_every
        )
        app.run()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
