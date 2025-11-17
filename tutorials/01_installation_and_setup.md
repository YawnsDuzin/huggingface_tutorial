# 튜토리얼 1: 설치 및 기본 설정

## 목표
이 튜토리얼에서는 Hugging Face를 시작하기 위한 환경을 설정하고 기본적인 사용법을 배웁니다.

## 1. 환경 요구사항

### 최소 요구사항
- Python 3.8 이상
- pip 또는 conda 패키지 매니저
- 8GB 이상의 RAM (권장: 16GB)
- (선택) NVIDIA GPU (CUDA 지원)

### 권장 환경
- Python 3.9-3.11
- CUDA 11.8 이상 (GPU 사용 시)
- 가상 환경 (venv 또는 conda)

## 2. 가상 환경 설정

### venv 사용 (Python 기본)

```bash
# 가상 환경 생성
python -m venv huggingface_env

# 가상 환경 활성화 (Linux/Mac)
source huggingface_env/bin/activate

# 가상 환경 활성화 (Windows)
huggingface_env\Scripts\activate
```

### conda 사용

```bash
# conda 환경 생성
conda create -n huggingface_env python=3.10

# 환경 활성화
conda activate huggingface_env
```

## 3. 필수 라이브러리 설치

### 기본 설치

```bash
# Transformers 라이브러리 설치
pip install transformers

# PyTorch 설치 (CPU 버전)
pip install torch

# Datasets 라이브러리 설치
pip install datasets
```

### GPU 지원 (NVIDIA CUDA)

```bash
# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 또는 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow 사용자

```bash
# TensorFlow 설치
pip install tensorflow

# Transformers with TensorFlow
pip install transformers[tf]
```

### 전체 설치 (권장)

```bash
# 모든 의존성 포함
pip install transformers[torch]
pip install datasets
pip install accelerate
pip install evaluate
pip install sentencepiece
```

## 4. 설치 확인

다음 Python 코드로 설치를 확인합니다:

```python
import transformers
import torch
import datasets

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Datasets version: {datasets.__version__}")

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
```

## 5. 첫 번째 모델 실행

### 간단한 감성 분석 예제

```python
from transformers import pipeline

# 파이프라인 생성 (자동으로 모델 다운로드)
classifier = pipeline("sentiment-analysis")

# 텍스트 분석
result = classifier("I love learning about AI!")
print(result)
# 출력: [{'label': 'POSITIVE', 'score': 0.9998}]

# 여러 텍스트 분석
texts = [
    "This is amazing!",
    "I'm not sure about this.",
    "This is terrible."
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text} -> {result}")
```

## 6. 모델 캐시 이해하기

Hugging Face는 다운로드한 모델을 로컬에 캐시합니다.

### 기본 캐시 위치
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\<username>\.cache\huggingface\`

### 캐시 위치 변경

```bash
# 환경 변수 설정
export HF_HOME=/path/to/custom/cache

# 또는 Python 코드에서
import os
os.environ['HF_HOME'] = '/path/to/custom/cache'
```

### 캐시 관리

```python
from huggingface_hub import scan_cache_dir

# 캐시 정보 확인
cache_info = scan_cache_dir()
print(f"Total cache size: {cache_info.size_on_disk_str}")

# 캐시 삭제 (필요 시)
# cache_info.delete_revisions(*revisions_to_delete).execute()
```

## 7. Hugging Face Hub 인증 (선택)

일부 모델이나 기능을 사용하려면 Hugging Face 계정이 필요합니다.

### 계정 생성
1. https://huggingface.co/ 에서 회원가입
2. Settings에서 Access Token 생성

### 로그인 방법

```bash
# CLI를 통한 로그인
pip install huggingface_hub
huggingface-cli login

# 토큰 입력 후 엔터
```

또는 Python에서:

```python
from huggingface_hub import login

login(token="your_token_here")
```

## 8. 추가 유틸리티 설치

### Jupyter Notebook

```bash
pip install jupyter notebook
pip install ipywidgets  # 위젯 지원
```

### 시각화 도구

```bash
pip install matplotlib seaborn
pip install plotly  # 인터랙티브 차트
```

### 진행률 표시

```bash
pip install tqdm
```

### 모델 분석 도구

```bash
pip install tensorboard
pip install wandb  # Weights & Biases
```

## 9. requirements.txt 생성

프로젝트 의존성을 관리하기 위한 파일:

```txt
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
accelerate>=0.24.0
evaluate>=0.4.0
sentencepiece>=0.1.99
huggingface-hub>=0.19.0
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
```

설치:
```bash
pip install -r requirements.txt
```

## 10. 문제 해결

### 일반적인 문제

**문제: ModuleNotFoundError**
```bash
# 해결: 라이브러리 재설치
pip install --upgrade transformers
```

**문제: CUDA out of memory**
```python
# 해결: 작은 배치 사이즈 사용
# 또는 CPU 사용
device = "cpu"
```

**문제: 느린 다운로드**
```bash
# 해결: 미러 사용 (중국 사용자)
export HF_ENDPOINT=https://hf-mirror.com
```

## 11. 다음 단계

환경 설정이 완료되었습니다! 이제 다음 튜토리얼로 진행하세요:

- **튜토리얼 2**: 텍스트 분류 실습
- **튜토리얼 3**: 텍스트 생성
- **튜토리얼 4**: 모델 파인튜닝

## 연습 문제

1. 가상 환경을 생성하고 필수 라이브러리를 설치하세요.
2. GPU가 사용 가능한지 확인하는 스크립트를 작성하세요.
3. 세 가지 다른 파이프라인을 사용해보세요 (sentiment-analysis, text-generation, question-answering).
4. 다운로드된 모델의 캐시 크기를 확인하세요.

## 참고 자료

- [Transformers 설치 가이드](https://huggingface.co/docs/transformers/installation)
- [PyTorch 설치](https://pytorch.org/get-started/locally/)
- [Hugging Face Hub 문서](https://huggingface.co/docs/hub/index)
