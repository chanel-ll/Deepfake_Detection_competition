# Deepfake_Detection_competition
2025 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회
# Deepfake Detection using SwinV2 (OpenFake-based)

## 🚀 Overview
본 프로젝트는 **Deepfake Detection** 문제를 대상으로,  
OpenFake 데이터셋과 Swin Transformer V2 기반 모델을 활용하여  
영상/이미지 레벨의 위·변조 판별 성능을 향상시키는 것을 목표로 한 프로젝트입니다.

특히 **Inference 단계의 파이프라인 튜닝**과 **Threshold 최적화**를 중심으로  
실제 환경에서 성능을 개선하는 방법론을 탐구하였습니다.

> ⚠️ 본 저장소는 개인 연구 및 학습 목적의 정리 자료입니다.  
> 특정 경진대회 제출용 코드/데이터와는 구현 설정이 상이할 수 있으며,  
> 해당 제출용 코드 및 데이터는 포함되어 있지 않습니다.

---

## 📚 References

### 📄 논문
- **OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection**  
  본 연구는 위 논문의 접근을 기반으로 진행되었습니다.

### 🔗 모델 & 데이터셋
- **Backbone Model (Hugging Face)**  
  SwinV2-Base Transformer  
  🔗 https://huggingface.co/microsoft/swinv2-base-patch4-window16-256?utm_source=chatgpt.com

- **OpenFake Dataset (Hugging Face Dataset)**  
  🔗 https://huggingface.co/datasets/ComplexDataLab/OpenFake?utm_source=chatgpt.com

---

## 🗂 Dataset

### 📌 OpenFake
- 공개 Deepfake Detection 데이터셋
- Face Forensics 기반의 다양한 영상/프레임으로 구성
- 본 저장소에는 데이터가 포함되어 있지 않으며,
  각 사용자가 위 링크를 통해 직접 획득해야 합니다.

👉 데이터 라이선스 및 사용 조건은 원 소스의 규정을 준수하세요.

---

## 🧠 Model Architecture

- **Backbone Model**: Swin Transformer V2 Base  
  논문은 SwinV2 Small 사용, 본 프로젝트에서는 더 큰 모델(SwinV2 Base)을 활용  
  영상 내 미세한 위조 징후까지 더 정교하게 캡처할 수 있도록 설계

---

## 🔄 Inference Pipeline Optimization

### 1. Frame Sampling
- **Uniform Frame Sampling**  
  영상 전체에서 균등 간격으로 **10~15 Frame**을 추출하여 inference 수행

### 2. Logit-based Aggregation
- Softmax 대신 **logit 값**을 aggregation에 사용  
- Fake class 관련 logit만 모아 frame-level score로 활용

### 3. Top-K Frame Aggregation
- 전체 평균 대신 **Top-K (K=5)** frame의 logit만 평균하여 최종 score 산출

### 4. Threshold Optimization
- Default(0.5) 대신 Validation 기반 **Threshold Sweep** 수행
- 최적 threshold **0.07** 적용

---
