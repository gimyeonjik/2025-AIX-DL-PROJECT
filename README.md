# 딥러닝 기반 야구 영상 분석 파이프라인

## Members

| 이름 | 소속 | 학번 | 이메일 |
| --- | --- | --- | --- |
| 김현준 | 에리카 인공지능학과 | 2023054639 | gizxmk@hanyang.ac.kr |
| 도기훈 | 에리카 인공지능학과 | 2021009407 | support7417@hanyang.ac.kr |

---

## I. Proposal (Option A)

### Motivation

야구 경기 중계와 분석은 전통적으로 해설자와 분석가들이 실시간으로 경기 상황을 파악하고 설명하는 방식에 의존해왔다. 그러나 이러한 수동 분석 방식에는 몇 가지 한계가 존재한다.

**기존 방식의 한계점**

- 전문 인력의 지속적인 투입이 필요하여 비용이 높음
- 인간의 인지 한계로 인해 빠르게 진행되는 플레이의 세부 동작을 놓칠 수 있음
- 경기 후 분석 시 대량의 영상을 수동으로 검토해야 하는 비효율성
- 다양한 카메라 앵글의 영상을 동시에 분석하기 어려움

**딥러닝 기반 접근의 필요성**

최근 Object Detection과 Pose Estimation 기술의 발전으로 영상 내 객체 인식과 인체 자세 추정의 정확도가 크게 향상되었다. 이러한 기술을 야구 영상 분석에 적용하면 경기 상황을 자동으로 인식하고 분류할 수 있어, 중계 자동화 및 경기 분석 효율화에 기여할 수 있다.

### Project Goal

본 프로젝트의 최종 목표는 야구 중계 영상에서 다양한 경기 상황을 자동으로 인식하는 딥러닝 파이프라인을 개발하는 것이다.

**구체적인 목표**

1. 영상 내 야구 선수 및 객체를 정확하게 탐지 (Object Detection)
2. 탐지된 선수의 신체 키포인트를 추정하여 자세 정보 추출 (Pose Estimation)
3. 시계열 분석을 통해 연속된 프레임에서 경기 상황을 분류 (Action Classification)
4. 실제 중계 영상에 적용 가능한 End-to-End 파이프라인 구축

---

## II. Datasets

### 데이터셋 개요

본 프로젝트에서는 AI Hub에서 제공하는 "야구 스포츠 영상" 데이터셋을 활용하였다.

| 항목 | 내용 |
| --- | --- |
| 데이터셋 출처 | AI Hub - 야구 스포츠 영상 |
| 구축년도 | 2021년 |
| 총 데이터 규모 | 1,220,000장 이상 |
| 이미지 해상도 | 1920 × 1080 (Full HD) |
| 데이터 형식 | JPG (이미지), JSON (라벨) |

### 실제 학습에 사용된 데이터 규모

**1. Object Detection (YOLOv13)**

| 구분 | 이미지 수 | 용량 |
| --- | --- | --- |
| Train | 1,007,489 | 4.0 GB |
| Validation | 135,314 | 538 MB |
| **Total** | **1,142,803** | **~4.5 GB** |

**2. Pose Estimation (ViTPose)**

| 구분 | 이미지 수 | 비율 |
| --- | --- | --- |
| Train | 157,039 | 88.8% |
| Validation | 19,912 | 11.2% |
| **Total** | **176,951** | **100%** |

**3. Action Classification (LSTM)**

| 구분 | 시퀀스 수 | 배치 수 |
| --- | --- | --- |
| Train | 3,888 | 121 |
| Validation | 504 | 16 |
| **Total** | **4,392** | - |
- 시퀀스 길이: 최대 64 프레임
- 입력 차원: 72 (24 keypoints × 3 [x, y, confidence])

### 객체 클래스 구성 (15개)

| ID | 클래스명 | 한글명 |
| --- | --- | --- |
| 0 | pitcher | 투수 |
| 1 | hitter | 타자 |
| 2 | catcher | 포수 |
| 3 | the_chief_umpire | 주심 |
| 4 | normal_umpire | 일반 심판 |
| 5 | center_fielder | 중견수 |
| 6 | runner | 주자 |
| 7 | second_baseman | 2루수 |
| 8 | short_stop | 유격수 |
| 9 | right_fielder | 우익수 |
| 10 | left_fielder | 좌익수 |
| 11 | first_baseman | 1루수 |
| 12 | third_baseman | 3루수 |
| 13 | others | 기타 |
| 14 | fielder | 일반 수비수 |

### 동작 클래스 구성 (13개)

| ID | 영문명 | 한글명 |
| --- | --- | --- |
| 0 | catch_throw | 포수 송구 |
| 1 | foul_fly | 파울 플라이 |
| 2 | fly_ground | 플라이/땅볼 아웃 |
| 3 | foul_tip | 파울 팁 |
| 4 | hit_by_pitch | 몸에 맞는 공 |
| 5 | home_run | 홈런 타격 |
| 6 | swing | 스윙 |
| 7 | passed_ball | 패스트볼 |
| 8 | pick_off | 견제 |
| 9 | pitch | 투구 |
| 10 | pitch_setup | 투구 준비 |
| 11 | play_up | 플레이 업 |
| 12 | runner_run | 주자 달리기 |

### Pose Keypoint 구성 (24개)

| 영역 | 키포인트 |
| --- | --- |
| 머리/눈 (3개) | head, eye_right, eye_left |
| 목/몸통 (2개) | neck, chest |
| 상체 (8개) | right/left_shoulder, right/left_elbow, right/left_wrist, right/left_fingertips |
| 하체 (11개) | waist, right/left_hip, right/left_knee, right/left_ankle, right/left_tiptoe, right/left_heel |

### Annotation 형식 예시

**Object Detection**

```json
{
  "image": {
    "filename": "baseball_rg_d01_00005_00001_00001.jpg",
    "resolution": [1920, 1080]
  },
  "annotations": [
    {
      "class": "player",
      "attribute": [{"position": "hitter"}],
      "bbox": [1030, 692, 1103, 909]
    }
  ]
}
```

---

## III. Methodology

본 프로젝트의 방법론은 세 가지 주요 단계로 구성되며, 각 단계의 출력이 다음 단계의 입력으로 연결되는 파이프라인 구조를 갖는다.

```
Video Frame → [Stage 1: Object Detection] → Bounding Boxes
                                                    ↓
                        [Stage 2: Pose Estimation] → Keypoint Sequences
                                                    ↓
                      [Stage 3: Action Classification] → Action Class

```

### Stage 1: Object Detection - YOLOv13

**목적:** 비디오 프레임에서 선수, 심판 등의 객체를 탐지하고 바운딩 박스를 추출

**모델 선정:** YOLOv13s (Small)

- 실시간 처리가 가능한 높은 추론 속도
- 최신 YOLO 아키텍처의 향상된 성능

**하이퍼파라미터 설정**

| 파라미터 | 값 |
| --- | --- |
| Model | YOLOv13s (Small) |
| Epochs | 10 |
| Batch Size | 128 (8 GPU × 16) |
| Image Size | 640×640 |
| Optimizer | SGD |
| Initial LR | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Warmup Epochs | 3.0 |
| AMP (Mixed Precision) | True |
| Early Stopping Patience | 50 |

**데이터 증강 기법**

- Mosaic: 1.0
- HSV-H: 0.015, HSV-S: 0.7, HSV-V: 0.4
- Flip LR: 0.5, Scale: 0.5, Translate: 0.1
- Copy-Paste: 0.1, Erasing: 0.4
- AutoAugment: randaugment

**입력/출력**

- Input: Video frame (1920 × 1080 × 3) → Resize to (640 × 640 × 3)
- Output: Bounding boxes with class labels (15 classes)

### Stage 2: Pose Estimation - ViTPose

**목적:** 탐지된 선수의 바운딩 박스 영역 내에서 신체 키포인트를 추정

**모델 선정:** ViTPose-Base (Vision Transformer for Pose Estimation)

- Transformer 기반의 높은 정확도
- 다양한 자세와 가려짐(occlusion)에 robust
- AI Hub 데이터셋의 24개 키포인트 체계에 맞게 fine-tuning

**모델 구조**

| 구성요소 | 상세 |
| --- | --- |
| Backbone | ViT-Base (12 layers) |
| Patch Size | 16 |
| Head | HeatmapHead (Deconv 256→256) |
| Decoder | UDPHeatmap (Heatmap size: 48×64, Sigma: 2) |
| Loss | KeypointMSELoss |
| Drop Path Rate | 0.3 |

**하이퍼파라미터 설정**

| 파라미터 | 값 |
| --- | --- |
| Image Size | 256×192 |
| Epochs | 100 |
| Batch Size | 256 (8 GPU × 32) |
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Weight Decay | 0.1 |
| LR Scheduler | MultiStepLR (milestones: 170, 200) |
| Layer Decay Rate | 0.75 |
| Gradient Clipping | max_norm=1.0 |

**입력/출력**

- Input: Cropped player image (bounding box 영역, 256×192)
- Output: 24개 keypoint 좌표 + confidence scores

### Stage 3: Action Classification - Bidirectional LSTM with Attention

**목적:** 연속된 프레임의 포즈 시퀀스를 분석하여 경기 상황(동작)을 분류

**모델 선정:** Bidirectional LSTM with Attention Mechanism

- Pose Stream을 통해 포즈 키포인트 시퀀스를 처리하여 동작의 시간적 패턴 학습
- Attention mechanism으로 중요한 시간 구간에 집중

**모델 구조**

| 구성요소 | 상세 |
| --- | --- |
| Input Dimension | 72 (24 keypoints × 3) |
| Hidden Dimension | 256 |
| LSTM Layers | 2 |
| Bidirectional | True |
| Attention | True |
| Dropout | 0.3 |
| 총 파라미터 수 | 2,915,597개 |

**하이퍼파라미터 설정**

| 파라미터 | 값 |
| --- | --- |
| Epochs | 100 (Early Stop at 40) |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 0.01 |
| LR Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Label Smoothing | 0.1 |
| Early Stopping Patience | 15 |
| Gradient Clipping | max_norm=1.0 |

**데이터 증강 기법**

- Horizontal Flip: 50%
- Noise Injection: 30% (σ=0.02)
- Temporal Dropout: 20% (drop_prob=0.1)
- Weighted Sampler: 클래스 불균형 처리

**클래스 가중치 (불균형 처리)**

- 가장 높음: passed_ball (3.40)
- 가장 낮음: foul_fly (0.64)

**입력/출력**

- Input: Pose sequence (최대 64 frames × 72 features)
- Output: Action class probability distribution (13 classes)

### 학습 환경

모든 모델은 동일한 하드웨어 환경에서 학습되었다.

| 항목 | 사양 |
| --- | --- |
| GPU | 8× NVIDIA RTX 3080 |
| 분산 학습 | DDP (Distributed Data Parallel) |

---

## IV. Evaluation & Analysis

### 전체 성능 요약

| 모델 | 주요 성능 지표 | 데이터 규모 | 클래스 수 | 학습 상태 |
| --- | --- | --- | --- | --- |
| **YOLOv13s** | mAP@50: **98.44%** | 1,142,803 이미지 | 15개 | 완료 (10 epoch) |
| **ViTPose-Base** | AP: **98.46%** | 176,951 이미지 | 24개 키포인트 | 완료 (100 epoch) |
| **LSTM** | Accuracy: **91.87%**, F1: **0.92** | 4,392 시퀀스 | 13개 동작 | 완료 (40 epoch) |

### Stage 1: YOLOv13 Object Detection 결과

**성능 지표**

| Metric | 값 |
| --- | --- |
| **Precision** | 96.85% |
| **Recall** | 96.20% |
| **mAP@50** | 98.44% |
| **mAP@75** | 95.39% |
| **mAP@50-95** | 85.60% |

**손실 함수 수렴**

| Loss Type | Train | Validation |
| --- | --- | --- |
| Box Loss | 0.629 | 0.449 |
| Cls Loss | 0.394 | 0.245 |

**분석:**

- mAP@50 98.44%의 높은 성능 달성
- Validation Loss가 Train Loss보다 낮아 과적합 없이 일반화 성능 양호
- 대규모 데이터셋(114만 장)과 효과적인 데이터 증강이 빠른 수렴에 기여

![image.png](attachment:31b7134c-b84b-4d6f-833b-9a1e95f11715:image.png)

### Stage 2: ViTPose Pose Estimation 결과

**성능 지표**

| Metric | Best (Epoch 40) | Final (Epoch 100) |
| --- | --- | --- |
| **AP @IoU=0.50:0.95** | **98.46%** | 98.18% |
| **AP @IoU=0.50** | 99.30% | 99.00% |
| **AP @IoU=0.75** | 99.00% | 99.00% |
| **AR @IoU=0.50:0.95** | 99.23% | 99.25% |
| **AR @IoU=0.50** | 99.40% | 100.00% |
| **AR @IoU=0.75** | 99.96% | 99.98% |

**분석:**

- Epoch 40에서 최고 성능(AP 98.46%) 달성 후 약간의 성능 하락 관찰
- AP@IoU=0.50에서 100%에 근접하는 매우 높은 정확도
- Vision Transformer의 global attention mechanism이 야구 선수의 다양한 자세를 효과적으로 학습
- 24개 키포인트 전체에 대해 안정적인 추정 성능 확보

![image.png](attachment:5788ab18-d1d1-49dc-9266-4ac36fc2b9b5:image.png)

### Stage 3: LSTM Action Classification 결과

**성능 지표 (Best: Epoch 40)**

| Metric | Train | Validation |
| --- | --- | --- |
| **Accuracy** | 99.97% | **91.87%** |
| **F1-Score (Macro)** | - | **0.9168** |
| **Loss** | 0.5191 | 0.8837 |

![image.png](attachment:fc232af3-dc0d-4687-9211-bddba537d2c5:image.png)

**분석:**

- Validation Accuracy 91.87%, F1-Score 0.9168로 13개 동작 클래스에 대해 우수한 분류 성능
- Train과 Validation 간 약 8%의 정확도 차이는 일정 수준의 과적합 존재 암시
- Early Stopping (Epoch 40)이 효과적으로 작동하여 과적합 방지
- Weighted Sampler를 통한 클래스 불균형 처리가 F1-Score 향상에 기여

### 파이프라인 종합 평가

**강점:**

1. 각 단계에서 95% 이상의 높은 정확도 달성
2. 대규모 데이터셋(총 130만 장 이상)을 활용한 robust한 학습
3. 8× GPU DDP 환경을 통한 효율적인 학습 진행

**개선 필요 사항:**

1. GT 키포인트 vs ViTPose 키포인트 도메인 갭 (핵심 원인)

| 구분 | 학습 데이터 | 추론 데이터 |
| --- | --- | --- |
| 키포인트 출처 | 사람이 수동 라벨링 (GT) | ViTPose 자동 추정 |
| 정확도 | 픽셀 단위 정확 | 추정 오차 존재 |
| 일관성 | 높음 | 포즈/각도에 따라 변동 |
- **문제**: 모델이 "깨끗한" GT 키포인트만 학습하여 ViTPose의 노이즈가 포함된 키포인트를 제대로 인식하지 못함
- **해결 방안**: ViTPose로 학습 영상에서 키포인트를 재추출하여 재학습 필요

---

2. 신뢰도(Confidence) 스케일 불일치

| 구분 | 학습 데이터 (JSON) | ViTPose 출력 |
| --- | --- | --- |
| 신뢰도 범위 | 1 또는 2 (정수) | 0.0 ~ 1.0 (실수) |
- LSTM 입력 72차원 중 24개가 신뢰도 값
- 스케일 변환(+1)을 적용해도 분포 특성이 다름

---

3. RGB 정보 미활용

- 현재 모델은 **키포인트 좌표(x, y, confidence)만** 사용
- 배경, 유니폼 색상, 공/배트 등 시각적 단서 무시
- 유사한 자세의 다른 동작 구분이 어려움 (예: 투구 준비 vs 타격 준비)

---

4. 학습/추론 환경 일반화 한계

- **학습 데이터**: 실내 또는 특정 야구장 환경
- **추론 영상**: 다양한 야구장, 중계 카메라 각도
- 카메라 앵글, 조명, 해상도 차이가 키포인트 추출 품질에 영향

---

## 권장 개선 방향

1. **ViTPose 기반 재학습**: 학습 영상을 ViTPose로 처리하여 키포인트 재추출 후 학습
2. **RGB + Keypoint 융합 모델**: 시각 정보와 키포인트를 함께 활용
3. **데이터 증강 강화**: 키포인트에 실제 ViTPose 수준의 노이즈 추가하여 학습

---

## V. Related Work

### 참고 논문

| 논문 | 활용 내용 |
| --- | --- |
| Redmon et al., "YOLOv3: An Incremental Improvement" (2018) | YOLO 아키텍처 기본 개념 |
| Ultralytics YOLOv8 (2023) | 최신 YOLO 구현 및 학습 기법 |
| Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation" (NeurIPS 2022) | Transformer 기반 포즈 추정 |
| Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) | LSTM 시계열 모델링 |

### 활용 라이브러리 및 도구

| 도구 | 용도 |
| --- | --- |
| **PyTorch** | 딥러닝 프레임워크 |
| **Ultralytics** | YOLOv13 구현 및 학습 |
| **MMPose** | ViTPose 구현 및 학습 |
| **OpenCV** | 이미지/비디오 처리 |
| **NumPy, Pandas** | 데이터 처리 및 분석 |

### 데이터셋 출처

- AI Hub 야구 스포츠 영상 데이터: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=488

---

## VI. Conclusion

### 프로젝트 성과

본 프로젝트에서는 야구 중계 영상에서 경기 상황을 자동으로 인식하는 3단계 딥러닝 파이프라인을 성공적으로 개발하였다.

**달성 목표:**

1. **Object Detection (YOLOv13s):** 15개 클래스에 대해 mAP@50 98.44% 달성
2. **Pose Estimation (ViTPose-Base):** 24개 키포인트에 대해 AP 98.46% 달성
3. **Action Classification (LSTM):** 13개 동작에 대해 Accuracy 91.87%, F1-Score 0.92 달성

### 기대 효과

1. **실용적 측면:** 야구 중계 자동화 및 경기 분석 효율화에 기여 가능
2. **기술적 측면:** Object Detection, Pose Estimation, Temporal Modeling을 결합한 멀티모달 파이프라인 구축 경험 확보
3. **학술적 측면:** 스포츠 영상 분석 분야의 딥러닝 적용 사례 제시

### 한계점 및 향후 과제

| 한계점 | 향후 개선 방향 |
| --- | --- |
| YOLOv13 학습 미완료 | 100 epoch 전체 학습 완료 |
| 데이터 불균형 | 추가적인 클래스 균형화 기법 적용 |
| 가려짐 문제 | Multi-view 또는 Tracking 기법 도입 |
| 실시간 처리 | 모델 경량화 및 TensorRT 최적화 |
| 일반화 성능 | 다양한 구장, 조명 조건에서의 추가 검증 |

### 역할 분담

| 팀원 | 담당 업무 |
| --- | --- |
| **김현준** | 데이터 수집, 코드 구현, 모델 구현 |
| **도기훈** | 데이터 수집, 코드 구현, 모델 구현 |

---

GitHub: https://github.com/gimyeonjik/2025-AIX-DL-PROJECT

YouTube: https://youtu.be/zBr_GU1Of08
