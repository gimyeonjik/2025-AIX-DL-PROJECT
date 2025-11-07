# 🥎 Object Detection과 Pose Estimation을 활용한 야구 경기 상황 인식 모델 구현

### Memebers:
김현준, 에리카 인공지능학과, 2023054639, gizxmk@hanyang.ac.kr<br>
도기훈, 에리카 인공지능학과, 2021009407, support7417@hanyang.ac.kr

# I. Proposal (Option A)
### Motivation: Why are you doing this?
야구 경기 중계와 분석은 전통적으로 해설자와 분석가들이 실시간으로 경기 상황을 파악하고 설명하는 방식에 크게 의존해왔다.
하지만 이러한 과정은 많은 인력이 필요하며, 자동화를 통해 효율성을 크게 높일 수 있다.

### What do you want to see at the end?
본 프로젝트의 최종 목표는 야구 중계 영상을 처리하여 경기 상황을 높은 정확도로 자동 식별할 수 있는 완전한 파이프라인을 개발하는 것이다.
최종 결과물은 실제 중계 영상에서 다양한 경기 상황을 인식하는 시스템의 능력을 보여주는 예시 결과물이 될 것이다.

# II. Datasets
-Describing your dataset-
본 프로젝트는 AI Hub에서 제공하는 야구 경기 영상 데이터셋을 활용한다.
https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=488

이 데이터셋은 실제 프로야구 경기의 포괄적인 비디오 영상, 시퀀스 이미지와 그에 대응하는 annotation을 포함하고 있다.
데이터셋에는 Player의 pose, 행동에 대한 annotation이 시퀀스 별, 프레임 단위로 포함되어 있어, 본 프로젝트의 학습 데이터로 적합하다.

# III. Methodology
-Explaining your choice of alogrithms (methods)-
-Explaining features (if any)-
본 연구의 방법론은 중계 영상에서 야구 경기 상황을 인식하기 위해 함께 작동하는 세 가지 주요 단계로 구성된다.
첫 번째 단계는 비디오 프레임에서 선수를 탐지하고, 두 번째 단계는 그들의 포즈를 추정하며, 세 번째 단계는 포즈 시퀀스와 시각적 시퀀스 모두를 기반으로 상황을 분류하기 위해 시계열 모델링을 사용한다.

1단계: 선수 탐지를 위한 Object Detection
2단계: 1단계에서 선수가 detection 되면, 각 bounding box 내에서 pose estimation을 적용하여 keypoint 정보를 추출
3단계: Pose sequence와 Image sequence를 동시 처리하는 dual stream 시계열 모델을 통한 추론

# IV. Evaluation & Analysis
-Graphs, tables, any statistics (if any)-

# V. Related Work
-Tools, libraries, blogs, or any documentation that you have used to do this project.

# VI. Conclusion: Discussion
...

김현준 : 데이터 수집, 코드 구현, 모델 구현<br>
도기훈 : 데이터 수집, 코드 구현, 모델 구현
