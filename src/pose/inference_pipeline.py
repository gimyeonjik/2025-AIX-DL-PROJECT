#!/usr/bin/env python

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mmpose.apis import inference_topdown, init_model
from ultralytics import YOLO
from src.action.model import BaseballActionLSTM
from src.action.config import get_config

KEYPOINT_NAMES = [
    'head', 'eye_right', 'eye_left', 'neck', 'chest',
    'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
    'right_wrist', 'left_wrist', 'right_fingertips', 'left_fingertips',
    'waist', 'right_hip', 'left_hip', 'right_knee', 'left_knee',
    'right_ankle', 'left_ankle', 'right_tiptoe', 'left_tiptoe',
    'right_heel', 'left_heel'
]

SKELETON = [
    (0, 3), (1, 2), (0, 1), (0, 2), (3, 4), (4, 5), (4, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
    (4, 13), (13, 14), (13, 15), (14, 16), (15, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (18, 22), (19, 23),
]

ACTION_NAMES = [
    'catch_throw', 'foul_fly', 'fly_ground', 'foul_tip', 'hit_by_pitch',
    'home_run', 'swing', 'passed_ball', 'pick_off', 'pitch',
    'pitch_setup', 'play_up', 'runner_run'
]

ACTION_NAMES_KR = [
    '포수 송구', '파울 플라이', '플라이/땅볼 아웃', '파울 팁', '몸에 맞는 공',
    '홈런 타격', '스윙', '패스트볼', '견제', '투구',
    '투구 준비', '플레이 업', '주자 달리기'
]


class ModelLoader:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.yolo_model = None
        self.pose_model = None
        self.lstm_model = None

    def load_yolo(self, model_path='yolov8n.pt'):
        print(f"YOLO 모델 로딩 중: {model_path}")
        self.yolo_model = YOLO(model_path)
        print("YOLO 모델 로딩 완료!")
        return self.yolo_model

    def load_vitpose(self, config_path, checkpoint_path):
        print(f"ViTPose 모델 로딩 중: {checkpoint_path}")
        self.pose_model = init_model(config_path, checkpoint_path, device=self.device)
        print("ViTPose 모델 로딩 완료!")
        return self.pose_model

    def load_lstm(self, checkpoint_path):
        print(f"LSTM 모델 로딩 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get('config', {})
        self.lstm_model = BaseballActionLSTM(
            input_dim=model_config.get('input_dim', 72),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            num_classes=model_config.get('num_classes', 13),
            dropout=model_config.get('dropout', 0.3),
            bidirectional=model_config.get('bidirectional', True),
            use_attention=model_config.get('use_attention', True)
        )
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model = self.lstm_model.to(self.device)
        self.lstm_model.eval()
        print(f"LSTM 모델 로딩 완료! (Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
        return self.lstm_model


class ActionClassificationPipeline:
    def __init__(self, yolo_model, pose_model, lstm_model, device='cuda', seq_len=64, conf_threshold=0.3, kpt_threshold=0.3):
        self.yolo_model = yolo_model
        self.pose_model = pose_model
        self.lstm_model = lstm_model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.seq_len = seq_len
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.keypoint_buffer = deque(maxlen=seq_len)
        self.bbox_buffer = deque(maxlen=seq_len)
        self.current_action = None
        self.current_confidence = 0.0

    def detect_persons(self, frame):
        results = self.yolo_model(frame, verbose=False)
        bboxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf >= self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    bboxes.append([x1, y1, w, h])
        return np.array(bboxes, dtype=np.float32) if bboxes else None

    def estimate_pose(self, frame, bboxes, temp_dir='/tmp'):
        if bboxes is None or len(bboxes) == 0:
            return None
        temp_path = os.path.join(temp_dir, 'temp_frame.jpg')
        cv2.imwrite(temp_path, frame)
        results = inference_topdown(self.pose_model, temp_path, bboxes=bboxes, bbox_format='xywh')
        os.remove(temp_path)
        return results

    def extract_keypoints(self, pose_results, bbox):
        if pose_results is None or len(pose_results) == 0:
            return None
        result = pose_results[0]
        keypoints = result.pred_instances.keypoints[0]
        scores = result.pred_instances.keypoint_scores[0]
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        normalized = np.zeros((24, 3), dtype=np.float32)
        normalized[:, 0] = (keypoints[:, 0] - cx) / max(w, 1)
        normalized[:, 1] = (keypoints[:, 1] - cy) / max(h, 1)
        normalized[:, 2] = scores
        return normalized

    def classify_action(self):
        if len(self.keypoint_buffer) < 16:
            return None, 0.0
        keypoints = list(self.keypoint_buffer)
        T = len(keypoints)
        if T < self.seq_len:
            padding = [np.zeros((24, 3), dtype=np.float32)] * (self.seq_len - T)
            keypoints = keypoints + padding
            mask = [1.0] * T + [0.0] * (self.seq_len - T)
        else:
            indices = np.linspace(0, T-1, self.seq_len, dtype=int)
            keypoints = [keypoints[i] for i in indices]
            mask = [1.0] * self.seq_len
        kps = np.stack(keypoints, axis=0)
        kps_flat = kps.reshape(self.seq_len, -1)
        x = torch.FloatTensor(kps_flat).unsqueeze(0).to(self.device)
        m = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.lstm_model(x, m)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        self.current_action = pred_idx
        self.current_confidence = confidence
        return pred_idx, confidence

    def process_frame(self, frame):
        bboxes = self.detect_persons(frame)
        if bboxes is None or len(bboxes) == 0:
            return None, None, None, None
        pose_results = self.estimate_pose(frame, bboxes)
        if pose_results is None:
            return bboxes, None, None, None
        keypoints = self.extract_keypoints(pose_results, bboxes[0])
        if keypoints is not None:
            self.keypoint_buffer.append(keypoints)
            self.bbox_buffer.append(bboxes[0])
        action_idx, confidence = self.classify_action()
        return bboxes, pose_results, action_idx, confidence

    def draw_results(self, frame, bboxes, pose_results, action_idx, confidence):
        frame_draw = frame.copy()
        if bboxes is None:
            return frame_draw
        for i, bbox in enumerate(bboxes):
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame_draw, (x, y), (x+int(w), y+int(h)), (255, 0, 0), 2)
        if pose_results is not None:
            for result in pose_results:
                keypoints = result.pred_instances.keypoints[0]
                scores = result.pred_instances.keypoint_scores[0]
                for (i, j) in SKELETON:
                    if scores[i] > self.kpt_threshold and scores[j] > self.kpt_threshold:
                        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(frame_draw, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
                    if score > self.kpt_threshold:
                        x, y = int(kpt[0]), int(kpt[1])
                        color = (0, 128, 255) if 'right' in KEYPOINT_NAMES[idx] else (0, 255, 0)
                        cv2.circle(frame_draw, (x, y), 5, color, -1, cv2.LINE_AA)
        if action_idx is not None and confidence > 0.3:
            action_en = ACTION_NAMES[action_idx]
            action_kr = ACTION_NAMES_KR[action_idx]
            label = f"{action_kr} ({action_en}): {confidence:.1%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame_draw, (10, 10), (20 + tw, 50 + th), (0, 0, 0), -1)
            cv2.putText(frame_draw, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        buffer_text = f"Buffer: {len(self.keypoint_buffer)}/{self.seq_len}"
        cv2.putText(frame_draw, buffer_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame_draw

    def reset_buffer(self):
        self.keypoint_buffer.clear()
        self.bbox_buffer.clear()
        self.current_action = None
        self.current_confidence = 0.0


def run_video_inference(video_path, output_path, pipeline, save_json=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return None
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{'='*60}")
    print(f"비디오 정보")
    print(f"{'='*60}")
    print(f"경로: {video_path}")
    print(f"해상도: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"총 프레임: {total_frames}")
    print(f"{'='*60}\n")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    results_data = {
        'video_path': video_path,
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'predictions': []
    }
    pipeline.reset_buffer()
    pbar = tqdm(total=total_frames, desc="추론 중")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, pose_results, action_idx, confidence = pipeline.process_frame(frame)
        frame_draw = pipeline.draw_results(frame, bboxes, pose_results, action_idx, confidence)
        out.write(frame_draw)
        if save_json and action_idx is not None:
            results_data['predictions'].append({
                'frame_idx': frame_idx,
                'action_idx': action_idx,
                'action_name': ACTION_NAMES[action_idx],
                'action_name_kr': ACTION_NAMES_KR[action_idx],
                'confidence': float(confidence)
            })
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    print(f"\n출력 비디오 저장 완료: {output_path}")
    if save_json:
        json_path = output_path.rsplit('.', 1)[0] + '_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"결과 JSON 저장 완료: {json_path}")
    return results_data


def main():
    parser = argparse.ArgumentParser(description='YOLO + ViTPose + LSTM 통합 추론 파이프라인')
    parser.add_argument('--video', type=str, required=True, help='입력 비디오 경로')
    parser.add_argument('--output', type=str, default=None, help='출력 비디오 경로')
    parser.add_argument('--yolo_model', type=str, default=str(PROJECT_ROOT / 'weights' / 'yolov13s.pt'), help='YOLO 모델 경로')
    parser.add_argument('--vitpose_config', type=str, default=str(PROJECT_ROOT / 'configs' / 'vitpose' / 'baseball_vitpose_base.py'), help='ViTPose config 경로')
    parser.add_argument('--vitpose_checkpoint', type=str, default=str(PROJECT_ROOT / 'weights' / 'vitpose_baseball.pth'), help='ViTPose checkpoint 경로')
    parser.add_argument('--lstm_checkpoint', type=str, default=str(PROJECT_ROOT / 'weights' / 'lstm_best_model.pth'), help='LSTM checkpoint 경로')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='YOLO 신뢰도 임계값')
    parser.add_argument('--kpt_threshold', type=float, default=0.3, help='Keypoint 신뢰도 임계값')
    parser.add_argument('--seq_len', type=int, default=64, help='LSTM 시퀀스 길이')
    parser.add_argument('--save_json', action='store_true', help='결과 JSON 저장')
    parser.add_argument('--device', type=str, default='cuda', help='추론 장치')
    args = parser.parse_args()
    if args.output is None:
        base = os.path.splitext(args.video)[0]
        args.output = f"{base}_result.mp4"
    os.chdir(PROJECT_ROOT)
    print("\n" + "="*60)
    print("YOLO + ViTPose + LSTM 통합 추론 파이프라인")
    print("="*60)
    loader = ModelLoader(device=args.device)
    yolo_model = loader.load_yolo(args.yolo_model)
    pose_model = loader.load_vitpose(args.vitpose_config, args.vitpose_checkpoint)
    lstm_model = loader.load_lstm(args.lstm_checkpoint)
    pipeline = ActionClassificationPipeline(
        yolo_model=yolo_model,
        pose_model=pose_model,
        lstm_model=lstm_model,
        device=args.device,
        seq_len=args.seq_len,
        conf_threshold=args.conf_threshold,
        kpt_threshold=args.kpt_threshold
    )
    run_video_inference(
        video_path=args.video,
        output_path=args.output,
        pipeline=pipeline,
        save_json=args.save_json
    )
    print("\n" + "="*60)
    print("추론 완료!")
    print("="*60)


if __name__ == '__main__':
    main()
