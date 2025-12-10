import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config, get_config
from dataset import BaseballActionDataset, create_dataloaders
from model import BaseballActionLSTM, create_model, count_parameters


class ActionClassificationTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, class_weights: Optional[torch.Tensor] = None, config: Optional[Config] = None):
        self.config = config or get_config()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        os.makedirs(self.config.save_dir, exist_ok=True)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.config.label_smoothing)
        self.optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
        self.class_names = list(self.config.CODE_TO_ACTION.values())

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch in pbar:
            keypoints = batch['keypoints'].to(self.device)
            labels = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(keypoints, mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            keypoints = batch['keypoints'].to(self.device)
            labels = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)
            logits = self.model(keypoints, mask)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True, zero_division=0)
        return avg_loss, accuracy, report

    def train(self, num_epochs: Optional[int] = None, early_stopping_patience: Optional[int] = None) -> Tuple[float, float]:
        num_epochs = num_epochs or self.config.num_epochs
        early_stopping_patience = early_stopping_patience or self.config.early_stopping_patience
        print(f"\n{'='*60}")
        print(f"Training Start")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, report = self.validate()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            val_f1 = report['macro avg']['f1-score']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            print(f'\nEpoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Macro F1: {val_f1:.4f}, LR: {current_lr:.6f}')
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc, val_f1, report, is_best=True)
                print(f'  [*] New best model saved! (F1: {val_f1:.4f})')
            else:
                self.patience_counter += 1
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_acc, val_f1, report, is_best=False)
            if self.patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break
        self._save_history()
        self._plot_training_curves()
        print(f"\n{'='*60}")
        print(f"Training Completed")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        return self.best_val_acc, self.best_val_f1

    def _save_checkpoint(self, epoch: int, val_acc: float, val_f1: float, report: Dict, is_best: bool):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'report': report,
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_classes': self.config.num_classes,
                'dropout': self.config.dropout,
                'bidirectional': self.config.bidirectional,
                'use_attention': self.config.use_attention
            }
        }
        if is_best:
            path = os.path.join(self.config.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.config.save_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(checkpoint, path)

    def _save_history(self):
        path = os.path.join(self.config.save_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy (%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[1, 0].set_title('Macro F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 1].plot(self.history['lr'], label='LR', color='orange')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        plt.tight_layout()
        path = os.path.join(self.config.save_dir, 'training_curves.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Training curves saved to: {path}")


def evaluate_model(model: nn.Module, val_loader: DataLoader, config: Optional[Config] = None, save_dir: Optional[str] = None):
    config = config or get_config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    save_dir = save_dir or config.save_dir
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            keypoints = batch['keypoints'].to(device)
            labels = batch['label'].to(device)
            mask = batch['mask'].to(device)
            logits = model(keypoints, mask)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    class_names = list(config.CODE_TO_ACTION.values())
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {path}")
    return all_preds, all_labels


def load_checkpoint(checkpoint_path: str, config: Optional[Config] = None) -> Tuple[nn.Module, Dict]:
    config = config or get_config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', {})
    model = BaseballActionLSTM(
        input_dim=model_config.get('input_dim', config.input_dim),
        hidden_dim=model_config.get('hidden_dim', config.hidden_dim),
        num_layers=model_config.get('num_layers', config.num_layers),
        num_classes=model_config.get('num_classes', config.num_classes),
        dropout=model_config.get('dropout', config.dropout),
        bidirectional=model_config.get('bidirectional', config.bidirectional),
        use_attention=model_config.get('use_attention', config.use_attention)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, checkpoint
