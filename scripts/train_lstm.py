#!/usr/bin/env python

import os
import sys
import argparse
import random
import numpy as np
import torch

from config import Config, get_config
from dataset import create_dataloaders
from model import create_model, count_parameters
from train import ActionClassificationTrainer, evaluate_model, load_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Baseball Action Classification LSTM Training'
    )
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_weighted_sampler', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    if args.data_root:
        config.data_root = args.data_root
    if args.save_dir:
        config.save_dir = args.save_dir

    config.max_seq_len = args.max_seq_len
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.use_attention = not args.no_attention
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.early_stopping_patience = args.patience
    config.num_workers = args.num_workers
    config.seed = args.seed
    config.use_weighted_sampler = not args.no_weighted_sampler
    config.augment = not args.no_augment

    set_seed(config.seed)

    print("\n" + "="*60)
    print("Baseball Action Classification - LSTM")
    print("="*60)
    print(f"Data root: {config.data_root}")
    print(f"Save dir: {config.save_dir}")
    print(f"Max seq len: {config.max_seq_len}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Num layers: {config.num_layers}")
    print(f"Dropout: {config.dropout}")
    print(f"Use attention: {config.use_attention}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Num epochs: {config.num_epochs}")
    print(f"Weighted sampler: {config.use_weighted_sampler}")
    print(f"Data augmentation: {config.augment}")
    print("="*60 + "\n")

    if args.eval_only:
        if not args.checkpoint:
            checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
        else:
            checkpoint_path = args.checkpoint

        print(f"Loading checkpoint: {checkpoint_path}")
        model, checkpoint = load_checkpoint(checkpoint_path, config)

        print(f"Checkpoint info:")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")

        _, val_loader, _ = create_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            num_workers=config.num_workers,
            use_weighted_sampler=False,
            config=config
        )

        evaluate_model(model, val_loader, config, config.save_dir)
        return

    print("Creating data loaders...")
    train_loader, val_loader, class_weights = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        num_workers=config.num_workers,
        use_weighted_sampler=config.use_weighted_sampler,
        config=config
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Class weights: {class_weights}")

    print("\nCreating model...")
    model = create_model(config)
    print(f"Total parameters: {count_parameters(model):,}")

    trainer = ActionClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config
    )

    best_acc, best_f1 = trainer.train(
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience
    )

    print("\nRunning final evaluation...")
    checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
    model, _ = load_checkpoint(checkpoint_path, config)
    evaluate_model(model, val_loader, config, config.save_dir)


if __name__ == '__main__':
    main()
