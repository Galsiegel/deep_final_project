"""
Training engine for stock prediction models.

Handles training loop, validation, checkpointing, and TensorBoard logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Trainer:
    """
    Handles model training, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        task: str,
        run_dir: Path,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 15,
        best_metric: str = 'val_loss',
        best_metric_mode: str = 'min'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            task: 'regression' or 'classification'
            run_dir: Directory to save checkpoints and logs
            grad_clip_norm: Gradient clipping norm
            early_stopping_patience: Epochs to wait before early stopping
            best_metric: Metric to use for saving best model
            best_metric_mode: 'min' or 'max' for best metric
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task = task
        self.run_dir = Path(run_dir)
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode
        
        # Create directories
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf') if best_metric_mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (now returns 3 values: X, day_open, y)
            X, day_open, y = batch
            X, day_open, y = X.to(self.device), day_open.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X, day_open)
            
            # Calculate loss
            if self.task == 'regression':
                loss = self.criterion(outputs, y)
            else:  # classification
                # CrossEntropyLoss expects [batch_size, num_classes] and [batch_size]
                loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if self.task == 'regression':
                all_preds.extend(outputs.detach().cpu().numpy())
                all_targets.extend(y.detach().cpu().numpy())
            else:  # classification
                pred_classes = torch.argmax(outputs, dim=1)
                all_preds.extend(pred_classes.detach().cpu().numpy())
                all_targets.extend(y.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = {'loss': avg_loss}
        
        if self.task == 'regression':
            # Calculate directional accuracy
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()
            # Simple directional accuracy (assumes sequential samples)
            dir_acc = self._calculate_directional_accuracy(all_preds, all_targets)
            metrics['dir_acc'] = dir_acc
        else:  # classification
            accuracy = accuracy_score(all_targets, all_preds)
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
            
            for batch in pbar:
                # Unpack batch (now returns 3 values: X, day_open, y)
                X, day_open, y = batch
                X, day_open, y = X.to(self.device), day_open.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = self.model(X, day_open)
                
                # Calculate loss
                if self.task == 'regression':
                    loss = self.criterion(outputs, y)
                else:  # classification
                    # CrossEntropyLoss expects [batch_size, num_classes] and [batch_size]
                    loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                
                if self.task == 'regression':
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                else:  # classification
                    pred_classes = torch.argmax(outputs, dim=1)
                    all_preds.extend(pred_classes.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = {'loss': avg_loss}
        
        if self.task == 'regression':
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()
            
            # MSE, MAE
            mse = np.mean((all_preds - all_targets) ** 2)
            mae = np.mean(np.abs(all_preds - all_targets))
            rmse = np.sqrt(mse)
            
            # Directional accuracy
            dir_acc = self._calculate_directional_accuracy(all_preds, all_targets)
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'directional_accuracy': dir_acc
            })
        else:  # classification
            accuracy = accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return metrics
    
    def _calculate_directional_accuracy(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate directional accuracy (simple version).
        Compares if prediction moved in same direction as target.
        """
        if len(preds) < 2:
            return 0.0
        
        # Calculate changes
        pred_changes = np.diff(preds)
        target_changes = np.diff(targets)
        
        # Check if signs match
        correct = np.sign(pred_changes) == np.sign(target_changes)
        
        return np.mean(correct)
    
    def save_checkpoint(self, filepath: Path, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['metrics']
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Task: {self.task}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
            
            # Log to TensorBoard
            self._log_metrics(train_metrics, val_metrics)
            
            # Save training history
            epoch_history = {
                'epoch': self.current_epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # Print epoch summary
            self._print_epoch_summary(train_metrics, val_metrics)
            
            # Save checkpoints
            self._save_checkpoints(val_metrics)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics):
                print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                break
        
        print("\nTraining complete!")
        print(f"Best val_loss: {self.best_val_loss:.4f}")
        print(f"Best {self.best_metric}: {self.best_val_metric:.4f}")
        
        self.writer.close()
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard."""
        # Log losses
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, self.current_epoch)
        
        # Log task-specific metrics
        if self.task == 'regression':
            if 'dir_acc' in train_metrics:
                self.writer.add_scalar('Train/DirectionalAccuracy', train_metrics['dir_acc'], self.current_epoch)
            if 'directional_accuracy' in val_metrics:
                self.writer.add_scalar('Val/DirectionalAccuracy', val_metrics['directional_accuracy'], self.current_epoch)
            if 'mae' in val_metrics:
                self.writer.add_scalar('Val/MAE', val_metrics['mae'], self.current_epoch)
        else:  # classification
            if 'accuracy' in train_metrics:
                self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], self.current_epoch)
            if 'accuracy' in val_metrics:
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], self.current_epoch)
            if 'f1' in val_metrics:
                self.writer.add_scalar('Val/F1', val_metrics['f1'], self.current_epoch)
        
        # Log learning rate
        self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
    
    def _print_epoch_summary(self, train_metrics: Dict, val_metrics: Dict):
        """Print epoch summary."""
        summary = f"\nEpoch {self.current_epoch}:"
        summary += f"\n  Train Loss: {train_metrics['loss']:.4f}"
        summary += f"\n  Val Loss: {val_metrics['loss']:.4f}"
        
        if self.task == 'regression':
            if 'directional_accuracy' in val_metrics:
                summary += f"\n  Val Dir Acc: {val_metrics['directional_accuracy']:.4f}"
            if 'mae' in val_metrics:
                summary += f"\n  Val MAE: {val_metrics['mae']:.4f}"
        else:
            if 'accuracy' in val_metrics:
                summary += f"\n  Val Accuracy: {val_metrics['accuracy']:.4f}"
            if 'f1' in val_metrics:
                summary += f"\n  Val F1: {val_metrics['f1']:.4f}"
        
        summary += f"\n  LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        print(summary)
    
    def _save_checkpoints(self, val_metrics: Dict):
        """Save model checkpoints."""
        # Always save last checkpoint
        self.save_checkpoint(
            self.checkpoint_dir / 'last.pth',
            val_metrics
        )
        
        # Save best val_loss checkpoint
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(
                self.checkpoint_dir / 'best_val_loss.pth',
                val_metrics
            )
            print("  [OK] Saved best_val_loss checkpoint")
        
        # Save best metric checkpoint
        metric_value = val_metrics.get(self.best_metric.replace('val_', ''), val_metrics['loss'])
        
        is_better = False
        if self.best_metric_mode == 'max':
            is_better = metric_value > self.best_val_metric
        else:
            is_better = metric_value < self.best_val_metric
        
        if is_better:
            self.best_val_metric = metric_value
            self.save_checkpoint(
                self.checkpoint_dir / 'best_val_metric.pth',
                val_metrics
            )
            print(f"  [OK] Saved best_{self.best_metric} checkpoint")
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check if early stopping should be triggered."""
        if val_metrics['loss'] < self.best_val_loss:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.early_stopping_patience:
            return True
        
        return False
    


if __name__ == "__main__":
    print("Trainer module ready")

