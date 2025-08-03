"""
Advanced Training Monitor with Early Stopping for YOLO Chess Detection
"""
import os
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class TrainingMonitor:
    def __init__(self, project_dir="chess_gpu_training"):
        self.project_dir = project_dir
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50_95': []
        }
        self.best_metric = 0
        self.patience_counter = 0
        
    def should_stop_early(self, current_metric, patience=30, min_delta=0.001):
        """
        Check if training should stop early
        
        Args:
            current_metric: Current validation mAP@0.5
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        
        if current_metric > (self.best_metric + min_delta):
            self.best_metric = current_metric
            self.patience_counter = 0
            return False, f"✅ New best mAP@0.5: {current_metric:.4f}"
        else:
            self.patience_counter += 1
            remaining = patience - self.patience_counter
            
            if self.patience_counter >= patience:
                return True, f"🛑 Early stopping triggered! No improvement for {patience} epochs"
            else:
                return False, f"⏳ No improvement for {self.patience_counter} epochs. {remaining} remaining."
    
    def plot_training_progress(self, save_path="training_progress.png"):
        """Plot training metrics"""
        
        if len(self.metrics_history['epoch']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Chess Detection Training Progress', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(self.metrics_history['epoch'], self.metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(self.metrics_history['epoch'], self.metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP plot
        axes[0, 1].plot(self.metrics_history['epoch'], self.metrics_history['mAP50'], 'g-', label='mAP@0.5')
        axes[0, 1].plot(self.metrics_history['epoch'], self.metrics_history['mAP50_95'], 'purple', label='mAP@0.5:0.95')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision & Recall
        axes[1, 0].plot(self.metrics_history['epoch'], self.metrics_history['precision'], 'orange', label='Precision')
        axes[1, 0].plot(self.metrics_history['epoch'], self.metrics_history['recall'], 'cyan', label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate and other metrics
        axes[1, 1].text(0.1, 0.8, f"Best mAP@0.5: {self.best_metric:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Current Epoch: {self.metrics_history['epoch'][-1] if self.metrics_history['epoch'] else 0}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Patience Counter: {self.patience_counter}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Info')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Progress plot saved: {save_path}")

def monitor_training_live(project_dir="chess_gpu_training", refresh_interval=30):
    """
    Monitor training progress in real-time
    """
    monitor = TrainingMonitor(project_dir)
    
    print("🔍 Starting live training monitor...")
    print(f"📁 Monitoring: {project_dir}")
    print(f"🔄 Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Look for results files
            results_files = list(Path(project_dir).rglob("results.csv"))
            
            if results_files:
                latest_results = max(results_files, key=os.path.getctime)
                
                try:
                    df = pd.read_csv(latest_results)
                    if not df.empty:
                        latest_row = df.iloc[-1]
                        
                        current_epoch = int(latest_row['epoch'])
                        mAP50 = float(latest_row['metrics/mAP50(B)']) if 'metrics/mAP50(B)' in latest_row else 0
                        
                        # Check for early stopping
                        should_stop, message = monitor.should_stop_early(mAP50)
                        
                        print(f"Epoch {current_epoch}: {message}")
                        
                        if should_stop:
                            print("🛑 Recommending early stopping!")
                            break
                        
                        # Update metrics history
                        monitor.metrics_history['epoch'].append(current_epoch)
                        monitor.metrics_history['mAP50'].append(mAP50)
                        
                        # Generate progress plot
                        plot_path = f"{project_dir}/live_progress.png"
                        monitor.plot_training_progress(plot_path)
                        
                except Exception as e:
                    print(f"⚠️ Error reading results: {e}")
            else:
                print("⏳ Waiting for training to start...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")

def main():
    """Main monitoring function"""
    print("🔍 YOLO Training Monitor with Early Stopping")
    print("=" * 50)
    
    choice = input("Select mode:\n1. Live monitoring\n2. Analyze existing results\nChoice (1-2): ")
    
    if choice == "1":
        project_dir = input("Project directory (default: chess_gpu_training): ").strip()
        if not project_dir:
            project_dir = "chess_gpu_training"
        
        refresh_interval = input("Refresh interval in seconds (default: 30): ").strip()
        refresh_interval = int(refresh_interval) if refresh_interval.isdigit() else 30
        
        monitor_training_live(project_dir, refresh_interval)
        
    elif choice == "2":
        print("📊 Analyzing existing results...")
        monitor = TrainingMonitor()
        monitor.plot_training_progress("final_training_analysis.png")
        print("✅ Analysis complete!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()