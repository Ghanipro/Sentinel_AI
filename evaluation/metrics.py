import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class SecurityMetrics:
    """Comprehensive security performance metrics for research evaluation"""
    
    def __init__(self):
        self.episode_metrics = []
        self.training_history = []
    
    def record_episode(self, episode_data: Dict[str, Any]):
        """Record metrics for a single episode"""
        self.episode_metrics.append(episode_data)
    
    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate performance metrics"""
        if not self.episode_metrics:
            return {}
        
        df = pd.DataFrame(self.episode_metrics)
        
        metrics = {
            # Effectiveness metrics
            'mean_defense_success_rate': df['successful_containments'].mean() / max(df['total_attacks'].mean(), 1),
            'mean_critical_breaches': df['critical_breaches'].mean(),
            'mean_false_positives': df['false_positives'].mean(),
            
            # Efficiency metrics
            'mean_detection_time': df['mean_detection_time'].mean(),
            'mean_episode_reward': df['episode_reward'].mean(),
            'mean_episode_length': df['episode_length'].mean(),
            
            # Security metrics
            'mean_assets_protected': df['assets_protected'].mean(),
            'mean_attack_containment_time': df.get('attack_containment_time', df['episode_length']).mean(),
            
            # Stability metrics
            'reward_std': df['episode_reward'].std(),
            'training_stability': df['episode_reward'].rolling(10, min_periods=1).std().mean()
        }
        
        return metrics
    
    def plot_training_progress(self, save_path: str = None):
        """Create comprehensive training progress plots"""
        if not self.episode_metrics:
            print("⚠️ No metrics data available")
            return
        
        df = pd.DataFrame(self.episode_metrics)
        episodes = range(len(df))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Reward progression
        ax1.plot(episodes, df['episode_reward'], alpha=0.6, label='Episode Reward')
        if len(df) > 10:
            reward_ma = df['episode_reward'].rolling(window=10, min_periods=1).mean()
            ax1.plot(episodes, reward_ma, 'r-', linewidth=2, label='Moving Avg (10)')
        ax1.set_title('Training Reward Progression')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Defense effectiveness
        ax2.plot(episodes, df['successful_containments'], 'g-', alpha=0.7, label='Successful Containments')
        ax2.plot(episodes, df['false_positives'], 'r-', alpha=0.7, label='False Positives')
        ax2.set_title('Defense Performance')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Security state
        ax3.plot(episodes, df['critical_breaches'], 'r-', label='Critical Breaches')
        ax3.plot(episodes, df['assets_protected'], 'b-', label='Assets Protected')
        ax3.set_title('Network Security State')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance distribution
        if len(df) >= 10:
            recent_rewards = df['episode_reward'][-min(20, len(df)):]
            ax4.hist(recent_rewards, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax4.axvline(recent_rewards.mean(), color='red', linestyle='--', label=f'Mean: {recent_rewards.mean():.2f}')
            ax4.set_title('Recent Reward Distribution')
            ax4.set_xlabel('Reward')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Training progress plot saved to {save_path}")
        
        plt.show()
    
    def plot_comparative_analysis(self, baseline_metrics, ppo_metrics, save_path: str = None):
        """Create comparative analysis between baseline and PPO"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics_to_compare = [
            ('mean_defense_success_rate', 'Defense Success Rate'),
            ('mean_false_positives', 'False Positives'),
            ('mean_critical_breaches', 'Critical Breaches'),
            ('mean_detection_time', 'Detection Time'),
            ('mean_episode_reward', 'Episode Reward'),
            ('training_stability', 'Training Stability')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_compare):
            ax = axes[idx // 3, idx % 3]
            
            baseline_val = baseline_metrics.get(metric, 0)
            ppo_val = ppo_metrics.get(metric, 0)
            
            values = [baseline_val, ppo_val]
            labels = ['Baseline', 'PPO Agent']
            colors = ['red', 'green']
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparative analysis saved to {save_path}")
        
        plt.show()
    
    def generate_research_metrics(self) -> Dict[str, Any]:
        """Generate metrics suitable for research paper"""
        aggregate_metrics = self.compute_aggregate_metrics()
        
        research_metrics = {
            'performance_metrics': {
                'defense_success_rate': aggregate_metrics.get('mean_defense_success_rate', 0),
                'false_positive_rate': aggregate_metrics.get('mean_false_positives', 0),
                'critical_breach_prevention': 1 - min(aggregate_metrics.get('mean_critical_breaches', 0), 1),
                'mean_detection_time': aggregate_metrics.get('mean_detection_time', 0)
            },
            'efficiency_metrics': {
                'mean_reward': aggregate_metrics.get('mean_episode_reward', 0),
                'training_stability': aggregate_metrics.get('training_stability', 0),
                'episode_length': aggregate_metrics.get('mean_episode_length', 0)
            },
            'learning_characteristics': {
                'convergence_speed': self._estimate_convergence_speed(),
                'performance_consistency': aggregate_metrics.get('reward_std', 0)
            }
        }
        
        return research_metrics
    
    def _estimate_convergence_speed(self) -> float:
        """Estimate how quickly the agent converged"""
        if len(self.episode_metrics) < 20:
            return 0.0
        
        rewards = [m['episode_reward'] for m in self.episode_metrics]
        target_performance = max(rewards) * 0.8
        
        for i in range(10, len(rewards)):
            if np.mean(rewards[i-5:i]) >= target_performance:
                return i / len(rewards)
        
        return 1.0