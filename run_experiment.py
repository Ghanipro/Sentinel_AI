import os
import sys
import yaml
import numpy as np
from typing import Dict, Any
import time
import json

import os
import sys
import yaml
import numpy as np
from typing import Dict, Any
import time
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from environments.corporate_network import CorporateNetworkEnv
from agents.ppo_agent import IntrusionResponsePPO
from agents.baseline_defender import BaselineDefender, HeuristicDefender
from evaluation.metrics import SecurityMetrics

# Import the callback - ADD THIS LINE
from training.callbacks import TrainingCallback

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from environments.corporate_network import CorporateNetworkEnv
from agents.ppo_agent import IntrusionResponsePPO
from agents.baseline_defender import BaselineDefender, HeuristicDefender
from evaluation.metrics import SecurityMetrics

class IntrusionResponseExperiment:
    """Main experiment runner for intrusion response research"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialize environment
        self.env = CorporateNetworkEnv()
        
        print("🔬 Intrusion Response Experiment Initialized!")
        print(f"   Network Nodes: {len(self.env.topology.nodes)}")
        print(f"   Action Space: {self.env.action_space.n}")
        print(f"   Observation Space: {self.env.observation_space.shape}")
    
    def run_ppo_training(self, total_timesteps: int = 50000):
        """Run PPO training experiment"""
        print(f"\n🎯 Starting PPO Training Experiment")
        print(f"   Timesteps: {total_timesteps}")
        
        # Initialize PPO agent
        ppo_agent = IntrusionResponsePPO(self.env, self.config['ppo'])
        
        # Create evaluation environment and callback
        eval_env = CorporateNetworkEnv()
        callback = TrainingCallback(
            eval_env=eval_env,
            save_path="models/ppo_intrusion_response",
            eval_freq=10000,  # Evaluate every 10,000 steps
            verbose=1
        )
        
        # Training
        start_time = time.time()
        ppo_agent.train(total_timesteps, callback=callback)
        training_time = time.time() - start_time
        
        # Save trained model
        ppo_agent.save("models/ppo_intrusion_response_final")
        
        eval_env.close()
        
        print(f"✅ PPO Training completed in {training_time/60:.2f} minutes")
        
        return ppo_agent
    
    def evaluate_agent(self, agent, num_episodes: int = 50, agent_type: str = "ppo"):
        """Evaluate agent performance"""
        print(f"\n📊 Evaluating {agent_type.upper()} Agent over {num_episodes} episodes...")
        
        episode_metrics = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            detection_times = []
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                if agent_type == "ppo":
                    action = agent.predict(obs, deterministic=True)
                else:  # baseline
                    action = agent.act(obs)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                # Track detection performance
                if info.get('successful_containments', 0) > 0 and not detection_times:
                    detection_times.append(episode_steps)
            
            # Record episode metrics
            episode_data = {
                'episode_reward': episode_reward,
                'episode_length': episode_steps,
                'successful_containments': info.get('successful_containments', 0),
                'false_positives': info.get('false_positives', 0),
                'critical_breaches': info.get('critical_assets_at_risk', 0),
                'total_attacks': len(info.get('attack_progression', [])),
                'assets_protected': len([n for n in self.env.topology.nodes 
                                       if not self.env.topology.node_attributes[n]['compromised']]),
                'mean_detection_time': np.mean(detection_times) if detection_times else episode_steps
            }
            episode_metrics.append(episode_data)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([m['episode_reward'] for m in episode_metrics[-10:]])
                print(f"   Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}")
        
        return episode_metrics
    
    def run_baseline_comparison(self, num_episodes: int = 50):
        """Run baseline agents for comparison"""
        print(f"\n🔄 Running Baseline Comparisons...")
        
        baseline_results = {}
        
        # Test different baseline strategies
        strategies = ['random', 'conservative', 'aggressive', 'targeted']
        
        for strategy in strategies:
            print(f"\n   Testing {strategy} strategy...")
            baseline_agent = BaselineDefender(self.env.action_space, strategy=strategy)
            metrics = self.evaluate_agent(baseline_agent, num_episodes, f"baseline_{strategy}")
            baseline_results[strategy] = metrics
            
            # Print strategy performance
            avg_reward = np.mean([m['episode_reward'] for m in metrics])
            success_rate = np.mean([1 if m['critical_breaches'] <= 1 else 0 for m in metrics])
            print(f"      Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}")
        
        return baseline_results
    
    def run_heuristic_comparison(self, num_episodes: int = 50):
        """Run heuristic defender for comparison"""
        print(f"\n🎯 Running Heuristic Defender Comparison...")
        
        heuristic_agent = HeuristicDefender(self.env.action_space, len(self.env.topology.nodes))
        metrics = self.evaluate_agent(heuristic_agent, num_episodes, "heuristic")
        
        avg_reward = np.mean([m['episode_reward'] for m in metrics])
        success_rate = np.mean([1 if m['critical_breaches'] <= 1 else 0 for m in metrics])
        print(f"   Heuristic Defender - Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}")
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save experiment results to file"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        with open(f"results/{filename}", 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy_types)
        
        print(f"💾 Results saved to results/{filename}")
    
    def run_comprehensive_experiment(self):
        """Run complete experiment suite"""
        print("=" * 60)
        print("🔬 AUTONOMOUS INTRUSION RESPONSE RESEARCH EXPERIMENT")
        print("=" * 60)
        
        all_results = {}
        
        # 1. Run baseline comparisons
        print("\n1. BASELINE STRATEGY COMPARISON")
        baseline_results = self.run_baseline_comparison(num_episodes=30)
        all_results['baseline'] = baseline_results
        
        # 2. Run heuristic comparison
        print("\n2. HEURISTIC DEFENDER EVALUATION")
        heuristic_results = self.run_heuristic_comparison(num_episodes=30)
        all_results['heuristic'] = heuristic_results
        
        # 3. Train PPO agent
        print("\n3. PPO AGENT TRAINING")
        ppo_agent = self.run_ppo_training(total_timesteps=50000)
        
        # 4. Evaluate PPO agent
        print("\n4. PPO AGENT EVALUATION")
        ppo_results = self.evaluate_agent(ppo_agent, num_episodes=50, agent_type="ppo")
        all_results['ppo'] = ppo_results
        
        # 5. Save all results
        self.save_results(all_results, "comprehensive_experiment_results.json")
        
        # 6. Print final comparison
        self._print_final_comparison(all_results)
        
        print("\n✅ Experiment completed successfully!")
        print("📊 Check 'results/' directory for detailed metrics")
        print("💾 Check 'models/' directory for trained models")
    
    def _print_final_comparison(self, results: Dict[str, Any]):
        """Print comprehensive experiment summary"""
        print("\n" + "=" * 60)
        print("📈 FINAL EXPERIMENT COMPARISON")
        print("=" * 60)
        
        # Calculate metrics for each approach
        comparison_data = []
        
        # Baseline strategies
        for strategy, metrics in results.get('baseline', {}).items():
            rewards = [m['episode_reward'] for m in metrics]
            success_rates = [1 if m['critical_breaches'] <= 1 else 0 for m in metrics]
            
            comparison_data.append({
                'approach': f"Baseline ({strategy})",
                'mean_reward': np.mean(rewards),
                'success_rate': np.mean(success_rates),
                'avg_episode_length': np.mean([m['episode_length'] for m in metrics])
            })
        
        # Heuristic
        heuristic_metrics = results.get('heuristic', [])
        if heuristic_metrics:
            rewards = [m['episode_reward'] for m in heuristic_metrics]
            success_rates = [1 if m['critical_breaches'] <= 1 else 0 for m in heuristic_metrics]
            
            comparison_data.append({
                'approach': "Heuristic Defender",
                'mean_reward': np.mean(rewards),
                'success_rate': np.mean(success_rates),
                'avg_episode_length': np.mean([m['episode_length'] for m in heuristic_metrics])
            })
        
        # PPO
        ppo_metrics = results.get('ppo', [])
        if ppo_metrics:
            rewards = [m['episode_reward'] for m in ppo_metrics]
            success_rates = [1 if m['critical_breaches'] <= 1 else 0 for m in ppo_metrics]
            
            comparison_data.append({
                'approach': "PPO Agent",
                'mean_reward': np.mean(rewards),
                'success_rate': np.mean(success_rates),
                'avg_episode_length': np.mean([m['episode_length'] for m in ppo_metrics])
            })
        
        # Print comparison table
        print(f"\n{'Approach':<25} {'Mean Reward':<12} {'Success Rate':<12} {'Avg Steps':<10}")
        print("-" * 65)
        
        for data in comparison_data:
            print(f"{data['approach']:<25} {data['mean_reward']:<12.2f} "
                  f"{data['success_rate']:<12.2%} {data['avg_episode_length']:<10.1f}")
        
        # Find best performer
        if comparison_data:
            best_by_reward = max(comparison_data, key=lambda x: x['mean_reward'])
            best_by_success = max(comparison_data, key=lambda x: x['success_rate'])
            
            print(f"\n🏆 Best by Reward: {best_by_reward['approach']} ({best_by_reward['mean_reward']:.2f})")
            print(f"🏆 Best by Success Rate: {best_by_success['approach']} ({best_by_success['success_rate']:.2%})")

def main():
    """Main execution function"""
    experiment = IntrusionResponseExperiment("configs/training_config.yaml")
    experiment.run_comprehensive_experiment()

if __name__ == "__main__":
    main()