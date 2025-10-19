import json
import numpy as np
import pandas as pd
from typing import Dict, Any

class ResultsAnalyzer:
    """Analyze and summarize experiment results for research paper"""
    
    def __init__(self, results_path: str = "results/comprehensive_experiment_results.json"):
        self.results_path = results_path
        self.results = self.load_results()
        self.analysis = {}
    
    def load_results(self) -> Dict[str, Any]:
        """Load experiment results"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_path}")
            return {}
    
    def analyze_performance(self):
        """Comprehensive performance analysis"""
        analysis = {}
        
        # Baseline strategies
        if 'baseline' in self.results:
            baseline_analysis = {}
            for strategy, metrics in self.results['baseline'].items():
                rewards = [m['episode_reward'] for m in metrics]
                success_rate = np.mean([1 if m['critical_breaches'] <= 1 else 0 for m in metrics])
                
                baseline_analysis[strategy] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'success_rate': success_rate,
                    'avg_episode_length': np.mean([m['episode_length'] for m in metrics]),
                    'avg_containments': np.mean([m['successful_containments'] for m in metrics]),
                    'avg_false_positives': np.mean([m['false_positives'] for m in metrics])
                }
            analysis['baseline'] = baseline_analysis
        
        # Heuristic defender
        if 'heuristic' in self.results:
            metrics = self.results['heuristic']
            rewards = [m['episode_reward'] for m in metrics]
            success_rate = np.mean([1 if m['critical_breaches'] <= 1 else 0 for m in metrics])
            
            analysis['heuristic'] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'success_rate': success_rate,
                'avg_episode_length': np.mean([m['episode_length'] for m in metrics]),
                'avg_containments': np.mean([m['successful_containments'] for m in metrics]),
                'avg_false_positives': np.mean([m['false_positives'] for m in metrics])
            }
        
        # PPO agent
        if 'ppo' in self.results:
            metrics = self.results['ppo']
            rewards = [m['episode_reward'] for m in metrics]
            success_rate = np.mean([1 if m['critical_breaches'] <= 1 else 0 for m in metrics])
            
            analysis['ppo'] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'success_rate': success_rate,
                'avg_episode_length': np.mean([m['episode_length'] for m in metrics]),
                'avg_containments': np.mean([m['successful_containments'] for m in metrics]),
                'avg_false_positives': np.mean([m['false_positives'] for m in metrics])
            }
        
        self.analysis = analysis
        return analysis
    
    def generate_research_summary(self):
        """Generate summary for research paper"""
        if not self.analysis:
            self.analyze_performance()
        
        summary = {
            'performance_comparison': {},
            'statistical_significance': {},
            'key_findings': []
        }
        
        # Performance comparison
        strategies = []
        if 'baseline' in self.analysis:
            for strategy, stats in self.analysis['baseline'].items():
                strategies.append(('Baseline_' + strategy, stats))
        
        if 'heuristic' in self.analysis:
            strategies.append(('Heuristic', self.analysis['heuristic']))
        
        if 'ppo' in self.analysis:
            strategies.append(('PPO', self.analysis['ppo']))
        
        # Find best performer
        if strategies:
            best_by_reward = max(strategies, key=lambda x: x[1]['mean_reward'])
            best_by_success = max(strategies, key=lambda x: x[1]['success_rate'])
            
            summary['best_performer_reward'] = {
                'strategy': best_by_reward[0],
                'mean_reward': best_by_reward[1]['mean_reward'],
                'improvement_over_worst': self._calculate_improvement(strategies, 'mean_reward')
            }
            
            summary['best_performer_success'] = {
                'strategy': best_by_success[0],
                'success_rate': best_by_success[1]['success_rate'],
                'improvement_over_worst': self._calculate_improvement(strategies, 'success_rate')
            }
        
        # Key findings
        if 'ppo' in self.analysis and 'baseline' in self.analysis:
            ppo_reward = self.analysis['ppo']['mean_reward']
            best_baseline = max([s[1]['mean_reward'] for s in strategies if 'Baseline' in s[0]])
            
            improvement = ((ppo_reward - best_baseline) / abs(best_baseline)) * 100
            
            summary['key_findings'].extend([
                f"PPO agent achieved {improvement:.1f}% higher average reward compared to the best baseline",
                f"Success rate of {self.analysis['ppo']['success_rate']:.1%} for mission completion",
                f"Average {self.analysis['ppo']['avg_containments']:.1f} successful containments per episode"
            ])
        
        return summary
    
    def _calculate_improvement(self, strategies, metric):
        """Calculate improvement of best over worst"""
        if not strategies:
            return 0
        
        best_value = max(s[1][metric] for s in strategies)
        worst_value = min(s[1][metric] for s in strategies)
        
        if worst_value == 0:
            return float('inf')
        
        return ((best_value - worst_value) / abs(worst_value)) * 100
    
    def print_detailed_report(self):
        """Print detailed analysis report"""
        if not self.analysis:
            self.analyze_performance()
        
        print("=" * 80)
        print("📊 AUTONOMOUS INTRUSION RESPONSE - EXPERIMENT ANALYSIS REPORT")
        print("=" * 80)
        
        # Performance Summary
        print("\n🎯 PERFORMANCE SUMMARY")
        print("-" * 40)
        
        all_strategies = []
        if 'baseline' in self.analysis:
            for strategy, stats in self.analysis['baseline'].items():
                all_strategies.append(('Baseline: ' + strategy.title(), stats))
        
        if 'heuristic' in self.analysis:
            all_strategies.append(('Heuristic Defender', self.analysis['heuristic']))
        
        if 'ppo' in self.analysis:
            all_strategies.append(('PPO Agent', self.analysis['ppo']))
        
        # Print performance table
        print(f"{'Strategy':<25} {'Avg Reward':<12} {'Success Rate':<12} {'Avg Steps':<10}")
        print("-" * 65)
        for name, stats in all_strategies:
            print(f"{name:<25} {stats['mean_reward']:<12.2f} "
                  f"{stats['success_rate']:<12.2%} {stats['avg_episode_length']:<10.1f}")
        
        # Statistical Analysis
        print("\n📈 STATISTICAL ANALYSIS")
        print("-" * 40)
        
        if len(all_strategies) >= 2:
            best_strategy = max(all_strategies, key=lambda x: x[1]['mean_reward'])
            worst_strategy = min(all_strategies, key=lambda x: x[1]['mean_reward'])
            
            improvement = ((best_strategy[1]['mean_reward'] - worst_strategy[1]['mean_reward']) 
                          / abs(worst_strategy[1]['mean_reward'])) * 100
            
            print(f"Best Strategy: {best_strategy[0]} ({best_strategy[1]['mean_reward']:.2f} avg reward)")
            print(f"Worst Strategy: {worst_strategy[0]} ({worst_strategy[1]['mean_reward']:.2f} avg reward)")
            print(f"Improvement: {improvement:.1f}%")
        
        # Defense Effectiveness
        print("\n🛡️ DEFENSE EFFECTIVENESS")
        print("-" * 40)
        
        if 'ppo' in self.analysis:
            ppo_stats = self.analysis['ppo']
            print(f"Successful Containments: {ppo_stats['avg_containments']:.2f} per episode")
            print(f"False Positives: {ppo_stats['avg_false_positives']:.2f} per episode")
            print(f"Containment Efficiency: {ppo_stats['avg_containments'] / max(ppo_stats['avg_containments'] + ppo_stats['avg_false_positives'], 1):.2%}")
        
        # Research Implications
        print("\n🎓 RESEARCH IMPLICATIONS")
        print("-" * 40)
        
        research_summary = self.generate_research_summary()
        for finding in research_summary.get('key_findings', []):
            print(f"• {finding}")
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete! Check visualization files for detailed graphs.")

def main():
    """Main analysis function"""
    analyzer = ResultsAnalyzer()
    analyzer.print_detailed_report()
    
    # Save analysis to file
    analysis = analyzer.analyze_performance()
    with open("results/detailed_analysis.json", 'w') as f:
        import json
        json.dump(analysis, f, indent=2)
    
    print("💾 Detailed analysis saved to results/detailed_analysis.json")

if __name__ == "__main__":
    main()