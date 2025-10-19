import numpy as np
import random
from gymnasium import spaces
from typing import Dict, Any

class BaselineDefender:
    """
    Baseline defense strategies for comparison with PPO agent
    Implements simple heuristic policies
    """
    
    def __init__(self, action_space: spaces.Space, strategy: str = "random"):
        self.action_space = action_space
        self.strategy = strategy
        self.action_history = []
        
        print(f"🔄 Initializing Baseline Defender with strategy: {strategy}")
    
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action based on chosen strategy"""
        if self.strategy == "random":
            return self._random_action()
        elif self.strategy == "conservative":
            return self._conservative_action(observation)
        elif self.strategy == "aggressive":
            return self._aggressive_action(observation)
        elif self.strategy == "targeted":
            return self._targeted_action(observation)
        else:
            return self._random_action()
    
    def _random_action(self) -> int:
        """Take completely random actions"""
        action = self.action_space.sample()
        self.action_history.append(("random", action))
        return action
    
    def _conservative_action(self, observation: np.ndarray) -> int:
        """Conservative strategy: mostly monitor, rarely take disruptive actions"""
        if random.random() < 0.7:  # 70% chance to monitor
            # Find monitor action (action_type=1 for monitor)
            num_nodes = (self.action_space.n - 3) // 8  # Calculate number of nodes
            if num_nodes > 0:
                node_idx = random.randint(0, num_nodes - 1)
                monitor_action = 3 + (1 * num_nodes) + node_idx  # monitor action
                if monitor_action < self.action_space.n:
                    self.action_history.append(("monitor", monitor_action))
                    return monitor_action
        
        # Fallback to random action
        return self._random_action()
    
    def _aggressive_action(self, observation: np.ndarray) -> int:
        """Aggressive strategy: frequently isolate and block"""
        if random.random() < 0.6:  # 60% chance for aggressive action
            num_nodes = (self.action_space.n - 3) // 8
            if num_nodes > 0:
                # Prefer isolate (action_type=2) and block (action_type=4)
                action_type = random.choice([2, 4])  # isolate or block
                node_idx = random.randint(0, num_nodes - 1)
                aggressive_action = 3 + (action_type * num_nodes) + node_idx
                if aggressive_action < self.action_space.n:
                    self.action_history.append(("aggressive", aggressive_action))
                    return aggressive_action
        
        return self._random_action()
    
    def _targeted_action(self, observation: np.ndarray) -> int:
        """Targeted strategy: focus on critical assets"""
        num_nodes = (self.action_space.n - 3) // 8
        if num_nodes <= 0:
            return self._random_action()
        
        # Extract node states from observation
        # Observation format: [compromised, defended, patched, monitored, value, vulnerability, connectivity] × nodes
        features_per_node = 7
        
        # Find critical nodes (high value)
        critical_nodes = []
        for i in range(num_nodes):
            start_idx = i * features_per_node
            if start_idx + 4 < len(observation):  # value is at index 4
                node_value = observation[start_idx + 4]
                if node_value > 0.6:  # High value node
                    critical_nodes.append(i)
        
        if critical_nodes and random.random() < 0.5:
            # Protect a critical node
            node_idx = random.choice(critical_nodes)
            action_type = random.choice([1, 2, 3])  # monitor, isolate, or patch
            targeted_action = 3 + (action_type * num_nodes) + node_idx
            if targeted_action < self.action_space.n:
                self.action_history.append(("targeted", targeted_action))
                return targeted_action
        
        return self._random_action()
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about actions taken"""
        if not self.action_history:
            return {}
        
        action_types = [action[0] for action in self.action_history]
        stats = {
            'total_actions': len(action_types),
            'random_actions': action_types.count('random'),
            'monitor_actions': action_types.count('monitor'),
            'aggressive_actions': action_types.count('aggressive'),
            'targeted_actions': action_types.count('targeted'),
        }
        
        # Calculate percentages
        for key in list(stats.keys()):
            if key != 'total_actions' and stats['total_actions'] > 0:
                stats[f'{key}_percent'] = stats[key] / stats['total_actions']
        
        return stats
    
    def save(self, path: str):
        """Save baseline agent configuration"""
        import json
        config = {
            'strategy': self.strategy,
            'action_space_n': self.action_space.n,
            'total_actions_taken': len(self.action_history)
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Baseline defender configuration saved to {path}")
    
    def load(self, path: str):
        """Load baseline agent configuration"""
        import json
        import os
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = json.load(f)
            
            self.strategy = config.get('strategy', 'random')
            print(f"📂 Baseline defender configuration loaded from {path}")
        else:
            print(f"⚠️  No baseline configuration found at {path}, using default")

class HeuristicDefender:
    """More sophisticated heuristic defender with rule-based strategies"""
    
    def __init__(self, action_space: spaces.Space, num_nodes: int):
        self.action_space = action_space
        self.num_nodes = num_nodes
        self.features_per_node = 7
        self.attack_history = []
        
        print("🎯 Initializing Heuristic Defender")
    
    def act(self, observation: np.ndarray) -> int:
        """Heuristic strategy based on network state analysis"""
        
        # Analyze current network state
        network_state = self._analyze_network_state(observation)
        
        # Strategy selection based on threat level
        if network_state['threat_level'] == 'high':
            return self._emergency_response(network_state)
        elif network_state['threat_level'] == 'medium':
            return self._targeted_defense(network_state)
        else:
            return self._preventive_defense(network_state)
    
    def _analyze_network_state(self, observation: np.ndarray) -> Dict[str, Any]:
        """Analyze the current network security state"""
        compromised_nodes = []
        critical_nodes_at_risk = []
        
        for i in range(self.num_nodes):
            start_idx = i * self.features_per_node
            if start_idx + 4 < len(observation):
                is_compromised = observation[start_idx] > 0.5
                node_value = observation[start_idx + 4]
                
                if is_compromised:
                    compromised_nodes.append(i)
                    if node_value > 0.6:  # Critical node
                        critical_nodes_at_risk.append(i)
        
        threat_level = 'low'
        if len(critical_nodes_at_risk) >= 2:
            threat_level = 'high'
        elif len(compromised_nodes) >= 2:
            threat_level = 'medium'
        elif len(compromised_nodes) > 0:
            threat_level = 'low'
        
        return {
            'threat_level': threat_level,
            'compromised_nodes': compromised_nodes,
            'critical_at_risk': critical_nodes_at_risk,
            'total_compromised': len(compromised_nodes)
        }
    
    def _emergency_response(self, network_state: Dict[str, Any]) -> int:
        """Emergency response for high threat levels"""
        # Use global emergency containment
        if network_state['critical_at_risk']:
            return 0  # Emergency containment action
        
        # Isolate compromised critical nodes
        if network_state['compromised_nodes']:
            node_idx = network_state['compromised_nodes'][0]
            isolate_action = 3 + (2 * self.num_nodes) + node_idx  # isolate action
            if isolate_action < self.action_space.n:
                return isolate_action
        
        return self.action_space.sample()
    
    def _targeted_defense(self, network_state: Dict[str, Any]) -> int:
        """Targeted defense for medium threat levels"""
        # Focus on compromised nodes
        if network_state['compromised_nodes']:
            node_idx = network_state['compromised_nodes'][0]
            # Prefer patch over isolate to maintain functionality
            patch_action = 3 + (3 * self.num_nodes) + node_idx  # patch action
            if patch_action < self.action_space.n:
                return patch_action
        
        # Monitor vulnerable areas
        monitor_node = random.randint(0, self.num_nodes - 1)
        monitor_action = 3 + (1 * self.num_nodes) + monitor_node
        if monitor_action < self.action_space.n:
            return monitor_action
        
        return self.action_space.sample()
    
    def _preventive_defense(self, network_state: Dict[str, Any]) -> int:
        """Preventive defense for low threat levels"""
        # Random monitoring and patching
        action_type = random.choice([1, 3])  # monitor or patch
        node_idx = random.randint(0, self.num_nodes - 1)
        preventive_action = 3 + (action_type * self.num_nodes) + node_idx
        
        if preventive_action < self.action_space.n:
            return preventive_action
        
        return self.action_space.sample()