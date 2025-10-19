import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import yaml
import networkx as nx

class NetworkTopology:
    """Modern corporate IT infrastructure simulation"""
    
    def __init__(self, config_path: str = "configs/network_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.graph = nx.Graph()
        self.nodes = []
        self.node_attributes = {}
        self.subnets = {}
        
        self._initialize_topology()
    
    def _initialize_topology(self):
        """Initialize realistic corporate network"""
        print("🏗️ Building Corporate IT Infrastructure...")
        
        # Node specifications
        node_specs = {
            # DMZ components
            "web_server": {"type": "server", "os": "linux", "services": ["http", "https"], "value": 2, "vulnerability": 0.7},
            "load_balancer": {"type": "network", "os": "specialized", "services": ["load_balancing"], "value": 3, "vulnerability": 0.4},
            
            # Internal services
            "app_server": {"type": "server", "os": "linux", "services": ["api", "application"], "value": 3, "vulnerability": 0.6},
            "file_server": {"type": "server", "os": "windows", "services": ["smb", "fileshare"], "value": 3, "vulnerability": 0.8},
            
            # Restricted zone
            "database_server": {"type": "database", "os": "linux", "services": ["mysql"], "value": 5, "vulnerability": 0.3},
            "domain_controller": {"type": "server", "os": "windows", "services": ["active_directory"], "value": 4, "vulnerability": 0.5},
            
            # User endpoints
            "workstation": {"type": "endpoint", "os": "windows", "services": ["rdp", "browser"], "value": 1, "vulnerability": 0.9},
            "developer_pc": {"type": "endpoint", "os": "linux", "services": ["ssh"], "value": 2, "vulnerability": 0.6}
        }
        
        # Create nodes based on configuration
        node_id = 0
        for subnet in self.config['network']['topology']['subnets']:
            subnet_nodes = []
            for node_template in subnet['nodes']:
                if "web_server" in node_template:
                    for i in range(2):
                        node_id = f"web_server_{i+1}"
                        self._add_node(node_id, node_specs["web_server"], subnet['name'])
                        subnet_nodes.append(node_id)
                
                elif "app_server" in node_template:
                    for i in range(2):
                        node_id = f"app_server_{i+1}"
                        self._add_node(node_id, node_specs["app_server"], subnet['name'])
                        subnet_nodes.append(node_id)
                
                elif "database_server" in node_template:
                    node_id = "database_server"
                    self._add_node(node_id, node_specs["database_server"], subnet['name'])
                    subnet_nodes.append(node_id)
                
                elif "workstation" in node_template:
                    for i in range(2):
                        node_id = f"workstation_{i+1}"
                        self._add_node(node_id, node_specs["workstation"], subnet['name'])
                        subnet_nodes.append(node_id)
                
                elif "file_server" in node_template:
                    node_id = "file_server"
                    self._add_node(node_id, node_specs["file_server"], subnet['name'])
                    subnet_nodes.append(node_id)
                
                elif "domain_controller" in node_template:
                    node_id = "domain_controller"
                    self._add_node(node_id, node_specs["domain_controller"], subnet['name'])
                    subnet_nodes.append(node_id)
                
                elif "developer_pc" in node_template:
                    node_id = "developer_pc_1"
                    self._add_node(node_id, node_specs["developer_pc"], subnet['name'])
                    subnet_nodes.append(node_id)
                
                elif "load_balancer" in node_template:
                    node_id = "load_balancer"
                    self._add_node(node_id, node_specs["load_balancer"], subnet['name'])
                    subnet_nodes.append(node_id)
            
            self.subnets[subnet['name']] = subnet_nodes
        
        # Establish network connectivity
        self._establish_connections()
        
        print("✅ IT Infrastructure built successfully!")
        print(f"   Total Nodes: {len(self.nodes)}")
        print(f"   Nodes: {sorted(self.nodes)}")
        print(f"   Subnets: {list(self.subnets.keys())}")
    
    def _add_node(self, node_id: str, specs: Dict, subnet: str):
        """Add a node to the network"""
        self.nodes.append(node_id)
        self.graph.add_node(node_id)
        
        self.node_attributes[node_id] = {
            'subnet': subnet,
            'type': specs['type'],
            'os': specs['os'],
            'services': specs['services'],
            'value': specs['value'],
            'vulnerability': specs['vulnerability'],
            'compromised': False,
            'defended': False,
            'patched': False,
            'monitored': False,
            'last_compromise_time': -1
        }
    
    def _establish_connections(self):
        """Establish realistic network connections"""
        print("🔗 Establishing network connections...")
        
        # Only create connections for nodes that exist
        existing_nodes = set(self.nodes)
        
        # DMZ connectivity
        if "web_server_1" in existing_nodes and "load_balancer" in existing_nodes:
            self.graph.add_edge("web_server_1", "load_balancer")
        if "web_server_2" in existing_nodes and "load_balancer" in existing_nodes:
            self.graph.add_edge("web_server_2", "load_balancer")
        if "load_balancer" in existing_nodes and "app_server_1" in existing_nodes:
            self.graph.add_edge("load_balancer", "app_server_1")
        if "load_balancer" in existing_nodes and "app_server_2" in existing_nodes:
            self.graph.add_edge("load_balancer", "app_server_2")
        
        # Application tier connectivity
        if "app_server_1" in existing_nodes and "file_server" in existing_nodes:
            self.graph.add_edge("app_server_1", "file_server")
        if "app_server_2" in existing_nodes and "file_server" in existing_nodes:
            self.graph.add_edge("app_server_2", "file_server")
        if "app_server_1" in existing_nodes and "database_server" in existing_nodes:
            self.graph.add_edge("app_server_1", "database_server")
        if "app_server_2" in existing_nodes and "database_server" in existing_nodes:
            self.graph.add_edge("app_server_2", "database_server")
        
        # User connectivity
        if "workstation_1" in existing_nodes and "app_server_1" in existing_nodes:
            self.graph.add_edge("workstation_1", "app_server_1")
        if "workstation_2" in existing_nodes and "app_server_2" in existing_nodes:
            self.graph.add_edge("workstation_2", "app_server_2")
        if "developer_pc_1" in existing_nodes and "app_server_1" in existing_nodes:
            self.graph.add_edge("developer_pc_1", "app_server_1")
        
        # Administrative connectivity
        if "domain_controller" in existing_nodes and "database_server" in existing_nodes:
            self.graph.add_edge("domain_controller", "database_server")
        if "domain_controller" in existing_nodes and "file_server" in existing_nodes:
            self.graph.add_edge("domain_controller", "file_server")
        if "domain_controller" in existing_nodes and "app_server_1" in existing_nodes:
            self.graph.add_edge("domain_controller", "app_server_1")
        
        print(f"   Created {len(self.graph.edges)} connections")
    
    def get_attack_surface(self, current_node: str) -> List[str]:
        """Get reachable nodes for lateral movement"""
        if current_node not in self.graph:
            return []
        return list(self.graph.neighbors(current_node))
    
    def get_critical_assets(self) -> List[str]:
        """Get high-value targets"""
        return [node for node, attrs in self.node_attributes.items() 
                if attrs['value'] >= 4]
    
    def get_vulnerable_nodes(self) -> List[str]:
        """Get nodes with high vulnerability scores"""
        return [node for node, attrs in self.node_attributes.items()
                if attrs['vulnerability'] > 0.6 and not attrs['defended']]
    
    def update_node_security_state(self, node: str, **kwargs):
        """Update node security attributes"""
        for key, value in kwargs.items():
            if key in self.node_attributes[node]:
                self.node_attributes[node][key] = value
    
    def get_network_state_vector(self) -> np.ndarray:
        """Get numerical state representation for RL"""
        state_features = []
        
        for node in sorted(self.nodes):
            attrs = self.node_attributes[node]
            state_features.extend([
                # Security state (binary)
                1.0 if attrs['compromised'] else 0.0,
                1.0 if attrs['defended'] else 0.0,
                1.0 if attrs['patched'] else 0.0,
                1.0 if attrs['monitored'] else 0.0,
                
                # Node properties (continuous)
                attrs['value'] / 5.0,  # Normalized value
                attrs['vulnerability'],  # Vulnerability score
                len(list(self.graph.neighbors(node))) / 6.0  # Normalized connectivity
            ])
        
        return np.array(state_features, dtype=np.float32)
    
    def get_compromised_nodes(self) -> List[str]:
        return [node for node in self.nodes if self.node_attributes[node]['compromised']]
    
    def get_defended_nodes(self) -> List[str]:
        return [node for node in self.nodes if self.node_attributes[node]['defended']]
    
    def get_node_value(self, node: str) -> float:
        return self.node_attributes[node]['value']

class CorporateNetworkEnv(gym.Env):
    """
    Realistic corporate network environment for intrusion response training
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, config_path: str = "configs/network_config.yaml", render_mode: Optional[str] = None):
        super().__init__()
        
        # Initialize network topology
        self.topology = NetworkTopology(config_path)
        self.config = self.topology.config
        
        # Environment parameters
        self.max_steps = self.config['environment']['max_steps']
        self.initial_compromised = self.config['environment']['initial_compromised']
        
        # Define action space
        self.num_nodes = len(self.topology.nodes)
        self.defender_actions = [
            "no_op", "monitor", "isolate", "patch", "block_traffic",
            "deceive", "quarantine", "backup"
        ]
        
        # Defender action space: action_type × target_node + global actions
        self.action_space = spaces.Discrete(
            len(self.defender_actions) * self.num_nodes + 3  # +3 for global actions
        )
        
        # Observation space: comprehensive network state
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_nodes * 7,), dtype=np.float32
        )
        
        # Attack simulation
        self.attack_progression = []
        self.attacker_skill = 0.7  # Base attacker capability
        self.current_threat_level = 1.0
        
        # Performance tracking
        self.metrics = {
            'detection_times': [],
            'false_positives': 0,
            'successful_containments': 0,
            'critical_breaches': 0,
            'service_disruption': 0
        }
        
        self.render_mode = render_mode
        self.current_step = 0
        
        print("🎮 Corporate Network Environment Ready!")
        print(f"   Observation space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space.n}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset network state
        for node in self.topology.nodes:
            self.topology.update_node_security_state(
                node, compromised=False, defended=False, 
                patched=False, monitored=False
            )
        
        # Initialize security incidents
        for node in self.initial_compromised:
            if node in self.topology.nodes:
                self.topology.update_node_security_state(node, compromised=True)
        
        self.current_step = 0
        self.attack_progression = self.initial_compromised.copy()
        self.current_threat_level = 1.0
        
        # Reset metrics for episode
        self.metrics = {
            'detection_times': [],
            'false_positives': 0,
            'successful_containments': 0,
            'critical_breaches': 0,
            'service_disruption': 0
        }
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get comprehensive network state observation"""
        return self.topology.get_network_state_vector()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info for monitoring"""
        compromised = self.topology.get_compromised_nodes()
        defended = self.topology.get_defended_nodes()
        critical_assets = self.topology.get_critical_assets()
        critical_at_risk = len([node for node in critical_assets 
                              if node in compromised])
        
        return {
            "compromised_nodes": compromised,
            "defended_nodes": defended,
            "critical_assets": critical_assets,
            "critical_assets_at_risk": critical_at_risk,
            "attack_progression": self.attack_progression.copy(),
            "current_threat_level": self.current_threat_level,
            "successful_containments": self.metrics['successful_containments'],
            "false_positives": self.metrics['false_positives'],
            "current_step": self.current_step
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step
        Returns: observation, reward, terminated, truncated, info
        """
        # Execute defender action
        defender_reward = self._execute_defender_action(action)
        
        # Simulate attacker progression
        attacker_reward = self._simulate_attacker_behavior()
        
        # Calculate combined reward
        reward = defender_reward - attacker_reward * 0.3
        
        # Update environment
        self.current_step += 1
        self.current_threat_level *= 1.02  # Gradually increasing threat
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get new state
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _execute_defender_action(self, action: int) -> float:
        """Execute defender action and return reward"""
        reward = 0.0
        reward_config = self.config['environment']['rewards']
        
        if action == 0:  # Global: Emergency containment
            reward = self._emergency_containment()
        
        elif action == 1:  # Global: Enhance monitoring
            reward = self._enhance_monitoring()
        
        elif action == 2:  # Global: System-wide patch
            reward = self._system_wide_patch()
        
        else:  # Node-specific actions
            action_idx = action - 3
            action_type_idx = action_idx // self.num_nodes
            node_idx = action_idx % self.num_nodes
            
            if node_idx < len(self.topology.nodes):
                node = self.topology.nodes[node_idx]
                action_type = self.defender_actions[action_type_idx]
                reward = self._execute_node_action(node, action_type)
        
        return reward
    
    def _execute_node_action(self, node: str, action_type: str) -> float:
        """Execute node-specific defense action"""
        reward_config = self.config['environment']['rewards']
        node_attrs = self.topology.node_attributes[node]
        node_value = node_attrs['value']
        
        if action_type == "no_op":
            return -0.1  # Small penalty for inaction
            
        elif action_type == "monitor":
            self.topology.update_node_security_state(node, monitored=True)
            return 0.2  # Small reward for monitoring
            
        elif action_type == "isolate":
            if node_attrs['compromised']:
                self.topology.update_node_security_state(node, compromised=False, defended=True)
                self.metrics['successful_containments'] += 1
                return reward_config['successful_defense'] * node_value
            else:
                self.metrics['false_positives'] += 1
                self.metrics['service_disruption'] += 1
                return reward_config['false_positive']
                
        elif action_type == "patch":
            self.topology.update_node_security_state(node, patched=True)
            if node_attrs['compromised']:
                self.topology.update_node_security_state(node, compromised=False)
                self.metrics['successful_containments'] += 1
                return reward_config['successful_defense'] * node_value * 0.7
            else:
                return reward_config['prevented_attack'] * 0.5
                
        elif action_type == "block_traffic":
            self.topology.update_node_security_state(node, defended=True)
            self.metrics['service_disruption'] += 1
            return 0.3  # Preventive measure
            
        elif action_type == "deceive":
            # Deception: honeypot-like action
            return 0.4
            
        elif action_type == "quarantine":
            # Isolate and investigate
            was_compromised = node_attrs['compromised']
            self.topology.update_node_security_state(node, defended=True)
            self.metrics['service_disruption'] += 1
            if was_compromised:
                self.topology.update_node_security_state(node, compromised=False)
                self.metrics['successful_containments'] += 1
                return reward_config['successful_defense'] * node_value
            return -0.2
            
        elif action_type == "backup":
            # Backup critical data
            if node_value >= 3:
                return 0.3
            return 0.1
            
        return 0.0
    
    def _emergency_containment(self) -> float:
        """Emergency containment of critical assets"""
        critical_assets = self.topology.get_critical_assets()
        contained = 0
        
        for asset in critical_assets:
            if self.topology.node_attributes[asset]['compromised']:
                self.topology.update_node_security_state(asset, compromised=False, defended=True)
                contained += 1
                self.metrics['service_disruption'] += 1
        
        if contained > 0:
            self.metrics['successful_containments'] += contained
            return self.config['environment']['rewards']['successful_defense'] * contained * 2
        else:
            return -1.0  # Unnecessary emergency action
    
    def _enhance_monitoring(self) -> float:
        """Enhance monitoring on vulnerable nodes"""
        vulnerable_nodes = self.topology.get_vulnerable_nodes()
        monitored = 0
        
        for node in vulnerable_nodes[:3]:  # Monitor up to 3 nodes
            self.topology.update_node_security_state(node, monitored=True)
            monitored += 1
        
        return 0.2 * monitored
    
    def _system_wide_patch(self) -> float:
        """Apply patches to multiple nodes"""
        nodes_to_patch = [node for node in self.topology.nodes 
                         if not self.topology.node_attributes[node]['patched']]
        patched = 0
        
        for node in nodes_to_patch[:4]:  # Patch up to 4 nodes
            self.topology.update_node_security_state(node, patched=True)
            if self.topology.node_attributes[node]['compromised']:
                self.topology.update_node_security_state(node, compromised=False)
                self.metrics['successful_containments'] += 1
            patched += 1
        
        return 0.3 * patched
    
    def _simulate_attacker_behavior(self) -> float:
        """Simulate realistic attacker behavior"""
        attacker_reward = 0.0
        compromised_nodes = self.topology.get_compromised_nodes()
        
        # Existing compromised nodes attempt lateral movement
        for source_node in compromised_nodes:
            if random.random() < 0.6:  # 60% chance to attempt lateral movement
                target_nodes = self.topology.get_attack_surface(source_node)
                # Filter only existing nodes
                target_nodes = [node for node in target_nodes if node in self.topology.nodes]
                if target_nodes:
                    target = random.choice(target_nodes)
                    if self._attempt_compromise(target):
                        self.attack_progression.append(target)
                        target_value = self.topology.node_attributes[target]['value']
                        attacker_reward += target_value * 0.5
        
        # New attack attempts from external
        if random.random() < 0.3:  # 30% chance of new attack
            external_targets = [node for node in self.topology.nodes 
                            if node.startswith('web_server') or node.startswith('workstation')]
            if external_targets:
                target = random.choice(external_targets)
                if self._attempt_compromise(target):
                    self.attack_progression.append(target)
                    target_value = self.topology.node_attributes[target]['value']
                    attacker_reward += target_value
        
        return attacker_reward
    
    def _attempt_compromise(self, target_node: str) -> bool:
        """Attempt to compromise a target node"""
        node_attrs = self.topology.node_attributes[target_node]
        
        if node_attrs['compromised']:
            return False  # Already compromised
        
        # Base success probability based on vulnerability
        success_prob = node_attrs['vulnerability'] * self.attacker_skill * self.current_threat_level
        
        # Adjust based on defense state
        if node_attrs['defended']:
            success_prob *= 0.3
        if node_attrs['patched']:
            success_prob *= 0.4
        if node_attrs['monitored']:
            success_prob *= 0.7
        
        if random.random() < success_prob:
            self.topology.update_node_security_state(target_node, compromised=True)
            return True
        return False
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        critical_assets = self.topology.get_critical_assets()
        critical_compromised = [asset for asset in critical_assets 
                              if self.topology.node_attributes[asset]['compromised']]
        
        # Terminate if too many critical assets are compromised
        if len(critical_compromised) >= 2:
            self.metrics['critical_breaches'] += 1
            return True
        
        # Terminate if attack is completely contained for a while
        if len(self.topology.get_compromised_nodes()) == 0 and self.current_step > 15:
            return True
        
        return False
    
    def render(self):
        """Render the current network state"""
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            print(f"Compromised: {self.topology.get_compromised_nodes()}")
            print(f"Defended: {self.topology.get_defended_nodes()}")
            print(f"Critical at risk: {self._get_info()['critical_assets_at_risk']}")
            print(f"Threat level: {self.current_threat_level:.2f}")
    
    def close(self):
        """Clean up environment"""
        pass