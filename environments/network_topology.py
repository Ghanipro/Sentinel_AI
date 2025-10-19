import numpy as np
import networkx as nx
from typing import Dict, List, Set
import yaml

class NetworkTopology:
    """Modern corporate IT infrastructure with realistic components"""
    
    def __init__(self, config_path: str = "configs/network_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.graph = nx.Graph()
        self.nodes = []
        self.node_attributes = {}
        self.subnets = {}
        self.services = {}
        
        self._initialize_topology()
    
    def _initialize_topology(self):
        """Initialize realistic corporate network"""
        print("🏗️ Building Corporate IT Infrastructure...")
        
        # Define node types and their properties
        node_specs = {
            # DMZ components
            "web_server": {"type": "server", "os": "linux", "services": ["http", "https"], "value": 2},
            "load_balancer": {"type": "network", "os": "specialized", "services": ["load_balancing"], "value": 3},
            
            # Internal services
            "app_server": {"type": "server", "os": "linux", "services": ["api", "application"], "value": 3},
            "file_server": {"type": "server", "os": "windows", "services": ["smb", "fileshare"], "value": 3},
            
            # Restricted zone
            "database_server": {"type": "database", "os": "linux", "services": ["mysql", "postgresql"], "value": 5},
            "domain_controller": {"type": "server", "os": "windows", "services": ["active_directory", "dns"], "value": 4},
            
            # User endpoints
            "workstation": {"type": "endpoint", "os": "windows", "services": ["rdp", "browser"], "value": 1},
            "developer_pc": {"type": "endpoint", "os": "linux", "services": ["ssh", "git"], "value": 2}
        }
        
        # Create network structure
        subnets_config = self.config['network']['topology']['subnets']
        for subnet in subnets_config:
            subnet_nodes = []
            for node_template in subnet['nodes']:
                # Create multiple instances of each node type
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
                    for i in range(3):
                        node_id = f"workstation_{i+1}"
                        self._add_node(node_id, node_specs["workstation"], subnet['name'])
                        subnet_nodes.append(node_id)
            
            self.subnets[subnet['name']] = subnet_nodes
        
        # Establish network connectivity
        self._establish_connections()
        print("✅ IT Infrastructure built successfully!")
    
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
            'compromised': False,
            'defended': False,
            'patched': False,
            'monitored': False,
            'last_compromise_time': -1
        }
    
    def _establish_connections(self):
        """Establish realistic network connections"""
        # DMZ connectivity
        self.graph.add_edge("web_server_1", "load_balancer")
        self.graph.add_edge("web_server_2", "load_balancer")
        self.graph.add_edge("load_balancer", "app_server_1")
        self.graph.add_edge("load_balancer", "app_server_2")
        
        # Application tier connectivity
        self.graph.add_edge("app_server_1", "file_server")
        self.graph.add_edge("app_server_2", "file_server")
        self.graph.add_edge("app_server_1", "database_server")
        self.graph.add_edge("app_server_2", "database_server")
        
        # User connectivity
        self.graph.add_edge("workstation_1", "app_server_1")
        self.graph.add_edge("workstation_2", "app_server_2")
        self.graph.add_edge("developer_pc_1", "app_server_1")
        
        # Administrative connectivity
        self.graph.add_edge("domain_controller", "database_server")
        self.graph.add_edge("domain_controller", "file_server")
    
    def get_attack_surface(self, current_node: str) -> List[str]:
        """Get reachable nodes for lateral movement"""
        if current_node not in self.graph:
            return []
        
        return list(self.graph.neighbors(current_node))
    
    def get_critical_assets(self) -> List[str]:
        """Get high-value targets"""
        return [node for node, attrs in self.node_attributes.items() 
                if attrs['value'] >= 4]
    
    def get_vulnerable_services(self, node: str) -> List[str]:
        """Get services with known vulnerabilities"""
        service_vulnerabilities = {
            "http": 0.7, "https": 0.5, "smb": 0.8, "rdp": 0.6,
            "mysql": 0.4, "active_directory": 0.3
        }
        
        services = self.node_attributes[node]['services']
        return [svc for svc in services if svc in service_vulnerabilities]
    
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
                # Security state
                1.0 if attrs['compromised'] else 0.0,
                1.0 if attrs['defended'] else 0.0,
                1.0 if attrs['patched'] else 0.0,
                1.0 if attrs['monitored'] else 0.0,
                
                # Node properties
                attrs['value'] / 5.0,  # Normalized value
                len(self.get_vulnerable_services(node)) / 5.0,
                len(list(self.graph.neighbors(node))) / 10.0  # Connectivity
            ])
        
        return np.array(state_features, dtype=np.float32)
    
    def get_compromised_nodes(self) -> List[str]:
        return [node for node in self.nodes if self.node_attributes[node]['compromised']]
    
    def get_defended_nodes(self) -> List[str]:
        return [node for node in self.nodes if self.node_attributes[node]['defended']]