import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from environments.corporate_network import CorporateNetworkEnv

class NetworkVisualizer:
    """Visualize the network topology and security state"""
    
    def __init__(self):
        self.env = CorporateNetworkEnv()
        self.colors = {
            'compromised': '#FF6B6B',
            'defended': '#4ECDC4', 
            'normal': '#45B7D1',
            'critical': '#FFE66D'
        }
    
    def visualize_network_topology(self, save_path: str = "results/network_topology.png"):
        """Visualize the network topology"""
        G = self.env.topology.graph
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(14, 10))
        
        # Get node attributes for coloring
        node_colors = []
        node_sizes = []
        
        for node in sorted(self.env.topology.nodes):
            attrs = self.env.topology.node_attributes[node]
            
            if attrs['value'] >= 4:  # Critical assets
                color = self.colors['critical']
                size = 1200
            elif attrs['compromised']:
                color = self.colors['compromised']
                size = 800
            elif attrs['defended']:
                color = self.colors['defended']
                size = 800
            else:
                color = self.colors['normal']
                size = 600
                
            node_colors.append(color)
            node_sizes.append(size)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.9, edgecolors='black', linewidths=2)
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['normal'], 
                      markersize=10, label='Normal Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['critical'], 
                      markersize=10, label='Critical Asset'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['compromised'], 
                      markersize=10, label='Compromised'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['defended'], 
                      markersize=10, label='Defended')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        plt.title('Corporate Network Topology and Security State', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Network topology saved to {save_path}")
        plt.show()
    
    def visualize_attack_progression(self, save_path: str = "results/attack_progression.png"):
        """Visualize how attacks propagate through the network"""
        # Simulate an attack progression
        self.env.reset()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        steps_to_show = min(6, self.env.max_steps)
        
        for step in range(steps_to_show):
            ax = axes[step]
            
            # Take a random action
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Visualize current state
            self._plot_network_state(ax, step, info)
            
            if terminated or truncated:
                break
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Attack progression saved to {save_path}")
        plt.show()
    
    def _plot_network_state(self, ax, step: int, info: dict):
        """Plot network state at a given step"""
        G = self.env.topology.graph
        pos = nx.spring_layout(G, seed=42)
        
        node_colors = []
        for node in sorted(self.env.topology.nodes):
            attrs = self.env.topology.node_attributes[node]
            
            if node in info['compromised_nodes']:
                color = self.colors['compromised']
            elif node in info['defended_nodes']:
                color = self.colors['defended']
            elif self.env.topology.node_attributes[node]['value'] >= 4:
                color = self.colors['critical']
            else:
                color = self.colors['normal']
                
            node_colors.append(color)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, 
                              alpha=0.9, ax=ax, edgecolors='black')
        nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax, font_weight='bold')
        
        ax.set_title(f'Step {step + 1}\nCompromised: {len(info["compromised_nodes"])}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')

def visualize_network():
    """Main function to generate network visualizations"""
    print("🌐 Generating network visualizations...")
    
    visualizer = NetworkVisualizer()
    visualizer.visualize_network_topology()
    visualizer.visualize_attack_progression()
    
    print("✅ Network visualizations generated!")

if __name__ == "__main__":
    visualize_network()