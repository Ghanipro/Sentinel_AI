import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.corporate_network import CorporateNetworkEnv
from agents.ppo_agent import IntrusionResponsePPO
import yaml

def test_ppo_training_simple():
    """Test PPO training with minimal steps"""
    print("🧪 Testing PPO training (simple)...")
    
    try:
        # Load config
        with open("configs/training_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print("🏗️ Creating environment...")
        env = CorporateNetworkEnv()
        
        print("🤖 Initializing PPO agent...")
        ppo_agent = IntrusionResponsePPO(env, config['ppo'])
        
        # Test with very small number of steps
        print("🚀 Starting PPO training with 500 steps...")
        ppo_agent.train(total_timesteps=500)
        
        print("✅ PPO training test completed successfully!")
        
        # Test that we can save the model
        ppo_agent.save("models/test_ppo_model")
        print("💾 Test model saved successfully!")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ PPO training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ppo_training_simple()