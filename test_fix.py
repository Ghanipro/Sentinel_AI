import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dependencies():
    """Test if all dependencies are available"""
    print("🧪 Testing dependencies...")
    
    try:
        import gymnasium
        print("✅ gymnasium: OK")
        
        import stable_baselines3
        print("✅ stable-baselines3: OK")
        
        import torch
        print("✅ torch: OK")
        
        import numpy as np
        print("✅ numpy: OK")
        
        import yaml
        print("✅ pyyaml: OK")
        
        import tensorboard
        print("✅ tensorboard: OK")
        
        # Try to import tqdm and rich (for progress bar)
        try:
            import tqdm
            print("✅ tqdm: OK")
        except ImportError:
            print("❌ tqdm: Missing")
            
        try:
            import rich
            print("✅ rich: OK")
        except ImportError:
            print("❌ rich: Missing")
            
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_basic_environment():
    """Test basic environment functionality"""
    print("\n🧪 Testing environment...")
    
    try:
        from environments.corporate_network import CorporateNetworkEnv
        
        env = CorporateNetworkEnv()
        obs, info = env.reset()
        
        print(f"✅ Environment created - Obs shape: {obs.shape}, Action space: {env.action_space.n}")
        
        # Test a few random steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i}: Reward = {reward:.2f}")
            
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_initialization():
    """Test PPO agent initialization"""
    print("\n🧪 Testing PPO initialization...")
    
    try:
        from environments.corporate_network import CorporateNetworkEnv
        from agents.ppo_agent import IntrusionResponsePPO
        import yaml
        
        # Load config
        with open("configs/training_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        env = CorporateNetworkEnv()
        ppo_agent = IntrusionResponsePPO(env, config['ppo'])
        
        print("✅ PPO agent initialized successfully!")
        
        # Test prediction
        obs, info = env.reset()
        action = ppo_agent.predict(obs)
        print(f"✅ PPO prediction test - Action: {action}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ PPO initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 INTRUSION RESPONSE SYSTEM DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Run tests
    deps_ok = test_dependencies()
    env_ok = test_basic_environment()
    ppo_ok = test_ppo_initialization()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"   Environment:  {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   PPO Agent:    {'✅ PASS' if ppo_ok else '❌ FAIL'}")
    
    if deps_ok and env_ok and ppo_ok:
        print("\n🎉 All tests passed! The system is ready.")
        print("💡 You can now run: python run_experiment.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")