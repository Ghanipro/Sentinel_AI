import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Any, List

class NetworkFeatureExtractor(BaseFeaturesExtractor):
    """Custom neural network for processing network security states"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)

class IntrusionResponsePPO:
    """Advanced PPO agent for autonomous intrusion response"""
    
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        
        # Policy configuration with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=NetworkFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 128, 64]
        )
        
        print("🤖 Initializing PPO Agent for Intrusion Response...")
        
        # Create tensorboard log directory if it doesn't exist
        import os
        tensorboard_log_dir = "./ppo_tensorboard/"
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('learning_rate', 0.0003),
            n_steps=config.get('n_steps', 1024),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log_dir
        )
        
        self.training_history = []
        print("✅ PPO Agent initialized successfully!")
    
    def train(self, total_timesteps: int = 100000, callback=None):
        """Train the PPO agent with progress tracking"""
        print(f"🚀 Starting PPO training for {total_timesteps} timesteps...")
        
        try:
            # Disable progress bar to avoid dependency issues
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,  # Add callback parameter
                reset_num_timesteps=True,
                tb_log_name="ppo_intrusion_response",
                progress_bar=False  # Explicitly disable progress bar
            )
            
            print("✅ PPO training completed!")
            
        except Exception as e:
            print(f"❌ PPO training failed: {e}")
            # Save model even if training fails
            try:
                self.model.save("models/ppo_intrusion_response_emergency")
                print("💾 Emergency model saved")
            except:
                print("⚠️ Could not save emergency model")
            raise e
    
    def save(self, path: str):
        """Save trained model"""
        self.model.save(path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env)
        print(f"📂 Model loaded from {path}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict action for given observation"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action