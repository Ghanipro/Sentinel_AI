import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Any, Dict

class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress and saving best model
    """
    
    def __init__(self, eval_env, save_path: str, eval_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        
        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to env.step()
        """
        if self.n_calls % self.eval_freq == 0:
            try:
                # Evaluate the policy
                mean_reward, std_reward = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=3,  # Reduced for speed
                    deterministic=False
                )
                
                if self.verbose > 0:
                    print(f"Eval at step {self.n_calls}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"New best model with mean reward {mean_reward:.2f}!")
                    self.model.save(f"{self.save_path}_best")
                
                # Save checkpoint every 2 evaluations
                if self.n_calls % (2 * self.eval_freq) == 0:
                    self.model.save(f"{self.save_path}_checkpoint_{self.n_calls}")
                    
            except Exception as e:
                print(f"⚠️ Evaluation failed at step {self.n_calls}: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """
        This method is called before training ends
        """
        # Save final model
        self.model.save(f"{self.save_path}_final")
        if self.verbose > 0:
            print("Training completed! Final model saved.")