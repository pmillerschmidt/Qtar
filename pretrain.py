from agent import Qtar
import os
import matplotlib.pyplot as plt

from visualization import smooth_curve

PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"


def pretrain():
    # Initialize model
    qtar = Qtar(scale='C_MAJOR', progression_type='I_VI_IV_V', use_human_feedback=False)
    # Training parameters
    total_epochs = 100
    episodes_per_epoch = 100
    print("Starting pretraining...")
    try:
        # Single call to train_extensive
        training_history = qtar.train_extensive(
            total_epochs=total_epochs,
            episodes_per_epoch=episodes_per_epoch
        )
        print("\nTraining complete! Saving model and plots...")
        # Save final model
        os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
        qtar.save_model(PRETRAINED_MODEL_PATH)
        print(f"Model saved to {PRETRAINED_MODEL_PATH}")
        # Create the figure with the desired size
        plt.figure(figsize=(12, 8))
        # Plot rewards
        avg_rewards = [entry['avg_reward'] for entry in training_history]
        epochs = range(len(avg_rewards))
        # Plot raw data with low alpha
        plt.plot(epochs, avg_rewards, 'b-', alpha=0.2, label='Raw')
        # Plot smoothed data
        smoothed_rewards = smooth_curve(points=avg_rewards)
        plt.plot(epochs, smoothed_rewards, 'b-', label='Smoothed')
        # Fill the area under the smoothed rewards plot
        plt.fill_between(epochs, smoothed_rewards, color='b', alpha=0.2)
        plt.title('Average Reward per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        # Save the plot to a file
        plt.savefig('average_reward_plot.png')
    except KeyboardInterrupt:
        print("\nPretraining interrupted by user")
        # Save model if interrupted
        os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
        qtar.save_model(PRETRAINED_MODEL_PATH)
        print(f"Model saved to {PRETRAINED_MODEL_PATH}")


if __name__ == "__main__":
    pretrain()