import matplotlib.pyplot as plt
import os

VISUALIZATION_DIR = 'visualizations'

def smooth_curve(points, factor=0.8):
    """Smooth points using exponential moving average"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def save_epoch(history, current_epoch, include_raw_minmax=False, training_phase=1):
    """Save visualization of training progress at specific epoch"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
    epochs = [entry['epoch'] for entry in history]
    avg_rewards = [entry['avg_reward'] for entry in history]
    max_rewards = [entry['max_reward'] for entry in history]
    min_rewards = [entry['min_reward'] for entry in history]
    epsilons = [entry['epsilon'] for entry in history]
    # Plot raw rewards with low alpha
    ax1.plot(epochs, avg_rewards, 'b-', alpha=0.2, label='Raw Average')
    if include_raw_minmax:
        ax1.plot(epochs, max_rewards, 'g-', alpha=0.2, label='Raw Max')
        ax1.plot(epochs, min_rewards, 'r-', alpha=0.2, label='Raw Min')
        ax1.fill_between(epochs, min_rewards, max_rewards, alpha=0.1, color='blue')
    else:
        ax1.fill_between(epochs, smooth_curve(min_rewards), smooth_curve(max_rewards), alpha=0.1, color='blue')
    # Plot smoothed rewards
    ax1.plot(epochs, smooth_curve(avg_rewards), 'b-', label='Smoothed Average', linewidth=2)
    ax1.plot(epochs, smooth_curve(max_rewards), 'g--', label='Smoothed Max', linewidth=2)
    ax1.plot(epochs, smooth_curve(min_rewards), 'r--', label='Smoothed Min', linewidth=2)
    ax1.set_title('Training Rewards Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Plot epsilon with smoothing
    ax2.plot(epochs, epsilons, 'purple', alpha=0.2, label='Raw Epsilon')
    ax2.plot(epochs, smooth_curve(epsilons), 'purple',
             label='Smoothed Epsilon', linewidth=2)
    ax2.set_title('Exploration Rate (Epsilon) Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    # Save with epoch number
    filepath = os.path.join(VISUALIZATION_DIR, f'training_progress_phase_{training_phase}_epoch_{current_epoch}.png')
    plt.savefig(filepath)
    plt.close()
    return filepath

