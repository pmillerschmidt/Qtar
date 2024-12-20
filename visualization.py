import matplotlib.pyplot as plt
import os

VISUALIZATION_DIR = 'visualizations'


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def save_epoch(history, current_epoch):
    # save viz
    fig = plt.figure(figsize=(15, 10), dpi=100)
    gs = plt.GridSpec(2, 2)
    epochs = list(range(1, len(history) + 1))
    # metrics
    avg_rewards = [entry['avg_reward'] for entry in history]
    max_rewards = [entry['max_reward'] for entry in history]
    min_rewards = [entry['min_reward'] for entry in history]
    epsilons = [entry['epsilon'] for entry in history]
    losses = [entry['avg_loss'] for entry in history]
    # rewards
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, avg_rewards, 'b-', alpha=0.2, label='Raw Average')
    ax1.plot(epochs, max_rewards, 'g-', alpha=0.2, label='Raw Max')
    ax1.plot(epochs, min_rewards, 'r-', alpha=0.2, label='Raw Min')
    ax1.fill_between(epochs, min_rewards, max_rewards, alpha=0.1, color='blue')
    # smooth rewards
    ax1.plot(epochs, smooth_curve(avg_rewards), 'b-', label='Smoothed Average', linewidth=2)
    ax1.plot(epochs, smooth_curve(max_rewards), 'g--', label='Smoothed Max', linewidth=2)
    ax1.plot(epochs, smooth_curve(min_rewards), 'r--', label='Smoothed Min', linewidth=2)
    ax1.set_title('Training Rewards Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # epsilon
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, epsilons, 'purple', alpha=0.2, label='Raw Epsilon')
    ax2.plot(epochs, smooth_curve(epsilons), 'purple', label='Smoothed Epsilon', linewidth=2)
    ax2.set_title('Exploration Rate (Epsilon)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    # loss
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, losses, 'orange', alpha=0.2, label='Raw Loss')
    ax3.plot(epochs, smooth_curve(losses), 'orange', label='Smoothed Loss', linewidth=2)
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    # save
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    filepath = os.path.join(VISUALIZATION_DIR, f'training_progress_epoch_{current_epoch}.png')
    plt.savefig(filepath)
    plt.close()
    return filepath