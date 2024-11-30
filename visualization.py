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

def save_epoch(history, current_epoch, include_raw_minmax=False):
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
    # Plot smoothed rewards
    ax1.plot(epochs, smooth_curve(avg_rewards), 'b-', label='Smoothed Average', linewidth=2)
    ax1.plot(epochs, smooth_curve(max_rewards), 'g--', label='Smoothed Max', linewidth=2)
    ax1.plot(epochs, smooth_curve(min_rewards), 'r--', label='Smoothed Min', linewidth=2)
    ax1.fill_between(epochs, min_rewards, max_rewards, alpha=0.1, color='blue')
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
    filepath = os.path.join(VISUALIZATION_DIR, f'training_progress_epoch_{current_epoch}.png')
    plt.savefig(filepath)
    plt.close()
    return filepath


def create_training_visualization(training_history, phase_history):
    """Create a visualization showing rewards and phase transitions"""
    plt.figure(figsize=(15, 10))

    # Create main reward plot
    ax1 = plt.gca()

    # Plot rewards
    avg_rewards = [entry['avg_reward'] for entry in training_history]
    epochs = range(len(avg_rewards))

    # Plot raw data with low alpha
    ax1.plot(epochs, avg_rewards, 'b-', alpha=0.2, label='Raw Rewards')

    # Plot smoothed data
    smoothed_rewards = smooth_curve(points=avg_rewards)
    ax1.plot(epochs, smoothed_rewards, 'b-', label='Smoothed Rewards')

    # Fill the area under the smoothed rewards plot
    ax1.fill_between(epochs, smoothed_rewards, color='b', alpha=0.2)

    # Add phase transition markers
    colors = ['r', 'g', 'y', 'm']  # Different colors for each phase
    for phase_change in phase_history:
        epoch = phase_change['epoch']
        phase = phase_change['phase']
        color = colors[min(phase - 2, len(colors) - 1)]  # -2 because we start at phase 1

        # Add vertical line for phase transition
        ax1.axvline(x=epoch, color=color, linestyle='--', alpha=0.5)

        # Add text annotation
        ax1.text(epoch, ax1.get_ylim()[1], f'Phase {phase}',
                 rotation=90, verticalalignment='bottom')

    # Add phase performance indicators
    for phase_change in phase_history:
        epoch = phase_change['epoch']
        avg_reward = phase_change['avg_reward']
        stability = phase_change['stability']

        # Add marker for phase performance
        ax1.scatter(epoch, avg_reward, color='red', s=100, zorder=5)

        # Add annotation with stability
        ax1.annotate(f'σ={stability:.2f}',
                     (epoch, avg_reward),
                     xytext=(10, 10),
                     textcoords='offset points')

    # Customize plot
    ax1.set_title('Training Progress with Phase Transitions')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True)
    ax1.legend()

    # Add training phase information
    phase_info = "Training Phases:\n"
    phase_info += "1: Basic Harmony & Rhythm\n"
    phase_info += "2: Voice Leading\n"
    phase_info += "3: Rhythm Patterns & Variety\n"
    phase_info += "4: Basic Motifs\n"
    phase_info += "5: Full ABAB Form"

    plt.text(1.15, 0.5, phase_info,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='center')

    # Adjust layout to prevent text overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig('training_progress.png', bbox_inches='tight', dpi=300)

    # Create additional phase analysis plot
    create_phase_analysis(training_history, phase_history)


def create_phase_analysis(training_history, phase_history):
    """Create a separate visualization for phase-specific analysis"""
    plt.figure(figsize=(15, 8))

    # Collect phase-specific data
    phase_data = {}
    current_phase = 1

    for entry in training_history:
        phase = entry['phase']
        if phase not in phase_data:
            phase_data[phase] = {'rewards': [], 'epsilon': []}

        phase_data[phase]['rewards'].append(entry['avg_reward'])
        phase_data[phase]['epsilon'].append(entry['epsilon'])

    # Create subplots for each phase
    num_phases = len(phase_data)
    fig, axs = plt.subplots(2, num_phases, figsize=(15, 8))

    for phase in sorted(phase_data.keys()):
        data = phase_data[phase]
        idx = phase - 1

        # Plot reward distribution
        axs[0, idx].hist(data['rewards'], bins=20, alpha=0.7)
        axs[0, idx].set_title(f'Phase {phase} Reward Distribution')
        axs[0, idx].set_xlabel('Reward')
        axs[0, idx].set_ylabel('Frequency')

        # Plot epsilon decay
        axs[1, idx].plot(data['epsilon'])
        axs[1, idx].set_title(f'Phase {phase} Epsilon Decay')
        axs[1, idx].set_xlabel('Episode')
        axs[1, idx].set_ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig('phase_analysis.png', bbox_inches='tight', dpi=300)


def create_phase_training_visualization(phase_history):
    """Create visualization showing training progress for each phase"""
    plt.figure(figsize=(15, 10))

    # Plot rewards for each phase in different colors
    colors = ['blue', 'green', 'red', 'purple']
    for phase in range(1, 5):
        phase_data = [entry for entry in phase_history if entry['phase'] == phase]
        if phase_data:
            epochs = range(len(phase_data))
            rewards = [entry['avg_reward'] for entry in phase_data]

            # Plot raw data with low alpha
            plt.plot(epochs, rewards, color=colors[phase - 1], alpha=0.2,
                     label=f'Phase {phase} Raw')

            # Plot smoothed data
            smoothed = smooth_curve(rewards)
            plt.plot(epochs, smoothed, color=colors[phase - 1],
                     label=f'Phase {phase} Smoothed')

    plt.title('Training Progress by Phase')
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()

    # Add phase descriptions
    phase_info = (
        "Phase 1: Basic Harmony & Rhythm\n"
        "Phase 2: Voice Leading\n"
        "Phase 3: Rhythm Patterns & Variety\n"
        "Phase 4: Motifs & Structure"
    )
    plt.text(1.15, 0.5, phase_info, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig('phase_training_progress.png', bbox_inches='tight', dpi=300)


