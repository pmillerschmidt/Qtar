import numpy as np

from agent import Qtar
import os

from visualization import smooth_curve, create_phase_training_visualization

PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"
PHASE_CHECKPOINTS_DIR = "models/phase_checkpoints"

MIN_EPOCHS = {
    1: 400,  # Much longer for fundamental skills
    2: 200,  # Voice leading needs significant practice
    3: 200,  # Rhythm patterns need time to develop
    4: 200   # Motifs and structure are complex
}

def train_single_phase(qtar, phase_number, epochs=200, episodes_per_epoch=100):
    """Train a single phase until performance criteria are met"""
    print(f"\nStarting Phase {phase_number} Training...")

    best_avg_reward = float('-inf')
    patience = 0
    max_patience = 50  # Much longer patience
    phase_history = []

    # Must complete minimum epochs regardless of performance
    min_epochs_completed = False

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} (Phase {phase_number})")

        training_history, _ = qtar.train_extensive(total_epochs=1, episodes_per_epoch=episodes_per_epoch)

        # Calculate metrics
        avg_reward = np.mean([entry['avg_reward'] for entry in training_history])
        phase_history.append({
            'epoch': epoch,
            'avg_reward': avg_reward,
            'phase': phase_number
        })

        print(f"Average Reward: {avg_reward:.2f}")

        # Only consider advancement after minimum epochs
        if len(phase_history) >= MIN_EPOCHS[phase_number]:
            min_epochs_completed = True
            if phase_meets_criteria(phase_number, avg_reward, phase_history):
                print(f"\nPhase {phase_number} criteria met after thorough training!")
                break

        # Early stopping only after minimum epochs and with long patience
        if min_epochs_completed:
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience = 0
            else:
                patience += 1

            if patience >= max_patience:
                print(f"\nNo improvement for {max_patience} epochs after minimum training")
                break

    return phase_history


def phase_meets_criteria(phase_number, avg_reward, training_history):
    """Define completion criteria for each phase"""
    if len(training_history) < MIN_EPOCHS[phase_number]:
        return False

    # Must show sustained performance
    if not check_sustained_performance(training_history):
        return False

    # Phase-specific criteria
    if phase_number == 1:
        # Basic harmony must be very solid
        return (avg_reward > 50.0 and  # Was 1.2
                check_consecutive_windows(training_history, 3) and
                min_last_n_rewards(training_history, 100) > 30.0)  # Was 0.8

    elif phase_number == 2:
        return (avg_reward > 75.0 and  # Was 1.5
                check_consecutive_windows(training_history, 3) and
                min_last_n_rewards(training_history, 100) > 50.0)  # Was 1.0

    elif phase_number == 3:
        return (avg_reward > 100.0 and  # Was 1.8
                check_consecutive_windows(training_history, 3) and
                min_last_n_rewards(training_history, 100) > 75.0)  # Was 1.2

    elif phase_number == 4:
        return (avg_reward > 150.0 and  # Was 2.0
                check_consecutive_windows(training_history, 3) and
                min_last_n_rewards(training_history, 100) > 100.0)  # Was 1.5

    return False


def check_sustained_performance(training_history, window_size=100):
    """Check if performance has been consistently good over a long window"""
    if len(training_history) < window_size:
        return False

    recent_rewards = [entry['avg_reward'] for entry in training_history[-window_size:]]
    avg_reward = np.mean(recent_rewards)
    stability = np.std(recent_rewards)
    min_reward = min(recent_rewards)

    # Adjusted thresholds for higher reward scale
    return (avg_reward > 50.0 and  # Was 1.0
            stability < (avg_reward * 0.3) and  # Relative stability instead of absolute
            min_reward > 25.0)  # Was 0.5


def check_consecutive_windows(training_history, num_windows, window_size=100):
    """Check if performance criteria are met for multiple consecutive windows"""
    if len(training_history) < window_size * num_windows:
        return False

    for i in range(num_windows):
        start_idx = -(window_size * (i + 1))
        end_idx = -(window_size * i) if i > 0 else None
        window = training_history[start_idx:end_idx]
        if not check_sustained_performance(window):
            return False
    return True


def min_last_n_rewards(training_history, n):
    """Get minimum reward over last n episodes"""
    rewards = [entry['avg_reward'] for entry in training_history[-n:]]
    return min(rewards) if rewards else float('-inf')

def reward_stability(training_history):
    """Calculate the stability of recent rewards"""
    recent_rewards = [entry['avg_reward'] for entry in training_history[-50:]]
    return np.std(recent_rewards) if recent_rewards else float('inf')


def pretrain():
    """Pretrain through phases 1-4 sequentially"""
    all_phase_history = []
    # Initialize model
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        use_human_feedback=False,
        training_phase=1
    )

    try:
        # Train each phase sequentially
        for phase in range(1, 5):
            print(f"\n{'=' * 50}")
            print(f"Starting Phase {phase} Training")
            print(f"{'=' * 50}")

            # Train the current phase
            phase_history = train_single_phase(qtar, phase)
            all_phase_history.extend(phase_history)

            # Save phase completion checkpoint
            os.makedirs(PHASE_CHECKPOINTS_DIR, exist_ok=True)
            qtar.save_model(
                f"{PHASE_CHECKPOINTS_DIR}/phase_{phase}_complete.pt",
                metadata={
                    'completed_phases': list(range(1, phase + 1)),
                    'phase_history': phase_history
                }
            )

            # Advance to next phase if not final phase
            if phase < 4:
                qtar.current_phase = phase + 1
                print(f"\nAdvancing to Phase {phase + 1}")

        # Save final pretrained model
        print("\nPretraining complete! Saving final model...")
        os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
        qtar.save_model(
            PRETRAINED_MODEL_PATH,
            metadata={
                'completed_phases': [1, 2, 3, 4],
                'ready_for_human_feedback': True
            }
        )

        # Create visualization
        create_phase_training_visualization(all_phase_history)

    except KeyboardInterrupt:
        print("\nPretraining interrupted by user")
        qtar.save_model(PRETRAINED_MODEL_PATH)
        print(f"Progress saved to {PRETRAINED_MODEL_PATH}")


if __name__ == "__main__":
    pretrain()