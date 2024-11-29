import numpy as np

from agent import Qtar
import os

from visualization import smooth_curve, create_phase_training_visualization

PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"
PHASE_CHECKPOINTS_DIR = "models/phase_checkpoints"


def train_single_phase(qtar, phase_number, epochs=200, episodes_per_epoch=100):
    """Train a single phase until performance criteria are met"""
    print(f"\nStarting Phase {phase_number} Training...")

    best_avg_reward = float('-inf')
    patience = 0
    max_patience = 10  # Number of epochs without improvement before stopping

    phase_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} (Phase {phase_number})")

        # Train for one epoch
        training_history, _ = qtar.train_extensive(
            total_epochs=1,
            episodes_per_epoch=episodes_per_epoch
        )

        # Calculate metrics
        avg_reward = np.mean([entry['avg_reward'] for entry in training_history])
        phase_history.append({
            'epoch': epoch,
            'avg_reward': avg_reward,
            'phase': phase_number
        })

        print(f"Average Reward: {avg_reward:.2f}")

        # Check if performance improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            patience = 0
            # Save best model for this phase
            os.makedirs(PHASE_CHECKPOINTS_DIR, exist_ok=True)
            qtar.save_model(
                f"{PHASE_CHECKPOINTS_DIR}/phase_{phase_number}_best.pt",
                metadata={'phase': phase_number, 'avg_reward': avg_reward}
            )
        else:
            patience += 1

        # Check phase completion criteria
        if phase_meets_criteria(phase_number, avg_reward, training_history):
            print(f"\nPhase {phase_number} criteria met! Moving to next phase.")
            break

        # Early stopping check
        if patience >= max_patience:
            print(f"\nEarly stopping triggered for Phase {phase_number}")
            break

    return phase_history


def phase_meets_criteria(phase_number, avg_reward, training_history):
    """Define completion criteria for each phase"""
    min_epochs = {
        1: 50,  # Minimum 100 epochs for phase 1
        2: 50,
        3: 50,
        4: 50
    }
    # Check if we've done minimum epochs
    if len(training_history) < min_epochs[phase_number]:
        return False
    # Get reward stats from recent history
    recent_rewards = [entry['avg_reward'] for entry in training_history[-50:]]
    stability = np.std(recent_rewards) if recent_rewards else float('inf')
    recent_avg = np.mean(recent_rewards) if recent_rewards else float('-inf')
    if phase_number == 1:
        return avg_reward > 1.0 and stability < 0.2 and recent_avg > 0.8
    elif phase_number == 2:
        return avg_reward > 1.0 and stability < 0.3
    elif phase_number == 3:
        return avg_reward > 1.5 and stability < 0.3
    elif phase_number == 4:
        return avg_reward > 2.0 and stability < 0.3
    return False

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