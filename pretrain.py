import numpy as np

from agent import Qtar
import os

from visualization import smooth_curve, create_phase_training_visualization

PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"
PHASE_CHECKPOINTS_DIR = "models/phase_checkpoints"

MIN_EPOCHS = {1: 200, 2: 100}


def train_single_phase(qtar, phase_number, epochs=200, episodes_per_epoch=100):
    """Train a single phase until performance criteria are met"""
    print(f"\nStarting Phase {phase_number} Training...")
    print("Phase 1: Learning motifs on single chord" if phase_number == 1
          else "Phase 2: Developing patterns across progression")

    best_avg_reward = float('-inf')
    patience = 0
    max_patience = 50
    phase_history = []
    min_epochs_completed = False

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} (Phase {phase_number})")

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

        # Only consider advancement after minimum epochs
        if len(phase_history) >= MIN_EPOCHS[phase_number]:
            min_epochs_completed = True
            if phase_meets_criteria(phase_number, avg_reward, phase_history):
                print(f"\nPhase {phase_number} criteria met!")
                break

        # Early stopping only after minimum epochs
        if min_epochs_completed:
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience = 0
                # Save best model for this phase
                os.makedirs(PHASE_CHECKPOINTS_DIR, exist_ok=True)
                qtar.save_model(
                    f"{PHASE_CHECKPOINTS_DIR}/phase_{phase_number}_best.pt",
                    metadata={
                        'phase': phase_number,
                        'avg_reward': avg_reward,
                        'epoch': epoch,
                        'motif_count': len(qtar.env.motif_memory)
                    }
                )
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

    if not check_sustained_performance(training_history):
        return False

    # Phase-specific criteria
    if phase_number == 1:
        # Motif learning phase
        min_motifs_learned = 10  # Require at least 10 good motifs
        return (avg_reward > 50.0 and
                check_consecutive_windows(training_history, 3) and
                min_last_n_rewards(training_history, 100) > 30.0)

    else:  # Phase 2
        # Pattern development phase
        return (avg_reward > 100.0 and
                check_consecutive_windows(training_history, 3) and
                min_last_n_rewards(training_history, 100) > 75.0)

def check_sustained_performance(training_history, window_size=100):
    """Check if performance has been consistently good"""
    if len(training_history) < window_size:
        return False

    recent_rewards = [entry['avg_reward'] for entry in training_history[-window_size:]]
    avg_reward = np.mean(recent_rewards)
    stability = np.std(recent_rewards)
    min_reward = min(recent_rewards)

    return (avg_reward > 50.0 and
            stability < (avg_reward * 0.3) and
            min_reward > 25.0)


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
    """Pretrain through both phases sequentially"""
    all_phase_history = []

    # Initialize model in phase 1 (single chord motifs)
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        use_human_feedback=False,
        training_phase=1
    )

    try:
        # Train each phase
        for phase in range(1, 3):  # Only two phases now
            print(f"\n{'=' * 50}")
            print(f"Starting Phase {phase}")
            print("Learning motifs on single chord" if phase == 1
                  else "Developing patterns across progression")
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
                    'phase_history': phase_history,
                    'motif_memory_size': len(qtar.env.motif_memory)
                }
            )

            # Advance to next phase if not final
            if phase == 1:
                print("\nPhase 1 complete - learned motifs:")
                print(f"Total successful motifs: {len(qtar.env.motif_memory)}")
                qtar.current_phase = 2
                print("\nAdvancing to Phase 2: Pattern Development")

        # Save final pretrained model
        print("\nPretraining complete! Saving final model...")
        os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
        qtar.save_model(
            PRETRAINED_MODEL_PATH,
            metadata={
                'completed_phases': [1, 2],
                'ready_for_human_feedback': True,
                'total_motifs_learned': len(qtar.env.motif_memory)
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