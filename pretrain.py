import argparse

import numpy as np

from agent import Qtar
import os

from visualization import create_phase_training_visualization, save_epoch

PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"
PHASE_CHECKPOINTS_DIR = "models/phase_checkpoints"

EPOCHS = {1: 100,
          2: 100}


def train_single_phase(qtar, phase_number, episodes_per_epoch=100):
    """Train a single phase until performance criteria are met"""
    print(f"\nStarting Phase {phase_number} Training...")
    print("Phase 1: Learning motifs on single chord" if phase_number == 1
          else "Phase 2: Developing patterns across progression")

    best_avg_reward = float('-inf')
    patience = 0
    max_patience = 50
    phase_history = []
    epochs = EPOCHS[phase_number]
    min_epochs_completed = False

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} (Phase {phase_number})")

        training_history, _ = qtar.train_extensive(
            total_epochs=1,
            episodes_per_epoch=episodes_per_epoch
        )

        # Calculate metrics
        rewards = [entry['avg_reward'] for entry in training_history]
        avg_reward = np.mean(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        # story history
        phase_history.append({
            'epoch': epoch,
            'avg_reward': avg_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'phase': phase_number,
            'epsilon': qtar.epsilon  # Also track epsilon for visualization
        })
        print(f"Average Reward: {avg_reward:.2f}")
        # Save visualization every 100 epochs
        if (epoch + 1) % 25 == 0:
            filepath = save_epoch(phase_history, epoch + 1)
            print(f"Saved training visualization at epoch {epoch + 1} to {filepath}")
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


def pretrain(
        training_phase: int = 1,
        model_path=None
):
    """Pretrain through both phases sequentially"""
    all_phase_history = []

    # Initialize model in phase 1
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        use_human_feedback=False,
        training_phase=training_phase
    )
    # train
    try:

        if training_phase == 1:
            # Train phase 1
            print("\nPhase 1: Learning motifs on single chord")
            phase1_history = train_single_phase(qtar, 1)
            all_phase_history.extend(phase1_history)
            # Save phase 1 completion with motifs
            learned_motifs = len(qtar.env.motif_memory)
            qtar.save_model(f"{PHASE_CHECKPOINTS_DIR}/phase_1_complete.pt")
        else:
            # load pre-existing model
            # --training_phase 2 --model_path models/phase_2_start.pt
            qtar.load_model(model_path)
            learned_motifs = qtar.env.motif_memory

        # Advance to phase 2
        print(f"\nPhase 1 complete - learned {len(learned_motifs)} motifs")
        qtar.advance_phase()

        # Train phase 2
        print("\nPhase 2: Developing patterns from learned motifs")
        phase2_history = train_single_phase(qtar, 2)
        all_phase_history.extend(phase2_history)

        # Save final model
        qtar.save_model(
            PRETRAINED_MODEL_PATH,
            metadata={
                'completed_phases': [1, 2],
                'learned_motifs': learned_motifs,
                'ready_for_human_feedback': True
            }
        )
        create_phase_training_visualization(all_phase_history)
    except KeyboardInterrupt:
        print("\nPretraining interrupted by user")
        qtar.save_model(PRETRAINED_MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music with Q-tar')
    parser.add_argument('--model_path', type=str, default='models/pretrained_qtar_model.pt',
                        help='Path to saved model')
    parser.add_argument('--training_phase', type=int, default=1, help='Training phase')
    args = parser.parse_args()
    training_phase = args.training_phase
    model_path = args.model_path

    assert os.path.exists(args.model_path), f"Error: Model file not found at {args.model_path}"

    pretrain(training_phase=training_phase, model_path=model_path)