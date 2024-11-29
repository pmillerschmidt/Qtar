from agent import Qtar
import os
import matplotlib.pyplot as plt

from visualization import smooth_curve, create_training_visualization

PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"


def pretrain():
    # Initialize model
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        use_human_feedback=False,
        training_phase=1
    )
    # Training parameters
    total_epochs = 100
    episodes_per_epoch = 100
    print("Starting pretraining...")
    try:
        # Single call to train_extensive
        training_history, phase_history = qtar.train_extensive(
            total_epochs=total_epochs,
            episodes_per_epoch=episodes_per_epoch
        )
        print("\nTraining complete! Saving model and plots...")
        # Save final model
        os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
        qtar.save_model(PRETRAINED_MODEL_PATH)
        print(f"Model saved to {PRETRAINED_MODEL_PATH}")

        # Create visualization with phases
        create_training_visualization(training_history, phase_history)

    except KeyboardInterrupt:
        print("\nPretraining interrupted by user")
        # Save model if interrupted
        os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
        qtar.save_model(PRETRAINED_MODEL_PATH)
        print(f"Model saved to {PRETRAINED_MODEL_PATH}")


if __name__ == "__main__":
    pretrain()