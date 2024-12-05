from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from agent import Qtar
import threading
import queue
import os
import json
from datetime import datetime

# Create a queue for feedback
feedback_queue = queue.Queue()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize in phase 2 for pattern training with human feedback
qtar = Qtar(
    scale='C_MAJOR',
    progression_type='I_VI_IV_V',
    use_human_feedback=True,
    training_phase=2  # Phase 2 for pattern development
)

# Load pretrained model and ensure phase 1 was completed
PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"
MODEL_PATH = "models/trained_qtar_model.pt"

if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
    metadata = qtar.load_model(PRETRAINED_MODEL_PATH)
    if metadata and 'completed_phases' in metadata:
        if 1 not in metadata['completed_phases']:
            raise ValueError("Pretrained model hasn't completed motif learning phase")
        print(f"Loaded model with {metadata.get('total_motifs_learned', 0)} learned motifs")
    qtar.current_phase = 2  # Ensure we're in phase 2
else:
    raise ValueError("No pretrained model found. Please run pretraining first.")


@app.get("/get-phase-info")
async def get_phase_info():
    """Get current training phase information"""
    motif_count = len(qtar.env.motif_memory) if hasattr(qtar.env, 'motif_memory') else 0
    return {
        "current_phase": 2,
        "phase_description": "Pattern Development Phase",
        "learned_motifs": motif_count,
        "training_mode": "Learning to create ABAB/AABB patterns from motifs"
    }


@app.get("/get-current-solo")
async def get_current_solo():
    """Get the current solo that needs feedback"""
    if not hasattr(get_current_solo, 'current_solo'):
        return {"status": "waiting", "message": "No solo available for feedback yet"}
    try:
        solo = get_current_solo.current_solo
        notes = []
        current_beat = 0
        for note, duration, _ in solo:
            # Handle two octaves
            octave = note // 12  # 0 or 1 for lower/upper octave
            note_in_octave = note % 12
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note_in_octave]
            notes.append({
                # TODO: fix feedback.tsx
                "note": f"{note_name}{octave + 4}",  # Octave 4 and 5
                # "note": f"{note_name}",  # Octave 4 and 5
                "duration": duration,
                "beat": current_beat
            })
            current_beat += duration
        return {
            "status": "ready",
            "chords": qtar.chord_progression,
            "notes": notes,
            "phase": "Pattern Development",
            "motifs_used": len(qtar.env.motif_memory)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/submit-feedback")
async def submit_feedback(feedback: dict):
    """Submit feedback for the current solo"""
    try:
        rating = feedback["rating"]
        qtar.env.human_feedback.add_feedback(get_current_solo.current_solo, rating)
        feedback_queue.put(rating)
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train")
async def start_training():
    try:
        print("Starting pattern development training...")
        solo = qtar.generate_solo()
        get_current_solo.current_solo = solo
        qtar.train_extensive(total_epochs=1, episodes_per_epoch=1)
        new_solo = qtar.generate_solo()
        get_current_solo.current_solo = new_solo

        # Include motif information in stats
        return {
            "status": "success",
            "message": "Training completed",
            "current_stats": {
                "epsilon": qtar.epsilon,
                "memory_size": len(qtar.memory),
                "motifs_available": len(qtar.env.motif_memory)
            }
        }
    except Exception as e:
        print(f"Training error: {str(e)}")
        return {"status": "error", "message": str(e)}


def train_model():
    """Main training loop for pattern development"""
    total_epochs = 50
    episodes_per_epoch = 20
    try:
        for epoch in range(total_epochs):
            print(f"\nEpoch {epoch + 1}/{total_epochs} (Pattern Development)")

            for episode in range(episodes_per_epoch):
                # Get human feedback every 10 episodes
                if episode % 10 == 0:
                    solo = qtar.generate_solo()
                    get_current_solo.current_solo = solo
                    print(f"\nWaiting for human feedback on pattern development...")
                    rating = feedback_queue.get()
                    print(f"Received feedback: {rating}")

                    # Save progress
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    metadata = {
                        'epoch': epoch,
                        'episode': episode,
                        'feedback_count': len(qtar.env.human_feedback.buffer),
                        'motifs_available': len(qtar.env.motif_memory)
                    }
                    qtar.save_model(MODEL_PATH, metadata=metadata)

                # Regular training step
                qtar.train_extensive(total_epochs=1, episodes_per_epoch=1)

            print(f"Completed epoch {epoch + 1}")
            print(f"Available motifs: {len(qtar.env.motif_memory)}")

    except Exception as e:
        print(f"Training error: {str(e)}")


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=5001)


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    print("Starting pattern development training...")
    print("Open http://localhost:3000 to provide feedback")
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        if os.path.exists(os.path.dirname(MODEL_PATH)):
            qtar.save_model(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")