import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from agent import Qtar
import threading
import queue
import os
import numpy as np
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

# Initialize Qtar with human feedback
qtar = Qtar(
    scale='C_MAJOR',
    progression_type='I_VI_IV_V',
    use_human_feedback=True
)

# Load pretrained model
PRETRAINED_MODEL_PATH = "models/pretrained_qtar_model.pt"
MODEL_PATH = "models/trained_qtar_model.pt"

@app.get("/get-training-info")
async def get_training_info():
    """Get current training information"""
    return {
        "learned_motifs": len(qtar.env.motif_memory),
        "training_mode": "Melody Generation with Human Feedback",
        "note_entropy": qtar.env._calculate_entropy_bonus(0, 0),  # Get current entropy
        "total_episodes": qtar.total_episodes
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
            octave = note // 12
            note_in_octave = note % 12
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note_in_octave]

            notes.append({
                "note": f"{note_name}{octave + 4}",
                "duration": duration,
                "beat": current_beat
            })
            current_beat += duration

        # Calculate musical metrics
        metrics = analyze_solo(solo)

        return {
            "status": "ready",
            "chords": qtar.chord_progression,
            "notes": notes,
            "metrics": metrics,
            "motifs_used": len(qtar.env.motif_memory)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/submit-feedback")
async def submit_feedback(feedback: dict):
    """Submit feedback for the current solo"""
    try:
        rating = feedback["rating"]
        comments = feedback.get("comments", "")

        # Store feedback with additional context
        context = {
            "timestamp": datetime.now().isoformat(),
            "training_progress": qtar.total_episodes,
            "comments": comments
        }

        qtar.env.human_feedback.add_feedback(get_current_solo.current_solo, rating, context)
        feedback_queue.put((rating, context))

        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train")
async def start_training():
    try:
        print("Training with human feedback...")
        solo = qtar.generate_solo(temperature=1.0)  # Add temperature for exploration
        get_current_solo.current_solo = solo

        # Train with curriculum learning
        progress = min(1.0, qtar.total_episodes / 1000)
        exploration_temp = max(0.5, 1.0 - progress * 0.5)  # Decrease temperature over time

        qtar.train(num_episodes=1)
        new_solo = qtar.generate_solo(temperature=exploration_temp)
        get_current_solo.current_solo = new_solo

        return {
            "status": "success",
            "message": "Training completed",
            "current_stats": {
                "epsilon": qtar.epsilon,
                "memory_size": len(qtar.memory),
                "motifs_available": len(qtar.env.motif_memory),
                "note_entropy": qtar.env._calculate_entropy_bonus(0, 0),
                "exploration_temperature": exploration_temp
            }
        }
    except Exception as e:
        print(f"Training error: {str(e)}")
        return {"status": "error", "message": str(e)}


def analyze_solo(solo):
    """Analyze musical characteristics of a solo"""
    notes = [note for note, _, _ in solo]
    durations = [duration for _, duration, _ in solo]

    # Calculate metrics
    note_variety = len(set(notes)) / len(notes)
    rhythm_variety = len(set(durations)) / len(durations)

    # Calculate melodic contour
    intervals = [b - a for a, b in zip(notes[:-1], notes[1:])]
    direction_changes = sum(1 for i in range(len(intervals) - 1)
                            if intervals[i] * intervals[i + 1] < 0)

    return {
        "note_variety": note_variety,
        "rhythm_variety": rhythm_variety,
        "direction_changes": direction_changes,
        "avg_interval": np.mean(np.abs(intervals)) if intervals else 0
    }


def train_model():
    """Main training loop with human feedback"""
    try:
        episode = 0
        while True:
            # Generate and get feedback every 5 episodes
            if episode % 5 == 0:
                progress = min(1.0, qtar.total_episodes / 1000)
                temperature = max(0.5, 1.0 - progress * 0.5)

                solo = qtar.generate_solo(temperature=temperature)
                get_current_solo.current_solo = solo
                print(f"\nWaiting for human feedback (Episode {episode})...")

                rating, context = feedback_queue.get()
                print(f"Received feedback: {rating}")

                # Save progress
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                metadata = {
                    'episode': episode,
                    'feedback_count': len(qtar.env.human_feedback.buffer),
                    'motifs_available': len(qtar.env.motif_memory),
                    'note_entropy': qtar.env._calculate_entropy_bonus(0, 0),
                    'temperature': temperature
                }
                qtar.save_model(MODEL_PATH, metadata=metadata)

            # Regular training step
            qtar.train(num_episodes=1)
            episode += 1

            if episode % 50 == 0:
                print(f"\nCompleted {episode} episodes")
                print(f"Available motifs: {len(qtar.env.motif_memory)}")
                print(f"Note entropy: {qtar.env._calculate_entropy_bonus(0, 0):.3f}")

    except Exception as e:
        print(f"Training error: {str(e)}")


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=5001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', action='store_true')
    args = parser.parse_args()
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    model_path = PRETRAINED_MODEL_PATH if args.from_pretrained else MODEL_PATH

    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        metadata = qtar.load_model(model_path)
        print(f"Loaded model with {len(qtar.env.motif_memory)} learned motifs")
    else:
        raise ValueError("No pretrained model found. Please run pretraining first.")

    print("Starting training with human feedback...")
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