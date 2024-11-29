from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from agent import Qtar
import threading
import queue
import os

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

# Initialize model
chord_progression = ['C', 'Am', 'F', 'G']
qtar = Qtar(chord_progression, use_human_feedback=True)

# Load model if exists
PRETRAINED_MODEL_PATH =  "models/pretrained_qtar_model.pt"
MODEL_PATH = "models/trained_qtar_model.pt"
if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading existing model from {PRETRAINED_MODEL_PATH}")
    qtar.load_model(PRETRAINED_MODEL_PATH)
else:
    print("No existing model found, starting fresh")


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
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note % 12]
            notes.append({
                "note": note_name,
                "duration": duration,
                "beat": current_beat
            })
            current_beat += duration

        return {
            "status": "ready",
            "chords": chord_progression,
            "notes": notes
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/submit-feedback")
async def submit_feedback(feedback: dict):
    """Submit feedback for the current solo"""
    try:
        rating = feedback["rating"]
        # Add feedback to the environment's buffer
        qtar.env.human_feedback.add_feedback(get_current_solo.current_solo, rating)
        feedback_queue.put(rating)
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train")
async def start_training():
    try:
        print("Starting training episode...")
        # Generate a solo for feedback first
        solo = qtar.generate_solo(chord_progression)
        get_current_solo.current_solo = solo

        # Do a small training step
        qtar.train_extensive(total_epochs=1, episodes_per_epoch=1)

        # Generate new solo after training
        new_solo = qtar.generate_solo(chord_progression)
        get_current_solo.current_solo = new_solo

        return {
            "status": "success",
            "message": "Training completed",
            "current_stats": {
                "epsilon": qtar.epsilon,
                "memory_size": len(qtar.memory)
            }
        }
    except Exception as e:
        print(f"Training error: {str(e)}")
        return {"status": "error", "message": str(e)}

def train_model():
    """Main training loop"""
    total_epochs = 50  # Changed from 100
    episodes_per_epoch = 20  # Changed from 50

    try:
        for epoch in range(total_epochs):
            print(f"\nEpoch {epoch + 1}/{total_epochs}")

            for episode in range(episodes_per_epoch):
                # Every 10 episodes, get human feedback
                if episode % 10 == 0:
                    # Generate a solo for feedback
                    solo = qtar.generate_solo(chord_progression)
                    get_current_solo.current_solo = solo

                    print(f"\nWaiting for human feedback on episode {episode}...")
                    # Wait for feedback
                    rating = feedback_queue.get()
                    print(f"Received feedback: {rating}")

                    # Save model after receiving feedback
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    metadata = {
                        'epoch': epoch,
                        'episode': episode,
                        'feedback_count': len(qtar.env.human_feedback.buffer)
                    }
                    qtar.save_model(MODEL_PATH, metadata=metadata)

                # Regular training step
                qtar.train_extensive(total_epochs=1, episodes_per_epoch=1)

            print(f"Completed epoch {epoch + 1}")

    except Exception as e:
        print(f"Training error: {str(e)}")


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=5001)


if __name__ == "__main__":
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True  # This ensures the thread will shut down with the main program
    server_thread.start()

    # Run training in main thread
    print("Starting training... Open http://localhost:3000 to provide feedback")
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        # Save model before exiting
        if os.path.exists(os.path.dirname(MODEL_PATH)):
            qtar.save_model(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")