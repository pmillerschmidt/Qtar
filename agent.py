import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from environment import QtarEnvironment
from model import QtarNetwork
from visualization import TrainingVisualizer
from music_theory import PROGRESSIONS

class Qtar:
    def __init__(self,
                 scale='C_MAJOR',
                 progression_type='I_VI_IV_V',
                 beats_per_chord=4,
                 use_human_feedback=False):
        self.scale = scale
        self.chord_progression = PROGRESSIONS[progression_type]
        self.env = QtarEnvironment(
            chord_progression=self.chord_progression,
            scale=scale,
            beats_per_chord=beats_per_chord,
            use_human_feedback=use_human_feedback)
        self.state_size = len(self.env._get_state())
        self.note_size = 12  # 12 semitones
        self.rhythm_size = len(self.env.rhythm_values)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0005
        self.model = QtarNetwork(self.state_size, self.note_size, self.rhythm_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.training_history = []
        self.visualizer = TrainingVisualizer()

    def save_model(self, filepath, metadata=None):
        """Save model weights and training metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_history': self.training_history
        }

        if metadata:
            model_state['metadata'] = metadata

        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights and training metadata"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")

        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint.get('training_history', [])

        print(f"Model loaded from {filepath}")
        return checkpoint.get('metadata', None)

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            # For random actions, only choose from scale tones
            valid_notes = [i for i in range(12) if self.env.scale_mask[i] == 1]
            note_action = random.choice(valid_notes)
            rhythm_action = random.randrange(self.rhythm_size)
            return note_action, rhythm_action

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            note_values, rhythm_values = self.model(state_tensor)
            note_action = torch.argmax(note_values).item()
            rhythm_action = torch.argmax(rhythm_values).item()
            return note_action, rhythm_action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, (note_action, rhythm_action), reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    next_note_values, next_rhythm_values = self.model(next_state_tensor)
                    target = reward + self.gamma * (torch.max(next_note_values).item() +
                                                    torch.max(next_rhythm_values).item()) / 2

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self.optimizer.zero_grad()
            note_outputs, rhythm_outputs = self.model(state_tensor)

            target_note = note_outputs.clone()
            target_rhythm = rhythm_outputs.clone()
            target_note[0][note_action] = target
            target_rhythm[0][rhythm_action] = target

            loss = nn.MSELoss()(note_outputs, target_note) + nn.MSELoss()(rhythm_outputs, target_rhythm)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_extensive(self, total_epochs, episodes_per_epoch=100):
        best_reward = float('-inf')
        patience_counter = 0

        """Train the model over multiple epochs"""
        for epoch in range(total_epochs):
            epoch_rewards = []
            print(f"\nEpoch {epoch + 1}/{total_epochs}")

            for episode in range(episodes_per_epoch):
                state = self.env.reset()
                total_reward = 0
                steps = 0

                while True:
                    note_action, rhythm_action = self.act(state)
                    next_state, reward, done = self.env.step(note_action, rhythm_action)
                    self.remember(state, (note_action, rhythm_action), reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1

                    if done:
                        break

                self.replay(32)
                epoch_rewards.append(total_reward)

                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}/{episodes_per_epoch}, "
                          f"Reward: {total_reward:.2f}, Steps: {steps}, "
                          f"Epsilon: {self.epsilon:.4f}")


            # Calculate epoch statistics
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            max_reward = max(epoch_rewards)
            min_reward = min(epoch_rewards)

            epoch_stats = {
                'epoch': epoch + 1,
                'avg_reward': avg_reward,
                'max_reward': max_reward,
                'min_reward': min_reward,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory)
            }

            self.training_history.append(epoch_stats)

            print(f"\nEpoch {epoch + 1} Statistics:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Max Reward: {max_reward:.2f}")
            print(f"Min Reward: {min_reward:.2f}")
            print(f"Epsilon: {self.epsilon:.4f}")

            # Save visualization every 100 epochs
            if (epoch + 1) % 100 == 0:
                filepath = self.visualizer.save_epoch(self.training_history, epoch + 1)
                print(f"Saved training visualization at epoch {epoch + 1} to {filepath}")

            # Learning rate scheduling
            self.scheduler.step(avg_reward)

            # Save best model and check early stopping
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_model('models/best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 50:  # Early stopping
                print("Early stopping triggered")
                break

        return self.training_history

    def generate_solo(self):
        """Generate a solo over the given chord progression"""
        self.env = QtarEnvironment(scale=self.scale, chord_progression=self.chord_progression)
        state = self.env.reset()
        melody = []

        while True:
            note_action, rhythm_action = self.act(state)
            next_state, _, done = self.env.step(note_action, rhythm_action)
            melody.append((note_action, self.env.rhythm_values[rhythm_action], self.env.current_beat))
            if done:
                break
            state = next_state

        return melody