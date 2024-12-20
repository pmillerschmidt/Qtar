import os

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from environment import QtarEnvironment
from model import QtarNetwork
from music_theory import PROGRESSIONS
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Qtar:
    def __init__(self,
                 scale='C_MAJOR',
                 progression_type='I_VI_IV_V',
                 beats_per_chord=4,
                 use_human_feedback=False,
                 batch_size=32,
                 learning_rate=0.0005,
                 noise_std=0.1):
        # init environment
        self.chord_progression = PROGRESSIONS[progression_type]
        self.env = QtarEnvironment(
            chord_progression=self.chord_progression,
            scale=scale,
            beats_per_chord=beats_per_chord,
            use_human_feedback=use_human_feedback)
        # base parameters
        self.state_size = len(self.env._get_state())
        self.note_size = 24
        self.rhythm_size = len(self.env.rhythm_values)
        self.batch_size = batch_size
        self.noise_std = noise_std
        # train parameters
        self.total_episodes = 0
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        # networks and optimization
        self.model = QtarNetwork(self.state_size, self.note_size, self.rhythm_size)
        self.target_model = QtarNetwork(self.state_size, self.note_size, self.rhythm_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        # progress tracking
        self.reward_history = deque(maxlen=100)
        self.motif_memory = []
        self.update_target_freq = 100
        self.steps = 0

    def remember(self, state, actions, reward, next_state, done):
        # Calculate TD error for priority
        with torch.no_grad():
            current_q = self._get_q_values(state, actions)
            next_q = self._get_next_q_values(next_state)
            td_error = abs(reward + (1 - done) * self.gamma * next_q - current_q)
        # Add experience to memory
        experience = (state, actions, reward, next_state, done)
        self.memory.append(experience)

    def _get_q_values(self, state, actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        note_values, rhythm_values = self.model(state_tensor)
        note_action, rhythm_action = actions
        # average Q-values from both heads
        return (note_values[0][note_action] + rhythm_values[0][rhythm_action]) / 2

    def _get_next_q_values(self, next_state):
        state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        # target network for stability
        note_values, rhythm_values = self.target_model(state_tensor)
        # max from both heads
        return (torch.max(note_values) + torch.max(rhythm_values)) / 2

    def act(self, state):
        # select action (explore vs. exploit)
        if random.random() <= self.epsilon:
            return self._explore(state)
        return self._exploit(state)

    def _explore(self):
        # explore with context
        progress = min(1.0, self.total_episodes / 1000)
        # early: focus on scale tones and basic rhythm
        if progress < 0.3:
            valid_notes = [n for n in range(24) if self.env.scale_mask[n % 12] == 1]
            note_action = random.choice(valid_notes)
            rhythm_action = random.randrange(self.rhythm_size)
        # mid: Consider voice leading and motifs
        elif progress < 0.7:
            if len(self.env.current_melody) > 0:
                last_note = self.env.current_melody[-1][0]
                valid_notes = [n for n in range(24)
                               if abs(n - last_note) <= 4 and self.env.scale_mask[n % 12] == 1]
                note_action = random.choice(valid_notes or range(24))
            else:
                note_action = random.randrange(24)
            rhythm_action = self._get_contextual_rhythm()
        # late: Use learned patterns (from earlier phases)
        else:
            if self.motif_memory and random.random() < 0.5:
                motif = random.choice(self.motif_memory)
                note_action = motif[0][0]
                rhythm_action = self.env.rhythm_values.index(motif[0][1])
            else:
                note_action = random.randrange(24)
                rhythm_action = self._get_contextual_rhythm()
        return note_action, rhythm_action

    def _exploit(self, state):
        # select action based on model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            note_values, rhythm_values = self.model(state_tensor)
            # add noise
            note_noise = torch.randn_like(note_values) * self.noise_std
            rhythm_noise = torch.randn_like(rhythm_values) * self.noise_std
            note_values += note_noise
            rhythm_values += rhythm_noise
            # get action
            note_action = torch.argmax(note_values).item()
            rhythm_action = torch.argmax(rhythm_values).item()
            return note_action, rhythm_action

    def _get_contextual_rhythm(self):
        # rhythm based on context
        if not self.env.current_melody:
            return random.randrange(self.rhythm_size)
        recent_rhythms = [n[1] for n in self.env.current_melody[-3:]]
        if len(recent_rhythms) >= 3 and len(set(recent_rhythms)) == 1:
            available_rhythms = list(range(self.rhythm_size))
            available_rhythms.remove(self.env.rhythm_values.index(recent_rhythms[0]))
            return random.choice(available_rhythms)
        return random.randrange(self.rhythm_size)

    def train(self, num_episodes):
        # train with curriculum
        epoch_history = []
        for episode in range(num_episodes):
            # reset the state and reward
            state = self.env.reset()
            total_reward = 0
            episode_steps = 0

            while True:
                note_action, rhythm_action = self.act(state)
                next_state, reward, done = self.env.step(note_action, rhythm_action)
                # store experience
                self.remember(state, (note_action, rhythm_action), reward, next_state, done)
                # once we've made enough actions -> replay
                if len(self.memory) >= self.batch_size:
                    self.replay(self.batch_size)
                # update state
                state = next_state
                total_reward += reward
                episode_steps += 1
                if done:
                    break
            # track progress
            self.total_episodes += 1
            self.reward_history.append(total_reward)
            # update network
            if self.steps % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            # update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(list(self.reward_history))
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}")
                # save good motifs
                if avg_reward > 50 and len(self.env.current_melody) >= 4:
                    self._store_good_motif(self.env.current_melody)
        return epoch_history


    def _store_good_motif(self, melody):
        # save motifs for future comparison
        if len(melody) >= 4:
            motif = melody[-4:]
            if not any(self._is_similar_motif(motif, m) for m in self.motif_memory):
                self.motif_memory.append(motif)

    def _is_similar_motif(self, motif1, motif2):
        # get if motifs are the same
        if len(motif1) != len(motif2):
            return False
        notes1 = [n[0] for n in motif1]
        notes2 = [n[0] for n in motif2]
        intervals1 = [b - a for a, b in zip(notes1[:-1], notes1[1:])]
        intervals2 = [b - a for a, b in zip(notes2[:-1], notes2[1:])]

        return intervals1 == intervals2

    def replay(self, batch_size):
        # update network with double Q
        if len(self.memory) < batch_size:
            return 0
        # get batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        # current Q values
        current_note_q, current_rhythm_q = self.model(states)
        # next Q values from target network
        with torch.no_grad():
            next_note_q, next_rhythm_q = self.target_model(next_states)
            max_next_note_q = torch.max(next_note_q, dim=1)[0]
            max_next_rhythm_q = torch.max(next_rhythm_q, dim=1)[0]
        # calc targets
        note_targets = current_note_q.clone()
        rhythm_targets = current_rhythm_q.clone()
        # for sample in batch, get actions + targets and update
        for i in range(batch_size):
            note_action, rhythm_action = actions[i]
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * (max_next_note_q[i] + max_next_rhythm_q[i]) / 2
            note_targets[i][note_action] = target
            rhythm_targets[i][rhythm_action] = target
        # loss and update
        note_loss = nn.MSELoss()(current_note_q, note_targets)
        rhythm_loss = nn.MSELoss()(current_rhythm_q, rhythm_targets)
        total_loss = note_loss + rhythm_loss
        # backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        # update target network (if needed)
        if self.steps % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return total_loss.item()

    def save_model(self, filepath, metadata=None):
        # save model weights and data
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert numpy arrays to lists for serialization
        note_history = self.env.note_history.tolist() if isinstance(self.env.note_history,
                                                                    np.ndarray) else self.env.note_history
        rhythm_history = self.env.rhythm_history.tolist() if isinstance(self.env.rhythm_history,
                                                                        np.ndarray) else self.env.rhythm_history
        # state
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'memory_size': len(self.memory),
            'note_history': note_history,
            'rhythm_history': rhythm_history,
            'motif_memory': self.env.motif_memory
        }
        if metadata:
            model_state['metadata'] = metadata
        # save
        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")
        try:
            # First try loading with weights_only=True for safety
            checkpoint = torch.load(filepath, weights_only=False, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint.get('steps', 0)
            # load env state if available
            if 'note_history' in checkpoint:
                self.env.note_history = checkpoint['note_history']
            if 'rhythm_history' in checkpoint:
                self.env.rhythm_history = checkpoint['rhythm_history']
            if 'motif_memory' in checkpoint:
                self.env.motif_memory = checkpoint['motif_memory']
            print(f"Loaded model with {len(self.env.motif_memory)} motifs")
            # Return metadata if present
            return checkpoint.get('metadata', None)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting to load with reduced security restrictions...")
            # If that fails, try loading with weights_only=False
            checkpoint = torch.load(filepath, weights_only=False, map_location='cpu')
            # load model components
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint.get('steps', 0)
            print(f"Successfully loaded model components.")
            return checkpoint.get('metadata', None)

    def generate_solo(self, temperature=1.0):
        # generate a single solo with temp
        state = self.env.reset()
        melody = []
        while True:
            # get action logits
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                note_logits, rhythm_logits = self.model(state_tensor)
                # apply temperature
                note_logits = note_logits / temperature
                rhythm_logits = rhythm_logits / temperature
                # Convert -> probabilities
                note_probs = torch.softmax(note_logits, dim=1)
                rhythm_probs = torch.softmax(rhythm_logits, dim=1)
                # sample actions
                note_action = torch.multinomial(note_probs[0], 1).item()
                rhythm_action = torch.multinomial(rhythm_probs[0], 1).item()
            # step
            next_state, _, done = self.env.step(note_action, rhythm_action)
            # add new note
            melody.append((note_action, self.env.rhythm_values[rhythm_action], self.env.current_beat))
            if done:
                break
            state = next_state
        return melody
