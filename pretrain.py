import argparse
import os
import numpy as np
from datetime import datetime
from collections import deque
from visualization import save_epoch
from agent import Qtar


class TrainingManager:
    def __init__(self,
                 base_dir="models",
                 checkpoint_frequency=5,
                 early_stopping_patience=10,
                 min_epochs=50):
        # saving info
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        self.pretrained_path = os.path.join(base_dir, "pretrained_qtar_model.pt")
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        self.min_epochs = min_epochs
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # init training history
        self.training_history = []
        #  config
        self.config = {
            'name': 'Melody Generation',
            'target_reward': 75.0,
            'max_epochs': 500,  # Increased from 200
            'episodes_per_epoch': 100,
            'stability_threshold': 6.0
        }

    def train(self, qtar):
        print(f"\nStarting Training: {self.config['name']}")
        # init history and rewards
        training_history = []
        best_reward = float('-inf')
        no_improvement = 0
        recent_rewards = deque(maxlen=20)
        # train loop
        for epoch in range(self.config['max_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['max_epochs']}")
            # train for one epoch
            epoch_metrics = self._train_epoch(qtar, self.config['episodes_per_epoch'])
            training_history.append(epoch_metrics)
            # monitoring metrics
            avg_reward = epoch_metrics['avg_reward']
            recent_rewards.append(avg_reward)
            reward_stability = np.std(list(recent_rewards)) if len(recent_rewards) >= 10 else float('inf')
            # save checkpoint and visualization
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self._save_checkpoint(qtar, epoch + 1, epoch_metrics)
                save_epoch(training_history, epoch + 1)
            # check if finished
            if self._check_completion(epoch, avg_reward, reward_stability):
                print("\nTraining completed successfully!")
                break
            # early stopping
            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improvement = 0
                self._save_best_model(qtar, epoch_metrics)
            else:
                no_improvement += 1
            if no_improvement >= self.early_stopping_patience and epoch >= self.min_epochs:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                qtar.load_model(self._get_best_model_path())
                break
        return training_history

    def _train_epoch(self, qtar, episodes):
        # train for one epoch
        episode_rewards = []
        episode_losses = []
        motif_metrics = []
        for episode in range(episodes):
            state = qtar.env.reset()
            total_reward = 0
            steps = 0
            while True:
                # get action
                note_action, rhythm_action = qtar.act(state)
                next_state, reward, done = qtar.env.step(note_action, rhythm_action)
                qtar.remember(state, (note_action, rhythm_action), reward, next_state, done)
                # replay once batch is finished
                if len(qtar.memory) >= qtar.batch_size:
                    loss = qtar.replay(qtar.batch_size)
                    episode_losses.append(loss)
                # update state
                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break
            # update rewards
            episode_rewards.append(total_reward)
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1, episodes, total_reward, steps, qtar.epsilon)
            # calc metrics
            if len(qtar.env.current_melody) > 0:
                motif_stats = self._calculate_motif_metrics(qtar.env)
                motif_metrics.append(motif_stats)
        # metrics dict
        metrics = {
            'epoch': len(self.training_history) + 1,  # Add explicit epoch number
            'avg_reward': np.mean(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'avg_loss': np.mean(episode_losses) if episode_losses else 0,
            'epsilon': qtar.epsilon,
            'motif_stats': np.mean(motif_metrics, axis=0) if motif_metrics else None,
            'note_entropy': self._calculate_note_entropy(qtar.env),
            'rhythm_entropy': self._calculate_rhythm_entropy(qtar.env)
        }
        return metrics

    def _calculate_motif_metrics(self, env):
        # calc motif specific metrics
        melody = env.current_melody
        notes = [n[0] for n in melody]
        rhythms = [n[1] for n in melody]
        note_variety = len(set(notes)) / len(notes)
        rhythm_variety = len(set(rhythms)) / len(rhythms)
        # Calculate motif coherence
        motif_coherence = 0
        if len(env.motif_memory) > 0:
            coherence_scores = [env._evaluate_motif_coherence(m) for m in env.motif_memory[-3:]]
            motif_coherence = np.mean(coherence_scores)
        return [note_variety, rhythm_variety, motif_coherence]

    def _calculate_note_entropy(self, env):
        # get entropy/randomness of notes
        if not env.note_history:
            return 0
        counts = np.bincount(env.note_history, minlength=24)
        probs = counts / len(env.note_history)
        return -np.sum(p * np.log(p + 1e-10) for p in probs if p > 0)

    def _calculate_rhythm_entropy(self, env):
        # get entropy/randomness of rhythm
        if not env.rhythm_history:
            return 0
        counts = np.bincount([env.rhythm_values.index(r) for r in env.rhythm_history])
        probs = counts / len(env.rhythm_history)
        return -np.sum(p * np.log(p + 1e-10) for p in probs if p > 0)

    def _check_completion(self, epoch, avg_reward, stability):
        # check if complete
        if epoch < self.min_epochs:
            return False
        return (avg_reward >= self.config['target_reward'] and
                stability <= self.config['stability_threshold'])

    def _save_checkpoint(self, qtar, epoch, metrics):
        # save training
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"epoch_{epoch}.pt"
        )
        qtar.save_model(checkpoint_path, metadata={
            'epoch': epoch,
            'metrics': metrics
        })

    def _save_best_model(self, qtar, metrics):
        best_model_path = self._get_best_model_path()
        qtar.save_model(best_model_path, metadata={
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    def _get_best_model_path(self):
        return os.path.join(self.base_dir, "best_model.pt")

    def _log_progress(self, episode, total_episodes, reward, steps, epsilon):
        print(f"Episode {episode}/{total_episodes}, "
              f"Reward: {reward:.2f}, Steps: {steps}, "
              f"Epsilon: {epsilon:.4f}")


def train(model_path=None):
    manager = TrainingManager()
    # init model
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        use_human_feedback=False
    )
    # load model
    if model_path:
        qtar.load_model(model_path)
    # train
    try:
        training_history = manager.train(qtar)
        # Save final model
        qtar.save_model(
            manager.pretrained_path,
            metadata={
                'training_history': training_history,
                'num_motifs': len(qtar.env.motif_memory),
                'ready_for_human_feedback': True
            }
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        qtar.save_model(manager.pretrained_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music with Q-tar')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    args = parser.parse_args()
    if args.model_path:
        assert os.path.exists(args.model_path), f"Model file not found at {args.model_path}"
    train(model_path=args.model_path)