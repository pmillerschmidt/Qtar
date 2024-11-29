import numpy as np
from collections import defaultdict
from music_theory import CHORD_TONES, SCALE_DEGREES
from feedback import HumanFeedbackBuffer


class QtarEnvironment:
    def __init__(self, chord_progression, scale='C', beats_per_chord=4, use_human_feedback=False):
        self.chord_progression = chord_progression
        self.beats_per_chord = beats_per_chord
        self.use_human_feedback = use_human_feedback
        self.current_position = 0
        self.current_beat = 0

        # histories
        self.solo_history = []  # Stores the history of notes in the solo
        self.rhythm_history = []  # Stores the history of rhythm patterns in the solo
        self.motif_history = []  # Stores motifs for comparison

        self.rhythm_values = [
            0.25,  # Sixteenth note
            0.5,  # Eighth note
            0.75,  # Dotted eighth note
            1.0,  # Quarter note
            1.5,  # Dotted quarter note
            2.0,  # Half note
        ]

        self.scale_tones = SCALE_DEGREES[scale]
        self.motif_memory = defaultdict(int)
        self.human_feedback = HumanFeedbackBuffer() if use_human_feedback else None
        self.reset()

    def reset(self):
        self.current_position = 0
        self.current_beat = 0
        self.beat_history = []
        self.current_melody = []  # (note, duration, beat_position)
        self.solo_history = []
        self.rhythm_history = []
        self.motif_history = []
        self.motif_memory.clear()

        return self._get_state()

    def _get_state(self):
        current_chord = self.chord_progression[self.current_position]

        state = [
            *self._one_hot_encode_chord(current_chord),
            self.current_beat / self.beats_per_chord,  # Normalized beat position
            self._is_strong_beat(),
            self._get_current_phrase_position() / 16,  # Normalized phrase position
        ]

        melody_context = []
        for i in range(min(4, len(self.current_melody))):
            note, duration, _ = self.current_melody[-(i + 1)]
            melody_context.extend([note % 12 / 12, duration / 2])

        while len(melody_context) < 8:  # 4 notes * 2 values each
            melody_context.extend([0, 0])

        return np.array(state + melody_context)

    def _one_hot_encode_chord(self, chord):
        chord_vocab = ['C', 'Am', 'F', 'G']
        encoding = [0] * len(chord_vocab)
        encoding[chord_vocab.index(chord)] = 1
        return encoding

    def _get_current_phrase_position(self):
        total_beats = sum(duration for _, duration, _ in self.current_melody)
        return total_beats % 16

    def _is_strong_beat(self):
        return self.current_beat % 2 == 0

    def _get_melodic_direction(self, window=3):
        if len(self.current_melody) < window:
            return 0
        recent_notes = [note for note, _, _ in self.current_melody[-window:]]
        return sum(b - a for a, b in zip(recent_notes[:-1], recent_notes[1:])) / (window - 1)

    def step(self, note_action, rhythm_action):
        current_chord = self.chord_progression[self.current_position]
        duration = self.rhythm_values[rhythm_action]

        beats_remaining = self.beats_per_chord - self.current_beat
        if duration > beats_remaining:
            duration = beats_remaining

        reward = self._calculate_reward(note_action, rhythm_action, current_chord)

        progression_bonus = self._calculate_progression_bonus()
        total_reward = self._normalize_reward(reward + progression_bonus)

        self.current_melody.append((note_action, duration, self.current_beat))

        self.current_beat += duration
        if self.current_beat >= self.beats_per_chord:
            self.current_beat = 0
            self.current_position = (self.current_position + 1) % len(self.chord_progression)

        done = self.current_position == 0 and self.current_beat == 0
        return self._get_state(), total_reward, done

    def _normalize_reward(self, raw_reward):
        MAX_REWARD = 10.0
        return np.clip(raw_reward, -MAX_REWARD, MAX_REWARD)

    def _calculate_progression_bonus(self):
        current_sequence = self.current_melody[-8:] if len(self.current_melody) >= 8 else self.current_melody
        if len(current_sequence) >= 4:
            intervals = [abs(b[0] - a[0]) for a, b in zip(current_sequence[:-1], current_sequence[1:])]
            if max(intervals) <= 4:
                return 2.0
            rhythms = [note[1] for note in current_sequence]
            if len(set(rhythms)) <= 2:
                return 1.5
        return 0.0

    def _calculate_reward(self, note, rhythm, chord):
        base_reward = self._calculate_base_reward(note, rhythm, chord)

        if self.use_human_feedback and self.human_feedback and len(self.current_melody) >= 4:
            recent_sequence = self.current_melody[-4:]
            similar_melodies = self.human_feedback.get_similar_melodies(recent_sequence)
            if similar_melodies:
                weighted_feedback = sum(sim * rating for sim, _, rating in similar_melodies)
                total_sim = sum(sim for sim, _, _ in similar_melodies)
                if total_sim > 0:
                    human_reward = weighted_feedback / total_sim
                    return 0.7 * base_reward + 0.3 * human_reward

        return base_reward

    def _calculate_base_reward(self, note, rhythm, chord):
        reward = 0
        reward += self._reward_chord_tone(note, chord)
        reward += self._reward_scale_tone(note)
        reward += self._penalize_repetition(note)
        reward += self._reward_rhythmic_variation(rhythm)
        reward += self._reward_melodic_direction()
        return reward

    def _reward_chord_tone(self, note, chord):
        chord_tones = CHORD_TONES[chord]
        note_in_octave = note % 12
        if note_in_octave in chord_tones:
            return 4.0 if self._is_strong_beat() else 2.0
        return 0

    def _reward_scale_tone(self, note):
        scale_reward = 1 if note % 12 in self.scale_tones else -0.5
        return scale_reward

    def _penalize_repetition(self, note):
        if len(self.current_melody) > 0 and note == self.current_melody[-1][0]:
            return -2.0
        return 0

    def _reward_melodic_direction(self):
        # Get the melodic direction for the last 3 notes (adjustable window)
        direction = self._get_melodic_direction(window=3)
        if direction > 0:
            return 1  # Reward ascending direction
        elif direction < 0:
            return 1  # Reward descending direction
        else:
            return -0.5  # Neutral reward for flat direction


    def _reward_melodic_variation(self, note):
        motif_length = 4  # Adjust this length for motif detection
        current_motif = [note]  # Start with the current note (could include previous notes)
        if len(self.solo_history) >= motif_length:
            current_motif = self.solo_history[-motif_length:]
        for motif in self.motif_history:
            if self._is_motif_transformation(motif, current_motif):
                # Reward if the motif is a transformation of an earlier motif
                return 1  # Reward for transformation
        self.motif_history.append(current_motif)
        if current_motif in self.motif_history[:-1]:
            return -0.5  # Penalty for repetition
        self.solo_history.append(note)  # Add note to history
        return 1  # Reward for melodic variation and motif transformation

    def _reward_rhythmic_variation(self, rhythm):
        # Penalize if rhythm is too repetitive
        if rhythm in self.rhythm_history:
            return -0.5  # Penalty for repeated rhythm
        else:
            self.rhythm_history.append(rhythm)  # Add the new rhythm to the history
            return 1  # Reward for rhythmic variation

    def _is_motif_transformation(self, motif1, motif2):
        if len(motif1) != len(motif2):
            return False
        intervals1 = [b - a for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b - a for a, b in zip(motif2[:-1], motif2[1:])]
        if intervals1 == intervals2 or intervals1 == [-x for x in reversed(intervals2)]:
            return True
        if motif1 == tuple(reversed(motif2)):
            return True
        return False
