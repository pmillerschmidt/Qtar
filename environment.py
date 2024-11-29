import torch
import numpy as np
from collections import defaultdict
from music_theory import CHORD_TONES, SCALE_DEGREES, SCALE_MASKS
from feedback import HumanFeedbackBuffer


class QtarEnvironment:
    def __init__(self,
                 chord_progression,
                 scale='C_MAJOR',
                 beats_per_chord=4,
                 use_human_feedback=False):
        self.chord_progression = chord_progression
        self.beats_per_chord = beats_per_chord
        self.use_human_feedback = use_human_feedback
        self.current_position = 0
        self.current_beat = 0
        self.scale_mask = SCALE_MASKS[scale]

        self.rhythm_values = [
            0.25,  # Sixteenth note
            0.5,  # Eighth note
            0.75,  # Dotted eighth note
            1.0,  # Quarter note
            1.5,  # Dotted quarter note
            2.0  # Half note
        ]

        self.motif_memory = defaultdict(int)

        # Initialize human feedback if enabled
        self.human_feedback = HumanFeedbackBuffer() if use_human_feedback else None

        self.reset()

    def reset(self):
        self.current_position = 0
        self.current_beat = 0
        self.current_melody = []  # List of (note, duration, beat_position) tuples
        self.motif_memory.clear()
        return self._get_state()

    def _get_state(self):
        current_chord = self.chord_progression[self.current_position]

        # Include more musical context in state
        state = [
            *self._one_hot_encode_chord(current_chord),
            self.current_beat / self.beats_per_chord,  # Normalized beat position
            self._is_strong_beat(),
            self._get_current_phrase_position() / 16,  # Normalized phrase position
        ]

        # Add recent melodic context
        melody_context = []
        for i in range(min(4, len(self.current_melody))):
            note, duration, _ = self.current_melody[-(i + 1)]
            melody_context.extend([
                note % 12 / 12,  # Normalized note
                duration / 2,  # Normalized duration
            ])

        # Pad if needed
        while len(melody_context) < 8:  # 4 notes * 2 values each
            melody_context.extend([0, 0])

        return np.array(state + melody_context)

    def _one_hot_encode_chord(self, chord):
        chord_vocab = ['C', 'Am', 'F', 'G']
        encoding = [0] * len(chord_vocab)
        encoding[chord_vocab.index(chord)] = 1
        return encoding

    def _get_current_phrase_position(self):
        """Track position within 4-bar phrase"""
        total_beats = sum(duration for _, duration, _ in self.current_melody)
        return total_beats % 16  # 16 beats in 4 bars

    def _is_strong_beat(self):
        """Check if current beat is 1 or 3 in 4/4 time"""
        phrase_position = self._get_current_phrase_position()
        return phrase_position % 2 == 0

    def _get_melodic_direction(self, window=3):
        """Get recent melodic direction"""
        if len(self.current_melody) < window:
            return 0
        recent_notes = [note for note, _, _ in self.current_melody[-window:]]
        return sum(b - a for a, b in zip(recent_notes[:-1], recent_notes[1:])) / (window - 1)

    def step(self, note_action, rhythm_action):
        current_chord = self.chord_progression[self.current_position]
        duration = self.rhythm_values[rhythm_action]

        # Check if note duration fits within remaining beats
        beats_remaining = self.beats_per_chord - self.current_beat
        if duration > beats_remaining:
            duration = beats_remaining

        # Calculate reward (includes human feedback if enabled)
        reward = self._calculate_reward(note_action, rhythm_action, current_chord)

        # Add progression bonus
        progression_bonus = self._calculate_progression_bonus()

        # Combine and normalize rewards
        total_reward = self._normalize_reward(reward + progression_bonus)


        # Add note to melody
        self.current_melody.append((note_action, duration, self.current_beat))

        # Update beat position
        self.current_beat += duration
        if self.current_beat >= self.beats_per_chord:
            self.current_beat = 0
            self.current_position = (self.current_position + 1) % len(self.chord_progression)

        # Check if episode is done (one complete progression)
        done = self.current_position == 0 and self.current_beat == 0

        return self._get_state(), total_reward, done

    def _normalize_reward(self, raw_reward):
        """Normalize rewards to prevent explosion/vanishing"""
        MAX_REWARD = 10.0
        return np.clip(raw_reward, -MAX_REWARD, MAX_REWARD)

    def _calculate_progression_bonus(self):
        # Reward longer coherent phrases
        current_sequence = self.current_melody[-8:] if len(self.current_melody) >= 8 else self.current_melody
        if len(current_sequence) >= 4:
            # Check for consistent intervals
            intervals = [abs(b[0] - a[0]) for a, b in zip(current_sequence[:-1], current_sequence[1:])]
            if max(intervals) <= 4:  # All intervals are small
                return 2.0

            # Check for consistent rhythm
            rhythms = [note[1] for note in current_sequence]
            if len(set(rhythms)) <= 2:  # Using only 1-2 rhythm values
                return 1.5

        return 0.0

    def _calculate_reward(self, note, rhythm, chord):
        base_reward = self._calculate_base_reward(note, rhythm, chord)

        if self.use_human_feedback and self.human_feedback and len(self.current_melody) >= 4:
            # Get human feedback influence from similar sequences
            recent_sequence = self.current_melody[-4:]
            similar_melodies = self.human_feedback.get_similar_melodies(recent_sequence)

            if similar_melodies:
                # Weight human feedback by similarity
                weighted_feedback = sum(sim * rating for sim, _, rating in similar_melodies)
                total_sim = sum(sim for sim, _, _ in similar_melodies)

                if total_sim > 0:
                    human_reward = weighted_feedback / total_sim
                    # Combine base reward with human feedback
                    return 0.7 * base_reward + 0.3 * human_reward

        return base_reward

    def _calculate_base_reward(self, note, rhythm, chord):
        reward = 0
        note_in_octave = note % 12

        # 1. Chord Tone Emphasis (Increased rewards)
        chord_tones = CHORD_TONES[chord]
        if note_in_octave in chord_tones:
            if note_in_octave == chord_tones[0]:  # root
                reward += 4.0 if self._is_strong_beat() else 2.0  # Doubled from previous
            else:  # third and fifth
                reward += 3.0 if self._is_strong_beat() else 1.0  # Doubled from previous

        # 2. Scale Context (Increased punishment)
        scale = SCALE_DEGREES[chord]
        if note_in_octave in scale:
            reward += 1.0  # Increased from 0.5
        else:
            reward -= 3.0  # Much stronger penalty for out-of-scale notes

        # 3. Variation Reward
        if len(self.current_melody) > 0:
            recent_notes = [n[0] % 12 for n in self.current_melody[-4:]]
            if note_in_octave not in recent_notes:
                reward += 2.0  # Reward for playing different notes
            else:
                # Count repetitions in recent notes
                repetitions = recent_notes.count(note_in_octave)
                if repetitions > 1:
                    reward -= repetitions * 3.0  # Increased penalty (was 1.0)

                # Additional penalty for immediate repetition
                if note_in_octave == (self.current_melody[-1][0] % 12):
                    reward -= 4.0  # Strong penalty for repeating the previous note

                # Even stronger penalty for three same notes in a row
                if len(recent_notes) >= 2 and note_in_octave == recent_notes[-1] == recent_notes[-2]:
                    reward -= 6.0  # Very strong penalty for three repetitions

        # 4. Enhanced Motif Development
        if len(self.current_melody) >= 4:
            # Get current motif (last 4 notes)
            current_motif = tuple(n[0] % 12 for n in self.current_melody[-4:])

            # Reward for motif creation
            if current_motif not in self.motif_memory:
                reward += 3.0  # New motif bonus
                self.motif_memory[current_motif] = 1
            else:
                # Check for motif transformation
                for stored_motif in self.motif_memory:
                    if self._is_motif_transformation(current_motif, stored_motif):
                        reward += 4.0  # Reward for creative variation
                        break

        # 5. Voice Leading (Keep existing logic but adjust rewards)
        if len(self.current_melody) > 0:
            prev_note = self.current_melody[-1][0]
            interval = abs(note - prev_note)

            if interval <= 2:
                reward += 3.0  # Increased from 2.0
            elif interval <= 4:
                reward += 1.5  # Increased from 1.0
            elif interval > 7:
                reward -= 3.0  # Increased from 2.0

            # Resolution of large leaps
            if len(self.current_melody) >= 2:
                previous_interval = abs(self.current_melody[-1][0] - self.current_melody[-2][0])
                if previous_interval > 4 and interval <= 2:
                    reward += 3.0  # Increased from 2.0

        # 4. Phrase Structure
        current_phrase_length = sum(duration for _, duration, _ in self.current_melody[-8:])
        if 4 <= current_phrase_length <= 8:  # Reward natural phrase lengths
            reward += 1.0

        # 5. Penalize repetition of same note
        if len(self.current_melody) > 0 and note == self.current_melody[-1][0]:
            reward -= 1.0

        # 6. Scale Context
        scale = SCALE_DEGREES[chord]
        if note_in_octave in scale:
            reward += 0.5

        # 7. Melodic Direction Changes
        if len(self.current_melody) >= 3:
            direction = self._get_melodic_direction()
            if -2 <= direction <= 2:  # Reward balanced melodic contour
                reward += 0.5

        # Add new musical evaluations
        tension = self._calculate_tension(note_in_octave,
                                          [n[0] for n in self.current_melody[-2:]])
        reward -= tension  # Penalize excessive tension

        reward += self._evaluate_rhythm_pattern(rhythm, self.current_beat)
        reward += self._analyze_contour()
        reward += self._calculate_target_resolution(note_in_octave, chord)

        # Add long-term structure rewards
        if len(self.current_melody) >= 16:
            # Reward for completing full phrases
            if len(self.current_melody) % 16 == 0:
                reward += 5.0

            # Reward for theme and variation
            if self._is_theme_and_variation():
                reward += 10.0

        return reward

    def _is_motif_transformation(self, motif1, motif2):
        """Check if motif2 is a transformation of motif1"""
        if len(motif1) != len(motif2):
            return False

        # Check for transposition
        intervals1 = [b - a for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b - a for a, b in zip(motif2[:-1], motif2[1:])]
        if intervals1 == intervals2:
            return True

        # Check for inversion
        if intervals1 == [-x for x in reversed(intervals2)]:
            return True

        # Check for retrograde
        if motif1 == tuple(reversed(motif2)):
            return True

        return False

    def _calculate_tension(self, note, prev_notes):
        """Calculate tension level of current note in context"""
        tension = 0
        if len(prev_notes) >= 2:
            # Tension increases with dissonant intervals
            intervals = [abs(note - n) % 12 for n in prev_notes[-2:]]
            dissonances = [i for i in intervals if i in [1, 2, 6, 10, 11]]
            tension += len(dissonances) * 0.5

            # Tension from repeated patterns
            if intervals[0] == intervals[1]:
                tension += 0.5
        return tension

    def _evaluate_rhythm_pattern(self, rhythm, current_beat):
        """Evaluate rhythmic interest"""
        reward = 0
        if len(self.current_melody) >= 4:
            recent_rhythms = [n[1] for n in self.current_melody[-4:]]

            # Reward syncopation
            if current_beat % 1 != 0 and rhythm > 0.5:
                reward += 1.0

            # Reward rhythmic variety
            unique_rhythms = len(set(recent_rhythms))
            reward += unique_rhythms * 0.5

            # Penalize too many short notes
            if rhythm <= 0.25 and all(r <= 0.25 for r in recent_rhythms):
                reward -= 2.0
        return reward

    def _analyze_contour(self):
        """Analyze melodic contour patterns"""
        if len(self.current_melody) >= 5:
            notes = [n[0] for n in self.current_melody[-5:]]
            directions = []
            for i in range(len(notes) - 1):
                if notes[i + 1] > notes[i]:
                    directions.append(1)
                elif notes[i + 1] < notes[i]:
                    directions.append(-1)
                else:
                    directions.append(0)

            # Reward for arch shapes
            if directions == [1, 1, -1, -1]:  # ascending then descending
                return 3.0
            elif directions == [-1, -1, 1, 1]:  # descending then ascending
                return 2.0
        return 0

    def _calculate_target_resolution(self, note, chord):
        """Calculate reward for resolving to target notes"""
        chord_tones = CHORD_TONES[chord]
        prev_notes = [n[0] % 12 for n in self.current_melody[-3:]] if len(self.current_melody) >= 3 else []

        if prev_notes:
            # Resolution to root after leading tone
            if prev_notes[-1] % 12 == (chord_tones[0] - 1) % 12 and note % 12 == chord_tones[0]:
                return 4.0
            # Resolution to third after suspension
            if prev_notes[-1] % 12 == (chord_tones[1] + 1) % 12 and note % 12 == chord_tones[1]:
                return 3.0
        return 0

    def _is_theme_and_variation(self):
        """Check if current phrase is a variation of an earlier phrase"""
        if len(self.current_melody) < 32:
            return False

        current_phrase = self.current_melody[-16:]
        previous_phrase = self.current_melody[-32:-16]

        # Compare rhythmic profile
        current_rhythms = [n[1] for n in current_phrase]
        previous_rhythms = [n[1] for n in previous_phrase]
        rhythm_similarity = sum(1 for a, b in zip(current_rhythms, previous_rhythms) if abs(a - b) < 0.25)

        # Compare melodic profile
        current_intervals = [b[0] - a[0] for a, b in zip(current_phrase[:-1], current_phrase[1:])]
        previous_intervals = [b[0] - a[0] for a, b in zip(previous_phrase[:-1], previous_phrase[1:])]
        interval_similarity = sum(1 for a, b in zip(current_intervals, previous_intervals) if abs(a - b) <= 2)

        return rhythm_similarity >= 12 and interval_similarity >= 10