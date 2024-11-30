import numpy as np
from collections import defaultdict, Counter
from music_theory import CHORD_TONES, SCALE_MASKS, RHYTHM_VALUES
from feedback import HumanFeedbackBuffer


class QtarEnvironment:
    def __init__(self,
                 chord_progression,
                 scale='C_MAJOR',
                 beats_per_chord=4,
                 beats_per_motif=2,
                 training_phase=1,
                 use_human_feedback=False):

        self.beats_per_chord = beats_per_chord
        self.beats_per_motif = beats_per_motif

        self.scale_mask = SCALE_MASKS[scale]
        self.rhythm_values = RHYTHM_VALUES

        # Note and rhythm parameters
        self.note_size = 24  # Two octaves
        self.base_octave = 60  # Middle C (MIDI note 60)
        self.rhythm_values = RHYTHM_VALUES

        self.training_phase = training_phase
        # Phase-specific initialization
        if self.training_phase == 1:
            self.chord_progression = chord_progression[0]
        else:
            self.chord_progression = chord_progression

        # State tracking
        self.current_position = 0
        self.current_beat = 0
        self.current_melody = []
        self.motif_memory = []  # Successful motifs
        self.used_notes = set()

        # Human feedback
        self.human_feedback = HumanFeedbackBuffer() if use_human_feedback else None

    def reset(self):
        """Reset environment maintaining phase-specific settings"""
        self.current_position = 0
        self.current_beat = 0
        self.current_melody = []
        self.used_notes.clear()
        # Only clear motif memory in phase 1 since phase 2 builds on learned motifs
        if self.training_phase == 1:
            self.motif_memory.clear()
        return self._get_state()

    def _get_state(self):
        """Get state representation based on training phase"""
        current_chord = self.chord_progression[self.current_position]
        if self.training_phase == 1:
            # Simpler state for phase 1 (single chord)
            state = [
                1.0,  # Always C major in phase 1
                self.current_beat / self.beats_per_chord,
                self._is_strong_beat(),
            ]
        else:
            # Full state for phase 2
            state = [
                *self._one_hot_encode_chord(current_chord),
                self.current_beat / self.beats_per_chord,
                self._is_strong_beat(),
            ]
        # Add melodic context (same for both phases)
        melody_context = []
        for i in range(min(4, len(self.current_melody))):
            note, duration, _ = self.current_melody[-(i + 1)]
            melody_context.extend([
                note / 24,  # Normalize over two octaves
                duration / 2,
            ])
        while len(melody_context) < 8:
            melody_context.extend([0, 0])
        # Add octave context
        if len(self.current_melody) > 0:
            last_note = self.current_melody[-1][0]
            in_upper_octave = 1.0 if last_note >= 12 else 0.0
            state.append(in_upper_octave)
        else:
            state.append(0.0)
        return np.array(state + melody_context)


    def _one_hot_encode_chord(self, chord):
        # TODO: update chord vocab if using other progressions
        chord_vocab = ['C', 'Am', 'F', 'G']
        encoding = [0] * len(chord_vocab)
        encoding[chord_vocab.index(chord)] = 1
        return encoding

    def _is_strong_beat(self):
        return self.current_beat % 2 == 0

    def step(self, note_action, rhythm_action):
        """Execute step based on current phase"""
        current_chord = self.chord_progression[self.current_position]
        duration = self.rhythm_values[rhythm_action]
        note_action = max(0, min(note_action, 23))
        # Adjust duration for remaining beats
        beats_remaining = self.beats_per_chord - self.current_beat
        if duration > beats_remaining:
            duration = beats_remaining

        # Calculate reward based on phase
        if self.training_phase == 1:
            base_reward = self._calculate_motif_reward(note_action, rhythm_action, current_chord)
        else:
            base_reward = self._calculate_pattern_reward(note_action, rhythm_action, current_chord)

        # Add human feedback if enabled
        final_reward = self._incorporate_human_feedback(base_reward)
        # Update melody and position
        self.current_melody.append((note_action, duration, self.current_beat))
        self.used_notes.add(note_action)
        self.current_beat += duration
        if self.current_beat >= self.beats_per_chord:
            self.current_beat = 0
            self.current_position = (self.current_position + 1) % len(self.chord_progression)
        # Episode is done when:
        # Phase 1: One chord is complete
        # Phase 2: Full progression is complete
        if self.training_phase == 1:
            done = self.current_beat == 0
        else:
            done = self.current_position == 0 and self.current_beat == 0
        return self._get_state(), final_reward, done

    def advance_phase(self):
        """Advance from phase 1 to phase 2"""
        if self.training_phase == 1:
            self.training_phase = 2
            self.chord_progression = ['C', 'Am', 'F', 'G']
            # Keep learned motifs for use in phase 2
            print(f"Advancing to Phase 2 with {len(self.motif_memory)} learned motifs")
            return True
        return False

    def _incorporate_human_feedback(self, base_reward):
        """Combine base reward with human feedback when available"""
        if not self.human_feedback or len(self.current_melody) < 4:
            return base_reward
        # Get human feedback for similar melodic sequences
        recent_sequence = self.current_melody[-4:]
        similar_melodies = self.human_feedback.get_similar_melodies(recent_sequence)
        if similar_melodies:
            # Weight human feedback by similarity
            weighted_feedback = sum(sim * rating for sim, _, rating in similar_melodies)
            total_sim = sum(sim for sim, _, _ in similar_melodies)
            if total_sim > 0:
                human_reward = weighted_feedback / total_sim
                # Combine base reward with human feedback (70-30 split)
                return 0.7 * base_reward + 0.3 * human_reward

        return base_reward

    def _calculate_motif_reward(self, note, rhythm, chord):
        """Phase 1: Reward for creating good motifs"""
        reward = 0
        note_in_octave = note % 12

        # Get current motif (last 2-4 beats of notes)
        current_motif = self._get_current_motif()

        reward += self._chord_tone_reward(note, chord) * 2.0  # Weight of 2.0

        # Basic musical rewards
        reward += self._voice_leading_reward(note) * 2.0
        reward += self._rhythm_coherence_reward(rhythm) * 2.0

        # penalize repetition
        reward += self._repetition_penalty(note)

        # # Motif-specific rewards
        # if len(current_motif) >= 2:  # Only evaluate once we have enough notes
        #     reward += self._motif_structure_reward(current_motif) * 3.0
        #     reward += self._motif_uniqueness_reward(current_motif) * 2.0
        #
        #     # If motif is complete and good, store it
        #     if self._is_motif_complete(current_motif):
        #         if self._is_good_motif(current_motif):
        #             self.motif_memory.append(current_motif)
        #             reward += 20.0  # Big reward for good motif

        return reward

    def _repetition_penalty(self, note):
        """Penalize note repetition and alternating patterns"""
        if len(self.current_melody) == 0:
            return 0

        penalty = 0
        note_in_octave = note % 12

        # Immediate repetition penalty
        if note == self.current_melody[-1][0]:
            penalty -= 15.0

            # Extra penalty for three repeated notes
            if len(self.current_melody) >= 2:
                if note == self.current_melody[-2][0]:
                    penalty -= 25.0

        # Check for alternating patterns (like C-E-C-E)
        if len(self.current_melody) >= 3:
            recent_notes = [n[0] % 12 for n in self.current_melody[-3:]] + [note_in_octave]
            if len(recent_notes) >= 4:
                if (recent_notes[-1] == recent_notes[-3] and
                        recent_notes[-2] == recent_notes[-4]):
                    penalty -= 20.0  # Penalty for alternating pattern

        # Penalize overuse in recent context (last 8 notes)
        if len(self.current_melody) >= 7:
            recent_notes = [n[0] % 12 for n in self.current_melody[-7:]] + [note_in_octave]
            note_counts = {}

            for n in recent_notes:
                note_counts[n] = note_counts.get(n, 0) + 1

            max_occurrences = max(note_counts.values())
            if max_occurrences > 3:
                penalty -= (max_occurrences - 3) * 10.0

        return penalty


    def _calculate_pattern_reward(self, note, rhythm, chord):
        """Phase 2: Reward for patterns across multiple chords"""
        reward = 0

        # Only evaluate when a chord is complete
        if self._is_chord_complete():
            current_motif = self._get_current_chord_notes()

            # Get previous motifs in the current phrase
            previous_motifs = self._get_previous_chord_motifs()

            if previous_motifs:
                # Check for patterns (ABAB or AABB)
                pattern_reward = self._evaluate_motif_patterns(current_motif, previous_motifs)
                reward += pattern_reward

        return reward

    def _is_good_motif(self, motif):
        """Evaluate if a motif is musically good"""
        if len(motif) < 2:
            return False

        # Check melodic shape
        intervals = [b[0] - a[0] for a, b in zip(motif[:-1], motif[1:])]
        if max(abs(i) for i in intervals) > 7:  # No large leaps
            return False

        # Check rhythmic coherence
        rhythms = [note[1] for note in motif]
        if not self._has_consistent_rhythm(rhythms):
            return False

        # Check note variety
        notes = [note[0] % 12 for note in motif]
        if len(set(notes)) < 2:  # Need at least 2 different notes
            return False

        return True

    def _evaluate_pattern_structure(self, motifs):
        """Evaluate ABAB or AABB pattern structure"""
        if len(motifs) < 4:
            return 0

        reward = 0

        # Check ABAB
        if (self._are_similar_motifs(motifs[0], motifs[2]) and
                self._are_similar_motifs(motifs[1], motifs[3])):
            reward += 50.0

        # Check AABB
        elif (self._are_similar_motifs(motifs[0], motifs[1]) and
              self._are_similar_motifs(motifs[2], motifs[3])):
            reward += 40.0

        return reward

    def _are_similar_motifs(self, motif1, motif2):
        """Check if two motifs are similar (allowing for transposition)"""
        if len(motif1) != len(motif2):
            return False

        # Compare intervals (allows for transposition)
        intervals1 = [b[0] - a[0] for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b[0] - a[0] for a, b in zip(motif2[:-1], motif2[1:])]

        # Allow for small rhythmic variations
        rhythms1 = [note[1] for note in motif1]
        rhythms2 = [note[1] for note in motif2]

        return (intervals1 == intervals2 and
                all(abs(r1 - r2) < 0.25 for r1, r2 in zip(rhythms1, rhythms2)))

    def _evaluate_motif_development(self, motifs):
        """Reward for good motif transformations"""
        reward = 0
        for i in range(1, len(motifs)):
            if self._is_good_transformation(motifs[i - 1], motifs[i]):
                reward += 20.0
        return reward

    def _is_good_transformation(self, motif1, motif2):
        """Check if motif2 is a good transformation of motif1"""
        # Check for transposition
        if self._is_transposition(motif1, motif2):
            return True

        # Check for rhythmic variation
        if self._is_rhythmic_variation(motif1, motif2):
            return True

        return False

    def _is_chord_complete(self):
        """Check if current note completes a chord (4 beats)"""
        current_chord_notes = self._get_current_chord_notes()
        total_duration = sum(note[1] for note in current_chord_notes)
        return abs(total_duration - self.beats_per_chord) < 0.01  # Allow small rounding errors

    def _get_current_chord_notes(self):
        """Get all notes from the current chord"""
        current_chord_notes = []
        total_duration = 0

        # Work backwards through melody until we hit previous chord
        for note in reversed(self.current_melody):
            if total_duration + note[1] > self.beats_per_chord:
                break
            current_chord_notes.insert(0, note)
            total_duration += note[1]

        return current_chord_notes

    def _evaluate_complete_motif(self, motif, chord):
        """Evaluate a complete chord's worth of notes as a motif"""
        reward = 0

        # 1. Check basic properties
        if not self._has_valid_motif_structure(motif):
            return -10.0  # Penalty for invalid structure

        # 2. Melodic properties
        reward += self._evaluate_melodic_shape(motif) * 3.0
        reward += self._evaluate_note_variety(motif) * 2.0

        # 3. Rhythmic properties
        reward += self._evaluate_rhythmic_structure(motif) * 3.0

        # 4. Harmony properties
        reward += self._evaluate_harmony(motif, chord) * 2.0

        return reward

    def _has_valid_motif_structure(self, motif):
        """Check if motif has valid basic structure"""
        if not motif:
            return False

        # Must have at least 2 notes
        if len(motif) < 2:
            return False

        # Must start on beat 1
        if motif[0][2] % self.beats_per_chord != 0:
            return False

        # Must fill exactly one chord (4 beats)
        total_duration = sum(note[1] for note in motif)
        if abs(total_duration - self.beats_per_chord) > 0.01:
            return False

        return True

    def _evaluate_melodic_shape(self, motif):
        """Evaluate the melodic shape of a complete motif"""
        reward = 0
        notes = [note[0] % 12 for note in motif]

        # Check for good melodic contour
        intervals = [b - a for a, b in zip(notes[:-1], notes[1:])]

        # Reward stepwise motion
        stepwise_count = sum(1 for i in intervals if abs(i) <= 2)
        reward += stepwise_count * 5

        # Penalize large leaps
        large_leaps = sum(1 for i in intervals if abs(i) > 7)
        reward -= large_leaps * 10

        # Reward clear direction (ascending or descending)
        if all(i >= 0 for i in intervals) or all(i <= 0 for i in intervals):
            reward += 10

        return reward

    def _evaluate_rhythmic_structure(self, motif):
        """Evaluate the rhythmic structure of a complete motif"""
        reward = 0
        rhythms = [note[1] for note in motif]
        positions = [note[2] % self.beats_per_chord for note in motif]

        # Reward longer notes on strong beats
        for note, pos in zip(motif, positions):
            if pos in [0, 2]:  # Strong beats
                if note[1] >= 1.0:  # Long note
                    reward += 10

        # Reward rhythmic patterns
        if len(set(rhythms)) >= 2:  # At least two different note lengths
            reward += 10

        # Penalize too many short notes in succession
        short_notes = sum(1 for r in rhythms if r <= 0.25)
        if short_notes > 3:
            reward -= 10

        return reward

    def _evaluate_harmony(self, motif, chord):
        """Evaluate harmonic properties of the motif"""
        reward = 0
        chord_tones = CHORD_TONES[chord]

        # Check if strong beats use chord tones
        for note, pos in zip(motif, [note[2] % self.beats_per_chord for note in motif]):
            if pos in [0, 2]:  # Strong beats
                if note[0] % 12 in chord_tones:
                    reward += 10
                    if note[0] % 12 == chord_tones[0]:  # Root on strong beat
                        reward += 5

        return reward

    def _evaluate_note_variety(self, motif):
        """Evaluate variety of notes used in the motif"""
        notes = [note[0] % 12 for note in motif]
        unique_notes = len(set(notes))

        if unique_notes == 1:
            return -10  # Penalize single-note motifs
        elif unique_notes == 2:
            return 5
        elif unique_notes == 3:
            return 15
        else:
            return 10  # Good variety but not too much

    def _chord_tone_reward(self, note, chord):
        """Reward for using chord tones appropriately"""
        note_in_octave = note % 12
        chord_tones = CHORD_TONES[chord]
        reward = 0
        # Check if it's a strong beat
        if self._is_strong_beat():
            if note_in_octave == chord_tones[0]:  # root note
                reward += 5.0
            elif note_in_octave in chord_tones:  # other chord tones
                reward += 2.0
            else:  # non-chord tone on strong beat
                reward -= 1.0
        else:  # weak beat
            if note_in_octave in chord_tones:
                reward += 3.0
        return reward

    def _voice_leading_reward(self, note):
        """Reward good voice leading across full range"""
        if len(self.current_melody) == 0:
            return 0

        prev_note = self.current_melody[-1][0]
        interval = abs(note - prev_note)

        # Basic voice leading rewards
        if interval == 0:
            return 0.0  # no reward for repetition
        if interval <= 2:
            return 5.0  # Stepwise motion
        elif interval <= 4:
            return 3.0  # Small leap
        elif interval <= 7:
            return 1.0  # Medium leap
        elif interval == 12:
            # Octave leaps are ok if prepared/resolved well
            if len(self.current_melody) >= 2:
                prev_interval = abs(self.current_melody[-1][0] - self.current_melody[-2][0])
                if prev_interval <= 2:  # Prepared by step
                    return 2.0
        elif interval > 12:
            return -5.0  # Penalize leaps larger than an octave
        return 0.0

    def _is_rhythmic_variation(self, motif1, motif2):
        """Check if motif2 is a rhythmic variation of motif1"""
        if len(motif1) != len(motif2):
            return False

        # Get pitch intervals (should be same)
        intervals1 = [b[0] - a[0] for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b[0] - a[0] for a, b in zip(motif2[:-1], motif2[1:])]

        if intervals1 != intervals2:
            return False

        # Compare rhythms (should be different but related)
        rhythms1 = [note[1] for note in motif1]
        rhythms2 = [note[1] for note in motif2]

        if rhythms1 == rhythms2:
            return False  # Not a variation if rhythms are identical

        # Check if total duration is same
        if abs(sum(rhythms1) - sum(rhythms2)) > 0.01:
            return False

        # Check if rhythms are related (e.g., divided or combined)
        return True

    def _is_transposition(self, motif1, motif2):
        """Check if motif2 is a transposition of motif1"""
        if len(motif1) != len(motif2):
            return False

        # Get intervals between consecutive notes
        intervals1 = [b[0] - a[0] for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b[0] - a[0] for a, b in zip(motif2[:-1], motif2[1:])]

        # Compare intervals (should be identical)
        return intervals1 == intervals2

    def _has_consistent_rhythm(self, rhythms):
        """Check if a rhythm pattern is consistent and musical"""
        if not rhythms:
            return False

        # Total duration should be beats_per_chord
        if abs(sum(rhythms) - self.beats_per_chord) > 0.01:
            return False

        # Check for too many short notes in succession
        short_notes_count = sum(1 for r in rhythms if r <= 0.25)
        if short_notes_count > 3:
            return False

        # Check if longer notes occur on strong beats
        beat_pos = 0
        for dur in rhythms:
            if self._is_strong_beat() and dur < 1.0:
                return False
            beat_pos += dur

        return True

    def _evaluate_motif_patterns(self, current_motif, previous_motifs):
        """Evaluate how current motif forms patterns with previous motifs"""
        if len(previous_motifs) < 1:
            return 0

        reward = 0

        # Check for ABAB pattern
        if len(previous_motifs) >= 3:
            if (self._is_transposition(previous_motifs[0], previous_motifs[2]) and
                    self._is_transposition(previous_motifs[1], current_motif)):
                reward += 50.0

        # Check for AABB pattern
        if len(previous_motifs) >= 2:
            if self._is_transposition(previous_motifs[-1], current_motif):
                reward += 40.0

        return reward

    def _get_previous_chord_motifs(self):
        """Get motifs from previous chords in current phrase"""
        motifs = []
        current_motif = []
        total_duration = 0

        for note in reversed(self.current_melody[:-1]):  # Exclude current incomplete motif
            current_motif.insert(0, note)
            total_duration += note[1]

            if abs(total_duration - self.beats_per_chord) < 0.01:
                motifs.insert(0, current_motif)
                current_motif = []
                total_duration = 0

        return motifs

    def _is_motif_complete(self, current_motif):
        """Check if current motif fills exactly one chord"""
        total_duration = sum(note[1] for note in current_motif)
        return abs(total_duration - self.beats_per_chord) < 0.01

    def _motif_uniqueness_reward(self, motif):
        """Reward for creating unique motifs"""
        if not self.motif_memory:
            return 20.0  # High reward for first motif

        # Check similarity with existing motifs
        for stored_motif in self.motif_memory:
            if self._is_transposition(motif, stored_motif):
                return -10.0  # Penalize exact repetition

        return 10.0  # Reward for unique motif

    def _motif_structure_reward(self, motif):
        """Evaluate structural properties of a motif"""
        reward = 0

        # Check start and end notes
        if motif[0][2] == 0:  # Starts on beat 1
            reward += 5.0

        # Check phrase shape
        notes = [note[0] for note in motif]
        if len(notes) >= 3:
            # Reward for clear contour
            if all(b >= a for a, b in zip(notes[:-1], notes[1:])):  # Ascending
                reward += 10.0
            elif all(b <= a for a, b in zip(notes[:-1], notes[1:])):  # Descending
                reward += 10.0
            elif (all(b >= a for a, b in zip(notes[:len(notes) // 2], notes[1:len(notes) // 2])) and
                  all(b <= a for a, b in zip(notes[len(notes) // 2:-1], notes[len(notes) // 2:]))):  # Arch
                reward += 15.0

        return reward

    def _rhythm_coherence_reward(self, rhythm):
        """Evaluate coherence of rhythm in current context"""
        if len(self.current_melody) == 0:
            return 0
        reward = 0
        # Reward longer notes on strong beats
        if self._is_strong_beat() and rhythm >= 1.0:
            reward += 5.0
        # Penalize very short notes in succession
        if len(self.current_melody) >= 2:
            recent_rhythms = [n[1] for n in self.current_melody[-2:]] + [rhythm]
            if all(r <= 0.25 for r in recent_rhythms):
                reward -= 10.0
        # Reward rhythmic patterns
        if len(self.current_melody) >= 3:
            recent_rhythms = [n[1] for n in self.current_melody[-3:]]
            if self._has_consistent_rhythm(recent_rhythms):
                reward += 5.0
        return reward

    def _get_current_motif(self):
        """Get notes in current incomplete motif"""
        current_motif = []
        total_duration = 0

        for note in reversed(self.current_melody):
            current_motif.insert(0, note)
            total_duration += note[1]

            if total_duration >= self.beats_per_chord:
                break

        return current_motif
