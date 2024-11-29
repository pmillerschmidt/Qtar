import numpy as np
from collections import defaultdict, Counter
from music_theory import CHORD_TONES, SCALE_MASKS, RHYTHM_VALUES
from feedback import HumanFeedbackBuffer


class QtarEnvironment:
    def __init__(self,
                 chord_progression,
                 scale='C_MAJOR',
                 beats_per_chord=4,
                 use_human_feedback=False):  # Added feedback flag
        self.chord_progression = chord_progression
        self.beats_per_chord = beats_per_chord
        self.use_human_feedback = use_human_feedback
        self.current_position = 0
        self.current_beat = 0
        self.scale_mask = SCALE_MASKS[scale]
        self.rhythm_values = RHYTHM_VALUES
        # Track motifs per chord position
        self.chord_motifs = defaultdict(list)  # {chord_position: [(motif, count)]}
        self.rhythm_patterns = defaultdict(list)
        self.used_notes = set()
        # Initialize human feedback if enabled
        self.human_feedback = HumanFeedbackBuffer() if use_human_feedback else None
        self.reset()

    def reset(self):
        self.current_position = 0
        self.current_beat = 0
        self.current_melody = []
        self.chord_motifs.clear()
        self.rhythm_patterns.clear()
        self.used_notes.clear()
        return self._get_state()

    def _get_state(self):
        current_chord = self.chord_progression[self.current_position]
        state = [
            *self._one_hot_encode_chord(current_chord),
            self.current_beat / self.beats_per_chord,  # Normalized beat position
            self._is_strong_beat(),
        ]
        # Add recent melodic context (last 4 notes)
        melody_context = []
        for i in range(min(4, len(self.current_melody))):
            note, duration, _ = self.current_melody[-(i + 1)]
            melody_context.extend([
                note % 12 / 12,  # Normalized note
                duration / 2,  # Normalized duration
            ])
        while len(melody_context) < 8:  # Pad if needed
            melody_context.extend([0, 0])
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
        current_chord = self.chord_progression[self.current_position]
        duration = self.rhythm_values[rhythm_action]
        # Adjust duration if it exceeds remaining beats
        beats_remaining = self.beats_per_chord - self.current_beat
        if duration > beats_remaining:
            duration = beats_remaining
        # Calculate base reward
        base_reward = self._calculate_base_reward(note_action, rhythm_action, current_chord)
        # Add human feedback if enabled and available
        final_reward = self._incorporate_human_feedback(base_reward)
        # Add note to melody
        self.current_melody.append((note_action, duration, self.current_beat))
        # Update motif memory if we have enough notes
        if len(self.current_melody) >= 4:
            self._update_chord_motifs()
        # Update position
        self.current_beat += duration
        if self.current_beat >= self.beats_per_chord:
            self.current_beat = 0
            self.current_position = (self.current_position + 1) % len(self.chord_progression)
        # Check if episode is done (one complete progression)
        done = self.current_position == 0 and self.current_beat == 0
        if done:
            phrase_reward = self._evaluate_phrase_structure()
            final_reward += phrase_reward

        return self._get_state(), final_reward, done

    def _incorporate_human_feedback(self, base_reward):
        """Combine base reward with human feedback when available"""
        if not self.use_human_feedback or not self.human_feedback or len(self.current_melody) < 4:
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

    def _update_chord_motifs(self):
        # Get last 4 notes as a motif
        current_motif = tuple(n[0] % 12 for n in self.current_melody[-4:])
        # Store motif for current chord position
        motifs = self.chord_motifs[self.current_position]
        # Check if motif exists for this position
        for i, (motif, count) in enumerate(motifs):
            if self._is_similar_motif(current_motif, motif):
                motifs[i] = (motif, count + 1)
                break
        else:
            motifs.append((current_motif, 1))

    def _is_similar_motif(self, motif1, motif2):
        """Check if motifs are similar (allowing for small variations)"""
        if len(motif1) != len(motif2):
            return False
        # Compare intervals between notes
        intervals1 = [b - a for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b - a for a, b in zip(motif2[:-1], motif2[1:])]
        # Allow for small variations in intervals
        differences = [abs(a - b) for a, b in zip(intervals1, intervals2)]
        return all(diff <= 2 for diff in differences)

    def _calculate_base_reward(self, note, rhythm, chord):
        reward = 0
        note_in_octave = note % 12
        # Core Harmonic Reward (weighted heavily)
        reward += self._harmony_reward(chord, note_in_octave) * 2.0
        # Voice Leading Reward
        reward += self._voice_leading_reward(note) * 1.5
        # Motif Development Reward (increased importance)
        reward += self._motif_reward() * 2.0
        # Basic Rhythm Reward
        reward += self._enhanced_rhythm_reward(rhythm) * 1.5
        # note variety and repetition reward
        reward += self._note_variety_reward(note)
        reward += self._repetition_penalty(note)
        return reward

    def _harmony_reward(self, chord, note_in_octave):
        chord_tones = CHORD_TONES[chord]
        if self._is_strong_beat():
            if note_in_octave == chord_tones[0]:  # root
                return 4.0
            elif note_in_octave in chord_tones:  # other chord tones
                return 2.0
            return -1.0
        else:
            if note_in_octave in chord_tones:
                return 2.0
            return 0.0

    def _voice_leading_reward(self, note):
        if len(self.current_melody) == 0:
            return 0.0
        interval = abs(note - self.current_melody[-1][0])
        if interval <= 2:  # stepwise motion
            return 2.0
        elif interval <= 4:  # small leap
            return 1.0
        elif interval > 7:  # large leap
            return -1.0
        return 0.0

    def _motif_reward(self):
        if len(self.current_melody) < 4:
            return 0.0
        current_motif = tuple(n[0] % 12 for n in self.current_melody[-4:])
        # Check if this is a repeat of a motif in the same chord position
        motifs = self.chord_motifs[self.current_position]
        for stored_motif, count in motifs:
            if self._is_similar_motif(current_motif, stored_motif):
                # Reward for maintaining motif structure (ABAB form)
                if self.current_position % 2 == 0:  # A sections
                    return 4.0 if count == 2 else 2.0
                else:  # B sections
                    return 3.0 if count == 2 else 1.5
        return 0.0

    def _enhanced_rhythm_reward(self, rhythm):
        """Complex rhythm reward system considering patterns and context"""
        reward = 0
        # 1. Basic rhythmic appropriateness
        reward += self._basic_rhythm_reward(rhythm)
        # 2. Rhythmic pattern development
        if len(self.current_melody) >= 4:
            reward += self._rhythm_pattern_reward()
        # 3. Syncopation reward
        reward += self._syncopation_reward(rhythm)
        # 4. Rhythmic variety reward
        reward += self._rhythmic_variety_reward(rhythm)
        return reward

    def _basic_rhythm_reward(self, rhythm):
        """Reward appropriate note lengths for beat positions"""
        if self._is_strong_beat():
            if rhythm >= 1.0:  # longer notes on strong beats
                return 1.5
            return 0.5
        else:
            if rhythm <= 0.5:  # shorter notes on weak beats
                return 1.0
            return 0.0

    def _rhythm_pattern_reward(self):
        """Reward rhythmic motifs and their development"""
        # Get the rhythm pattern for the last four notes
        current_pattern = self._get_current_rhythm_pattern()
        # Store pattern for current chord position
        patterns = self.rhythm_patterns[self.current_position]
        # Check for pattern matches
        for stored_pattern, count in patterns:
            if self._is_similar_rhythm_pattern(current_pattern, stored_pattern):
                # Update pattern count
                patterns.append((current_pattern, count + 1))

                # Reward based on pattern position and repetition
                if self.current_position % 2 == 0:  # A sections
                    return 3.0 if count >= 2 else 1.5  # Higher reward for established patterns
                else:  # B sections
                    return 2.0 if count >= 2 else 1.0  # Slightly lower to encourage variation
        # New pattern
        patterns.append((current_pattern, 1))
        return 0.5  # Small reward for introducing new patterns

    def _get_current_rhythm_pattern(self):
        """Extract rhythm pattern from recent notes"""
        if len(self.current_melody) < 4:
            return tuple()
        # Get the last 4 durations and their beat positions
        recent_rhythms = [(n[1], n[2] % self.beats_per_chord)
                          for n in self.current_melody[-4:]]
        return tuple(recent_rhythms)

    def _is_similar_rhythm_pattern(self, pattern1, pattern2):
        """Compare rhythm patterns allowing for small variations"""
        if len(pattern1) != len(pattern2):
            return False
        # Compare durations and positions
        for (dur1, pos1), (dur2, pos2) in zip(pattern1, pattern2):
            # Allow for small duration differences
            if abs(dur1 - dur2) > 0.25:
                return False
            # Allow for small position shifts
            if abs(pos1 - pos2) > 0.25:
                return False
        return True

    def _syncopation_reward(self, rhythm):
        """Reward intentional syncopation"""
        reward = 0
        # Check if we're on an off-beat
        is_off_beat = (self.current_beat % 1) >= 0.5
        if is_off_beat:
            # Reward longer notes starting on off-beats (syncopation)
            if rhythm >= 0.5:
                reward += 1.0
            # Extra reward if previous note was shorter
            if len(self.current_melody) > 0 and self.current_melody[-1][1] <= 0.25:
                reward += 0.5
            # Additional reward if creating a syncopated pattern
            if self._is_syncopated_pattern():
                reward += 1.5
        return reward

    def _is_syncopated_pattern(self):
        """Check if recent notes form a syncopated pattern"""
        if len(self.current_melody) < 3:
            return False
        recent_positions = [n[2] % 1 for n in self.current_melody[-3:]]
        recent_durations = [n[1] for n in self.current_melody[-3:]]
        # Look for alternating on-beat/off-beat pattern
        alternating = all(abs(pos - prev_pos) >= 0.5
                          for pos, prev_pos in zip(recent_positions[1:], recent_positions[:-1]))
        # Check if durations support syncopation
        supports_syncopation = any(dur >= 0.5 for dur in recent_durations)
        return alternating and supports_syncopation

    def _rhythmic_variety_reward(self, rhythm):
        """Reward rhythmic variety while avoiding excessive complexity"""
        if len(self.current_melody) < 4:
            return 0
        # Get recent unique rhythm values
        recent_rhythms = [n[1] for n in self.current_melody[-4:]]
        unique_rhythms = len(set(recent_rhythms))
        # Reward moderate variety (2-3 different values)
        if unique_rhythms == 2:
            return 1.0
        elif unique_rhythms == 3:
            return 1.5
        elif unique_rhythms == 4:
            return 0.5  # Small reward for high variety
        return 0.0  # No reward for complete repetition

    def _note_variety_reward(self, note):
        """Reward for using new notes and maintaining variety"""
        reward = 0
        note_in_octave = note % 12
        # Reward for using a completely new note
        if note_in_octave not in self.used_notes:
            reward += 4.0  # Significant reward for new notes
            self.used_notes.add(note_in_octave)
        if len(self.current_melody) >= 8:
            # Check variety in recent context (last 8 notes)
            recent_notes = [n[0] % 12 for n in self.current_melody[-7:]] + [note_in_octave]
            unique_notes = len(set(recent_notes))
            # Reward for maintaining good variety
            if unique_notes >= 5:  # Using 5+ different notes in last 8 notes
                reward += 2.0
            elif unique_notes >= 3:  # Using 3-4 different notes
                reward += 1.0
            # Penalize for low variety
            if unique_notes <= 2:  # Using only 1-2 different notes
                reward -= 2.0
        return reward

    def _repetition_penalty(self, note):
        """Penalty for repeated notes"""
        if len(self.current_melody) == 0:
            return 0
        note_in_octave = note % 12
        penalty = 0
        # Check for immediate repetition
        if note_in_octave == self.current_melody[-1][0] % 12:
            penalty -= 4.0  # Penalty for immediate repetition
            # Extra penalty if this would be third note in a row
            if len(self.current_melody) >= 2:
                if note_in_octave == self.current_melody[-2][0] % 12:
                    penalty -= 6.0  # Severe penalty for three in a row
        # Check for pattern repetition (like A-B-A-B-A)
        if len(self.current_melody) >= 4:
            last_5_notes = [n[0] % 12 for n in self.current_melody[-4:]] + [note_in_octave]
            if last_5_notes[-1] == last_5_notes[-3] == last_5_notes[-5]:
                penalty -= 2.0  # Penalty for repetitive patterns
        return penalty

    def _evaluate_phrase_structure(self):
        """Evaluate if the complete sequence follows ABAB structure"""
        if len(self.current_melody) < 16:  # Need enough notes for a full phrase
            return 0
        # Split the sequence into 4 bars
        total_beats = len(self.chord_progression) * self.beats_per_chord
        beats_per_bar = total_beats // 4
        bars = []
        current_bar = []
        current_beat_sum = 0
        for note in self.current_melody:
            current_bar.append(note)
            current_beat_sum += note[1]  # Add duration
            if current_beat_sum >= beats_per_bar:
                bars.append(current_bar)
                current_bar = []
                current_beat_sum = 0
        if len(bars) != 4:  # Ensure we have exactly 4 bars
            return 0
        # Compare bars for ABAB structure
        similarity_score = self._compare_bars(bars[0], bars[2])  # A-A comparison
        similarity_score += self._compare_bars(bars[1], bars[3])  # B-B comparison
        difference_score = self._verify_contrast(bars[0], bars[1])  # A-B contrast
        # Calculate final phrase structure reward
        if similarity_score >= 0.7 and difference_score >= 0.4:  # Clear ABAB structure
            return 10.0  # Large reward for good phrase structure
        elif similarity_score >= 0.5 and difference_score >= 0.3:  # Moderate ABAB structure
            return 5.0  # Moderate reward for attempted phrase structure
        return 0.0

    def _compare_bars(self, bar1, bar2):
        """Compare two bars for similarity in both melody and rhythm"""
        if not bar1 or not bar2:
            return 0
        # Compare melodic contour
        contour1 = self._get_melodic_contour(bar1)
        contour2 = self._get_melodic_contour(bar2)
        melodic_similarity = self._compare_contours(contour1, contour2)
        # Compare rhythm
        rhythm1 = tuple(note[1] for note in bar1)  # Get durations
        rhythm2 = tuple(note[1] for note in bar2)
        rhythm_similarity = self._compare_rhythms(rhythm1, rhythm2)
        # Weighted combination
        return 0.6 * melodic_similarity + 0.4 * rhythm_similarity

    def _get_melodic_contour(self, bar):
        """Extract melodic contour from a bar"""
        notes = [note[0] for note in bar]
        contour = []
        for i in range(1, len(notes)):
            if notes[i] > notes[i - 1]:
                contour.append(1)  # up
            elif notes[i] < notes[i - 1]:
                contour.append(-1)  # down
            else:
                contour.append(0)  # same
        return contour

    def _compare_contours(self, contour1, contour2):
        """Compare two melodic contours"""
        if not contour1 or not contour2:
            return 0
        # Pad shorter contour if needed
        max_len = max(len(contour1), len(contour2))
        contour1 = contour1 + [0] * (max_len - len(contour1))
        contour2 = contour2 + [0] * (max_len - len(contour2))
        # Calculate similarity
        matches = sum(1 for a, b in zip(contour1, contour2) if a == b)
        return matches / max_len

    def _compare_rhythms(self, rhythm1, rhythm2):
        """Compare two rhythm patterns"""
        if not rhythm1 or not rhythm2:
            return 0
        # Convert to rhythm strings for comparison
        def quantize_rhythm(rhythm):
            return tuple(round(dur * 4) / 4 for dur in rhythm)  # Quantize to quarter beats
        r1 = quantize_rhythm(rhythm1)
        r2 = quantize_rhythm(rhythm2)

        if len(r1) != len(r2):
            return 0.5  # Partial similarity if different lengths
        # Compare individual durations
        matches = sum(1 for a, b in zip(r1, r2) if abs(a - b) <= 0.25)  # Allow small differences
        return matches / len(r1)

    def _verify_contrast(self, bar1, bar2):
        """Verify that A and B sections are sufficiently different"""
        # Compare for difference rather than similarity
        similarity = self._compare_bars(bar1, bar2)
        return 1.0 - similarity  # Convert similarity to difference score
