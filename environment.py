import numpy as np
from collections import defaultdict, Counter
from music_theory import CHORD_TONES, SCALE_MASKS, RHYTHM_VALUES
from feedback import HumanFeedbackBuffer


class QtarEnvironment:
    def __init__(self,
                 chord_progression,
                 scale='C_MAJOR',
                 beats_per_chord=4,
                 training_phase=1,
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
        # phase training
        self.training_phase = training_phase
        self._setup_phase_weights()
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

    def _setup_phase_weights(self):
        """Initialize weights based on training phase"""
        if self.training_phase == 1:
            self.weights = {
                'harmony': 3.0,
                'basic_rhythm': 2.0,
                'enhanced_rhythm': 0.0,
                'voice_leading': 0.0,
                'note_variety': 1.0,
                'motifs': 0.0,
                'phrase': 0.0
            }
        elif self.training_phase == 2:
            self.weights = {
                'harmony': 2.5,
                'basic_rhythm': 2.0,
                'enhanced_rhythm': 0.0,
                'voice_leading': 1.5,
                'note_variety': 2.0,
                'motifs': 0.0,
                'phrase': 0.0
            }
        elif self.training_phase == 3:
            self.weights = {
                'harmony': 2.0,
                'basic_rhythm': 1.5,
                'enhanced_rhythm': 2.0,
                'voice_leading': 1.5,
                'note_variety': 1.0,
                'motifs': 0.0,
                'phrase': 0.0
            }
        elif self.training_phase == 4:
            self.weights = {
                'harmony': 2.0,
                'basic_rhythm': 1.5,
                'enhanced_rhythm': 2.0,
                'voice_leading': 1.5,
                'note_variety': 1.0,
                'motifs': 1.5,
                'phrase': 0.5
            }
        else:  # Phase 5 (full)
            self.weights = {
                'harmony': 2.0,
                'basic_rhythm': 1.5,
                'enhanced_rhythm': 2.0,
                'voice_leading': 1.5,
                'note_variety': 1.0,
                'motifs': 2.0,
                'phrase': 1.5
            }

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
        w = self.weights
        reward = 0
        note_in_octave = note % 12
        # harmonic and rhythmic reward
        reward += self._basic_harmony_reward(chord, note_in_octave) * w['harmony']
        reward +=  self._basic_rhythm_reward(rhythm) * w['basic_rhythm']
        # phase two -> introduce voice leading
        if self.training_phase >= 2:
            reward += self._voice_leading_reward(note) * w['voice_leading']
            reward += self._enhanced_harmony_reward(chord, note) * w['harmony']
        # Phase 3: Add Enhanced Rhythm and Note Variety
        if self.training_phase >= 3:
            reward += self._rhythm_pattern_reward(rhythm) * w['enhanced_rhythm']
            reward += self._note_variety_reward(note) * w['note_variety']
        # Phase 4: Add Basic Motifs
        if self.training_phase >= 4:
            reward += self._basic_motif_reward() * w['motifs']
            reward += self._basic_phrase_reward() * w['phrase']
        # Phase 5: Full Complexity
        if self.training_phase >= 5:
            reward += self._complex_motif_reward() * w['motifs']
            reward += self._phrase_structure_reward() * w['phrase']
        return reward

    def _basic_harmony_reward(self, chord, note):
        """Phase 1: Simple chord tone rewards"""
        note_in_octave = note % 12
        chord_tones = CHORD_TONES[chord]

        if note_in_octave == chord_tones[0]:  # root
            return 2.0
        elif note_in_octave in chord_tones:  # other chord tones
            return 1.0
        return -1.0

    def _basic_rhythm_reward(self, rhythm):
        """Phase 1: Simple strong/weak beat rewards"""
        if self._is_strong_beat():
            return 1.0 if rhythm >= 1.0 else -0.5
        return 0.5 if rhythm <= 0.5 else 0.0

    def _enhanced_harmony_reward(self, chord, note):
        """Phase 2: More sophisticated harmony rules"""
        note_in_octave = note % 12
        chord_tones = CHORD_TONES[chord]
        if self._is_strong_beat():
            if note_in_octave == chord_tones[0]:  # root
                return 3.0
            elif note_in_octave == chord_tones[2]:  # fifth
                return 2.0
            elif note_in_octave in chord_tones:  # third
                return 1.5
            return -1.0
        else:
            if note_in_octave in chord_tones:
                return 1.0
            elif note_in_octave in self.scale_mask:
                return 0.5
            return -0.5

    def _voice_leading_reward(self, note):
        """Phase 2: Voice leading rules"""
        if len(self.current_melody) == 0:
            return 0
        interval = abs(note - self.current_melody[-1][0])
        if interval == 0:
            return -2.0  # Penalize repetition
        elif interval <= 2:
            return 2.0  # Reward stepwise motion
        elif interval <= 4:
            return 1.0  # Small leaps okay
        elif interval > 7:
            return -2.0  # Penalize large leaps
        return 0.0

    def _rhythm_pattern_reward(self, rhythm):
        """Evaluate rhythm patterns and their development"""
        if len(self.current_melody) < 4:
            return 0.0
        # Get recent rhythm pattern
        recent_rhythms = [(n[1], n[2] % self.beats_per_chord)
                          for n in self.current_melody[-4:]]
        current_pattern = tuple(recent_rhythms)
        reward = 0.0
        # Check for basic rhythmic patterns
        if self._has_consistent_short_long(current_pattern):
            reward += 1.0
        # Reward syncopation when well-executed
        if self._is_effective_syncopation(current_pattern):
            reward += 1.5
        # Penalize excessive subdivision
        if self._is_overly_subdivided(current_pattern):
            reward -= 1.0
        # Check for rhythmic coherence with previous patterns
        if len(self.current_melody) >= 8:
            previous_pattern = tuple((n[1], n[2] % self.beats_per_chord)
                                     for n in self.current_melody[-8:-4])
            if self._are_rhythms_related(current_pattern, previous_pattern):
                reward += 2.0
        return reward

    def _basic_motif_reward(self):
        """Evaluate simple melodic motifs"""
        if len(self.current_melody) < 4:
            return 0.0

        # Extract recent notes
        recent_notes = [n[0] % 12 for n in self.current_melody[-4:]]

        reward = 0.0

        # Check for simple patterns (e.g., repetition, sequence)
        if recent_notes[0] == recent_notes[2] and recent_notes[1] == recent_notes[3]:
            reward += 2.0  # Direct repetition

        # Check for sequence (same pattern transposed)
        if len(recent_notes) >= 4:
            if (recent_notes[1] - recent_notes[0] ==
                    recent_notes[3] - recent_notes[2]):
                reward += 1.5  # Sequential pattern

        # Check for contour patterns
        contour = [1 if b > a else (-1 if b < a else 0)
                   for a, b in zip(recent_notes[:-1], recent_notes[1:])]
        if len(set(contour)) == 1:  # Consistent direction
            reward += 1.0

        return reward

    def _basic_phrase_reward(self):
        """Evaluate basic phrase structure"""
        if len(self.current_melody) < 8:
            return 0.0

        reward = 0.0

        # Get the last 8 notes
        recent_notes = self.current_melody[-8:]

        # Check for phrase ending on strong beat
        last_note = recent_notes[-1]
        if last_note[2] % self.beats_per_chord == 0:  # Lands on strong beat
            reward += 1.0

        # Check for descending motion at phrase end
        last_three_notes = [n[0] for n in recent_notes[-3:]]
        if last_three_notes[0] >= last_three_notes[1] >= last_three_notes[2]:
            reward += 1.5

        # Check for longer note values at phrase end
        if recent_notes[-1][1] > 1.0:  # Last note is long
            reward += 1.0

        return reward

    def _note_variety_reward(self, note):
        """
        Reward for melodic variety and penalize excessive repetition
        Analyzes both local and global note variety
        """
        note_in_octave = note % 12
        reward = 0.0

        # Track unique notes used (global variety)
        if note_in_octave not in self.used_notes:
            reward += 3.0  # Significant reward for first use of a note
            self.used_notes.add(note_in_octave)

        if len(self.current_melody) >= 1:
            # Penalize immediate repetition
            if note_in_octave == self.current_melody[-1][0] % 12:
                reward -= 2.0

                # Extra penalty for three repeated notes
                if len(self.current_melody) >= 2:
                    if note_in_octave == self.current_melody[-2][0] % 12:
                        reward -= 4.0  # Severe penalty

        # Analyze recent context (last 8 notes)
        if len(self.current_melody) >= 7:
            recent_notes = [n[0] % 12 for n in self.current_melody[-7:]] + [note_in_octave]
            note_counts = {}

            # Count occurrences of each note
            for n in recent_notes:
                note_counts[n] = note_counts.get(n, 0) + 1

            # Reward for variety in recent context
            unique_notes = len(note_counts)
            if unique_notes >= 5:  # Good variety
                reward += 2.0
            elif unique_notes >= 3:  # Moderate variety
                reward += 1.0
            elif unique_notes <= 2:  # Poor variety
                reward -= 2.0

            # Penalize overuse of any single note
            max_occurrences = max(note_counts.values())
            if max_occurrences > 3:  # Note appears more than 3 times in last 8 notes
                reward -= (max_occurrences - 3) * 1.5

        # Check for alternating patterns (like A-B-A-B-A)
        if len(self.current_melody) >= 4:
            recent_notes = [n[0] % 12 for n in self.current_melody[-4:]] + [note_in_octave]
            if len(recent_notes) >= 5:
                if recent_notes[-1] == recent_notes[-3] == recent_notes[-5]:
                    if recent_notes[-2] == recent_notes[-4]:  # Alternating pattern
                        reward -= 2.0

        # Scale the reward to keep it proportional with other rewards
        return reward

    def _complex_motif_reward(self):
        """Evaluate sophisticated motif development"""
        if len(self.current_melody) < 8:
            return 0.0

        reward = 0.0

        # Get last 8 notes for analysis
        recent_melody = self.current_melody[-8:]

        # Extract note patterns and rhythms
        notes = [n[0] % 12 for n in recent_melody]
        rhythms = [n[1] for n in recent_melody]

        # Check for motivic development techniques
        reward += self._evaluate_augmentation(notes, rhythms)
        reward += self._evaluate_diminution(notes, rhythms)
        reward += self._evaluate_inversion(notes)
        reward += self._evaluate_retrograde(notes)

        # Check relations to earlier material
        if len(self.current_melody) >= 16:
            previous_section = self.current_melody[-16:-8]
            reward += self._evaluate_motif_relationship(
                recent_melody, previous_section)

        return reward

    def _phrase_structure_reward(self):
        """Evaluate overall phrase structure and form"""
        if len(self.current_melody) < 16:  # Need full 4-bar phrase
            return 0.0

        reward = 0.0

        # Split into 4-bar sections
        phrase = self.current_melody[-16:]
        bars = []
        current_bar = []
        beat_count = 0

        for note in phrase:
            current_bar.append(note)
            beat_count += note[1]
            if beat_count >= self.beats_per_chord:
                bars.append(current_bar)
                current_bar = []
                beat_count = 0

        if len(bars) != 4:  # Ensure we have 4 complete bars
            return 0.0

        # Compare first and third bars (A sections)
        similarity_aa = self._compare_bars(bars[0], bars[2])
        reward += similarity_aa * 3.0  # Higher weight for A section similarity

        # Compare second and fourth bars (B sections)
        similarity_bb = self._compare_bars(bars[1], bars[3])
        reward += similarity_bb * 2.0

        # Check contrast between A and B sections
        contrast_ab = 1.0 - self._compare_bars(bars[0], bars[1])
        reward += contrast_ab * 2.0  # Reward contrast between A and B

        # Additional rewards for good phrase structure
        reward += self._evaluate_cadence(bars[-1])  # Ending
        reward += self._evaluate_phrase_coherence(bars)  # Overall coherence

        return reward

    def _compare_bars(self, bar1, bar2):
        """Compare two bars for similarity"""
        if not bar1 or not bar2:
            return 0.0

        # Compare melodic contour
        notes1 = [n[0] % 12 for n in bar1]
        notes2 = [n[0] % 12 for n in bar2]

        # Get contours
        contour1 = [b - a for a, b in zip(notes1[:-1], notes1[1:])]
        contour2 = [b - a for a, b in zip(notes2[:-1], notes2[1:])]

        # Compare rhythms
        rhythm1 = [n[1] for n in bar1]
        rhythm2 = [n[1] for n in bar2]

        # Calculate similarities
        melodic_similarity = self._get_contour_similarity(contour1, contour2)
        rhythm_similarity = self._get_rhythm_similarity(rhythm1, rhythm2)

        return (melodic_similarity * 0.6 + rhythm_similarity * 0.4)

    def _get_contour_similarity(self, contour1, contour2):
        """Compare melodic contours"""
        if not contour1 or not contour2:
            return 0.0

        # Normalize lengths
        max_len = max(len(contour1), len(contour2))
        c1 = contour1 + [0] * (max_len - len(contour1))
        c2 = contour2 + [0] * (max_len - len(contour2))

        # Compare directions
        matches = sum(1 for a, b in zip(c1, c2)
                      if (a > 0 and b > 0) or (a < 0 and b < 0) or (a == b == 0))

        return matches / max_len

    def _get_rhythm_similarity(self, rhythm1, rhythm2):
        """Compare rhythm patterns"""
        if not rhythm1 or not rhythm2:
            return 0.0

        # Quantize rhythms
        def quantize(rhythm):
            return [round(r * 4) / 4 for r in rhythm]

        r1 = quantize(rhythm1)
        r2 = quantize(rhythm2)

        # Compare quantized rhythms
        max_len = max(len(r1), len(r2))
        r1 = r1 + [0] * (max_len - len(r1))
        r2 = r2 + [0] * (max_len - len(r2))

        similarity = sum(1 for a, b in zip(r1, r2) if abs(a - b) < 0.25)
        return similarity / max_len

    def _evaluate_cadence(self, final_bar):
        """Evaluate the strength of the ending"""
        if not final_bar:
            return 0.0

        reward = 0.0

        # Check final note length
        if final_bar[-1][1] >= 1.0:  # Long final note
            reward += 1.0

        # Check if ends on chord tone
        final_chord = self.chord_progression[-1]
        final_note = final_bar[-1][0] % 12
        if final_note in CHORD_TONES[final_chord]:
            reward += 1.0
            if final_note == CHORD_TONES[final_chord][0]:  # Root
                reward += 1.0

        return reward

    def _has_consistent_short_long(self, pattern):
        """Check for alternating short-long rhythm patterns"""
        durations = [dur for dur, _ in pattern]
        if len(durations) < 2:
            return False

        # Look for alternating pattern
        for i in range(0, len(durations) - 1, 2):
            if durations[i] >= durations[i + 1]:
                return False

        return True

    def _is_effective_syncopation(self, pattern):
        """Evaluate if syncopation is musically effective"""
        durations, positions = zip(*pattern)

        # Check for off-beat emphasis
        for dur, pos in zip(durations, positions):
            beat_position = pos % 1
            # Effective syncopation: longer notes on weak beats
            if 0.25 <= beat_position <= 0.75 and dur >= 1.0:
                # Check if preceded by shorter note
                prev_idx = pattern.index((dur, pos)) - 1
                if prev_idx >= 0 and pattern[prev_idx][0] <= 0.5:
                    return True

        return False

    def _is_overly_subdivided(self, pattern):
        """Check if rhythm has too many short notes in succession"""
        durations = [dur for dur, _ in pattern]
        short_notes_count = sum(1 for dur in durations if dur <= 0.25)
        return short_notes_count >= 3  # Three or more very short notes in a row

    def _are_rhythms_related(self, pattern1, pattern2):
        """Check if two rhythm patterns are related (augmentation, diminution, or similar)"""
        dur1, pos1 = zip(*pattern1)
        dur2, pos2 = zip(*pattern2)

        # Convert to relative duration ratios
        def get_ratios(durations):
            return [a / b for a, b in zip(durations[:-1], durations[1:])]

        ratios1 = get_ratios(dur1)
        ratios2 = get_ratios(dur2)

        # Check for similar proportions
        if len(ratios1) == len(ratios2):
            differences = [abs(a - b) for a, b in zip(ratios1, ratios2)]
            return sum(differences) / len(differences) < 0.25

        return False

    def _evaluate_augmentation(self, notes, rhythms):
        """Check for rhythmic augmentation of motifs"""
        if len(notes) < 4 or len(rhythms) < 4:
            return 0.0

        # Look for same note pattern with doubled durations
        half_notes = notes[:len(notes) // 2]
        half_rhythms = rhythms[:len(rhythms) // 2]

        for i in range(1, len(notes) - len(half_notes) + 1):
            current_notes = notes[i:i + len(half_notes)]
            current_rhythms = rhythms[i:i + len(half_rhythms)]

            if half_notes == current_notes:
                # Check if rhythms are doubled
                if all(abs(a * 2 - b) < 0.25 for a, b in zip(half_rhythms, current_rhythms)):
                    return 2.0

        return 0.0

    def _evaluate_diminution(self, notes, rhythms):
        """Check for rhythmic diminution of motifs"""
        if len(notes) < 4 or len(rhythms) < 4:
            return 0.0

        # Look for same note pattern with halved durations
        half_notes = notes[:len(notes) // 2]
        half_rhythms = rhythms[:len(rhythms) // 2]

        for i in range(1, len(notes) - len(half_notes) + 1):
            current_notes = notes[i:i + len(half_notes)]
            current_rhythms = rhythms[i:i + len(half_rhythms)]

            if half_notes == current_notes:
                # Check if rhythms are halved
                if all(abs(a / 2 - b) < 0.25 for a, b in zip(half_rhythms, current_rhythms)):
                    return 2.0

        return 0.0

    def _evaluate_inversion(self, notes):
        """Check for melodic inversion"""
        if len(notes) < 4:
            return 0.0

        # Calculate intervals in original pattern
        intervals = [b - a for a, b in zip(notes[:-1], notes[1:])]

        # Look for inverted intervals
        for i in range(1, len(notes) - len(intervals)):
            current_intervals = [b - a for a, b in zip(notes[i:-1], notes[i + 1:])]
            if len(current_intervals) >= len(intervals):
                # Check if intervals are inverted (opposite direction)
                if all(abs(a + b) < 2 for a, b in zip(intervals, current_intervals[:len(intervals)])):
                    return 2.0

        return 0.0

    def _evaluate_retrograde(self, notes):
        """Check for retrograde (reverse) patterns"""
        if len(notes) < 4:
            return 0.0

        # Look for reversed patterns
        half_len = len(notes) // 2
        first_half = notes[:half_len]
        second_half = notes[half_len:]

        if first_half == list(reversed(second_half)):
            return 2.0

        return 0.0

    def _evaluate_motif_relationship(self, current_motif, previous_motif):
        """Evaluate relationship between current motif and previous material"""
        reward = 0.0

        # Extract notes and rhythms
        current_notes = [n[0] % 12 for n in current_motif]
        current_rhythms = [n[1] for n in current_motif]
        prev_notes = [n[0] % 12 for n in previous_motif]
        prev_rhythms = [n[1] for n in previous_motif]

        # Check for transposition
        intervals1 = [b - a for a, b in zip(current_notes[:-1], current_notes[1:])]
        intervals2 = [b - a for a, b in zip(prev_notes[:-1], prev_notes[1:])]
        if intervals1 == intervals2:
            reward += 1.5

        # Check for rhythmic similarity
        if self._are_rhythms_related(list(zip(current_rhythms, range(len(current_rhythms)))),
                                     list(zip(prev_rhythms, range(len(prev_rhythms))))):
            reward += 1.0

        # Check for contour similarity
        if self._get_contour_similarity(intervals1, intervals2) > 0.8:
            reward += 1.0

        return reward

    def _evaluate_phrase_coherence(self, bars):
        """Evaluate overall coherence of the phrase"""
        reward = 0.0

        # Check for consistent rhythmic density
        rhythmic_densities = []
        for bar in bars:
            total_duration = sum(note[1] for note in bar)
            num_notes = len(bar)
            density = num_notes / total_duration
            rhythmic_densities.append(density)

        # Reward similar densities in corresponding sections (A-A, B-B)
        if abs(rhythmic_densities[0] - rhythmic_densities[2]) < 0.5:  # A sections
            reward += 1.0
        if abs(rhythmic_densities[1] - rhythmic_densities[3]) < 0.5:  # B sections
            reward += 1.0

        # Check for motivic consistency
        for i in range(len(bars) - 1):
            notes1 = [n[0] % 12 for n in bars[i]]
            notes2 = [n[0] % 12 for n in bars[i + 1]]

            # Calculate melodic range for each bar
            range1 = max(notes1) - min(notes1)
            range2 = max(notes2) - min(notes2)

            # Reward similar ranges (coherent use of register)
            if abs(range1 - range2) <= 4:
                reward += 0.5

        return reward