from collections import deque

import numpy as np
from music_theory import CHORD_TONES, SCALE_MASKS, RHYTHM_VALUES
from feedback import HumanFeedbackBuffer


class QtarEnvironment:
    def __init__(self,
                 chord_progression,
                 scale='C_MAJOR',
                 beats_per_chord=4,
                 use_human_feedback=False,
                 entropy_weight=0.1):
        # setup
        self.chord_progression = chord_progression
        self.beats_per_chord = beats_per_chord
        self.scale_mask = SCALE_MASKS[scale]
        self.rhythm_values = RHYTHM_VALUES
        # note parameter
        self.note_size = 24  # two octaves
        self.base_octave = 60  # start at C
        # track state
        self.current_position = 0
        self.current_beat = 0
        self.current_melody = []
        self.motif_memory = []
        self.used_notes = set()
        # track progress
        self.total_episodes = 0
        self.last_rewards = deque(maxlen=100)
        self.note_history = []
        self.rhythm_history = []
        self.entropy_weight = entropy_weight
        # HF
        self.human_feedback = HumanFeedbackBuffer() if use_human_feedback else None

    def _calculate_entropy_bonus(self, note, rhythm):
        # entropy bonus -> diversity
        # note distribution
        note_counts = np.bincount(self.note_history if self.note_history else [], minlength=24)
        note_probs = note_counts / max(1, len(self.note_history))
        note_entropy = -np.sum(p * np.log(p + 1e-10) for p in note_probs if p > 0)
        # rhythm distribution
        rhythm_counts = np.bincount(
            [self.rhythm_values.index(r) for r in self.rhythm_history] if self.rhythm_history else [],
            minlength=len(self.rhythm_values)
        )
        rhythm_probs = rhythm_counts / max(1, len(self.rhythm_history))
        rhythm_entropy = -np.sum(p * np.log(p + 1e-10) for p in rhythm_probs if p > 0)
        # apply entropy weight
        return ((note_entropy + rhythm_entropy) / 2) * self.entropy_weight

    def _update_history(self, note, rhythm):
        # update note/rhythm histories
        self.note_history.append(note)
        self.rhythm_history.append(rhythm)
        # forget after a while
        if len(self.note_history) > 1000:
            self.note_history = self.note_history[-1000:]
            self.rhythm_history = self.rhythm_history[-1000:]

    def reset(self):
        # reset env
        self.current_position = 0
        self.current_beat = 0
        self.current_melody = []
        self.used_notes.clear()
        return self._get_state()

    def _get_state(self):
        # curr state representation
        current_chord = self.chord_progression[self.current_position]
        state = [
            *self._one_hot_encode_chord(current_chord),
            self.current_beat / self.beats_per_chord,
            self._is_strong_beat(),
        ]
        # melodic contexts
        melody_context = []
        for i in range(min(4, len(self.current_melody))):
            note, duration, _ = self.current_melody[-(i + 1)]
            melody_context.extend([
                note / 24,  # Normalize over two octaves
                duration / 2,
            ])
        # pad
        while len(melody_context) < 8:
            melody_context.extend([0, 0])
        # octave context
        if len(self.current_melody) > 0:
            last_note = self.current_melody[-1][0]
            in_upper_octave = 1.0 if last_note >= 12 else 0.0
            state.append(in_upper_octave)
        else:
            state.append(0.0)
        return np.array(state + melody_context)

    def get_learned_motifs(self):
        # motif mem for saving
        return self.motif_memory.copy()

    def set_learned_motifs(self, motifs):
        # load previously learned motifs
        self.learned_phase1_motifs = motifs

    def _one_hot_encode_chord(self, chord):
        # TODO: update chord vocab if using other progressions
        chord_vocab = ['C', 'Am', 'F', 'G']
        encoding = [0] * len(chord_vocab)
        encoding[chord_vocab.index(chord)] = 1
        return encoding

    def _is_strong_beat(self):
        return self.current_beat % 2 == 0

    def step(self, note_action, rhythm_action):
        # get reward and step
        current_chord = self.chord_progression[self.current_position]
        duration = self.rhythm_values[rhythm_action]
        note_action = max(0, min(note_action, 23))
        # calc reward
        reward = self._calculate_base_reward(note_action, duration, current_chord)
        # update state
        self.current_melody.append((note_action, duration, self.current_beat))
        self.used_notes.add(note_action)
        self.current_beat += duration
        # handle chord progression
        if self.current_beat >= self.beats_per_chord:
            self.current_beat = 0
            self.current_position = (self.current_position + 1) % len(self.chord_progression)
        # progression complete -> done
        done = (self.current_position == 0 and self.current_beat == 0)
        if done:
            self.total_episodes += 1
        return self._get_state(), reward, done

    def _incorporate_human_feedback(self, base_reward):
        # get human feedback and incorporate into reward
        if not self.human_feedback or len(self.current_melody) < 4:
            return base_reward
        # HF for similar melodies
        recent_sequence = self.current_melody[-4:]
        similar_melodies = self.human_feedback.get_similar_melodies(recent_sequence)
        if similar_melodies:
            # weighted HF by similarity
            weighted_feedback = sum(sim * rating for sim, _, rating in similar_melodies)
            total_sim = sum(sim for sim, _, _ in similar_melodies)
            if total_sim > 0:
                human_reward = weighted_feedback / total_sim
                # base reward + human feedback (70-30 split)
                return 0.7 * base_reward + 0.3 * human_reward
        return base_reward

    def _calculate_base_reward(self, note, rhythm, chord):
        # dynamically calc base reward based on training progress
        progress = min(1.0, self.total_episodes / 1000)
        reward = 0
        # Basic musicality rewards (decrease over time)
        basic_weight = 1.0 - (0.3 * progress)
        reward += self._chord_tone_reward(note, chord) * basic_weight
        reward += self._voice_leading_reward(note) * 2 * basic_weight
        reward += self._rhythm_coherence_reward(rhythm) * (1.5 + 0.5 * progress)
        reward += self._repetition_penalty(note) * (1.0 + 0.5 * progress)
        reward += self._calculate_position_penalty(note) * (0.8 + 0.4 * progress)
        # add cumulative leap penalty (want smaller jumps)
        if len(self.current_melody) >= 2:
            last_two_intervals = [abs(b[0] - a[0]) for a, b in zip(self.current_melody[-2:],
                                                                   self.current_melody[-1:] + [
                                                                       (note, rhythm, self.current_beat)])]
            if all(i > 4 for i in last_two_intervals):
                reward -= 5.0  # Strong penalty for consecutive leaps
        # motif development rewards (increase over time)
        motif_weight = 0.2 + (0.8 * progress)
        if len(self.current_melody) >= 3:
            reward += self._melodic_context_reward(note) * motif_weight
            reward += self._motif_development_reward(note, rhythm) * motif_weight
        # pattern formation rewards (increase over time)
        pattern_weight = 0.1 + (0.9 * progress)
        if self._is_chord_complete(rhythm):
            current_motif = self._get_current_chord_notes() + [(note, rhythm, self.current_beat)]
            reward += self._evaluate_pattern_formation(current_motif) * pattern_weight
            # save successful motifs
            if reward > 30:
                self._store_motif(current_motif)
        return reward

    def _melodic_context_reward(self):
        recent_notes = [n[0] for n in self.current_melody[-3:]]
        intervals = [b - a for a, b in zip(recent_notes[:-1], recent_notes)]
        # reward melodic direction changes
        direction_changes = sum(1 for i in range(len(intervals) - 1)
                                if intervals[i] * intervals[i + 1] < 0)
        # reward balanced interval sizes
        avg_interval = sum(abs(i) for i in intervals) / len(intervals)
        interval_balance = 1.0 if 2 <= avg_interval <= 4 else 0.5
        return direction_changes + interval_balance

    def _calculate_position_penalty(self, note):
        # balance melody around middle C
        center = 11.5
        distance_from_center = abs(note - center)
        # penalty for distance from center (stronger penalty for extreme ranges)
        position_penalty = -(distance_from_center ** 2) / 50
        # distribution across octaves
        if len(self.note_history) > 10:
            lower_octave_count = sum(1 for n in self.note_history[-20:] if n < 12)
            upper_octave_count = sum(1 for n in self.note_history[-20:] if n >= 12)
            # penalty for imbalanced octave usage
            octave_ratio = min(lower_octave_count, upper_octave_count) / max(lower_octave_count,
                                                                             upper_octave_count) if max(
                lower_octave_count, upper_octave_count) > 0 else 1
            distribution_bonus = octave_ratio * 2
            return position_penalty + distribution_bonus
        return position_penalty

    def _repetition_penalty(self, note):
        # penalize excessive repetition
        if len(self.current_melody) == 0:
            return 0
        penalty = 0
        note_in_octave = note % 12
        # short term repetition penalty
        if note == self.current_melody[-1][0]:
            penalty -= 4.0
            if len(self.current_melody) >= 2:
                if note == self.current_melody[-2][0]:
                    penalty -= 6.0
        # alternating pattern repetition penalty
        if len(self.current_melody) >= 3:
            recent_notes = [n[0] % 12 for n in self.current_melody[-3:]] + [note_in_octave]
            if len(recent_notes) >= 4:
                if (recent_notes[-1] == recent_notes[-3] and
                        recent_notes[-2] == recent_notes[-4]):
                    penalty -= 4.0
        return penalty

    def _is_good_transformation(self, motif1, motif2):
        # transposition
        if self._is_transposition(motif1, motif2):
            return True
        # rhythmic variation
        if self._is_rhythmic_variation(motif1, motif2):
            return True
        return False

    def _is_chord_complete(self, rhythm):
        return self.current_beat + rhythm == self.beats_per_chord

    def _get_current_chord_notes(self):
        # notes from chord
        current_chord_notes = []
        total_duration = 0
        # backwards until hit different chord
        for note in reversed(self.current_melody):
            if total_duration + note[1] > self.beats_per_chord:
                break
            current_chord_notes.insert(0, note)
            total_duration += note[1]
        return current_chord_notes

    def _chord_tone_reward(self, note, chord):
        # basic harmony reward
        note_in_octave = note % 12
        chord_tones = CHORD_TONES[chord]
        reward = 0
        if self._is_strong_beat():
            if note_in_octave == chord_tones[0]:  # root note
                reward += 2.0
            elif note_in_octave in chord_tones:  # other chord tones
                reward += 1.0
            else:  # non-chord tone on strong beat
                reward -= 1.0
        else:  # weak beat
            if note_in_octave in chord_tones:
                reward += 1.0
        return reward

    def _voice_leading_reward(self, note):
        # basic voice leading reward
        if len(self.current_melody) == 0:
            return 0
        prev_note = self.current_melody[-1][0]
        interval = abs(note - prev_note)
        if interval == 0:
            return -1.0  # penalty for repetition
        elif interval == 1:
            return 3.0  # reward for half steps
        elif interval == 2:
            return 2.0  # reward for whole steps
        elif interval <= 4:
            return 1.0  # low reward for small leaps (thirds)
        elif interval <= 7:
            return 0.0  # neutral for perfect fourths/fifths
        elif interval <= 10:
            return -2.0  # penalty for large leaps
        else:
            return -5.0  # large penalty for very large leaps

    def _is_rhythmic_variation(self, motif1, motif2):
        # motif2 is rhythmic variation of motif1
        if len(motif1) != len(motif2):
            return False
        # pitch intervals (should be same)
        intervals1 = [b[0] - a[0] for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b[0] - a[0] for a, b in zip(motif2[:-1], motif2[1:])]
        if intervals1 != intervals2:
            return False
        # rhythms (should be different but related)
        rhythms1 = [note[1] for note in motif1]
        rhythms2 = [note[1] for note in motif2]
        if rhythms1 == rhythms2:
            return False
        # total duration is same
        if abs(sum(rhythms1) - sum(rhythms2)) > 0.01:
            return False
        # rhythms are related (e.g., divided or combined)
        return True

    def _is_transposition(self, motif1, motif2):
        # motif2 is transposition of motif1 -> True
        if len(motif1) != len(motif2):
            return False
        # intervals between consecutive notes
        intervals1 = [b[0] - a[0] for a, b in zip(motif1[:-1], motif1[1:])]
        intervals2 = [b[0] - a[0] for a, b in zip(motif2[:-1], motif2[1:])]
        # intervals (should be identical)
        return intervals1 == intervals2

    def _get_previous_chord_motifs(self):
        # motifs from other chords in prog
        motifs = []
        current_motif = []
        total_duration = 0
        for note in reversed(self.current_melody[:-1]):  # exclude current motif
            current_motif.insert(0, note)
            total_duration += note[1]
            if abs(total_duration - self.beats_per_chord) < 0.01:
                motifs.insert(0, current_motif)
                current_motif = []
                total_duration = 0
        return motifs

    def _evaluate_pattern_formation(self, current_motif):
        reward = 0
        # motif evaluation
        reward += self._evaluate_motif_complexity(current_motif)
        reward += self._evaluate_motif_coherence(current_motif)
        # comparison with previous motifs
        previous_motifs = self._get_previous_chord_motifs()
        if previous_motifs:
            relation_reward = self._evaluate_motif_relationships(current_motif, previous_motifs)
            reward += relation_reward
        return reward

    def _evaluate_motif_complexity(self, motif):
        if not motif:
            return 0
        notes = [note[0] for note in motif]
        rhythms = [note[1] for note in motif]
        # melodic complexity
        intervals = [abs(b - a) for a, b in zip(notes[:-1], notes[1:])]
        unique_intervals = len(set(intervals))
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        # rhythmic complexity
        unique_rhythms = len(set(rhythms))
        rhythm_variety = unique_rhythms / len(rhythms)
        direction_changes = sum(1 for i in range(len(intervals) - 1)
                                if (intervals[i + 1] - intervals[i]) != 0)
        complexity = (unique_intervals * 2 +
                      (avg_interval if 2 <= avg_interval <= 5 else 0) +
                      rhythm_variety * 10 +
                      direction_changes * 3)
        return min(complexity, 30)  # clip reward

    def _evaluate_motif_relationships(self, current_motif, previous_motifs):
        # compare motif to previous motifs
        reward = 0
        # relationships with recent motifs
        for prev_motif in previous_motifs[-3:]:
            similarity = self._calculate_motif_similarity(current_motif, prev_motif)
            # reward moderate similarity
            if 0.3 <= similarity <= 0.7:
                reward += 15
            elif similarity > 0.9:  # penalize too much similarity
                reward -= 10
            #  reward good transformations
            if self._is_good_transformation(current_motif, prev_motif):
                reward += 20
        return reward

    def _rhythm_coherence_reward(self, rhythm):
        if len(self.current_melody) == 0:
            return 0
        recent_rhythms = [n[1] for n in self.current_melody[-4:]] + [rhythm]
        current_beat = self.current_beat
        reward = 0
        if current_beat % 2 == 0:  # strong beat
            if rhythm >= 1.0:  # long notes on strong beats
                reward += 5.0
            elif rhythm < 0.5:  # penalty for very short notes on strong beats
                reward -= 1.0
        else:  # weak beat
            if rhythm >= 1.0:  # penalty for long notes on weak beats
                reward -= 0.5
            elif rhythm == 0.5:  # reward eighth notes on weak beats
                reward += 1.5
        # penalty for off-beat syncopation
        if current_beat % 0.5 != 0:
            reward -= 1.0
        # reward consistent rhythmic groupings
        if len(recent_rhythms) >= 4:
            if all(r == 1.0 for r in recent_rhythms):  # quarter notes
                reward += 2.0
            elif all(r == 0.5 for r in recent_rhythms):  # eighth notes
                reward += 1.5
        return reward

    def _evaluate_motif_coherence(self, motif):
        notes = [n[0] for n in motif]
        rhythms = [n[1] for n in motif]
        beats = [n[2] for n in motif]
        reward = 0
        # reward for strong beat alignment
        strong_beat_notes = [note for note, _, beat in motif if beat % 2 == 0]
        if len(strong_beat_notes) >= 2:
            reward += 4.0
        # melodic patterns
        intervals = [b - a for a, b in zip(notes[:-1], notes[1:])]
        if len(set(intervals)) <= 2:
            reward += 3.0
        # rhythmic patterns
        rhythm_pairs = list(zip(rhythms[:-1], rhythms[1:]))
        if any(pair == (0.5, 0.5) for pair in rhythm_pairs):
            reward += 2.0
        # reward balanced use of note values
        quarter_notes = sum(1 for r in rhythms if r == 1.0)
        eighth_notes = sum(1 for r in rhythms if r == 0.5)
        if quarter_notes >= 2 and eighth_notes >= 2:
            reward += 3.0
        strong_beat_aligned = sum(1 for beat in beats if beat % 2 == 0)
        reward += strong_beat_aligned * 1.5
        # pattern development
        if len(set(intervals)) == 1:  # sequence
            reward += 3.0
        elif len(set(intervals)) == 2:  # alternating
            reward += 2.0
        return reward

    def _motif_development_reward(self, note, rhythm):
        if len(self.current_melody) < 4:
            return 0
        recent_notes = [n[0] for n in self.current_melody[-3:]] + [note]
        recent_rhythms = [n[1] for n in self.current_melody[-3:]] + [rhythm]
        reward = 0
        # motif recognition
        intervals = [b - a for a, b in zip(recent_notes[:-1], recent_notes[1:])]
        # reward interval patterns (e.g., consistent step-wise motion or arpeggios)
        if len(set(intervals)) <= 2:  # similar intervals
            reward += 3.0
        # reward melodic direction consistency
        if all(i > 0 for i in intervals) or all(i < 0 for i in intervals):
            reward += 2.5
        # reward  rhythmic patterns
        if recent_rhythms[-2:] == [0.5, 0.5]:
            reward += 2.0
        # strong beat emphasis with motivic context
        if self.current_beat % 2 == 0:
            if rhythm == 1.0:
                reward += 4.0
                if len(set(intervals)) == 1:
                    reward += 2.0
            elif rhythm == 0.5:  #
                reward += 2.0
        return reward

    def _calculate_motif_similarity(self, motif1, motif2):
        # similarity between motifs
        if len(motif1) != len(motif2):
            return 0.0
        # extract notes
        notes1 = motif1 if isinstance(motif1[0], (int, float)) else [note[0] for note in motif1]
        notes2 = motif2 if isinstance(motif2[0], (int, float)) else [note[0] for note in motif2]
        # compare intervals
        intervals1 = [b - a for a, b in zip(notes1[:-1], notes1[1:])]
        intervals2 = [b - a for a, b in zip(notes2[:-1], notes2[1:])]
        # calc interval similarity
        interval_similarity = sum(1 for a, b in zip(intervals1, intervals2)
                                  if abs(a - b) <= 1) / len(intervals1)
        # compare contours
        contour1 = [1 if i > 0 else -1 if i < 0 else 0 for i in intervals1]
        contour2 = [1 if i > 0 else -1 if i < 0 else 0 for i in intervals2]
        contour_similarity = sum(1 for a, b in zip(contour1, contour2)
                                 if a == b) / len(contour1)
        return (interval_similarity + contour_similarity) / 2

    def _store_motif(self, motif):
        # save unique motifs
        if len(motif) < 4:
            return
        # calc similarity with existing motifs
        max_similarity = 0
        if self.motif_memory:
            similarities = [self._calculate_motif_similarity(motif, m) for m in self.motif_memory]
            max_similarity = max(similarities)
        # store if unique
        if max_similarity < 0.7:
            self.motif_memory.append(motif)
            # update histories
            notes = [n[0] for n in motif]
            rhythms = [n[1] for n in motif]
            self.note_history.extend(notes)
            self.rhythm_history.extend(rhythms)
            # forget after a while
            if len(self.note_history) > 1000:
                self.note_history = self.note_history[-1000:]
                self.rhythm_history = self.rhythm_history[-1000:]
