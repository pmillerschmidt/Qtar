from collections import deque


class HumanFeedbackBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)

    def add_feedback(self, melody_sequence, rating):
        # save sequence + rating
        self.buffer.append((melody_sequence, rating))

    def get_similar_melodies(self, current_melody, n_samples=5):
        # given melody -> fetch similar
        if not self.buffer:
            return []

        similarities = []
        for stored_melody, rating in self.buffer:
            sim_score = self._calculate_similarity(current_melody, stored_melody)
            similarities.append((sim_score, stored_melody, rating))
        # top n similar
        similarities.sort(reverse=True)
        return similarities[:n_samples]

    def _calculate_similarity(self, melody1, melody2):
        # similarity score between melodies
        notes1 = [note for note, _, _ in melody1]
        notes2 = [note for note, _, _ in melody2]
        min_len = min(len(notes1), len(notes2))
        if min_len == 0:
            return 0
        # calc similarity metrics
        # 1. Note similarity
        note_sim = sum(1 for i in range(min_len) if notes1[i] % 12 == notes2[i] % 12) / min_len
        # 2. Interval similarity
        if min_len > 1:
            intervals1 = [notes1[i + 1] - notes1[i] for i in range(min_len - 1)]
            intervals2 = [notes2[i + 1] - notes2[i] for i in range(min_len - 1)]
            interval_sim = sum(1 for i in range(len(intervals1))
                               if abs(intervals1[i] - intervals2[i]) <= 2) / (min_len - 1)
        else:
            interval_sim = 0
        # Combine similarities
        return 0.6 * note_sim + 0.4 * interval_sim


def collect_human_feedback(solo, feedback_buffer):
    # get human feedback
    print("\nPlease rate the following musical sequences (1-5, or 'q' to finish):")
    print("1: Poor, 2: Fair, 3: Good, 4: Very Good, 5: Excellent")
    # break solo into phrases for rating
    phrase_length = 8  # notes / phrase
    for i in range(0, len(solo), phrase_length):
        phrase = solo[i:i + phrase_length]
        # info
        print("\nPhrase:")
        for note, duration, beat in phrase:
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note % 12]
            print(f"{note_name:<3} Duration: {duration:<4} Beat: {beat:<4}")
        # rating
        while True:
            rating = input("\nRate this phrase (1-5, or 'q' to quit): ")
            if rating.lower() == 'q':
                return
            try:
                rating = int(rating)
                if 1 <= rating <= 5:
                    # scale reward -> (-1, 1)
                    reward = (rating - 3) / 2
                    feedback_buffer.add_feedback(phrase, reward)
                    break
                else:
                    print("Please enter a rating between 1 and 5")
            except ValueError:
                print("Please enter a valid rating")