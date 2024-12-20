from agent import Qtar
from midi_utils import convert_to_midi
import argparse
import os
import numpy as np
from collections import defaultdict


def analyze_melody(solo):
    analysis = {}
    # Basic statistics
    analysis['total_notes'] = len(solo)
    analysis['unique_pitches'] = len(set(note for note, _, _ in solo))
    analysis['unique_pitch_classes'] = len(set(note % 12 for note, _, _ in solo))
    analysis['avg_duration'] = sum(duration for _, duration, _ in solo) / len(solo)
    # Melodic analysis
    intervals = [abs(b[0] - a[0]) for a, b in zip(solo[:-1], solo[1:])]
    analysis['avg_interval'] = np.mean(intervals)
    analysis['max_interval'] = max(intervals)
    analysis['stepwise_motion_percent'] = sum(1 for i in intervals if i <= 2) / len(intervals) * 100
    # Rhythmic analysis
    durations = [d for _, d, _ in solo]
    analysis['unique_rhythms'] = len(set(durations))
    analysis['rhythm_variety'] = analysis['unique_rhythms'] / len(durations)
    # Pattern analysis
    note_sequence = [n for n, _, _ in solo]
    patterns = find_patterns(note_sequence)
    analysis['repeated_patterns'] = len(patterns)
    return analysis


def find_patterns(sequence, min_length=3):
    patterns = defaultdict(int)
    for i in range(len(sequence) - min_length + 1):
        for j in range(i + min_length, len(sequence) + 1):
            pattern = tuple(sequence[i:j])
            if sequence.count(pattern) > 1:
                patterns[pattern] += 1
    return patterns


def format_note(note):
    # note + octave
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note // 12 + 4
    note_name = note_names[note % 12]
    return f"{note_name}{octave}"


def main():
    parser = argparse.ArgumentParser(description='Generate music with Q-tar')
    parser.add_argument('--model_path', type=str, default='models/pretrained_qtar_model.pt',
                        help='Path to saved model')
    parser.add_argument('--output', type=str, default='qtar_solo.mid',
                        help='Output MIDI file path')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--num_solos', type=int, default=1,
                        help='Number of solos to generate')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    # init qtar
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        use_human_feedback=False
    )
    # load model
    print(f"Loading model from {args.model_path}")
    qtar.load_model(args.model_path)
    best_solo = None
    best_score = float('-inf')
    for i in range(args.num_solos):
        print(f"\nGenerating solo {i + 1}/{args.num_solos}...")
        solo = qtar.generate_solo(temperature=args.temperature)
        # analyze
        analysis = analyze_melody(solo)
        score = evaluate_solo(analysis)
        if score > best_score:
            best_score = score
            best_solo = (solo, analysis)
        if args.num_solos > 1:
            print(f"Solo {i + 1} score: {score:.2f}")
    # get best solo for output
    solo, analysis = best_solo
    # analysis
    print("\nGenerated Solo Analysis:")
    print("-" * 50)
    print(f"Total notes: {analysis['total_notes']}")
    print(f"Unique pitches: {analysis['unique_pitches']}")
    print(f"Stepwise motion: {analysis['stepwise_motion_percent']:.1f}%")
    print(f"Average interval: {analysis['avg_interval']:.2f} semitones")
    print(f"Rhythm variety: {analysis['rhythm_variety']:.2f}")
    print(f"Repeated patterns: {analysis['repeated_patterns']}")
    # solo
    print("\nSolo Details:")
    print("-" * 50)
    print("Note    Duration  Beat  Chord")
    print("-" * 50)
    current_chord_idx = 0
    current_beat = 0
    for note, duration, beat in solo:
        if beat == 0 and current_beat != 0:
            current_chord_idx = (current_chord_idx + 1) % len(qtar.chord_progression)
        current_chord = qtar.chord_progression[current_chord_idx]
        note_name = format_note(note)
        print(f"{note_name:<7} {duration:<8.2f} {beat:<5.1f} {current_chord}")
        current_beat = beat
    # Save to MIDI file
    try:
        print(f"\nSaving to MIDI file: {args.output}")
        convert_to_midi(solo, qtar.chord_progression, args.output)
        print(f"Successfully saved to {args.output}")
    except Exception as e:
        print(f"Error saving MIDI file: {str(e)}")
        print("Make sure pretty_midi is installed: pip install pretty_midi")


def evaluate_solo(analysis):
    score = 0
    # Reward variety in pitches and rhythms
    score += analysis['unique_pitch_classes'] * 2
    score += analysis['rhythm_variety'] * 20
    # Reward balanced use of intervals
    if 2 <= analysis['avg_interval'] <= 4:
        score += 10
    # Reward stepwise motion
    score += min(analysis['stepwise_motion_percent'] / 10, 5)
    # Reward pattern usage but not too much
    pattern_score = min(analysis['repeated_patterns'], 5) * 2
    score += pattern_score
    return score


if __name__ == "__main__":
    main()