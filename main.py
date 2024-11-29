from agent import Qtar
from midi_utils import convert_to_midi
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Generate music with Q-tar')
    parser.add_argument('--model-path', type=str, default='models/qtar_model.pt',
                        help='Path to saved model')
    parser.add_argument('--output', type=str, default='qtar_solo.mid',
                        help='Output MIDI file path')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Create Q-tar instance and load model
    qtar = Qtar(scale='C_MAJOR', progression_type='I_VI_IV_V')
    chord_progression = qtar.chord_progression
    print(f"Loading model from {args.model_path}")
    metadata = qtar.load_model(args.model_path)
    if metadata:
        print("Model metadata:", metadata)

    # Generate solo
    print("\nGenerating solo...")
    solo = qtar.generate_solo(chord_progression)

    # Print the solo in a readable format
    print("\nGenerated Solo:")
    current_chord_idx = 0
    current_beat = 0

    # Note name mapping
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    print("\nChord Progression:", ' | '.join(chord_progression))
    print("\nNote  Duration  Beat  Chord")
    print("-" * 40)

    for note, duration, beat in solo:
        if beat == 0 and current_beat != 0:
            current_chord_idx = (current_chord_idx + 1) % len(chord_progression)
        current_chord = chord_progression[current_chord_idx]
        note_name = note_names[note % 12]
        print(f"{note_name:<6} {duration:<9.2f} {beat:<5.1f} {current_chord}")
        current_beat = beat

    # Save to MIDI file
    try:
        print(f"\nSaving to MIDI file: {args.output}")
        convert_to_midi(solo, chord_progression, args.output)
    except Exception as e:
        print(f"Could not save MIDI file: {str(e)}")
        print("Make sure pretty_midi is installed: pip install pretty_midi")

    # Print statistics about the solo
    total_notes = len(solo)
    unique_notes = len(set(note for note, _, _ in solo))
    avg_duration = sum(duration for _, duration, _ in solo) / total_notes

    print(f"\nSolo statistics:")
    print(f"Total notes: {total_notes}")
    print(f"Unique notes: {unique_notes}")
    print(f"Average note duration: {avg_duration:.2f} beats")


if __name__ == "__main__":
    main()