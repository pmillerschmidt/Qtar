import os
import pretty_midi


def convert_to_midi(melody, chord_progression, filename="qtar_solo.mid", base_note=60):
    pm = pretty_midi.PrettyMIDI()

    # Create guitar track for the solo
    guitar = pretty_midi.Instrument(program=24)  # 24 is acoustic guitar
    current_time = 0.0

    for note, duration, _ in melody:
        # Create a new note
        note_number = base_note + note  # Middle C (60) + offset
        note = pretty_midi.Note(
            velocity=100,
            pitch=note_number,
            start=current_time,
            end=current_time + duration
        )
        guitar.notes.append(note)
        current_time += duration

    # Create piano track for chords
    piano = pretty_midi.Instrument(program=0)  # 0 is acoustic grand piano

    # Define chord notes (relative to root)
    chord_types = {
        '': [0, 4, 7],  # Major triad
        'm': [0, 3, 7],  # Minor triad
        '7': [0, 4, 7, 10],  # Dominant 7th
    }

    # Define root notes
    root_notes = {
        'C': 60,
        'D': 62,
        'E': 64,
        'F': 65,
        'G': 67,
        'A': 69,
        'B': 71,
    }

    # Add chords
    chord_time = 0.0
    beats_per_chord = 4.0  # Assuming 4 beats per chord

    for chord in chord_progression:
        # Parse chord name and type
        if chord.endswith('m'):
            root = chord[:-1]
            chord_type = 'm'
        elif chord.endswith('7'):
            root = chord[:-1]
            chord_type = '7'
        else:
            root = chord
            chord_type = ''

        # Get root note number
        root_note = root_notes[root]

        # Add each note in the chord
        for note_offset in chord_types[chord_type]:
            note = pretty_midi.Note(
                velocity=80,  # Slightly softer than the solo
                pitch=root_note + note_offset,
                start=chord_time,
                end=chord_time + beats_per_chord
            )
            piano.notes.append(note)

        chord_time += beats_per_chord

    # Add both instruments to the MIDI file
    pm.instruments.append(guitar)
    pm.instruments.append(piano)

    # Write the MIDI file
    pm.write(filename)
    print(f"MIDI file saved as {filename}")

def save_midi(epoch, env):
    """Save the current melody as a MIDI file using pretty_midi"""
    if not os.path.exists('midi'):
        os.makedirs('midi')  # Ensure the midi folder exists

    midi_filename = f"midi/epoch_{epoch}.mid"

    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI()

    # Create an instrument (for example, Electric Piano)
    instrument = pretty_midi.Instrument(program=12)  # Program 12 corresponds to Electric Piano

    # Add notes from the current melody to the instrument
    for note, duration, beat_position in env.current_melody:
        # Convert the note and duration into a PrettyMIDI note
        start_time = beat_position * 60 / env.beats_per_chord  # Convert to seconds (assuming 120 bpm)
        end_time = start_time + (duration * 60 / env.beats_per_chord)
        midi_note = pretty_midi.Note(velocity=100, pitch=int(note), start=start_time, end=end_time)
        instrument.notes.append(midi_note)

    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(instrument)

    # Write the MIDI data to file
    midi_data.write(midi_filename)
    print(f"Saved MIDI file for epoch {epoch} to {midi_filename}")