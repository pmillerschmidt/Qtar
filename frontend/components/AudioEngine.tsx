import { useRef } from 'react';

const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const SECONDS_PER_BEAT = 0.5; // 120 BPM

export const AudioEngine = () => {
  const audioContext = useRef(null);
  const activeNodes = useRef([]);

  const CHORD_NOTES = {
    'C':  ['C4', 'E4', 'G4'],
    'Am': ['A4', 'C4', 'E4'],
    'F':  ['F4', 'A4', 'C5'],
    'G':  ['G4', 'B4', 'D5']
  };

  const init = () => {
    audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
  };

  const stopAllNodes = () => {
    activeNodes.current.forEach(node => {
      try {
        node.stop();
        node.disconnect();
      } catch (e) {
        // Node might have already stopped
      }
    });
    activeNodes.current = [];
  };

  const getNoteFrequency = (noteWithOctave) => {
    // Extract note name and octave (e.g., "C4" -> ["C", "4"])
    const noteName = noteWithOctave.slice(0, -1);
    const octave = parseInt(noteWithOctave.slice(-1));

    // Calculate semitones from A4 (which is 440Hz)
    const noteIndex = NOTES.indexOf(noteName);
    if (noteIndex === -1) return 440; // Default to A4 if invalid note

    // Calculate the number of semitones from A4
    const A4_OCTAVE = 4;
    const A4_INDEX = NOTES.indexOf('A');
    const semitones = (octave - A4_OCTAVE) * 12 + (noteIndex - A4_INDEX);

    // Calculate frequency using the equal temperament formula
    return 440 * Math.pow(2, semitones / 12);
  };

  const playNote = (noteWithOctave, time, duration, isChord = false) => {
    if (!audioContext.current) init();

    const freq = getNoteFrequency(noteWithOctave);
    const osc = audioContext.current.createOscillator();
    const gain = audioContext.current.createGain();

    // Use different waveforms for melody and chords
    osc.type = isChord ? 'triangle' : 'sine';
    osc.frequency.value = freq;

    osc.connect(gain);
    gain.connect(audioContext.current.destination);

    const startTime = audioContext.current.currentTime + time;

    // Different volumes for melody and chords
    const maxGain = isChord ? 0.1 : 0.2;
    gain.gain.setValueAtTime(0, startTime);
    gain.gain.linearRampToValueAtTime(maxGain, startTime + 0.01);
    gain.gain.linearRampToValueAtTime(0, startTime + duration - 0.01);

    osc.start(startTime);
    osc.stop(startTime + duration);

    activeNodes.current.push(osc);
    return osc;
  };

  const playChord = (chord, time, duration) => {
    CHORD_NOTES[chord].forEach(note => {
      playNote(note, time, duration, true);
    });
  };

  const playPhrase = (phrase, startTime = 0) => {
    if (!audioContext.current) init();

    stopAllNodes();

    const currentTime = audioContext.current.currentTime;

    // Play melody notes
    phrase.notes
      .filter(note => (note.beat * SECONDS_PER_BEAT) >= startTime)
      .forEach(note => {
        playNote(
          note.note,
          (note.beat * SECONDS_PER_BEAT) - startTime,
          note.duration * SECONDS_PER_BEAT
        );
      });

    // Play chord progression
    phrase.chords.forEach((chord, i) => {
      playChord(
        chord,
        (i * 4 * SECONDS_PER_BEAT) - startTime,
        4 * SECONDS_PER_BEAT
      );
    });

    return currentTime;
  };

  return { playPhrase, stopAllNodes };
};

export default AudioEngine;