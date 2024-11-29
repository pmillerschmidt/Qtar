"use client";

import React, { useRef, useEffect } from 'react';

const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const TOTAL_NOTES = 24; // Two octaves
const PIXELS_PER_BEAT = 40;
const NOTE_HEIGHT = 20;

const CHORD_NOTES = {
  'C':  ['C', 'E', 'G'],
  'Am': ['A', 'C', 'E'],
  'F':  ['F', 'A', 'C'],
  'G':  ['G', 'B', 'D']
};

interface PianoRollProps {
  phrase: {
    notes: Array<{
      note: string;
      duration: number;
      beat: number;
    }>;
    chords: string[];
  };
  isPlaying: boolean;
  currentBeat: number | null;
}

const PianoRoll: React.FC<PianoRollProps> = ({ phrase, isPlaying, currentBeat }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = PIXELS_PER_BEAT * 16; // 16 beats total
    const height = NOTE_HEIGHT * TOTAL_NOTES;

    // Set canvas size
    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';

    // Draw chord sections with subtle background colors
    phrase.chords.forEach((chord, i) => {
      ctx.fillStyle = `rgba(156, 163, 175, ${i % 2 ? 0.1 : 0.05})`;
      ctx.fillRect(i * 4 * PIXELS_PER_BEAT, 0, 4 * PIXELS_PER_BEAT, height);
    });

    // Vertical lines (beats)
    for (let i = 0; i <= 16; i++) {
      ctx.lineWidth = i % 4 === 0 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(i * PIXELS_PER_BEAT, 0);
      ctx.lineTo(i * PIXELS_PER_BEAT, height);
      ctx.stroke();
    }

    // Horizontal lines (notes)
    for (let i = 0; i <= TOTAL_NOTES; i++) {
      ctx.lineWidth = i % 12 === 0 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(0, i * NOTE_HEIGHT);
      ctx.lineTo(width, i * NOTE_HEIGHT);
      ctx.stroke();
    }

    // Draw note names on the left
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    for (let i = 0; i < TOTAL_NOTES; i++) {
      const noteName = NOTES[i % 12];
      ctx.fillText(noteName, 5, (TOTAL_NOTES - i) * NOTE_HEIGHT - 5);
    }

    // Draw chord names
    ctx.fillStyle = '#6b7280';
    ctx.font = '14px sans-serif';
    phrase.chords.forEach((chord, i) => {
      ctx.fillText(chord, (i * 4 + 0.5) * PIXELS_PER_BEAT, 20);
    });

    // Draw notes
    phrase.notes.forEach(note => {
      const noteIndex = NOTES.indexOf(note.note);
      const y = (TOTAL_NOTES - 1 - noteIndex) * NOTE_HEIGHT;
      const x = note.beat * PIXELS_PER_BEAT;
      const width = note.duration * PIXELS_PER_BEAT;

      // Note rectangle
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(x, y, width, NOTE_HEIGHT - 1);

      // Note label
      ctx.fillStyle = '#fff';
      ctx.font = '12px sans-serif';
      if (width > 30) {
        ctx.fillText(note.note, x + 5, y + 14);
      }
    });

    // Draw playhead if playing
    if (isPlaying && currentBeat !== null) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(currentBeat * PIXELS_PER_BEAT, 0);
      ctx.lineTo(currentBeat * PIXELS_PER_BEAT, height);
      ctx.stroke();
    }

  // Draw chord notes (lighter backgrounds)
    phrase.chords.forEach((chord, i) => {
      const chordNotes = CHORD_NOTES[chord];
      chordNotes.forEach(noteName => {
        const noteIndex = NOTES.indexOf(noteName);
        const y = (TOTAL_NOTES - 1 - noteIndex) * NOTE_HEIGHT;
        ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
        ctx.fillRect(i * 4 * PIXELS_PER_BEAT, y, 4 * PIXELS_PER_BEAT, NOTE_HEIGHT - 1);
      });
    });

  }, [phrase, isPlaying, currentBeat]);

  return (
    <div className="relative overflow-x-auto border rounded">
      <canvas ref={canvasRef} className="min-w-full" />
    </div>
  );
};

export default PianoRoll;