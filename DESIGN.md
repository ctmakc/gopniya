# Design decision: demo page

Subject: a clinic's text crosses from the trusted side (the clinic's own browser) to a
cloud model. The page has to make that boundary visible and show what the agent did on
each side of it.

## Three directions considered

1. **Carbon OPD slip** — the pale blue duplicate paper of a government hospital OPD slip,
   indigo carbon ink for type, redaction as a marker strike that wipes across each
   identifier and leaves a placeholder chip; one rubber-stamp red for the "sent" stamp.
   Anek Latin + Anek Devanagari (Ek Type, one family for both scripts).
2. **Lab report grid** — white thermal-print look, monospace everywhere, green dot-matrix
   accent, table rules. Rejected: reads as a terminal, and Devanagari in a mono face is weak.
3. **Night ward** — dark slate, teal monitor glow. Rejected: neon-on-dark is a ban, and
   clinics use this on bright screens.

## Locked: Carbon OPD slip

- Background: slip blue `--paper #E4EAF2`, card paper `--sheet #F3F6FA`; ink `--ink #1E2A5A`
  (carbon indigo), muted `--ink-2 #4A5680`, rule `--rule #B9C4D8`.
- Accent: stamp red `--stamp #B8322A`, used for the perimeter line label and the SENT stamp only.
- Redaction bar: `--bar #1E2A5A` (ink) with placeholder text in `--sheet`.
- Type: Anek Latin (UI, headings at wdth 75–100, weight 700) + Anek Devanagari for Hindi;
  Spline Sans Mono for placeholders and step timings.
- Radius: 2px (paper). Borders over shadows; one hard 2px offset shadow on the slip.
- Layout primitive: the two-sided slip — clinic side | perimeter line | cloud side, repeated
  for the note, the request and the answer.
- Signature: the dashed perimeter line with the agent's steps stamped along it.
- Motion: marker wipe (scaleX 0→1 from the left, 280ms expo-out, 40ms stagger) over each
  detected identifier; steps stamp in (scale 1.06→1, 160ms). Reduced motion: no wipe.

Ledger check: differs from RedactGate "REDACTION FIELD" (vellum + Geist + seal cyan + chip
tokens) on background, display face, accent, signature and motion.
