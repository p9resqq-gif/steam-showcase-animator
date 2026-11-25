# steam-showcase-animator

Turn a **still character art** into a **looping animated Steam artwork showcase** (GIF) with plain Python: OpenCV + NumPy + ffmpeg. No After Effects.

Effects:
- automatic cutout of the subject (dark backgrounds)
- violet color grade
- pulsing neon glow on bright details (eyes, lines) with random "electric" spikes
- subtle breathing motion
- rim light behind the subject
- drifting particle background (procedural, or your own video cropped at the showcase position)
- palette-optimized GIF to fit Steam's size limits

## Install
```bash
pip install opencv-python numpy scipy imageio-ffmpeg
```

## Usage
```bash
python animate.py art.jpg --out showcase.gif
# limit the glow to the face area (y0 y1 x0 x1 in source pixels)
python animate.py art.jpg --glow-box 240 385 240 480
# use your profile background video so the showcase lines up with it
python animate.py art.jpg --bg background.mp4 --bg-offset 495 256
```

Options: `--width/--height` (default 506×824, Steam showcase), `--fps` (8), `--seconds` (10), `--colors` (96, fewer = smaller file).

## How it works
1. Luminance threshold → largest connected blobs → closed + hole-filled → soft alpha.
2. Glow mask = bright pinkish pixels (optionally inside a box), dilated.
3. Each frame: `pulse = sine + random spikes`, scale ±1.2% via `warpAffine`, two Gaussian blooms (σ 7 and 2) tinted violet, composite over the background.
4. ffmpeg `palettegen` + `paletteuse` (no dither) → small looping GIF.

## Limitations
- Cutout expects the subject on a dark background.
- Glow mask is color-based; a face-landmark version would be nicer — PRs welcome.

MIT License
