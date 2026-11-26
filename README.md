# NeonSight

Real-time object detection for the MacBook camera, built on **YOLO26** and the Apple GPU (MPS).
Live video at camera rate (30 fps) with animated, stylized annotations, object tracking and
several extra YOLO26 vision layers you can switch on live.

## Features

**Detection & tracking**
- YOLO26 detector + ByteTrack: every object keeps an ID, a smoothed box and a motion trail
- Three model sizes, switchable live from the panel: Light (YOLO26-N, fastest), Medium (YOLO26-S),
  Heavy (YOLO26-M, most accurate)
- 10 selectable objects (checkbox panel, live count per object): Person, Phone, Cup, Bottle,
  Glass, Laptop, Mouse, Keyboard, Book, Scissors
- **Ask for anything**: type object names (e.g. `watch, glasses, hat`) and YOLOE-26 open-vocabulary
  detection finds them, including things outside the 80 COCO classes. Words autocomplete, typos and
  nonsense are rejected with "did you mean" suggestions, and each word shows a live status
  (looking… / found / not seen yet). Some words, such as body parts like `hair`, are known only
  weakly, and the app says so rather than staying silent

**AI layers** (YOLO26 tasks, toggle live)
- 🧬 **Masks**: instance segmentation with glowing outlines
- 🕺 **Pose**: neon skeletons
- 📏 **Depth**: metric depth view, plus the distance in meters on every object label
- 🌫️ **Portrait**: depth-based background blur that keeps the nearest subject sharp

**Look & feel**
- Three annotation styles: **Neon** (glowing corner brackets), **Glass** (frosted boxes with a
  confidence ring), **Target** (rotating sci-fi reticles with callouts)
- Effects: motion trails, spotlight (dims everything except detected objects), activity heatmap,
  mirror view
- Live scene caption (e.g. "1 person with Phone & Cup · Laptop · nearest: Phone 0.6 m")
- HUD with FPS, AI latency, headroom and objects seen

**Demo extras**
- 🎯 **Mission mode**: a short game of a few rounds (5 by default). Each round asks "Show me a CUP"
  with a countdown. Show it to score (faster = more points, with confetti), or press **Skip** (key N)
  if you don't have it. A progress row shows every round (found / skipped / time's up), and a
  results card at the end lists them all with the total score.
- 🙈 **Privacy**: pixelates faces, using pose keypoints to locate them
- 🔊 **Voice**: spoken call-outs when new objects appear (macOS `say`)
- 📸 **Snapshot**: saves the annotated view to `snapshots/`

## Setup

Requires macOS on Apple Silicon and Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights live in `models/`. Any missing file is downloaded automatically on first run.

## Run

```bash
source .venv/bin/activate
python main.py
```

The first time, macOS asks for camera access for your terminal app. Allow it in
**System Settings › Privacy & Security › Camera**, then start the app again.

Other sources:

```bash
python main.py --source 1            # another camera index (e.g. iPhone Continuity Camera)
python main.py --source demo.mp4     # a video file (loops)
python main.py --config my.json      # an alternative config file
```

## Configuration: `config/config.json`

| Key | Meaning |
| --- | --- |
| `model` | model the app starts with: `light` (YOLO26-N), `medium` (YOLO26-S, default) or `heavy` (YOLO26-M). The **Model** buttons in the panel switch between them live (all three are preloaded, so switching is instant) |
| `model_presets` | detector weights for each preset |
| `layers` | fixed models for the AI layers (masks, pose, depth, YOLOE open vocabulary); they don't change with the preset |
| `inference` | device (`mps`), image size, FP16, default confidence, tracker |
| `camera` | index, requested resolution / fps, mirror |
| `mission` | the game: `rounds`, `seconds_per_round`, and `objects` it asks for (names from `objects`; an empty list means the ticked objects + typed words). Mission objects are detected even when unticked |
| `objects` | the 10 checkbox objects: COCO class name, label, emoji, color, enabled |

Any of the 80 COCO class names can be used in `objects`.

`config/vocabulary.txt` lists the words the "Ask for anything" box accepts (about 2,400, built from the
COCO, LVIS, Objects365 and Open Images class names). Add a line to allow a new word. Phrases that end
in a known word, like `red cup`, are accepted too. Typed words are matched at 70% of the confidence
slider, because open-vocabulary scores run lower.

## Keyboard

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| Space | pause / resume | G | masks |
| S | snapshot | K | pose |
| M | mission mode | D | depth |
| 1 / 2 / 3 | Neon / Glass / Target | B | portrait |
| T | trails | L | spotlight |
| H | heatmap | P | privacy |
| V | voice | C | next camera |
| Tab | hide / show panel | F | fullscreen |
| N | skip mission round | Q | quit |

## Performance (MacBook Pro, M4 Pro, 1280×720)

Raw model speed on MPS, FP16, 640 px:

| Model | ms / frame | fps |
| --- | --- | --- |
| YOLO26-N (light) | 8.6 | 116 |
| YOLO26-S (medium) | 12.1 | 83 |
| YOLO26-M (heavy) | 18.4 | 54 |
| YOLO26-S-seg (masks) | 15.3 | 65 |
| YOLO26-N-pose | 9.1 | 107 |
| YOLO26-N-depth | 12.7 | 79 |
| YOLOE-26S (open vocab) | 16.4 | 61 |

Inside the app (tracking, drawing and UI included), detection runs at the full camera rate
(30 fps) with every preset. With `light` or `medium`, each AI layer on its own (and pose + depth
together) also stays at 30 fps. Pose and depth are scheduled adaptively: they use whatever time
is left in each frame and reuse their last result otherwise, so the video stays smooth. Stacking
every layer at once (masks + pose + depth + privacy) runs at about 25 fps on `medium`.

## Project layout

```
main.py                 entry point
config/config.json      model preset, layers, camera, the 10 objects
config/vocabulary.txt   words accepted by "Ask for anything"
neonsight/
  app.py                main window, shortcuts, wiring
  camera.py             threaded grabber (always the newest frame, no buffering lag)
  detector.py           inference worker: detect / segment / YOLOE + pose + depth
  tracks.py             smoothed tracks, trails, scene caption
  view.py               the video canvas: styles, HUD, mission banner, effects
  sidebar.py            control panel
  vocab.py              word check and suggestions for "Ask for anything"
  effects.py            heatmap, confetti, pulse rings
  mission.py            mission-mode game logic
  voice.py              spoken call-outs
models/                 YOLO26 weights
```
