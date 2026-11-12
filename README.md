# tictacrobo

A minimal YOLO object-detection example using the pretrained YOLO11 nano model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run detection

Pass an image, video, directory, or camera index. The first run downloads
`yolo11n.pt` automatically.

```bash
python detect.py path/to/image.jpg
python detect.py 0
```

Annotated results are written to `runs/detect/`.
