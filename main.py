"""NeonSight — real-time object detection with YOLO26 on Apple Silicon.

Run:  python main.py            (MacBook camera)
      python main.py --source demo.mp4
"""

import sys

from neonsight.app import main

if __name__ == "__main__":
    sys.exit(main())
