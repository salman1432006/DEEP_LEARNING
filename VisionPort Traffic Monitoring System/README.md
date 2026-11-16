# VisionPort – Tasks 1 to 20

## Run on Windows
1. Install Python 3.10 or 3.11 (64-bit).
2. Open this folder in VS Code.
3. In PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
4. Open http://127.0.0.1:8000/

The first run may download `yolov8n.pt`.

## Assignment coverage
1. Environment: requirements.txt and setup instructions.
2. Camera capture: POST /camera/capture (camera connected to the computer running backend).
3. Image reading: OpenCV decode and /detect/.
4. Video loading/playback: /analyze-video/ processes uploaded video and frontend plays output.
5. Frame extraction: saves one frame per second to extracted_frames/.
6-8. YOLO detects cars, motorcycles, buses, trucks and bicycles with labels/confidence.
9. Vehicle count per analyzed frame and average/max count report.
10. Low/Medium/High density heuristic.
11. Congestion classification/alert heuristic.
12. ByteTrack tracking IDs appear on annotated video.
13. Approximate speed from tracked centroid movement.
14. Traffic-light class detection and HSV colour heuristic.
15. Approximate stopped-track waiting-time estimate.

## Important limitations
- COCO-pretrained YOLOv8n is a general model, not specifically trained for all Indian vehicle categories. It does not have an auto-rickshaw class.
- Speed needs a fixed camera and calibration. Set PIXELS_PER_METER to a measured value before treating estimates as meaningful.
- Waiting time is inferred from tracks moving under 3 km/h; it is not a real traffic-signal cycle timer.
- Density/congestion thresholds are simple demonstration thresholds and should be calibrated for the camera scene.
- Traffic-light colour is a heuristic inside the detected traffic-light crop and can return Unknown.
- Video is processed synchronously; long videos can take time. VIDEO_FRAME_STEP defaults to 2. Set to 1 for every frame (slower, better temporal detail), or 3 for faster but less accurate tracking.
- Camera capture endpoint accesses a camera attached to the backend computer, not the browser user's camera.
