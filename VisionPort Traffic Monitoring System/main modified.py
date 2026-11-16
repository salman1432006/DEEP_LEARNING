"""
VisionPort: Intelligent Vehicle Detection and Traffic Monitoring
Tasks 1-20 implementation.

Run:
    python -m pip install -r requirements.txt
    python -m uvicorn main:app --host 127.0.0.1 --port 8000

Notes:
- YOLOv8n COCO classes include car, motorcycle, bus, truck, bicycle and traffic light.
- Speed/waiting-time values are estimates and require camera calibration.
- Density/congestion thresholds are configurable demonstration heuristics.
- Tasks 16-20 use virtual lines, trajectory heuristics and a configurable restricted zone.
- Configure the camera geometry and expected direction for reliable event counts.
- Motorcycle rider counts are approximate person-box overlap estimates.
"""
import os
import cv2
import json
import time
import uuid
import shutil
import threading
import numpy as np
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
FRAMES = BASE / "extracted_frames"
OUTPUTS.mkdir(exist_ok=True)
FRAMES.mkdir(exist_ok=True)

MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF = float(os.getenv("CONFIDENCE", "0.25"))
VIDEO_STEP = max(1, int(os.getenv("VIDEO_FRAME_STEP", "2")))
VIDEO_IMGSZ = int(os.getenv("VIDEO_IMGSZ", "480"))
# Set this after calibrating a fixed camera. A single scale is only approximate.
PIXELS_PER_METER = float(os.getenv("PIXELS_PER_METER", "20"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# COCO model class labels used by YOLOv8n.
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
# Task 16-20 demo configuration. Tune these for the fixed camera.
ENTRY_EXIT_LINE_RATIO = float(os.getenv("ENTRY_EXIT_LINE_RATIO", "0.50"))
EXPECTED_DIRECTION = os.getenv("EXPECTED_DIRECTION", "left_to_right").lower()
PARKING_SECONDS = float(os.getenv("PARKING_SECONDS", "10"))
TURN_ANGLE_DEGREES = float(os.getenv("TURN_ANGLE_DEGREES", "55"))
# Restricted parking polygon as normalized x,y points (0..1). Default: lower-middle area.
RESTRICTED_ZONE = [(0.35, 0.55), (0.75, 0.55), (0.90, 0.98), (0.20, 0.98)]
app = FastAPI(title="VisionPort - Traffic Detection and Monitoring", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
model = YOLO(MODEL_PATH)
model_lock = threading.Lock()


def decode_image(data: bytes):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image file.")
    return image


def density_for(count):
    if count <= 5:
        return "Low"
    if count <= 15:
        return "Medium"
    return "High"


def traffic_light_colour(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        "Red": (cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)) |
                cv2.inRange(hsv, (170, 90, 70), (180, 255, 255))),
        "Yellow": cv2.inRange(hsv, (15, 80, 90), (40, 255, 255)),
        "Green": cv2.inRange(hsv, (40, 60, 50), (90, 255, 255))
    }
    scores = {k: cv2.countNonZero(v) for k, v in masks.items()}
    winner = max(scores, key=scores.get)
    return winner if scores[winner] >= 2 else "Unknown"


def analyze_result(result, frame):
    names = result.names
    vehicles, lights, persons = [], [], []
    boxes = result.boxes
    if boxes is not None:
        for b in boxes:
            cls = int(b.cls[0])
            label = names.get(cls, str(cls))
            conf = float(b.conf[0])
            coords = b.xyxy[0].cpu().numpy().tolist()
            if label in VEHICLE_CLASSES:
                vehicles.append({
                    "class": label, "confidence": round(conf, 3),
                    "box": [round(float(x), 1) for x in coords]
                })
            elif label == "person":
                persons.append({
                    "confidence": round(conf, 3),
                    "box": [round(float(x), 1) for x in coords]
                })
            elif label == "traffic light":
                lights.append({
                    "status": traffic_light_colour(frame, coords),
                    "confidence": conf, "box": coords
                })
    signal = "Unknown"
    if lights:
        signal = max(lights, key=lambda x: x["confidence"])["status"]
    return vehicles, lights, signal, persons


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.int32), point, False
    ) >= 0


def centre_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def angle_between(v1, v2):
    a, b = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))))


def annotate(frame, result, count, density, signal, congestion=None):
    out = result.plot()
    cv2.putText(out, f"Vehicles: {count} | Density: {density}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(out, f"Signal: {signal}", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
    if congestion:
        cv2.putText(out, f"Congestion: {congestion}", (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)
    return out


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {
        "status": "online", "model": MODEL_PATH,
        "video_frame_step": VIDEO_STEP, "video_imgsz": VIDEO_IMGSZ
    }


# Task 2: capture a still image from the camera connected to the server.
@app.post("/camera/capture")
def capture_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise HTTPException(503, "Camera could not be opened. Check CAMERA_INDEX and camera permissions.")
    try:
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(503, "Camera opened but no frame was received.")
        filename = f"camera_{uuid.uuid4().hex[:8]}.jpg"
        path = OUTPUTS / filename
        cv2.imwrite(str(path), frame)
        return {"image_url": f"/outputs/{filename}", "message": "Camera image captured."}
    finally:
        cap.release()


# Tasks 3 and 8: image decode, inference, annotated image + labels/confidences.
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    frame = decode_image(await file.read())
    start = time.perf_counter()
    with model_lock:
        result = model.predict(frame, conf=CONF, imgsz=640, verbose=False)[0]
    elapsed = (time.perf_counter() - start) * 1000
    vehicles, lights, signal, persons = analyze_result(result, frame)
    count = len(vehicles)
    density = density_for(count)
    output = annotate(frame, result, count, density, signal)
    filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(OUTPUTS / filename), output)
    return {
        "annotated_image": f"/outputs/{filename}",
        "vehicle_count": count, "density": density,
        "signal_status": signal, "inference_ms": round(elapsed, 1),
        "detections": vehicles, "traffic_lights": lights,
        "person_detections": persons
    }


# Tasks 4-15: video playback, frame extraction, YOLO, count/density,
# congestion, ByteTrack IDs, approximate speed, signal and waiting-time report.
@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "traffic.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
        raise HTTPException(400, "Use MP4, AVI, MOV, MKV or M4V video.")
    job = uuid.uuid4().hex[:10]
    input_path = OUTPUTS / f"{job}_input{suffix}"
    output_path = OUTPUTS / f"{job}_annotated.mp4"
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read this video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video dimensions are invalid.")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        input_path.unlink(missing_ok=True)
        raise HTTPException(500, "Could not create output MP4. Try another video codec/format.")

    frame_no = 0
    frames_total = 0
    frames_analyzed = 0
    counts = []
    max_count = 0
    class_totals = defaultdict(int)
    signal_votes = defaultdict(int)
    last_annotated = None
    previous_centres = {}
    stopped_since = {}
    longest_stops = defaultdict(float)
    speed_values = []

    # Tasks 16-20: tracked-event state and counters.
    entered_ids, exited_ids = set(), set()
    wrong_turn_ids, wrong_way_ids, illegal_park_ids = set(), set(), set()
    overloaded_two_wheelers = set()
    track_history = defaultdict(list)
    parking_since = {}
    previous_side = {}
    frame_interval = max(1, int(round(fps)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1

            if frame_no % VIDEO_STEP == 0:
                with model_lock:
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml",
                        conf=CONF, imgsz=VIDEO_IMGSZ, verbose=False
                    )[0]
                vehicles, lights, signal, persons = analyze_result(result, frame)
                count = len(vehicles)
                counts.append(count)
                max_count = max(max_count, count)
                frames_analyzed += 1
                for item in vehicles:
                    class_totals[item["class"]] += 1
                if signal != "Unknown":
                    signal_votes[signal] += 1

                # Tracking IDs supplied by ByteTrack.
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    coords_list = boxes.xyxy.cpu().numpy()
                    cls_list = boxes.cls.int().cpu().tolist()
                    names = result.names
                    # Person boxes are used to estimate riders overlapping each motorcycle.
                    person_boxes = [p["box"] for p in persons]
                    h, w = frame.shape[:2]
                    line_x = int(w * ENTRY_EXIT_LINE_RATIO)
                    restricted_poly = np.array(
                        [(int(x * w), int(y * h)) for x, y in RESTRICTED_ZONE],
                        dtype=np.int32
                    )
                    for track_id, coords, class_id in zip(ids, coords_list, cls_list):
                        cx = float((coords[0] + coords[2]) / 2)
                        cy = float((coords[1] + coords[3]) / 2)
                        now = frame_no / fps
                        label = names.get(int(class_id), str(class_id))
                        current_point = (int(cx), int(cy))

                        # Task 16: count unique vehicles crossing the virtual entry/exit line.
                        if label in VEHICLE_CLASSES:
                            side = "left" if cx < line_x else "right"
                            old_side = previous_side.get(track_id)
                            if old_side and old_side != side:
                                if old_side == "left" and side == "right":
                                    entered_ids.add(track_id)
                                elif old_side == "right" and side == "left":
                                    exited_ids.add(track_id)
                            previous_side[track_id] = side

                            # Task 17: flag sharp trajectory changes as possible wrong turns.
                            hist = track_history[track_id]
                            hist.append((cx, cy, now))
                            if len(hist) > 12:
                                hist.pop(0)
                            if len(hist) >= 3:
                                p0, p1, p2 = hist[-3], hist[-2], hist[-1]
                                turn_angle = angle_between(
                                    (p1[0] - p0[0], p1[1] - p0[1]),
                                    (p2[0] - p1[0], p2[1] - p1[1])
                                )
                                if turn_angle >= TURN_ANGLE_DEGREES:
                                    wrong_turn_ids.add(track_id)

                            # Task 18: flag movement opposite to configured direction.
                            if len(hist) >= 4:
                                dx = hist[-1][0] - hist[0][0]
                                dy = hist[-1][1] - hist[0][1]
                                if EXPECTED_DIRECTION == "left_to_right" and dx < -25:
                                    wrong_way_ids.add(track_id)
                                elif EXPECTED_DIRECTION == "right_to_left" and dx > 25:
                                    wrong_way_ids.add(track_id)
                                elif EXPECTED_DIRECTION == "top_to_bottom" and dy < -25:
                                    wrong_way_ids.add(track_id)
                                elif EXPECTED_DIRECTION == "bottom_to_top" and dy > 25:
                                    wrong_way_ids.add(track_id)

                            # Task 19: vehicle inside restricted zone and stationary for threshold.
                            inside_restricted = point_in_polygon(current_point, restricted_poly)
                            if inside_restricted:
                                parking_since.setdefault(track_id, now)
                                if now - parking_since[track_id] >= PARKING_SECONDS:
                                    illegal_park_ids.add(track_id)
                            else:
                                parking_since.pop(track_id, None)

                            # Task 20: count persons whose centers fall inside motorcycle box.
                            if label == "motorcycle":
                                x1, y1, x2, y2 = map(float, coords)
                                riders = 0
                                for pb in person_boxes:
                                    px, py = centre_of(pb)
                                    if x1 <= px <= x2 and y1 <= py <= y2:
                                        riders += 1
                                if riders > 2:
                                    overloaded_two_wheelers.add(track_id)

                        if track_id in previous_centres:
                            old_x, old_y, old_t = previous_centres[track_id]
                            dt = max(now - old_t, 1.0 / fps)
                            # Approximate only; requires a calibrated fixed camera.
                            metres = np.hypot(cx - old_x, cy - old_y) / max(PIXELS_PER_METER, 0.01)
                            kmh = min((metres / dt) * 3.6, 180.0)
                            speed_values.append(kmh)
                            if kmh < 3:
                                stopped_since.setdefault(track_id, now)
                                longest_stops[track_id] = max(
                                    longest_stops[track_id], now - stopped_since[track_id]
                                )
                            else:
                                stopped_since.pop(track_id, None)
                        previous_centres[track_id] = (cx, cy, now)

                avg_now = float(np.mean(counts)) if counts else 0
                congestion_now = ("High" if count > 15 else
                                  "Medium" if count > 5 else "Low")
                last_annotated = annotate(
                    frame, result, count, density_for(count), signal, congestion_now
                )

            # Preserve original frame rate/duration; skipped frames reuse last annotated frame.
            writer.write(last_annotated if last_annotated is not None else frame)

            # Task 5: save one original frame per second.
            if frame_no % frame_interval == 0:
                cv2.imwrite(str(FRAMES / f"{job}_frame_{frame_no:06d}.jpg"), frame)
            frame_no += 1
    finally:
        cap.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    if frames_total == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(400, "No readable frames found in video.")

    avg_count = float(np.mean(counts)) if counts else 0.0
    avg_speed = float(np.mean(speed_values)) if speed_values else 0.0
    wait_list = list(longest_stops.values())
    avg_wait = float(np.mean(wait_list)) if wait_list else 0.0
    density = density_for(round(avg_count))
    congestion = ("High" if avg_count > 15 else
                  "Medium" if avg_count > 5 else "Low")
    signal = max(signal_votes, key=signal_votes.get) if signal_votes else "Unknown"

    return {
        "job_id": job,
        "processed_video": f"/outputs/{output_path.name}",
        "frames_processed": frames_total,
        "frames_analyzed_by_yolo": frames_analyzed,
        "frame_step": VIDEO_STEP,
        "vehicle_count_average": round(avg_count, 2),
        "maximum_vehicle_count": max_count,
        "density": density,
        "congestion": congestion,
        "congestion_alert": f"{congestion} traffic congestion" if congestion != "Low" else "No significant congestion",
        "vehicle_detections_by_class": dict(class_totals),
        "average_estimated_speed_kmh": round(avg_speed, 2),
        "estimated_waiting_time_seconds": round(avg_wait, 2),
        "traffic_signal_status": signal,
        "extracted_frames_saved": int(np.ceil(frames_total / frame_interval)),

        # Task 16: unique track IDs crossing the virtual line.
        "vehicles_entering": len(entered_ids),
        "vehicles_leaving": len(exited_ids),
        "entry_exit_line_x_ratio": ENTRY_EXIT_LINE_RATIO,

        # Task 17-20: event counts and track IDs for dashboard display.
        "wrong_turn_alert_count": len(wrong_turn_ids),
        "wrong_turn_track_ids": sorted(wrong_turn_ids),
        "wrong_way_alert_count": len(wrong_way_ids),
        "wrong_way_track_ids": sorted(wrong_way_ids),
        "illegal_parking_count": len(illegal_park_ids),
        "illegal_parking_track_ids": sorted(illegal_park_ids),
        "two_wheelers_carrying_more_than_two_count": len(overloaded_two_wheelers),
        "overloaded_two_wheeler_track_ids": sorted(overloaded_two_wheelers),
        "task_16_20_configuration": {
            "expected_direction": EXPECTED_DIRECTION,
            "parking_threshold_seconds": PARKING_SECONDS,
            "wrong_turn_angle_threshold_degrees": TURN_ANGLE_DEGREES,
            "restricted_zone_normalized_polygon": RESTRICTED_ZONE
        },
        "calibration_note": "Speed and waiting time are rough estimates. Calibrate PIXELS_PER_METER and use a fixed camera for meaningful values."
    }


@app.get("/frames")
def list_extracted_frames():
    files = sorted(FRAMES.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"count": len(files), "frames": [p.name for p in files[:200]]}
