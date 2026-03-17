# CV Pose Fitness Coach

A local Python computer vision project that uses a webcam, draws a stickman skeleton, coaches exercise form in real time, counts only correct reps, and saves session reports.

## Features
- Live webcam pose tracking with `MediaPipe Pose` and `OpenCV`
- Stickman overlay for body landmarks
- V1 exercise support for squat, push-up, and plank
- Rep counting only when squat and push-up form passes the exercise rules
- Real-time text + color feedback such as `Straighten back`, `Go lower`, and `Keep hips level`
- JSON session reports plus a rolling CSV session summary

## Setup
1. Create a virtual environment.
2. Install the package and runtime dependencies:
   ```bash
   pip install -r requirements.txt  # installs MediaPipe 0.10.14
   pip install -e .
   ```
3. Run the app:
   ```bash
   cv-pose
   ```
   or:
   ```bash
   python -m cv_pose
   ```

## Tests
```bash
pytest
```

## Controls
- `1`: select squat
- `2`: select push-up
- `3`: select plank
- `S`: start or stop the current session
- `R`: reset the current session without saving
- `Q`: quit the app

## Camera Guidance
- Use a single laptop or USB webcam.
- Stand in a side view relative to the camera.
- Keep shoulder, hip, knee, and ankle visible for the tracked side.
- Good lighting improves landmark confidence and feedback accuracy.

## Reports
Every completed session writes:
- `reports/<session_id>.json`: detailed session report with events and posture issues
- `reports/session_summary.csv`: one summary row per session

## Project Layout
- `src/cv_pose/pose.py`: MediaPipe integration and derived pose metrics
- `src/cv_pose/analyzers.py`: squat, push-up, and plank rules
- `src/cv_pose/workout.py`: session state, feedback tracking, and report generation
- `src/cv_pose/app.py`: webcam UI and keyboard controls
- `tests/`: geometry and exercise logic tests
