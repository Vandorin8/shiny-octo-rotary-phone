# Manga Flow — Intelligent Reading Engine (Android)

Native Kotlin / Jetpack Compose manga reader with hybrid OpenCV + on-device-ML panel detection,
layout-aware Japanese reading order, dialogue-safe framing, a personalised automatic zoom engine
and cinematic panel reveal. Fully offline: the manifest does not even request INTERNET.

## Build the APK

**From your phone (no computer needed)**
1. Create an empty repository on github.com (private is fine).
2. Upload the contents of this folder (keep the `.github` folder) — the GitHub app or the
   website's "Add file → Upload files" both work.
3. Open the repository's **Actions** tab → **Build APK**. It runs on every push (or press
   *Run workflow*). It runs the engine unit tests and the detector benchmark, then builds.
4. When it finishes, download the **MangaFlow-apk** artifact, unzip, install
   `app-release.apk` (allow "install unknown apps" for your browser/files app).

**With Android Studio** (Ladybug or newer, JDK 17): open the folder, let Gradle sync, run `app`.
CLI: `./gradlew :engine:test :app:assembleRelease`.

The release APK is signed with the debug key unless you set `MANGAFLOW_KEYSTORE`,
`MANGAFLOW_KEYSTORE_PASSWORD`, `MANGAFLOW_KEY_ALIAS`, `MANGAFLOW_KEY_PASSWORD`.

## Verification status (be aware before relying on it)

| Part | Status |
|---|---|
| `engine` module (layout fusion, reading order, confidence, bubble attribution, zoom engine, reveal planner, navigator, learning, manual-edit operations, settings) | Compiled with Kotlin 2.0.21 and **41/41 unit tests pass** |
| OpenCV panel + bubble detector algorithm | Validated in `tools/detector_prototype.py` on 88 synthetic panels across grid, tall-column, splash, borderless, diagonal and black-gutter pages: precision 1.00, recall 1.00, mean IoU 0.98; bubble recall 1.00 (60/60). **Synthetic pages only — accuracy on real manga has not been measured.** |
| `app` module (Android UI, file formats, Room, renderer, OpenCV/ONNX glue) | **Not yet compiled.** The environment this was written in had no access to Google's Android SDK/Maven repositories. The first CI or Android Studio build is the first compile; if it reports errors, they will be ordinary API/typing fixes, not missing features. |
| Real-device performance (detection time, memory, battery, frame stability) | Not measured. The Settings screen shows the measured average detection time per page on your device. |

## Machine-learning model

The ML stage (`MlPanelDetector.kt`) is complete — ONNX Runtime session, letterbox preprocessing,
mask and YOLO post-processing, fusion with OpenCV results — but **no trained weights are bundled**:
no appropriately licensed manga panel model or dataset was reachable while building this, and the
spec forbids simulated results. Without a model the app uses the OpenCV detector alone (the
Settings screen says so), and everything else works.

To add one, either put it at `app/src/main/assets/models/panel.onnx` before building, or use
**Settings → Import .onnx model** on the phone. Supported contracts (auto-detected):

* **Segmentation**: input `[1,3,S,S]` RGB 0–1; output `[1,1,S,S]` or `[1,S,S]` panel probability or logits.
* **YOLOv8 detection**: input `[1,3,S,S]`; output `[1, 4+C, N]` (`yolo export format=onnx`).

Changing the model changes the detector id, which invalidates cached automatic detections but
never touches manual corrections. Check the licence of any model and its training data (e.g.
Manga109 is research-only).

## Architecture

```
engine/  (pure Kotlin, unit-tested)
  geometry/      RectF, Polygon, union area — one source-pixel coordinate system for everything
  model/         PanelCandidate, Bubble, Panel (panel region + safe content region), PageLayout
  layout/        CandidateFusion (stage D), ReadingOrder (stage E, recursive XY-cut layout graph),
                 LayoutAssembler (stage F confidence + low-confidence fallbacks + bubble protection),
                 ManualCorrections (editor operations; corrections override detection)
  camera/        CameraState (uniform scale, log-space interpolation), ZoomEngine (reading-size
                 preference → per-panel framing), RevealPlanner (dialogue-aware reveal steps),
                 ReadingNavigator (page/panel/step state machine), PlanBuilder
  learning/      PreferenceLearner (local running statistics, opt-in)
  settings/      ReaderSettings (spec defaults), BookOverrides (per-book)
app/
  data/source    CBZ, CBR (RAR4 via junrar), EPUB (OPF spine → page images), PDF (PdfRenderer)
  data/db        Room: books + progress, versioned detection cache, manual corrections
  detection/     ClassicalDetector (OpenCV, stages A–B + bubbles), MlPanelDetector (stage C),
                 DetectionService (correction > cache > detect; single background lane)
  reader/        TileCache (base bitmap + BitmapRegionDecoder tiles), ReaderViewModel
                 (preloading, plans, learning hooks), ReaderScreen (taps, pinch/pan, camera animation)
  ui/            Library, Intelligent Reading settings, Choose Your Reading Comfort, Panel editor
tools/detector_prototype.py   reference implementation + synthetic benchmark of the OpenCV detector
```

Navigation: tap left = forward, tap right = backward, tap centre = controls, double-tap = whole
panel, pinch/drag = manual zoom (next tap resumes automatic framing).

## Known limitations

* RAR5 `.cbr` files are rejected with a message (junrar supports RAR4 only); re-pack as CBZ.
* EPUB support targets image-based (fixed-layout) manga EPUBs.
* Speech-bubble detection is a visual heuristic (white convex blobs containing glyph-like marks);
  no OCR. Irregular or borderless balloons may be missed — attribution uncertainty lowers page
  confidence, which triggers the configured fallback instead of cropping.
* Non-rectangular panels keep their polygon, but the camera frames the polygon's bounding box and
  the editor edits rectangles.
* Reading-size calibration uses procedurally drawn sample panels (no copyrighted art is bundled).
