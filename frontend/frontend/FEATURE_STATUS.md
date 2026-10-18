# Underwater Detection Project: Feature Status

This status note describes what the Flask application and its supplied model/data can actually do. It avoids treating entries in a requested feature list as already-trained capabilities.

## Available in the Flask application

- Registration and login, with password hashing for new registrations and transparent upgrade of existing plaintext passwords when they log in.
- User and administrator roles, profile editing, per-user detection history, CSV export, and printable reports.
- Administrator user-role management and persistent confidence/IoU inference settings.
- Image upload inference, browser-camera inference, and video upload inference using the currently selected Ultralytics model.
- Annotated image/video output and result lists containing the model class, confidence, and image bounding box.
- A basic visual image assessment for brightness, contrast, and color-channel averages. These are image heuristics, not water measurements.
- A capabilities endpoint which reports the class names from the model that is currently loaded.
- Automatic use of `best_model_expanded.pt` when that file exists, with fallback to the original `best_model+CSEA.pt`.

## Labels and model constraints

- The original checkpoint was trained for `echinus`, `holothurian`, `scallop`, and `starfish`.
- The expanded training set combines those labels with the downloaded debris archive. Its intended additional labels are `plastic`, `marine_life`, `rov`, and `other_debris`.
- The archive has no labeled rock bounding boxes. Rock detection is therefore not a supported capability.
- `marine_life` is a broad source label; it does not distinguish many fish species. The application must use the loaded checkpoint's actual class names rather than claim species-level recognition.
- The expanded checkpoint is not available until its training job completes and saves the weights. Until then, the app continues using the original checkpoint.
- Video track IDs use a simple position-and-class matching heuristic. They are approximate and are not a robust multi-object tracking system. Video summary counts are detections across sampled frames, not counts of unique animals or debris items.

## Not implemented as measured or validated outputs

- Chemical water-quality readings, contaminants, oxygen, pH, salinity, or turbidity in NTU require sensors or calibrated measurement data.
- Depth, distance, physical size, debris volume, and real-world area require camera calibration, known scale, stereo/depth sensors, or other reference measurements.
- Gold/mineral identity cannot be established from RGB appearance alone.
- Rock detection, underwater segmentation, reliable above-water rejection, and new fine-grained animal/debris categories require suitable labeled examples and evaluation.
- The image brightness/contrast assessment is a visual heuristic only and must not be reported as a scientific environmental measurement.

## Operational notes

- Configure `FLASK_SECRET_KEY` to keep browser sessions valid and secret across restarts. Without it, the application generates a new key on each start.
- Configure `ADMIN_EMAIL` to promote the matching registered user to administrator at startup.
- Database configuration currently remains the local MySQL settings in `app.py`; move credentials to environment variables before sharing or deploying the project.
- The Flask development server/debug mode is for local development and should not be exposed as a production server.
- The training helper and its expanded dataset are local artifacts and remain excluded from Git by `.gitignore`.

