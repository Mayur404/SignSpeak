# SignSpeak

SignSpeak is an offline Windows desktop app that turns ASL fingerspelling into text and speech. It uses a webcam, tracks up to two hands with MediaPipe, and runs a local classifier. The bundled model works immediately; record examples of your own handshapes to train a personal model.

![SignSpeak translate screen](screenshots/translate_real.png)

## Features

- Live ASL alphabet and digit recognition from a webcam, with up to two hands tracked together.
- Personal training with your own handshapes, including custom words and a backspace gesture.
- Transcript editing, word spacing, and optional offline speech.
- Windows Natural voices, including English (India), when installed on the device.
- Local processing and storage; camera frames are not uploaded.
- Supports both local USB webcams and network RTSP / IP CCTV cameras (including Sparsh SC-IM82NP-I(Z)(S)(H)).

## Run

Use Python 3.10 or later (Python 3.12 is the version used during development). In PowerShell, create an environment and install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start SignSpeak from the project folder:

```powershell
.\.venv\Scripts\python.exe app.py
```

The optional `--no-camera` and `--no-speech` flags are useful for UI checks on machines without those devices:

```powershell
.\.venv\Scripts\python.exe app.py --no-camera --no-speech
```

### IP / CCTV Camera (Sparsh / RTSP)

SignSpeak automatically discovers network RTSP streams on your local subnet (such as the directly connected Sparsh camera at `192.168.128.10:554`):
- **Sub-stream (720p @ 20 FPS, recommended)**: `rtsp://admin:admin123@192.168.128.10:554/h264/ch1/sub/av_stream`
- **Main stream (4K UHD)**: `rtsp://admin:admin123@192.168.128.10:554/h264/ch1/main/av_stream`

You can switch between local webcams and the Sparsh CCTV streams anytime using the camera dropdown on the Translate or Settings page. SignSpeak uses TCP for complete H.264 packets and continuously drains the network stream so slow hand tracking does not leave the preview seconds behind the camera. The 720p sub-stream is recommended for the smoothest response; the 4K main stream needs more decode power.

## Use

1. Select a camera and allow it to start. Keep the signing hand or both hands visible, palms toward the camera, with reasonable light and a plain background.
2. Hold a static sign steady. The app stabilizes predictions and types a character after the hold indicator completes. Lower your hand or change signs before repeating a character.
3. Use the thumbs-up control gesture to finish a word and the “I love you” handshape to finish a sentence. Both actions also have buttons on the Translate page.
4. Speak the transcript with **Speak sentence**. Word and sentence auto-speech can be switched independently.

The reference alphabet includes A-Z and digits 0-9. J and Z involve movement, so the static classifier may not recognize their motion reliably. Word, space, sentence and backspace controls can be trained as handshapes.

### Personal training

Open **Train**, choose a label, and record at least three separate takes per sign. Hold the handshape steady during each take, but vary hand position and distance slightly between takes. Keep both hands in frame when recording a two-hand sign so the personal model learns their joint shape. Each take captures up to 15 spaced-out frames, so aim for at least 30 samples from multiple takes for each sign. Use **Add word** to create a custom word or short phrase label, then record its handshape like any other sign. The **DEL** tile records a custom backspace handshape. Custom word and DEL handshapes work only with the trained personal model; training activates the updated model. Train with at least two signs. When multiple takes are available, SignSpeak compares local classifiers using held-out recording takes; otherwise the sample-level score is marked as an optimistic estimate. A personal model is stored under `data/` and can be removed from the Train page.

The live top-class score is a diagnostic probability, not model accuracy. It can decrease as more signs are added because the classifier has more choices. SignSpeak does not reject a sign based on this score; it commits using stable votes and hold time. Personal training improves adaptation to an individual hand and camera, but validation on recorded takes is not a guarantee of accuracy in new lighting or poses. Capture varied examples and test the signs you plan to use.

Training compares an RBF SVM, Extra Trees, Random Forest and K-Nearest Neighbours. It selects by balanced accuracy, so each sign class contributes equally to model selection. Separate recording takes are held out together when enough takes exist; otherwise the UI identifies the score as a sample-level estimate. The live pipeline uses MediaPipe Tasks in video mode, normalized hand landmarks, and temporal voting before entering text.

## Controls

| Action | Control |
| --- | --- |
| Type current sign | Enter or **Type sign** |
| Delete one character / last transcript token | Backspace key or **Backspace** button |
| Delete the current or last completed word | **Delete word** button or Ctrl+Backspace |
| Pause or resume camera | Esc |
| Clear transcript | Ctrl+R |
| Finish word | Thumbs-up gesture or **Finish word** |
| Finish sentence | “I love you” gesture |
| Insert a recorded custom word | Train and hold its custom handshape; the word is added as a finished word with spacing after it |
| Delete one character / last transcript token by gesture | Train and hold the **DEL** handshape |

## Local data and privacy

Camera frames and speech are processed locally. Recorded landmark samples, the trained model, and settings are stored in `data/`. The app does not upload these files.

## Troubleshooting

- **Camera offline:** close other camera applications, choose another camera index, and press **Retry camera**. Some USB cameras need a moment after being selected.
- **Signs are inconsistent:** improve lighting, keep the whole hand in view, record separate varied takes for each sign, and adjust vote agreement or hold time in Settings. The displayed model score does not control whether a sign is accepted.
- **Speech unavailable:** Windows uses installed Windows voices; other platforms use `pyttsx3`. For a more natural Indian English voice, install **Neerja** or **Prabhat** in Settings → Accessibility → Narrator → Add a natural voice, then choose it under SignSpeak Settings → Speech. Downloaded Windows Natural voices run on-device.
- **Missing model assets:** retain `assets/models/hand_landmarker.task` and `assets/models/asl_legacy_rf.pkl` alongside the app. The included MediaPipe task model is required for hand tracking.
- **Dependency installation fails:** run `python -m pip install -r requirements.txt` in the environment used to launch the app, then run `app.py` again.

## Project layout

- `app.py` and `run.bat`: app entry points.
- `signspeak/vision.py`: camera, MediaPipe tracking, overlay, and sample capture.
- `signspeak/recognition.py`: classifier loading and temporal stabilization.
- `signspeak/features.py`, `trainer.py`: landmark features, data storage, and personal training.
- `signspeak/composer.py`, `tts.py`: text composition and offline speech.
- `signspeak/ui/`: PySide6 interface.
- `assets/`: bundled models and sign reference images.

## Credits

The bundled classifier and sign reference material are based on [Sign-Language-to-Text-and-Speech](https://github.com/tanmayJivnani/Sign-Language-to-Text-and-Speech); see its README and LICENSE for original attribution and license terms. Hand tracking uses MediaPipe Tasks. SignSpeak's interface, landmark feature pipeline, temporal controls, and personal trainer are implemented in this project.

Third-party attribution and license details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
