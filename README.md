# SignSpeak

SignSpeak watches a video stream for American Sign Language (ASL) and custom hand gestures,
translates them into characters, words, and sentences in real time, and speaks them aloud.
It was built for the private 5G lab at IIIT Dharwad (EC368).

In the lab, the camera video travels over the 5G network from an edge Sparsh CCTV camera
to an edge server / VM (`172.16.0.10` / `172.16.0.11`). The server runs real-time MediaPipe
dual-hand landmark tracking, temporal voting, and machine learning inference directly at
the edge to keep gesture translation fast and responsive.

![SignSpeak Translate Screen](screenshots/translate_real.png)

## What it does

- Tracks up to two hands at once in real time using MediaPipe Tasks (video mode).
- Translates static ASL alphabet (A-Z) and digits (0-9) using a calibrated classifier.
- Supports **personal model training**: record your own handshapes, custom words/phrases,
  and a custom backspace gesture directly from the UI.
- Stabilizes signs with sliding-window temporal voting, hold-to-type ring, and release-to-repeat
  detection so you don't get accidental duplicate letters.
- Composes words and sentences with automated speech synthesis (Edge TTS, Windows SAPI,
  Linux system voices, or optional Sarvam AI for 11 Indian languages).
- Automatically connects to **5G network RTSP cameras** (Sparsh SC-IM82NP series) as well as
  local USB webcams.

## What's in this folder

| Item | What it is |
|---|---|
| `app.py` | Main application entry point. Run this to start the app. |
| `run.bat` | One-click Windows starter. Sets up the venv if missing and launches the app. |
| `run.sh`, `setup_ubuntu.sh` | Linux / Ubuntu VM launcher and dependency installer. |
| `signmodel.joblib` | Pre-trained sign language model ready to load. |
| `data/` | User settings (`settings.json`), trained model (`user_model.joblib`), and recordings (`user_dataset.npz`). |
| `signspeak/` | Python source code: vision pipeline, recognition engine, trainer, composer, UI. |
| `assets/models/` | Hand landmarker model (`hand_landmarker.task`) and bundled legacy model (`asl_legacy_rf.pkl`). |
| `assets/` | Alphabet charts and gesture guide images. |
| `requirements.txt` | Python library dependencies. |
| `tests/` | Automated unit test suite (`test_core.py`). |

---

## Try it on your own laptop

You don't need the 5G lab or a CCTV camera to test. You can run SignSpeak on your laptop
with any built-in or USB webcam.

### On Windows

1. Install Python 3.10 or newer (Python 3.12 recommended).
2. Open PowerShell in this `SignSpeak` folder.
3. Run the setup and start:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe app.py
   ```
   *(Or simply double-click `run.bat`)*

### On Linux / Ubuntu

1. Open terminal in this folder and install dependencies:
   ```bash
   chmod +x setup_ubuntu.sh run.sh
   ./setup_ubuntu.sh
   ./run.sh
   ```

Within a few seconds, the SignSpeak window opens. Place your hand in front of the camera,
hold an ASL letter steady, and watch it type onto the screen.

---

## Running it in the 5G lab

In the lab, SignSpeak runs on the edge server / VM at `172.16.0.10` (or `172.16.0.11`).
Because SignSpeak has a graphical interface, you run it inside the VM's desktop environment.

### Accessing the VM UI

1. Open your lab VM dashboard in a web browser.
2. Select your VM (`172.16.0.10` / `172.16.0.11`) and open the **Console** (NoVNC / SPICE).
3. Log in to the desktop session.
4. Open a terminal inside the desktop GUI and launch SignSpeak:
   ```bash
   cd ~/SignSpeak
   ./run.sh
   ```

---

### Connect the camera over 5G

In the standard 5G lab setup, the camera streams directly across the 5G radio link:

1. Plug the Sparsh camera's Ethernet cable into the LAN port of the **5G modem** (T99W240).
2. Power both on and wait about a minute for the modem to register with the 5G network.
3. Check the modem's 5G WAN IP (typically `10.101.0.70` or check the modem dashboard).
4. The camera stream is available over RTSP:
   ```text
   rtsp://admin:admin123@10.101.0.70:554/avstream/channel=1/stream=1.sdp
   ```
   *(SignSpeak automatically probes `10.101.0.70` and lists **Sparsh CCTV [5G]** in the camera picker dropdown).*
5. To test the 5G link from the VM terminal:
   ```bash
   tracepath -n 10.101.0.70
   ```
   The path routes through `172.16.0.7` (the 5G core UPF gateway) with `pmtu 1450`.

---

### Backup: when the camera is plugged directly into a laptop

If the 5G modem is not used and the camera is plugged directly into your laptop's Ethernet
port (`192.168.128.100`), the camera's IP is `192.168.128.10`.

Because the VM cannot access your laptop's physical Ethernet port directly, forward the port
from your laptop:

#### Option 1: Windows Port Proxy (Recommended)
Run this once in PowerShell on your laptop as **Administrator**:
```powershell
netsh interface portproxy add v4tov4 listenport=554 listenaddress=0.0.0.0 connectport=554 connectaddress=192.168.128.10
```
Then on the VM, point SignSpeak to your laptop's Wi-Fi IP:
```text
rtsp://admin:admin123@<YOUR_LAPTOP_IP>:554/avstream/channel=1/stream=1.sdp
```

To remove the proxy when done:
```powershell
netsh interface portproxy delete v4tov4 listenport=554 listenaddress=0.0.0.0
```

#### Option 2: Direct on Laptop
If you run SignSpeak directly on the laptop connected to the camera:
SignSpeak automatically detects `192.168.128.10:554` on the Ethernet link and connects.

---

## The Sign Models

SignSpeak includes two model tiers:

1. **Bundled ASL Reference Model** (`assets/models/asl_legacy_rf.pkl`):
   Pre-trained on 42-feature hand landmarks for standard ASL alphabet (A-Z) and digits (0-9).
   Works out of the box with zero setup.
2. **Personal Trained Model** (`signmodel.joblib` / `data/user_model.joblib`):
   Trained on 126-feature translation-, scale-, and rotation-normalized landmarks tracking
   both hands. Supports custom gestures, custom words, and the backspace (DEL) handshape.

### Loading the Model on a new machine / VM

The repository includes `signmodel.joblib`. When launching SignSpeak:
- SignSpeak checks `data/user_model.joblib` first, and falls back to `signmodel.joblib`.
- In the app, switch between the **Bundled model** and **Personal model** via the **Train** page
  or by clicking the model badge in the top header.
- To export or backup a newly trained model, click **Save a copy…** on the Train page.

---

## Settings

Settings are stored in `data/settings.json` and can be adjusted in the UI or by editing the file:

| Setting | What it does | Default |
|---|---|---|
| `camera_source` | Full RTSP stream URL or video source. | `rtsp://admin:admin123@10.101.0.70:554/avstream/channel=1/stream=1.sdp` |
| `camera_index` | Webcam device index (used when `camera_source` is empty). | `0` |
| `inference_width` | Frame width for MediaPipe inference (`640` or `800` px). | `800` |
| `confidence` | Detection confidence threshold (0.20 to 0.95). | `0.55` |
| `hold_ms` | Milliseconds a sign must stay stable before typing. | `500` ms |
| `smoothing` | Sliding-window frame count for temporal voting. | `8` frames |
| `min_vote_ratio` | Percentage agreement in window required to commit. | `0.60` (60%) |
| `cooldown_ms` | Minimum gap before another character can type. | `200` ms |
| `auto_word` | Speak word aloud when thumbs-up is shown. | `true` |
| `auto_sentence` | Speak entire sentence aloud when “I love you” sign is shown. | `true` |
| `active_model` | Active classifier (`user` for personal model, `legacy` for bundled). | `user` |

---

## Gesture Shortcuts

| Sign / Action | What it does |
|---|---|
| **Static Handshape (A-Z, 0-9)** | Held for 500 ms → Types the character. |
| **Thumbs Up 👍** | Space gesture → Completes the current word and speaks it. |
| **“I love you” 🤟** | Sentence end gesture → Appends period and speaks the full sentence. |
| **DEL Gesture (Custom)** | Trained backspace handshape → Deletes the last character. |
| **Custom Word Gesture** | Trained custom sign → Injects full word/phrase directly into transcript. |

---

## When something goes wrong

| Problem | Cause | Solution |
|---|---|---|
| **Camera shows OFFLINE** | RTSP stream not reachable. | Check 5G modem IP (`10.101.0.70:554`) or direct link (`192.168.128.10:554`). Press **Retry camera**. |
| **RTSP returns 403 Forbidden** | Wrong stream URL path used. | Use `/avstream/channel=1/stream=1.sdp` (substream) or `/avstream/channel=1/stream=0.sdp` (main). Do not use `/stream1`. |
| **Camera feed is lagging / delayed** | RTSP buffer accumulation over UDP. | SignSpeak forces `rtsp_transport;tcp` and buffersize 1. Use the 720p substream (`stream=1.sdp`) instead of 4K. |
| **Qt XCB error on Linux VM** | GUI display environment missing. | Run inside the VM's desktop console (NoVNC/SPICE). Ensure `libxcb-cursor0` is installed. |
| **Personal model won't load** | `signmodel.joblib` missing. | Ensure `signmodel.joblib` or `data/user_model.joblib` exists in the folder. |
| **Speech does not play on Linux** | No TTS voice installed. | Run `sudo apt install espeak-ng libespeak1` on the VM. |
