# SignSpeak

SignSpeak watches a video stream for American Sign Language (ASL) fingerspelling and hand gestures,
translates them into characters, words, and sentences in real time, and speaks them aloud using
native web speech synthesis. It was built for the private 5G lab at IIIT Dharwad (EC368).

There, a high-definition Sparsh CCTV camera transmits its video feed over the lab's dedicated 5G
wireless network to a server at the edge of that network (`172.16.0.10`), and the server does the
MediaPipe tracking and machine learning inference. Keeping the work at the edge keeps latency minimal:
from holding a hand shape to your phone or laptop speaking the word is under half a second.

The interface is a web dashboard running on **port 8080**. Anyone connected to the lab Wi-Fi or edge
network can open `http://172.16.0.10:8080` (or `http://localhost:8080` on a laptop) on their phone,
tablet, or PC to view the live annotated video stream, compose sentences, and hear speech synthesis
directly through their own device's speakers.

---

## What it does

- Looks at the camera picture at up to 30 frames a second with MediaPipe dual-hand landmark detection
  and a 35-class Extra Trees model (**99.4% validation accuracy**).
- Detects the full ASL alphabet (A–Z) and digits (0–9).
- Renders a real-time HUD directly on the video feed: dual-hand skeletons with colored bone links,
  joint highlights, corner framing brackets, confidence badge, and a circular hold-to-type countdown ring.
- Navigation & control gestures:
  - **Thumbs Up (`👍`)**: Commits Space, finishes current word, and triggers word-level speech synthesis.
  - **I Love You (`🤟`)**: Commits Period (`.`), finishes sentence, and triggers full sentence speech.
  - **Open Palm Wave (`✋`)**: Backspace (deletes previous letter).
- Built-in Browser Speech Synthesis (Web Speech API): Plays audio directly on your phone or laptop speakers
  with zero latency and natural voice selection without needing audio drivers on the server.
- One-click camera switcher: Switch seamlessly between the **5G RTSP stream** (`10.101.0.70`), **Direct Ethernet RTSP**
  (`192.168.128.10`), local **Webcam 0**, or custom RTSP URLs.

---

## What's in this folder

| Item | What it is |
|---|---|
| `app.py` | The program. Starts the Web Dashboard on port 8080. |
| `start.sh` | Starts the program in the background on the 5G edge server. |
| `stop.sh` | Stops the background program cleanly. |
| `run.sh` | Starts the program in the foreground (Linux / Mac). |
| `run.bat` | One-click Windows starter. |
| `signmodel.joblib` | The 35-class Extra Trees model (17.6 MB, 99.4% accuracy). |
| `signspeak/web/` | Web application: MJPEG video streamer, REST API, modern HTML5/JS dashboard. |
| `signspeak/` | Core Python engine: vision pipeline, hand tracking, recognition, composer. |
| `assets/models/` | MediaPipe hand landmarker (`hand_landmarker.task`) and bundled legacy model. |
| `data/` | User settings (`settings.json`), trained model (`user_model.joblib`), recordings. |
| `.env` / `.env.example` | Network settings: camera IP (`CAMERA_HOST`) and web port (`8080`). |
| `requirements.txt` | Python dependencies (Flask, OpenCV, MediaPipe, scikit-learn, numpy). |
| `tests/` | Automated unit test suite (`test_core.py`, `test_web.py`). |

---

## Try it on your own laptop

You don't need the 5G lab or a CCTV camera for this. You can run SignSpeak on your laptop with any
built-in webcam or USB camera and access the web UI from your laptop or your phone.

1. Install Python 3.10 or newer (Python 3.12 recommended). Check with:
   ```bash
   python3 --version
   ```
2. Open a terminal in this `SignSpeak` folder and install the dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Start the program:
   * On Linux / Mac:
     ```bash
     python3 app.py
     ```
     *(Or `./run.sh`)*
   * On Windows:
     ```cmd
     run.bat
     ```
     *(Or `python app.py`)*
4. Open **http://localhost:8080** in your web browser.
5. **Connect from your phone on the same Wi-Fi**:
   Find your laptop's local IP address (e.g. `192.168.1.45`) and open:
   ```text
   http://192.168.1.45:8080
   ```
   The camera feed and transcript will load on your phone, and speech will play right from your phone's speakers!

---

## Running it in the 5G lab

In the lab, SignSpeak runs on the edge server VM at `172.16.0.10` (or `172.16.0.11`), in the folder
`~/5Gusecase/SignSpeak` (or `~/SignSpeak`). That server sits right behind the 5G core UPF gateway (`172.16.0.7`),
so any video the 5G camera sends has crossed the private 5G air interface.

### 1. SSH into the edge server
```bash
ssh user@172.16.0.10
cd ~/5Gusecase/SignSpeak
```

### 2. Start the program in the background
```bash
./start.sh
```
You will see:
```text
==========================================================
  SignSpeak Web UI is running in background (PID: 12345)
==========================================================
  * Local browser:    http://localhost:8080
  * 5G Edge Server:   http://172.16.0.10:8080
  * View live logs:   tail -f signspeak.log
  * Stop server:      ./stop.sh
==========================================================
```

### 3. Open the Dashboard
Connect your laptop or phone to the lab Wi-Fi, open your browser, and navigate to:
```text
http://172.16.0.10:8080
```
The dashboard will connect immediately. You can view the live camera stream, see recognized ASL signs,
and click "Speak Now" or enable auto-speech.

### 4. When you're done:
```bash
./stop.sh
```

---

## Camera & Network Setup (Finding the Camera IP)

The Sparsh SC-IM82NP series IP camera is connected to the 5G modem (T99W240) in the lab.

### Why the camera IP changes:
The 5G modem establishes a cellular PDP context with the 5G core network. Each time the modem or 5G
cell restarts, the 5G core assigns a dynamic WAN IPv4 address to the modem (for example, `10.101.0.70`).
The modem automatically DMZ-forwards incoming RTSP port `554` traffic directly to the camera.

### How to get the current Camera IP:
> [!IMPORTANT]
> The 5G modem web administration page (`192.168.128.1`) is **NOT** accessible from the Wi-Fi or over
> the 5G network. You **MUST** connect your laptop directly to the modem via an **Ethernet cable**.

1. Connect an Ethernet cable between your laptop and the 5G modem's LAN switch.
2. Configure your laptop's Ethernet network adapter with a static IP:
   - **IP Address**: `192.168.128.100`
   - **Subnet Mask**: `255.255.255.0`
   - **Default Gateway**: `192.168.128.1`
3. Open your browser and navigate to:
   ```text
   http://192.168.128.1
   ```
4. Check the **"WAN IPv4 Address"** on the modem status page (e.g. `10.101.0.70`).
5. Update your `.env` file on the edge server:
   ```bash
   nano .env
   ```
   Set:
   ```ini
   CAMERA_HOST=10.101.0.70
   ```
6. Alternatively, click the **"⚡ 5G Modem (10.101.0.70)"** or **"✏️ Custom..."** button directly in the Web UI!

### Sparsh Topsvision v2.1 RTSP Stream Paths:
| Stream | Exact RTSP URL | Recommended Use |
|---|---|---|
| **Substream (720p)** | `rtsp://admin:admin123@<HOST>:554/avstream/channel=1/stream=1.sdp` | **Recommended** (Low latency, 30 FPS, light on CPU) |
| **Mainstream (4K)** | `rtsp://admin:admin123@<HOST>:554/avstream/channel=1/stream=0.sdp` | High resolution stream |

*(Note: Do not use `/stream1` as older firmware returns `403 Forbidden`).*

---

## Backup: Camera Connected Directly via Ethernet

If the 5G modem is turned off and the camera is plugged directly into a laptop's Ethernet port
(`192.168.128.100`), the camera's fixed IP is `192.168.128.10`.

* If running SignSpeak directly on that laptop:
  Click the **"🔌 Direct Ethernet (192.168.128.10)"** preset in the Web UI.
* If running SignSpeak on the VM, forward port 554 from the laptop using Windows portproxy:
  ```powershell
  netsh interface portproxy add v4tov4 listenport=554 listenaddress=0.0.0.0 connectport=554 connectaddress=192.168.128.10
  ```
  Then in the Web UI enter:
  `rtsp://admin:admin123@<LAPTOP_WIFI_IP>:554/avstream/channel=1/stream=1.sdp`

---

## ASL & Gesture Cheatsheet

| Gesture | Action | Description |
|---|---|---|
| **A – Z** | Letter typing | Hold hand shape steady for 0.5s to type character. |
| **0 – 9** | Digit typing | Static handshapes for numbers. |
| **Thumbs Up (`👍`)** | Space (` `) | Finishes current word and triggers word speech. |
| **I Love You (`🤟`)** | Period (`.`) | Finishes sentence and triggers full sentence speech. |
| **Open Palm Wave (`✋`)** | Backspace | Deletes the previous character from word buffer. |

---

## Web REST API Reference

The web backend exposes a clean JSON and streaming REST API:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main Web Dashboard. |
| `/video_feed` | `GET` | Multipart MJPEG stream with real-time HUD annotations. |
| `/api/status` | `GET` | JSON status: camera FPS, recognition metrics, word, transcript, model info. |
| `/api/action` | `POST` | Trigger action: `{"action": "space" \| "period" \| "backspace" \| "clear" \| "type", "param": "A"}`. |
| `/api/camera` | `POST` | Switch camera: `{"preset": "5g" \| "ethernet" \| "webcam"}` or `{"source": "<RTSP_URL>"}`. |
| `/api/settings` | `POST` | Live settings update: `{"hold_ms": 500, "confidence": 0.55, "mirror": true}`. |
| `/api/model` | `POST` | Switch model: `{"kind": "user" \| "legacy"}`. |
| `/api/dictionary`| `GET` | Full dictionary of ASL handshapes and descriptive hints. |
| `/api/system` | `GET` | Network configuration, edge server info, and IP lookup helper. |

---

## When something goes wrong

| Problem | Cause | Solution |
|---|---|---|
| **Camera shows "Reconnecting to camera..."** | RTSP stream not reachable. | Check modem IP via Ethernet at `192.168.128.1` or click the **Webcam 0** preset to test locally. |
| **RTSP returns 403 Forbidden** | Wrong stream path used. | Use `/avstream/channel=1/stream=1.sdp`. Never use `/stream1`. |
| **Laggy video feed** | UDP packet buffering. | SignSpeak enforces TCP transport (`rtsp_transport;tcp`) and buffer size 1. Use the 720p substream. |
| **Speech does not play** | Browser audio permissions. | Click anywhere on the webpage or press **"🔊 Speak Now"** to initialize the browser's Web Speech API. |
| **Port 8080 already in use** | Another service running. | Run `python app.py --port 8090` or stop the other process with `./stop.sh`. |
