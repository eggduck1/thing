# Mira — Local AI Companion

Mira is a Tauri desktop companion for macOS. Chat and camera snapshots run through a small local vision-language model with Ollama; voice input uses Whisper locally, and voice replies use macOS speech. After setup, prompts, images, audio, and profile data stay on the Mac. Mira does not need a Gemini key.

## Requirements

- macOS on Apple silicon or a supported Intel Mac
- Node.js 20 or newer and npm
- Rust and the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)
- [Ollama](https://ollama.com/download) for local chat and vision
- Homebrew `whisper.cpp` for local voice transcription: `brew install whisper-cpp`

The default Qwen3-VL 2B model is about 1.9 GB; the quantized Whisper English tiny model is about 32 MB. The model setup needs an internet connection once to download those weights. Inference is local afterward. A Mac with 8 GB RAM can run this small model; larger models may be too memory-intensive.

## Run Mira

```sh
npm install
npm run tauri dev
```

Open **Settings → Local AI → Set up local AI**. Mira starts Ollama locally, downloads Qwen3-VL 2B and the Whisper model if needed, and reports when each feature is ready. The browser preview is for UI development only and cannot run local inference.

To build the macOS app bundle:

```sh
npm run tauri build -- --bundles app
```

## Privacy and controls

- **Chat and camera:** text and captured JPEG images are sent only to Ollama at `127.0.0.1`. Mira does not continuously analyze the camera; a camera snapshot is sent only with a chat message.
- **Voice input:** Talk records after a click. Press to talk records microphone audio and shows the camera preview; when stopped (or after 30 seconds), one camera frame and the recording are transcribed and processed locally. Switching away or closing the app discards an active recording.
- **Voice output:** the Speak button uses macOS's built-in `say` voice locally.
- **Memory:** conversation review uses the local model. Profile notes and pending review messages are stored on this device and can be edited or deleted in Settings.
- **Safe code proposals:** selected source files are provided to the local model. Changes are staged for review and are not applied until explicitly approved; generated code is never executed.
- **No wake word or background listening.** Camera and microphone access starts only after an explicit click and permission.

## Checks

```sh
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml
```
