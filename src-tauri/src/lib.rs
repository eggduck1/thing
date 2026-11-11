use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    time::Duration,
};
use tauri::{Emitter, Manager, State};
use tempfile::{NamedTempFile, TempDir};

const LOCAL_MODEL: &str = "qwen3-vl:2b";
const OLLAMA_URL: &str = "http://127.0.0.1:11434";
#[cfg(any())]
const KEYRING_SERVICE: &str = "com.mira.companion";
#[cfg(any())]
const KEYRING_USER: &str = "gemini-api-key";
#[cfg(any())]
const GEMINI_ENDPOINT: &str =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent";
#[cfg(any())]
const GEMINI_TOKEN_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/auth_tokens";
const MAX_FILE_BYTES: u64 = 120_000;
const MAX_TOTAL_BYTES: usize = 500_000;
const MAX_FILES: usize = 25;
const MAX_LOCAL_AUDIO_BYTES: usize = 12_000_000;

struct AppState {
    proposals: Mutex<HashMap<String, StagedProposal>>,
}

struct StagedProposal {
    project_root: PathBuf,
    changes: Vec<StagedChange>,
    _validation_copy: TempDir,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileChange {
    path: String,
    content: String,
}

#[derive(Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalAiStatus {
    model_installed: bool,
    speech_installed: bool,
    model_name: &'static str,
}

fn ollama_binary() -> Option<PathBuf> {
    [
        "/opt/homebrew/bin/ollama",
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

fn start_ollama() -> Result<(), String> {
    if std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        Duration::from_millis(300),
    )
    .is_ok()
    {
        return Ok(());
    }
    let binary = ollama_binary().ok_or_else(|| {
        "Ollama is not installed. Install Ollama from ollama.com, then reopen Mira.".to_string()
    })?;
    Command::new(binary)
        .arg("serve")
        .env("OLLAMA_HOST", "127.0.0.1:11434")
        .env("OLLAMA_NO_CLOUD", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Could not start the local Ollama service: {error}"))?;

    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(250));
        if std::net::TcpStream::connect_timeout(
            &"127.0.0.1:11434".parse().unwrap(),
            Duration::from_millis(300),
        )
        .is_ok()
        {
            return Ok(());
        }
    }
    Err("Ollama did not start listening on the local device.".into())
}

fn whisper_binary() -> Option<PathBuf> {
    [
        "/opt/homebrew/bin/whisper-cli",
        "/opt/homebrew/opt/whisper.cpp/bin/whisper-cli",
        "/usr/local/bin/whisper-cli",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

fn whisper_model_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".local/share/mira/ggml-tiny.en-q5_1.bin")
}

fn whisper_model_is_ready(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.len() >= 20_000_000)
}

#[tauri::command]
async fn local_ai_status() -> Result<LocalAiStatus, String> {
    start_ollama()?;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|error| format!("Could not prepare local AI check: {error}"))?
        .get(format!("{OLLAMA_URL}/api/tags"))
        .send()
        .await
        .map_err(|error| format!("Could not reach local Ollama: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("Local Ollama returned HTTP {}.", response.status()));
    }
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not read the local model list: {error}"))?;
    let model_installed = payload
        .get("models")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                model.get("name").and_then(serde_json::Value::as_str) == Some(LOCAL_MODEL)
            })
        });
    Ok(LocalAiStatus {
        model_installed,
        speech_installed: whisper_binary().is_some()
            && whisper_model_is_ready(&whisper_model_path()),
        model_name: LOCAL_MODEL,
    })
}

#[tauri::command]
async fn install_local_ai(app: tauri::AppHandle) -> Result<(), String> {
    start_ollama()?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30 * 60))
        .build()
        .map_err(|error| format!("Could not prepare local model download: {error}"))?;
    let response = client
        .post(format!("{OLLAMA_URL}/api/pull"))
        .json(&serde_json::json!({ "name": LOCAL_MODEL, "stream": false }))
        .send()
        .await
        .map_err(|error| format!("Could not download the local vision model: {error}"))?;
    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not read the model download result: {error}"))?;
    if !status.is_success() {
        let detail = payload
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Ollama could not download the model.");
        return Err(format!("{detail} (HTTP {status})"));
    }

    let model_path = whisper_model_path();
    if !whisper_model_is_ready(&model_path) {
        if whisper_binary().is_none() {
            return Err(
                "Chat and vision are ready. Install whisper.cpp to enable fully local voice input."
                    .into(),
            );
        }
        let parent = model_path
            .parent()
            .ok_or("Could not locate the local speech-model directory.")?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("Could not create the speech-model directory: {error}"))?;
        app.emit(
            "local-ai-install-progress",
            "Downloading the local speech model…",
        )
        .map_err(|error| format!("Could not report speech model setup: {error}"))?;
        let response = client
            .get("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin")
            .send()
            .await
            .map_err(|error| format!("Could not download the local speech model: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "Could not download the local speech model (HTTP {status})."
            ));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| format!("Could not read the local speech model download: {error}"))?;
        if bytes.len() < 20_000_000 {
            return Err("The downloaded speech model was incomplete; try setup again.".into());
        }
        fs::write(&model_path, bytes)
            .map_err(|error| format!("Could not save the local speech model: {error}"))?;
    }
    Ok(())
}

#[tauri::command]
async fn local_ai_chat(
    messages: Vec<ChatMessage>,
    image_base64: Option<String>,
) -> Result<String, String> {
    if messages.is_empty() || messages.len() > 100 {
        return Err("A local chat request must contain between 1 and 100 messages.".into());
    }
    let total_bytes: usize = messages.iter().map(|message| message.content.len()).sum();
    if total_bytes > MAX_TOTAL_BYTES
        || messages.iter().any(|message| {
            !matches!(message.role.as_str(), "system" | "user" | "assistant")
                || message.content.len() > MAX_TOTAL_BYTES
        })
    {
        return Err("The local chat request exceeds the 500 KB text context limit.".into());
    }
    if image_base64
        .as_ref()
        .is_some_and(|image| image.len() > 8_000_000)
    {
        return Err("The image exceeds the 6 MB limit.".into());
    }
    start_ollama()?;
    let mut ollama_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|message| serde_json::json!({ "role": message.role, "content": message.content }))
        .collect();
    if let Some(image) = image_base64 {
        let last = ollama_messages
            .last_mut()
            .ok_or("The local chat request is empty.")?;
        if last.get("role").and_then(serde_json::Value::as_str) != Some("user") {
            return Err("An image must be attached to a user message.".into());
        }
        last["images"] = serde_json::json!([image]);
    }
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|error| format!("Could not prepare local inference: {error}"))?
        .post(format!("{OLLAMA_URL}/api/chat"))
        .json(&serde_json::json!({
            "model": LOCAL_MODEL,
            "messages": ollama_messages,
            "stream": false,
            "keep_alive": "5m",
            "options": { "num_ctx": 4096 }
        }))
        .send()
        .await
        .map_err(|error| format!("Could not reach the local model: {error}"))?;
    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not read the local model response: {error}"))?;
    if !status.is_success() {
        let detail = payload
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Local inference failed.");
        return Err(format!("{detail} (HTTP {status})"));
    }
    payload
        .pointer("/message/content")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "The local model returned an empty response.".into())
}

#[tauri::command]
fn local_transcribe_audio(audio_base64: String) -> Result<String, String> {
    if audio_base64.is_empty() || audio_base64.len() > MAX_LOCAL_AUDIO_BYTES {
        return Err("The local audio clip is empty or exceeds the 9 MB limit.".into());
    }
    let whisper = whisper_binary().ok_or_else(|| {
        "Local speech input needs whisper.cpp. Install it with Homebrew, then restart Mira."
            .to_string()
    })?;
    let model = whisper_model_path();
    if !whisper_model_is_ready(&model) {
        return Err("The local speech model is missing. Open Settings → Local AI and choose Set up local AI.".into());
    }
    let audio = BASE64
        .decode(audio_base64)
        .map_err(|_| "The recorded audio is invalid.".to_string())?;
    if audio.len() < 44 || &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return Err("The recorded audio is not a valid WAV file.".into());
    }
    let mut input = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|error| format!("Could not prepare local audio input: {error}"))?;
    input
        .write_all(&audio)
        .map_err(|error| format!("Could not write local audio input: {error}"))?;
    let output = Command::new(whisper)
        .args([
            "-m",
            model.to_str().ok_or("Invalid local speech-model path.")?,
            "-f",
        ])
        .arg(input.path())
        .args(["-nt", "-np"])
        .output()
        .map_err(|error| format!("Could not run local speech recognition: {error}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Local speech recognition failed: {}",
            detail.trim()
        ));
    }
    let transcript = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if transcript.is_empty() {
        return Err("No speech was recognized. Try speaking a little longer.".into());
    }
    Ok(transcript)
}

#[tauri::command]
fn local_speak(text: String) -> Result<(), String> {
    if text.trim().is_empty() || text.len() > 12_000 {
        return Err("The local speech text is empty or too long.".into());
    }
    let status = Command::new("/usr/bin/say")
        .args(["-v", "Samantha", "-r", "180"])
        .arg(text)
        .status()
        .map_err(|error| format!("Could not start macOS local speech: {error}"))?;
    if !status.success() {
        return Err("macOS local speech did not finish successfully.".into());
    }
    Ok(())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectFile {
    path: String,
    content: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct StagedChange {
    path: String,
    before: Option<String>,
    after: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StageResult {
    id: String,
    changes: Vec<StagedChange>,
    validation_message: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct MiraProfile {
    soul: String,
    identity: String,
    user: String,
    memory: Vec<String>,
    #[serde(default)]
    reviewed_at: u64,
}

impl Default for MiraProfile {
    fn default() -> Self {
        Self {
            soul: "Be warm, curious, honest, and respectful. You are an AI, not a human or conscious being. Never identify people from camera images.".into(),
            identity: "Mira is a playful AI companion. The user may call her Dad's AI daughter as a persona, not as a literal relationship.".into(),
            user: String::new(),
            memory: Vec::new(),
            reviewed_at: 0,
        }
    }
}

#[tauri::command]
#[cfg(any())]
async fn create_gemini_live_token(system_instruction: String) -> Result<String, String> {
    if system_instruction.trim().is_empty() || system_instruction.len() > 12_000 {
        return Err("The live session instructions are empty or too large.".into());
    }
    let api_key = keyring_entry()?
        .get_password()
        .map_err(|error| format!("Could not read the API key from the system keychain: {error}"))?;
    let request_body = serde_json::json!({ "uses": 1 });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("Could not prepare a secure Gemini connection: {error}"))?;
    let response = client
        .post(GEMINI_TOKEN_ENDPOINT)
        .header("x-goog-api-key", api_key)
        .json(&request_body)
        .send()
        .await
        .map_err(|error| format!("Could not request a Gemini live token: {error}"))?;
    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not read Gemini's token response: {error}"))?;
    if !status.is_success() {
        let detail = payload
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Gemini could not create a live token.");
        return Err(format!("{detail} (HTTP {status})"));
    }
    payload
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "Gemini returned an unexpected live token.".into())
}

#[tauri::command]
fn load_mira_profile(app: tauri::AppHandle) -> Result<MiraProfile, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Could not locate Mira's app data: {error}"))?
        .join("profile");
    let mut profile = MiraProfile::default();
    profile.soul = read_profile_part(&directory, "soul.json", &profile.soul)?;
    profile.identity = read_profile_part(&directory, "identity.json", &profile.identity)?;
    profile.user = read_profile_part(&directory, "user.json", &profile.user)?;
    profile.memory = read_profile_part(&directory, "memory.json", &profile.memory)?;
    validate_profile(&profile)?;
    Ok(profile)
}

#[tauri::command]
fn save_mira_profile(app: tauri::AppHandle, profile: MiraProfile) -> Result<(), String> {
    validate_profile(&profile)?;
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Could not locate Mira's app data: {error}"))?
        .join("profile");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Could not create Mira's local profile: {error}"))?;
    write_profile_part(&directory, "soul.json", &profile.soul)?;
    write_profile_part(&directory, "identity.json", &profile.identity)?;
    write_profile_part(&directory, "user.json", &profile.user)?;
    write_profile_part(&directory, "memory.json", &profile.memory)?;
    Ok(())
}

fn read_profile_part<T: for<'de> Deserialize<'de> + Serialize>(
    directory: &Path,
    filename: &str,
    default: &T,
) -> Result<T, String> {
    let path = directory.join(filename);
    if !path.exists() {
        return serde_json::from_slice(
            &serde_json::to_vec(default)
                .map_err(|error| format!("Could not initialize {filename}: {error}"))?,
        )
        .map_err(|error| format!("Could not initialize {filename}: {error}"));
    }
    let bytes = fs::read(&path).map_err(|error| format!("Could not read {filename}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("Could not parse {filename}: {error}"))
}

fn write_profile_part<T: Serialize>(
    directory: &Path,
    filename: &str,
    value: &T,
) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("Could not serialize {filename}: {error}"))?;
    let mut temporary = NamedTempFile::new_in(directory)
        .map_err(|error| format!("Could not prepare {filename}: {error}"))?;
    temporary
        .write_all(&bytes)
        .map_err(|error| format!("Could not write {filename}: {error}"))?;
    temporary
        .persist(directory.join(filename))
        .map_err(|error| format!("Could not save {filename}: {}", error.error))?;
    Ok(())
}

fn validate_profile(profile: &MiraProfile) -> Result<(), String> {
    if profile.soul.len() > 8_000 || profile.identity.len() > 8_000 || profile.user.len() > 8_000 {
        return Err("A Mira profile section exceeds the 8 KB limit.".into());
    }
    if profile.memory.len() > 50
        || profile
            .memory
            .iter()
            .any(|entry| entry.trim().is_empty() || entry.len() > 180)
    {
        return Err("Mira's memory must contain at most 50 short, non-empty notes.".into());
    }
    Ok(())
}

#[tauri::command]
#[cfg(any())]
fn has_api_key() -> Result<bool, String> {
    match keyring_entry()?.get_password() {
        Ok(_) => Ok(true),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(error) => Err(format!(
            "Could not read the API key from the system keychain: {error}"
        )),
    }
}

#[tauri::command]
#[cfg(any())]
fn set_api_key(api_key: String) -> Result<(), String> {
    let key = api_key.trim();
    if key.is_empty() || key.len() > 512 {
        return Err("Enter an API key between 1 and 512 characters.".into());
    }
    keyring_entry()?
        .set_password(key)
        .map_err(|error| format!("Could not save the API key in the system keychain: {error}"))
}

#[tauri::command]
#[cfg(any())]
fn clear_api_key() -> Result<(), String> {
    match keyring_entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(format!(
            "Could not remove the API key from the system keychain: {error}"
        )),
    }
}

#[tauri::command]
#[cfg(any())]
async fn gemini_chat(
    messages: Vec<ChatMessage>,
    image_base64: Option<String>,
    image_mime_type: Option<String>,
) -> Result<String, String> {
    if messages.is_empty() || messages.len() > 100 {
        return Err("A chat request must contain between 1 and 100 messages.".into());
    }
    let total_bytes: usize = messages.iter().map(|message| message.content.len()).sum();
    if total_bytes > MAX_TOTAL_BYTES
        || messages.iter().any(|message| {
            !matches!(message.role.as_str(), "system" | "user" | "assistant")
                || message.content.len() > MAX_TOTAL_BYTES
        })
    {
        return Err("The chat request is invalid or exceeds the 500 KB text context limit.".into());
    }
    if image_base64
        .as_ref()
        .is_some_and(|image| image.len() > 8_000_000)
    {
        return Err("The image exceeds the 6 MB limit.".into());
    }
    if image_base64.is_some()
        && !matches!(
            image_mime_type.as_deref(),
            Some("image/jpeg" | "image/png" | "image/webp")
        )
    {
        return Err("Only JPEG, PNG, and WebP images are supported.".into());
    }

    let api_key = keyring_entry()?
        .get_password()
        .map_err(|error| format!("Could not read the API key from the system keychain: {error}"))?;
    let mut system_instructions = Vec::new();
    let mut contents = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role == "system" {
            system_instructions.push(message.content.clone());
            continue;
        }
        let mut parts = vec![serde_json::json!({ "text": message.content })];
        if index == messages.len() - 1 && message.role == "user" {
            if let Some(image) = &image_base64 {
                parts.push(serde_json::json!({
                    "inline_data": {
                        "mime_type": image_mime_type.as_deref().unwrap_or("image/jpeg"),
                        "data": image
                    }
                }));
            }
        }
        contents.push(serde_json::json!({
            "role": if message.role == "assistant" { "model" } else { "user" },
            "parts": parts
        }));
    }
    if contents.is_empty() {
        return Err("A chat request must include a user message.".into());
    }
    let mut request_body = serde_json::json!({ "contents": contents });
    if !system_instructions.is_empty() {
        request_body["system_instruction"] = serde_json::json!({
            "parts": [{ "text": system_instructions.join("\n\n") }]
        });
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("Could not prepare a secure Gemini connection: {error}"))?;
    let response = client
        .post(GEMINI_ENDPOINT)
        .header("x-goog-api-key", api_key)
        .json(&request_body)
        .send()
        .await
        .map_err(|error| format!("Could not reach Gemini: {error}"))?;
    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not read Gemini's response: {error}"))?;
    if !status.is_success() {
        let detail = payload
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Gemini returned an error.");
        return Err(format!("{detail} (HTTP {status})"));
    }
    let response_text = payload
        .pointer("/candidates/0/content/parts")
        .and_then(serde_json::Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .filter(|text| !text.is_empty());
    response_text.ok_or_else(|| "Gemini returned an unexpected response.".into())
}

#[tauri::command]
#[cfg(any())]
async fn gemini_transcribe(
    audio_base64: String,
    audio_mime_type: String,
) -> Result<String, String> {
    if audio_base64.is_empty() || audio_base64.len() > 8_000_000 {
        return Err("The voice clip is empty or exceeds the 6 MB limit.".into());
    }
    if !matches!(
        audio_mime_type.as_str(),
        "audio/webm"
            | "audio/mp4"
            | "audio/aac"
            | "audio/wav"
            | "audio/ogg"
            | "audio/mpeg"
            | "audio/flac"
    ) {
        return Err("This browser's audio format is not supported for transcription.".into());
    }
    let api_key = keyring_entry()?
        .get_password()
        .map_err(|error| format!("Could not read the API key from the system keychain: {error}"))?;
    let request_body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [
                { "text": "Transcribe the spoken words in this audio faithfully. Return only the transcription, with no commentary." },
                { "inline_data": { "mime_type": audio_mime_type, "data": audio_base64 } }
            ]
        }]
    });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("Could not prepare a secure Gemini connection: {error}"))?;
    let response = client
        .post(GEMINI_ENDPOINT)
        .header("x-goog-api-key", api_key)
        .json(&request_body)
        .send()
        .await
        .map_err(|error| format!("Could not reach Gemini: {error}"))?;
    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not read Gemini's response: {error}"))?;
    if !status.is_success() {
        let detail = payload
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Gemini returned an error.");
        return Err(format!("{detail} (HTTP {status})"));
    }
    payload
        .pointer("/candidates/0/content/parts")
        .and_then(serde_json::Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "Gemini returned an unexpected transcription.".into())
}

#[cfg(any())]
fn keyring_entry() -> Result<Entry, String> {
    Entry::new(KEYRING_SERVICE, KEYRING_USER).map_err(|error| error.to_string())
}

#[tauri::command]
fn list_project_files(project_path: String) -> Result<Vec<ProjectFile>, String> {
    let root = canonical_project_root(&project_path)?;
    let mut files = Vec::new();
    collect_project_files(&root, &root, &mut files, 0)?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

#[tauri::command]
fn read_project_files(
    project_path: String,
    paths: Vec<String>,
) -> Result<Vec<ProjectFile>, String> {
    if paths.is_empty() || paths.len() > MAX_FILES {
        return Err(format!("Select between 1 and {MAX_FILES} files."));
    }
    let root = canonical_project_root(&project_path)?;
    let mut total = 0usize;
    let mut result = Vec::with_capacity(paths.len());
    for relative in paths {
        let path = checked_file_path(&root, &relative, false)?;
        let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
            return Err(format!(
                "{relative} is not a supported text file or is too large."
            ));
        }
        let content = fs::read_to_string(&path)
            .map_err(|_| format!("{relative} is not a readable UTF-8 text file."))?;
        total += content.len();
        if total > MAX_TOTAL_BYTES {
            return Err("The selected files exceed the 500 KB context limit.".into());
        }
        result.push(ProjectFile {
            path: relative,
            content: Some(content),
        });
    }
    Ok(result)
}

#[tauri::command]
fn stage_project_changes(
    project_path: String,
    changes: Vec<FileChange>,
    allowed_paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<StageResult, String> {
    if changes.is_empty() || changes.len() > MAX_FILES {
        return Err(format!(
            "A proposal must contain between 1 and {MAX_FILES} files."
        ));
    }
    if allowed_paths.is_empty() || allowed_paths.len() > MAX_FILES {
        return Err(format!(
            "Select between 1 and {MAX_FILES} files before staging a proposal."
        ));
    }
    let root = canonical_project_root(&project_path)?;
    let allowed: std::collections::HashSet<String> = allowed_paths
        .into_iter()
        .map(|path| {
            checked_file_path(&root, &path, false)?;
            Ok(path)
        })
        .collect::<Result<_, String>>()?;
    let mut unique = std::collections::HashSet::new();
    let mut total = 0usize;
    let mut staged = Vec::with_capacity(changes.len());
    for change in changes {
        if !allowed.contains(&change.path) {
            return Err(format!(
                "{} was not among the files selected for this proposal.",
                change.path
            ));
        }
        if !unique.insert(change.path.clone()) {
            return Err(format!(
                "The proposal contains duplicate path {}.",
                change.path
            ));
        }
        if change.content.len() as u64 > MAX_FILE_BYTES {
            return Err(format!(
                "{} exceeds the 120 KB per-file limit.",
                change.path
            ));
        }
        total += change.content.len();
        if total > MAX_TOTAL_BYTES {
            return Err("The proposal exceeds the 500 KB total limit.".into());
        }
        let destination = checked_file_path(&root, &change.path, true)?;
        let before = match fs::read_to_string(&destination) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => {
                return Err(format!(
                    "{} is not a readable UTF-8 text file.",
                    change.path
                ))
            }
        };
        staged.push(StagedChange {
            path: change.path,
            before,
            after: change.content,
        });
    }

    let validation_copy = tempfile::tempdir().map_err(|error| error.to_string())?;
    for change in &staged {
        if let Some(before) = &change.before {
            write_relative(validation_copy.path().join("before"), &change.path, before)?;
        }
        write_relative(
            validation_copy.path().join("after"),
            &change.path,
            &change.after,
        )?;
    }

    let id = uuid::Uuid::new_v4().to_string();
    let response_changes = staged.clone();
    state
        .proposals
        .lock()
        .map_err(|_| "Proposal storage is unavailable.".to_string())?
        .insert(
            id.clone(),
            StagedProposal {
                project_root: root,
                changes: staged,
                _validation_copy: validation_copy,
            },
        );
    Ok(StageResult {
        id,
        changes: response_changes,
        validation_message: "Path, file-type, and size checks passed. Before/after copies were staged in an isolated temporary workspace; no project commands were run.".into(),
    })
}

#[tauri::command]
fn discard_project_changes(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state
        .proposals
        .lock()
        .map_err(|_| "Proposal storage is unavailable.".to_string())?
        .remove(&id);
    Ok(())
}

#[tauri::command]
fn apply_project_changes(id: String, state: State<'_, AppState>) -> Result<(), String> {
    let proposal = state
        .proposals
        .lock()
        .map_err(|_| "Proposal storage is unavailable.".to_string())?
        .remove(&id)
        .ok_or_else(|| {
            "This proposal is no longer available. Stage it again before applying.".to_string()
        })?;

    let current_root = canonical_project_root(
        proposal
            .project_root
            .to_str()
            .ok_or_else(|| "The project path is invalid.".to_string())?,
    )?;
    if current_root != proposal.project_root {
        return Err("The project folder changed; stage the proposal again.".into());
    }
    for change in &proposal.changes {
        let destination = checked_file_path(&current_root, &change.path, true)?;
        let current = match fs::read_to_string(destination) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => {
                return Err(format!(
                    "{} is no longer a readable text file.",
                    change.path
                ))
            }
        };
        if current != change.before {
            return Err(format!(
                "{} changed since review. Stage the proposal again.",
                change.path
            ));
        }
    }

    let mut applied: Vec<&StagedChange> = Vec::new();
    for change in &proposal.changes {
        let destination = match checked_file_path(&current_root, &change.path, true) {
            Ok(destination) => destination,
            Err(error) => {
                return Err(with_rollback_status(&current_root, &applied, error));
            }
        };
        let current = match fs::read_to_string(&destination) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(with_rollback_status(
                    &current_root,
                    &applied,
                    format!("Could not recheck {}: {error}", change.path),
                ));
            }
        };
        if current != change.before {
            return Err(with_rollback_status(
                &current_root,
                &applied,
                format!(
                    "{} changed during apply. Review and stage the proposal again.",
                    change.path
                ),
            ));
        }
        let parent = destination
            .parent()
            .ok_or_else(|| "Invalid destination path.".to_string())?;
        if let Err(error) = fs::create_dir_all(parent) {
            return Err(with_rollback_status(
                &current_root,
                &applied,
                format!("Could not create the destination folder: {error}"),
            ));
        }
        let mut temporary = match NamedTempFile::new_in(parent) {
            Ok(temporary) => temporary,
            Err(error) => {
                return Err(with_rollback_status(
                    &current_root,
                    &applied,
                    error.to_string(),
                ));
            }
        };
        if let Err(error) = temporary.write_all(change.after.as_bytes()) {
            return Err(with_rollback_status(
                &current_root,
                &applied,
                error.to_string(),
            ));
        }
        match fs::metadata(&destination) {
            Ok(metadata) => {
                if let Err(error) = temporary.as_file().set_permissions(metadata.permissions()) {
                    return Err(with_rollback_status(
                        &current_root,
                        &applied,
                        error.to_string(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(with_rollback_status(
                    &current_root,
                    &applied,
                    format!("Could not inspect {}: {error}", change.path),
                ));
            }
        }
        if let Err(error) = temporary.persist(&destination) {
            return Err(with_rollback_status(
                &current_root,
                &applied,
                format!("Could not apply {}: {}", change.path, error.error),
            ));
        }
        applied.push(change);
    }
    Ok(())
}

fn canonical_project_root(path: &str) -> Result<PathBuf, String> {
    let root = fs::canonicalize(path)
        .map_err(|error| format!("Could not open project folder: {error}"))?;
    if !root.is_dir() {
        return Err("Choose a project folder.".into());
    }
    Ok(root)
}

fn checked_file_path(root: &Path, relative: &str, allow_missing: bool) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if relative.is_empty()
        || relative.contains('\\')
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        || !supported_source_path(path)
    {
        return Err(format!(
            "{} is not an allowed project source-file path.",
            relative
        ));
    }

    let mut current = root.to_path_buf();
    let components: Vec<_> = path.components().collect();
    for (index, component) in components.iter().enumerate() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!("{} traverses a symbolic link.", relative));
            }
            Ok(metadata) if index + 1 < components.len() && !metadata.is_dir() => {
                return Err(format!("{} has a non-folder parent.", relative));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && allow_missing => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(format!("{} does not exist in the project.", relative));
            }
            Err(error) => return Err(format!("Could not inspect {}: {error}", relative)),
        }
    }
    if !allow_missing && !current.is_file() {
        return Err(format!("{} is not a file.", relative));
    }
    Ok(current)
}

fn supported_source_path(path: &Path) -> bool {
    let excluded = [
        ".git",
        ".env",
        "node_modules",
        "target",
        "dist",
        "build",
        ".ssh",
        ".github",
        ".vscode",
        "package.json",
        "package-lock.json",
        "Cargo.toml",
        "tauri.conf.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    ];
    if path.components().any(|component| {
        let part = component.as_os_str().to_string_lossy();
        part.starts_with('.') || excluded.contains(&part.as_ref())
    }) {
        return false;
    }
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("ts" | "tsx" | "js" | "jsx" | "css" | "html" | "rs" | "md")
    )
}

fn collect_project_files(
    root: &Path,
    directory: &Path,
    result: &mut Vec<ProjectFile>,
    depth: usize,
) -> Result<(), String> {
    if result.len() >= 500 || depth >= 32 {
        return Ok(());
    }
    let entries = fs::read_dir(directory).map_err(|error| error.to_string())?;
    for entry in entries {
        if result.len() >= 500 {
            break;
        }
        let entry = entry.map_err(|error| error.to_string())?;
        let metadata = entry.file_type().map_err(|error| error.to_string())?;
        let path = entry.path();
        if metadata.is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            if supported_source_path(path.strip_prefix(root).unwrap_or(&path)) {
                collect_project_files(root, &path, result, depth + 1)?;
            }
            continue;
        }
        let relative = match path.strip_prefix(root) {
            Ok(relative) if supported_source_path(relative) => relative,
            _ => continue,
        };
        let metadata = entry.metadata().map_err(|error| error.to_string())?;
        if metadata.len() > MAX_FILE_BYTES {
            continue;
        }
        let relative = relative.to_string_lossy().replace('\\', "/");
        result.push(ProjectFile {
            path: relative,
            content: None,
        });
        if result.len() >= 500 {
            break;
        }
    }
    Ok(())
}

fn write_relative(root: PathBuf, relative: &str, content: &str) -> Result<(), String> {
    let path = root.join(relative);
    let parent = path
        .parent()
        .ok_or_else(|| "Invalid temporary path.".to_string())?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    fs::write(path, content).map_err(|error| error.to_string())
}

fn with_rollback_status(root: &Path, changes: &[&StagedChange], original_error: String) -> String {
    match rollback_changes(root, changes) {
        Ok(()) => original_error,
        Err(rollback_error) => {
            format!("{original_error}; rollback was incomplete: {rollback_error}")
        }
    }
}

fn rollback_changes(root: &Path, changes: &[&StagedChange]) -> Result<(), String> {
    let mut failures = Vec::new();
    for change in changes.iter().rev() {
        let path = match checked_file_path(root, &change.path, true) {
            Ok(path) => path,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        let result = match &change.before {
            Some(contents) => fs::write(path, contents),
            None => fs::remove_file(path),
        };
        if let Err(error) = result {
            failures.push(format!("could not restore {}: {error}", change.path));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            proposals: Mutex::new(HashMap::new()),
        })
        .invoke_handler(tauri::generate_handler![
            local_ai_status,
            install_local_ai,
            local_ai_chat,
            local_transcribe_audio,
            local_speak,
            load_mira_profile,
            save_mira_profile,
            list_project_files,
            read_project_files,
            stage_project_changes,
            discard_project_changes,
            apply_project_changes
        ])
        .run(tauri::generate_context!())
        .expect("error while running Mira");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_relative_source_files_and_rejects_traversal_and_configuration() {
        assert!(supported_source_path(Path::new("src/App.tsx")));
        assert!(!supported_source_path(Path::new("../outside.ts")));
        assert!(!supported_source_path(Path::new("package.json")));
        assert!(!supported_source_path(Path::new(".env")));
    }

    #[test]
    fn path_validation_rejects_parent_components_and_symbolic_links() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("src")).unwrap();
        fs::write(temp.path().join("src/App.tsx"), "export {}").unwrap();
        assert!(checked_file_path(temp.path(), "../outside.ts", true).is_err());
        assert!(checked_file_path(temp.path(), "src/App.tsx", false).is_ok());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(temp.path(), temp.path().join("linked")).unwrap();
            assert!(checked_file_path(temp.path(), "linked/App.tsx", true).is_err());
        }
    }
}
