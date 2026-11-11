fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "local_ai_status",
            "install_local_ai",
            "local_ai_chat",
            "local_transcribe_audio",
            "local_speak",
            "load_mira_profile",
            "save_mira_profile",
            "list_project_files",
            "read_project_files",
            "stage_project_changes",
            "discard_project_changes",
            "apply_project_changes",
        ]),
    ))
    .expect("failed to build Tauri application permissions");
}
