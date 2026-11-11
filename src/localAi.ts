import { invoke } from "@tauri-apps/api/core";
import { isNativeDesktop } from "./runtime";

export type LocalAiStatus = {
  modelInstalled: boolean;
  speechInstalled: boolean;
  modelName: string;
};

export type LocalMessage = { role: string; content: string };

function requireDesktop(): void {
  if (!isNativeDesktop()) {
    throw new Error("Local AI runs in the Mira desktop app, not the browser preview.");
  }
}

export async function getLocalAiStatus(): Promise<LocalAiStatus> {
  requireDesktop();
  return invoke<LocalAiStatus>("local_ai_status");
}

export async function installLocalAi(): Promise<void> {
  requireDesktop();
  await invoke("install_local_ai");
}

export async function localChat(
  messages: LocalMessage[],
  imageBase64?: string,
): Promise<string> {
  requireDesktop();
  return invoke<string>("local_ai_chat", {
    messages,
    imageBase64: imageBase64 ?? null,
  });
}

export async function localTranscribe(audioBase64: string): Promise<string> {
  requireDesktop();
  return invoke<string>("local_transcribe_audio", { audioBase64 });
}

export async function localSpeak(text: string): Promise<void> {
  requireDesktop();
  await invoke("local_speak", { text });
}
