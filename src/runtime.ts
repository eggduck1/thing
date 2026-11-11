type TauriWindow = Window & {
  __TAURI_INTERNALS__?: {
    metadata?: {
      currentWebview?: {
        label?: string;
      };
    };
  };
};

export function isNativeDesktop(): boolean {
  const label = (window as TauriWindow).__TAURI_INTERNALS__?.metadata?.currentWebview?.label;
  return typeof label === "string" && !label.startsWith("browser-preview-");
}
