import { afterEach, describe, expect, it, vi } from "vitest";
import { isNativeDesktop } from "./runtime";

type RuntimeWindow = Window & {
  __TAURI_INTERNALS__?: { metadata?: { currentWebview?: { label?: string } } };
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("desktop runtime detection", () => {
  it("recognizes the native Tauri window", () => {
    vi.stubGlobal("window", {
      __TAURI_INTERNALS__: { metadata: { currentWebview: { label: "main" } } },
    } as RuntimeWindow);
    expect(isNativeDesktop()).toBe(true);
  });

  it("does not mistake the browser preview shim for the desktop app", () => {
    vi.stubGlobal("window", {
      __TAURI_INTERNALS__: { metadata: { currentWebview: { label: "browser-preview-mira" } } },
    } as RuntimeWindow);
    expect(isNativeDesktop()).toBe(false);
  });
});
