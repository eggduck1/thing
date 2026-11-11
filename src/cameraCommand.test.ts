import { describe, expect, it } from "vitest";
import { isCameraOnRequest } from "./cameraCommand";

describe("isCameraOnRequest", () => {
  it.each([
    "Mira, turn on the camera",
    "Can you turn on my camera please?",
    "Please activate camera",
    "camera on",
  ])("recognizes a camera-on request: %s", (text) => {
    expect(isCameraOnRequest(text)).toBe(true);
  });

  it.each([
    "How do I turn on the camera?",
    "Turn off the camera",
    "Don't turn on the camera",
    "Tell me about camera permissions",
  ])("does not treat unrelated or camera-off text as a command: %s", (text) => {
    expect(isCameraOnRequest(text)).toBe(false);
  });
});
