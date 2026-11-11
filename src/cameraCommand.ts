const CAMERA_ON_REQUEST =
  /\b(?:turn|switch)\s+on\s+(?:(?:the|my)\s+)?camera\b|\b(?:activate|enable|open|start)\s+(?:(?:the|my)\s+)?camera\b|\bcamera\s+(?:on|enabled)\b/i;
const CAMERA_OFF_REQUEST =
  /\b(?:turn|switch)\s+off\s+(?:(?:the|my)\s+)?camera\b|\b(?:disable|close|stop)\s+(?:(?:the|my)\s+)?camera\b|\bcamera\s+off\b/i;

export function isCameraOnRequest(text: string): boolean {
  if (/\b(?:don't|do not|never|not)\b.{0,40}\b(?:turn|switch|activate|enable|open|start)\b/i.test(text)) {
    return false;
  }
  if (/^\s*how\s+(?:do|can|could)\s+i\b/i.test(text)) return false;
  return CAMERA_ON_REQUEST.test(text) && !CAMERA_OFF_REQUEST.test(text);
}
