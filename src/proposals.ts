export interface ProposedFile {
  path: string;
  content: string;
}

export function parseProposal(raw: string): ProposedFile[] {
  const cleaned = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "");
  const parsed: unknown = JSON.parse(cleaned);
  if (
    !Array.isArray(parsed) ||
    parsed.length === 0 ||
    parsed.length > 25 ||
    !parsed.every(
      (file): file is ProposedFile =>
        typeof file === "object" &&
        file !== null &&
        "path" in file &&
        "content" in file &&
        typeof file.path === "string" &&
        typeof file.content === "string",
    )
  ) {
    throw new Error("The response must be a JSON array of { path, content } files.");
  }

  const seen = new Set<string>();
  for (const file of parsed) {
    if (!file.path.trim() || seen.has(file.path)) {
      throw new Error("The response contains an empty or duplicate file path.");
    }
    seen.add(file.path);
  }
  return parsed;
}

export function buildProposalPrompt(
  request: string,
  files: Array<{ path: string; content: string }>,
): string {
  return [
    "Propose a focused code change for the user's request.",
    "Return only a JSON array of objects with exactly these fields: path (project-relative string), content (complete replacement file text).",
    "Do not include markdown fences, explanations, shell commands, or files the user did not select.",
    "Never suggest secrets, credentials, or destructive behavior. Keep the change limited to the selected files.",
    "",
    `Request:\n${request}`,
    "",
    "Selected project files:",
    ...files.map(({ path, content }) => `\n--- FILE ${path} ---\n${content}`),
  ].join("\n");
}
