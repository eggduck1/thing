import { describe, expect, it } from "vitest";
import { buildProposalPrompt, parseProposal } from "./proposals";

describe("parseProposal", () => {
  it("parses a JSON list and tolerates a JSON code fence", () => {
    expect(parseProposal('```json\n[{"path":"src/App.tsx","content":"new"}]\n```')).toEqual([
      { path: "src/App.tsx", content: "new" },
    ]);
  });

  it("rejects malformed, empty, oversized, and duplicate proposals", () => {
    expect(() => parseProposal("not json")).toThrow();
    expect(() => parseProposal("[]")).toThrow();
    expect(() => parseProposal('[{"path":"x.ts","content":""},{"path":"x.ts","content":""}]')).toThrow();
    expect(() =>
      parseProposal(JSON.stringify(Array.from({ length: 26 }, (_, i) => ({ path: `${i}.ts`, content: "" })))),
    ).toThrow();
  });
});

it("includes only selected files and requests structured output", () => {
  const prompt = buildProposalPrompt("Add a button", [{ path: "src/App.tsx", content: "export {}" }]);
  expect(prompt).toContain("JSON array");
  expect(prompt).toContain("--- FILE src/App.tsx ---");
  expect(prompt).not.toContain("unselected.ts");
});
