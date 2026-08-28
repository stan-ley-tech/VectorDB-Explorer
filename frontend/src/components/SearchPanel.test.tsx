import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../api/client";
import type { Collection, SearchResponse } from "../types";
import SearchPanel from "./SearchPanel";

vi.mock("../api/client");

const collection: Collection = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "docs",
  dim: 128,
  metric: "cosine",
  created_at: "2026-01-01T00:00:00Z",
};

const response: SearchResponse = {
  results: [
    { id: "a", external_id: "auth", document: "Authentication middleware", score: 0.94, metadata: {} },
    { id: "b", external_id: "jwt", document: "JWT implementation", score: 0.91, metadata: {} },
  ],
  latency_ms: 12.4,
  used_index: false,
};

beforeEach(() => {
  vi.mocked(api.searchCollection).mockResolvedValue(response);
});

describe("SearchPanel", () => {
  it("runs a search and renders ranked results with scores", async () => {
    const user = userEvent.setup();
    render(<SearchPanel collection={collection} />);

    await user.type(screen.getByLabelText(/query/i), "how do I authenticate an API request");
    await user.click(screen.getByRole("button", { name: /^search$/i }));

    expect(await screen.findByText(/Authentication middleware/)).toBeInTheDocument();
    expect(screen.getByText(/JWT implementation/)).toBeInTheDocument();
    expect(screen.getByText("0.940")).toBeInTheDocument();
    expect(api.searchCollection).toHaveBeenCalledWith(
      collection.id,
      expect.objectContaining({ query: "how do I authenticate an API request", top_k: 10, use_index: false }),
    );
  });

  it("shows whether the ANN index was used", async () => {
    vi.mocked(api.searchCollection).mockResolvedValue({ ...response, used_index: true });
    const user = userEvent.setup();
    render(<SearchPanel collection={collection} />);

    await user.type(screen.getByLabelText(/query/i), "test query");
    await user.click(screen.getByRole("button", { name: /^search$/i }));

    expect(await screen.findByText("indexed")).toBeInTheDocument();
  });
});
