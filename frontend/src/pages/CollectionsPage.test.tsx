import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../api/client";
import type { Collection } from "../types";
import CollectionsPage from "./CollectionsPage";

// Mocks only the network calls; keeps the real ApiError class so
// `instanceof` checks in the component still work against thrown errors.
vi.mock("../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof api>()),
  listCollections: vi.fn(),
  createCollection: vi.fn(),
}));

const sampleCollection: Collection = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "docs",
  dim: 256,
  metric: "cosine",
  created_at: "2026-01-01T00:00:00Z",
};

beforeEach(() => {
  vi.mocked(api.listCollections).mockResolvedValue([]);
  vi.mocked(api.createCollection).mockResolvedValue(sampleCollection);
});

function renderPage() {
  return render(
    <MemoryRouter>
      <CollectionsPage />
    </MemoryRouter>,
  );
}

describe("CollectionsPage", () => {
  it("shows an empty state when there are no collections", async () => {
    renderPage();
    expect(await screen.findByText(/no collections yet/i)).toBeInTheDocument();
  });

  it("lists collections returned by the API", async () => {
    vi.mocked(api.listCollections).mockResolvedValue([sampleCollection]);
    renderPage();

    expect(await screen.findByText("docs")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "cosine" })).toBeInTheDocument();
  });

  it("creates a collection and refreshes the list", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByText(/no collections yet/i);

    vi.mocked(api.listCollections).mockResolvedValue([sampleCollection]);

    await user.type(screen.getByLabelText(/name/i), "docs");
    await user.click(screen.getByRole("button", { name: /create collection/i }));

    await waitFor(() => {
      expect(api.createCollection).toHaveBeenCalledWith({ name: "docs", dim: 256, metric: "cosine" });
    });
    expect(await screen.findByText("docs")).toBeInTheDocument();
  });

  it("shows an error message when loading fails", async () => {
    vi.mocked(api.listCollections).mockRejectedValue(new api.ApiError(500, "server exploded"));
    renderPage();
    expect(await screen.findByText("server exploded")).toBeInTheDocument();
  });
});
