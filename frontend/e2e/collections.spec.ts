import { expect, test } from "@playwright/test";

// Exercises the full golden path against a real backend + Postgres, not
// mocks: create a collection, import a dataset, search it, then run a
// benchmark. Requires the API (see .github/workflows/ci.yml, job
// "e2e-tests") to be reachable at VITE_API_BASE_URL.

test("create a collection, import data, search it, and run a benchmark", async ({ page }) => {
  const collectionName = `e2e-${Date.now()}`;

  await page.goto("/collections");
  await page.getByLabel("Name").fill(collectionName);
  await page.getByLabel("Dimensions").fill("64");
  await page.getByRole("button", { name: "Create collection" }).click();

  const collectionLink = page.getByRole("link", { name: collectionName });
  await expect(collectionLink).toBeVisible();
  await collectionLink.click();

  await expect(page.getByText("Documents")).toBeVisible();

  const csv =
    "text,category\n" +
    "Authentication middleware validates API requests,security\n" +
    "Deploying containers to production,ops\n";
  await page.setInputFiles("#import-file", {
    name: "dataset.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(csv),
  });
  await expect(page.getByText(/Imported 2 record/)).toBeVisible();

  await page.locator(".tabs").getByRole("button", { name: "Search" }).click();
  await page.getByLabel("Query").fill("how do I authenticate an API request");
  await page.locator("form").getByRole("button", { name: "Search" }).click();
  await expect(page.getByText(/Authentication middleware/)).toBeVisible();
  await expect(page.getByText("exact (pgvector)")).toBeVisible();

  await page.locator(".tabs").getByRole("button", { name: "Benchmark" }).click();
  await page.getByLabel("Dataset size").fill("200");
  await page.getByLabel("Dimensions").fill("32");
  await page.getByLabel("Queries").fill("20");
  await page.getByRole("button", { name: "Run benchmark" }).click();

  await expect(page.getByText(/Results —/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("cell", { name: "Brute force" })).toBeVisible();
});

test("shows a validation-friendly empty state before any collection exists", async ({ page }) => {
  await page.goto("/collections");
  await expect(page.getByRole("button", { name: "Create collection" })).toBeDisabled();
});
