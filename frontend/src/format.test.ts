import { describe, expect, it } from "vitest";
import { formatBytes, formatMs, formatNumber } from "./format";

describe("formatNumber", () => {
  it("adds thousands separators", () => {
    expect(formatNumber(125430)).toBe("125,430");
  });
});

describe("formatBytes", () => {
  it("keeps small values in bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("converts to kilobytes", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
  });

  it("converts to gigabytes for large values", () => {
    expect(formatBytes(1.8 * 1024 * 1024 * 1024)).toBe("1.8 GB");
  });
});

describe("formatMs", () => {
  it("renders sub-millisecond durations in microseconds", () => {
    expect(formatMs(0.5)).toBe("500µs");
  });

  it("renders millisecond durations with one decimal place", () => {
    expect(formatMs(12.42)).toBe("12.4ms");
  });
});
