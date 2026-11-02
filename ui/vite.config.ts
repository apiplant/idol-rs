import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";
import tailwindcss from "@tailwindcss/vite";

// `pnpm dev` proxies the API to a running `idol serve` (IDOL_URL, default 127.0.0.1:8190).
const api = process.env.IDOL_URL ?? "http://127.0.0.1:8190";

export default defineConfig({
  plugins: [solid(), tailwindcss()],
  server: { port: 5290, proxy: { "/v1": api, "/health": api, "/openapi.json": api } },
  worker: { format: "es" },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
