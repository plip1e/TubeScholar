import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite is the dev server + bundler. The `proxy` block below is what lets the
// frontend call the FastAPI backend without CORS pain in dev: the browser only
// ever talks to :5173, and Vite forwards anything under /api to :8000 as-is
// (the backend's routes live under /api natively: /api/chat, /api/videos, ...).
// In production there is no Vite at all; FastAPI serves the built frontend
// itself, so /api requests land on the same server with no proxy involved.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
