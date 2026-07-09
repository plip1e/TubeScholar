import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite is the dev server + bundler. The `proxy` block below is what lets the
// frontend call the FastAPI backend without CORS pain in dev: the browser only
// ever talks to :5173, and Vite forwards anything under /api to :8000,
// stripping the /api prefix (the backend's routes are /chat, /ingest, /videos).
// In production a reverse proxy (nginx/Caddy) plays this same role.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
