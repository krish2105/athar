import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

// The built page is committed to docs/ so GitHub Pages can serve the repository
// without a build step. base is relative so the page works from a project path
// (krish2105.github.io/athar/) as well as from a file:// preview.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: resolve(__dirname, "../docs"),
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: { port: 5175 },
});
