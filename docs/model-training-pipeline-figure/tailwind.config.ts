import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Segoe UI", "Arial", "sans-serif"],
      },
      colors: {
        paper: "#f7f8f4",
        ink: "#173042",
      },
      boxShadow: {
        figure: "0 20px 60px rgba(42, 58, 75, 0.12)",
      },
    },
  },
  plugins: [],
} satisfies Config;
