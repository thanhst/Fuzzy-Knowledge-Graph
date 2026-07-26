import { useState } from "react";
import { ModelTestingPipelineFigure, ModelTrainingPipelineFigure } from "./components/ModelTrainingPipelineFigure";

const FIGURE_ID = "model-training-pipeline-svg";
type FigureVariant = "training" | "testing";

function getFigureGeometry(svg: SVGSVGElement): { width: number; height: number } {
  const [, , viewBoxWidth, viewBoxHeight] = svg.getAttribute("viewBox")?.split(/\s+/).map(Number) ?? [];
  return {
    width: Number.isFinite(viewBoxWidth) ? viewBoxWidth : 2300,
    height: Number.isFinite(viewBoxHeight) ? viewBoxHeight : 1050,
  };
}

function serializeFigure(): { svgText: string; width: number; height: number } {
  const svg = document.getElementById(FIGURE_ID);
  if (!(svg instanceof SVGSVGElement)) {
    throw new Error("Figure SVG was not found.");
  }

  const { width, height } = getFigureGeometry(svg);
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  return { svgText: new XMLSerializer().serializeToString(clone), width, height };
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function downloadSvg(variant: FigureVariant): void {
  const { svgText } = serializeFigure();
  downloadBlob(new Blob([svgText], { type: "image/svg+xml;charset=utf-8" }), `model-${variant}-pipeline.svg`);
}

async function downloadPng(variant: FigureVariant): Promise<void> {
  const { svgText, width, height } = serializeFigure();
  const svgBlob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);
  const image = new Image();

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = reject;
    image.src = url;
  });

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    URL.revokeObjectURL(url);
    throw new Error("Canvas context was not available.");
  }

  context.fillStyle = "#f7f8f4";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0);
  URL.revokeObjectURL(url);

  const pngBlob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("PNG export failed."));
    }, "image/png", 1);
  });

  downloadBlob(pngBlob, `model-${variant}-pipeline.png`);
}

export default function App() {
  const [variant, setVariant] = useState<FigureVariant>(() => {
    return new URLSearchParams(window.location.search).get("view") === "testing" ? "testing" : "training";
  });
  const isTraining = variant === "training";
  const selectVariant = (nextVariant: FigureVariant): void => {
    setVariant(nextVariant);
    const url = new URL(window.location.href);
    url.searchParams.set("view", nextVariant);
    window.history.replaceState(null, "", url);
  };

  return (
    <main className="min-h-screen px-5 py-6 md:px-8">
      <div className="mx-auto flex max-w-[2400px] flex-col gap-4">
        <header className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
              Publication Figure
            </p>
            <h1 className="mt-1 text-2xl font-semibold text-ink md:text-3xl">
              {isTraining ? "Model Training Pipeline" : "Model Testing Pipeline"}
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="mr-1 inline-flex rounded-md border border-slate-300 bg-white p-1 shadow-sm">
              <button
                className={`rounded px-4 py-2 text-sm font-semibold transition ${
                  isTraining ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-50"
                }`}
                type="button"
                onClick={() => selectVariant("training")}
              >
                Training
              </button>
              <button
                className={`rounded px-4 py-2 text-sm font-semibold transition ${
                  !isTraining ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-50"
                }`}
                type="button"
                onClick={() => selectVariant("testing")}
              >
                Testing
              </button>
            </div>
            <button
              className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50"
              type="button"
              onClick={() => downloadSvg(variant)}
            >
              Export SVG
            </button>
            <button
              className="rounded-md border border-slate-800 bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
              type="button"
              onClick={() => void downloadPng(variant)}
            >
              Export PNG
            </button>
          </div>
        </header>

        <section className="overflow-x-auto rounded-xl border border-slate-200 bg-white/90 p-3 shadow-figure">
          <div className="min-w-[2300px]">
            {isTraining ? <ModelTrainingPipelineFigure svgId={FIGURE_ID} /> : <ModelTestingPipelineFigure svgId={FIGURE_ID} />}
          </div>
        </section>
      </div>
    </main>
  );
}
