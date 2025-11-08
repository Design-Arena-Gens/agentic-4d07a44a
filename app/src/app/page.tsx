"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";

type FaceBox = { x: number; y: number; width: number; height: number };
type Metrics = {
  blurVariance: number;
  gradientEnergy: number;
  symmetryScore: number; // 0..1, higher is more symmetric
  spoofRisk: number; // 0..1, higher is riskier
};

const MODEL_CDN = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const [faces, setFaces] = useState<FaceBox[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [processing, setProcessing] = useState(false);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  // Lazy-load face-api in the browser only
  const loadModels = useCallback(async () => {
    if (modelsReady || loadingModels) return;
    setLoadingModels(true);
    const faceapi = await import("@vladmandic/face-api");
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_CDN),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_CDN),
    ]);
    setModelsReady(true);
    setLoadingModels(false);
  }, [modelsReady, loadingModels]);

  useEffect(() => {
    loadModels().catch(() => {});
  }, [loadModels]);

  const onSelectFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageSrc(reader.result as string);
    reader.readAsDataURL(file);
  }, []);

  const drawImageToCanvas = useCallback(async () => {
    if (!imageSrc || !canvasRef.current) return;
    const img = new Image();
    img.src = imageSrc;
    await img.decode().catch(() => {});
    const maxDim = 1024;
    const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    const canvas = canvasRef.current;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, w, h);
  }, [imageSrc]);

  useEffect(() => {
    drawImageToCanvas();
  }, [imageSrc, drawImageToCanvas]);

  const computeMetrics = useCallback((data: ImageData): Metrics => {
    // Grayscale
    const gray = new Float32Array(data.width * data.height);
    for (let i = 0, j = 0; i < data.data.length; i += 4, j += 1) {
      const r = data.data[i];
      const g = data.data[i + 1];
      const b = data.data[i + 2];
      gray[j] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // Variance of Laplacian (blur measure)
    const w = data.width, h = data.height;
    const lap = new Float32Array(w * h);
    let mean = 0;
    let count = 0;
    const idx = (x: number, y: number) => y * w + x;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const v = -gray[idx(x - 1, y)] - gray[idx(x + 1, y)] - gray[idx(x, y - 1)] - gray[idx(x, y + 1)] + 4 * gray[idx(x, y)];
        lap[idx(x, y)] = v;
        mean += v;
        count++;
      }
    }
    mean /= Math.max(1, count);
    let variance = 0;
    for (let i = 0; i < lap.length; i++) {
      const dv = lap[i] - mean;
      variance += dv * dv;
    }
    variance /= Math.max(1, count);

    // Gradient energy (Sobel magnitude)
    let gradEnergy = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const gx = -gray[idx(x - 1, y - 1)] - 2 * gray[idx(x - 1, y)] - gray[idx(x - 1, y + 1)] + gray[idx(x + 1, y - 1)] + 2 * gray[idx(x + 1, y)] + gray[idx(x + 1, y + 1)];
        const gy = -gray[idx(x - 1, y - 1)] - 2 * gray[idx(x, y - 1)] - gray[idx(x + 1, y - 1)] + gray[idx(x - 1, y + 1)] + 2 * gray[idx(x, y + 1)] + gray[idx(x + 1, y + 1)];
        gradEnergy += Math.hypot(gx, gy);
      }
    }
    gradEnergy /= Math.max(1, (w - 2) * (h - 2));

    // Symmetry: compare left-right halves
    const mid = Math.floor(w / 2);
    let sym = 0, pix = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < mid; x++) {
        const a = gray[idx(x, y)];
        const b = gray[idx(w - 1 - x, y)];
        sym += 1 - Math.min(1, Math.abs(a - b) / 255);
        pix++;
      }
    }
    const symmetryScore = sym / Math.max(1, pix);

    // Spoof risk heuristic
    // Low blur variance, very high gradient energy patterns, and high symmetry can indicate recaptured images
    const blurNorm = Math.min(1, variance / 2000);
    const gradNorm = Math.min(1, gradEnergy / 50);
    const symNorm = symmetryScore; // already 0..1
    const spoofRisk = Math.min(1, 0.6 * (1 - blurNorm) + 0.3 * gradNorm + 0.1 * symNorm);

    return { blurVariance: variance, gradientEnergy: gradEnergy, symmetryScore, spoofRisk };
  }, []);

  const boxBlur = (src: Uint8ClampedArray, w: number, h: number, r: number) => {
    // simple separable box blur for RGBA image
    const tmp = new Uint8ClampedArray(src.length);
    const dst = new Uint8ClampedArray(src.length);
    const chan = 4;
    const iarr = 1 / (r + r + 1);
    // horizontal
    for (let y = 0; y < h; y++) {
      for (let c = 0; c < chan; c++) {
        let ti = (y * w) * chan + c, li = ti, ri = ti + r * chan;
        let fv = src[ti], lv = src[(y * w + (w - 1)) * chan + c];
        let val = (r + 1) * fv;
        for (let j = 0; j < r; j++) val += src[ti + j * chan];
        for (let x = 0; x <= r; x++) { tmp[ti] = Math.round(val * iarr); ti += chan; val += src[ri] - fv; ri += chan; }
        for (let x = r + 1; x < w - r; x++) { tmp[ti] = Math.round(val * iarr); ti += chan; val += src[ri] - src[li]; ri += chan; li += chan; }
        for (let x = w - r; x < w; x++) { tmp[ti] = Math.round(val * iarr); ti += chan; val += lv - src[li]; li += chan; }
      }
    }
    // vertical
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < chan; c++) {
        let ti = (x * chan) + c, li = ti, ri = ti + r * w * chan;
        let fv = tmp[ti], lv = tmp[((h - 1) * w + x) * chan + c];
        let val = (r + 1) * fv;
        for (let j = 0; j < r; j++) val += tmp[ti + j * w * chan];
        for (let y = 0; y <= r; y++) { dst[ti] = Math.round(val * iarr); ti += w * chan; val += tmp[ri] - fv; ri += w * chan; }
        for (let y = r + 1; y < h - r; y++) { dst[ti] = Math.round(val * iarr); ti += w * chan; val += tmp[ri] - tmp[li]; ri += w * chan; li += w * chan; }
        for (let y = h - r; y < h; y++) { dst[ti] = Math.round(val * iarr); ti += w * chan; val += lv - tmp[li]; li += w * chan; }
      }
    }
    return dst;
  };

  const median3x3 = (src: Uint8ClampedArray, w: number, h: number) => {
    const dst = new Uint8ClampedArray(src.length);
    const chan = 4;
    const idx = (x: number, y: number, c: number) => (y * w + x) * chan + c;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 3; c++) {
          const vals: number[] = [];
          for (let j = -1; j <= 1; j++) {
            for (let i = -1; i <= 1; i++) {
              const xx = Math.max(0, Math.min(w - 1, x + i));
              const yy = Math.max(0, Math.min(h - 1, y + j));
              vals.push(src[idx(xx, yy, c)]);
            }
          }
          vals.sort((a, b) => a - b);
          dst[idx(x, y, c)] = vals[4];
        }
        dst[idx(x, y, 3)] = src[idx(x, y, 3)];
      }
    }
    return dst;
  };

  const unsharpMask = (src: Uint8ClampedArray, w: number, h: number, radius: number, amount: number) => {
    const blurred = boxBlur(src, w, h, radius);
    const dst = new Uint8ClampedArray(src.length);
    for (let i = 0; i < src.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const s = src[i + c];
        const b = blurred[i + c];
        let v = s + amount * (s - b);
        dst[i + c] = Math.max(0, Math.min(255, Math.round(v)));
      }
      dst[i + 3] = src[i + 3];
    }
    return dst;
  };

  const equalizeLuma = (src: Uint8ClampedArray, w: number, h: number) => {
    // Convert to YUV, equalize Y histogram, convert back
    const yArr = new Uint8Array(w * h);
    const uArr = new Float32Array(w * h);
    const vArr = new Float32Array(w * h);
    let k = 0;
    for (let i = 0; i < src.length; i += 4, k++) {
      const r = src[i] / 255, g = src[i + 1] / 255, b = src[i + 2] / 255;
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      const u = -0.14713 * r - 0.28886 * g + 0.436 * b;
      const v = 0.615 * r - 0.51499 * g - 0.10001 * b;
      yArr[k] = Math.round(y * 255);
      uArr[k] = u; vArr[k] = v;
    }
    const hist = new Uint32Array(256);
    for (let i = 0; i < yArr.length; i++) hist[yArr[i]]++;
    const cdf = new Uint32Array(256);
    let cum = 0;
    for (let i = 0; i < 256; i++) { cum += hist[i]; cdf[i] = cum; }
    const cdfMin = cdf.find(v => v > 0) ?? 0;
    const total = w * h;
    const eqY = new Float32Array(w * h);
    for (let i = 0; i < yArr.length; i++) {
      eqY[i] = Math.max(0, Math.min(1, (cdf[yArr[i]] - cdfMin) / Math.max(1, total - cdfMin)));
    }
    const dst = new Uint8ClampedArray(src.length);
    for (let i = 0, p = 0; i < dst.length; i += 4, p++) {
      const y = eqY[p];
      const u = uArr[p], v = vArr[p];
      let r = y + 1.13983 * v;
      let g = y - 0.39465 * u - 0.58060 * v;
      let b = y + 2.03211 * u;
      dst[i] = Math.max(0, Math.min(255, Math.round(r * 255)));
      dst[i + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      dst[i + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
      dst[i + 3] = src[i + 3];
    }
    return dst;
  };

  const processImage = useCallback(async () => {
    if (!canvasRef.current) return;
    setProcessing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // Denoise
    const denoised = median3x3(imgData.data, canvas.width, canvas.height);
    // Equalize luma
    const equalized = equalizeLuma(denoised, canvas.width, canvas.height);
    // Unsharp mask
    const sharpened = unsharpMask(equalized, canvas.width, canvas.height, 1, 0.8);
    const outData = new ImageData(sharpened, canvas.width, canvas.height);
    ctx.putImageData(outData, 0, 0);

    // Metrics
    const m = computeMetrics(outData);
    setMetrics(m);

    // Face detection
    setFaces([]);
    if (modelsReady) {
      const faceapi = await import("@vladmandic/face-api");
      const detections = await faceapi.detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 }));
      const bbs: FaceBox[] = detections.map(d => ({ x: d.box.x, y: d.box.y, width: d.box.width, height: d.box.height }));
      setFaces(bbs);
      if (overlayRef.current) {
        const ol = overlayRef.current;
        ol.width = canvas.width; ol.height = canvas.height;
        const octx = ol.getContext("2d")!;
        octx.clearRect(0, 0, ol.width, ol.height);
        octx.strokeStyle = "#22c55e";
        octx.lineWidth = 2;
        octx.font = "12px ui-sans-serif";
        octx.fillStyle = "#22c55e";
        bbs.forEach((b, i) => {
          octx.strokeRect(b.x, b.y, b.width, b.height);
          octx.fillText(`Face ${i + 1}`, b.x + 4, b.y + 14);
        });
      }
    }
    setProcessing(false);
  }, [computeMetrics, modelsReady]);

  useEffect(() => {
    if (imageSrc) processImage();
  }, [imageSrc, processImage]);

  const onExportJSON = useCallback(() => {
    if (!metrics) return;
    const blob = new Blob([JSON.stringify({
      timestamp: new Date().toISOString(),
      faces,
      metrics,
    }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "analysis.json"; a.click();
    URL.revokeObjectURL(url);
  }, [metrics, faces]);

  const onExportImage = useCallback(() => {
    if (!canvasRef.current) return;
    const url = canvasRef.current.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url; a.download = "enhanced.png"; a.click();
  }, []);

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-black dark:text-zinc-50">
      <div className="mx-auto max-w-5xl px-6 py-10">
        <h1 className="text-2xl font-semibold">AI-Powered Forensic Face Reconstruction</h1>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">Live photo detector and enhancement for blurred or partial faces. Client-side processing.</p>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <div className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
              <label className="block text-sm font-medium">Upload image</label>
              <input ref={inputRef} type="file" accept="image/*" className="mt-2 w-full cursor-pointer rounded border border-zinc-200 bg-white p-2 text-sm dark:border-zinc-700 dark:bg-zinc-900" onChange={onSelectFile} />
              <button onClick={() => inputRef.current?.click()} className="mt-2 rounded bg-zinc-900 px-3 py-2 text-sm text-white dark:bg-zinc-100 dark:text-black">Choose File</button>
              <div className="mt-4 text-xs text-zinc-500">Supported: PNG/JPG. Images are processed locally in your browser.</div>
              <div className="mt-4 flex items-center gap-2 text-xs">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: modelsReady ? "#22c55e" : "#f59e0b" }} />
                {modelsReady ? "Face models loaded" : loadingModels ? "Loading face models?" : "Models not loaded"}
              </div>
            </div>

            <div className="mt-6 rounded-lg border border-zinc-200 p-4 text-sm dark:border-zinc-800">
              <div className="font-medium">Metrics</div>
              {metrics ? (
                <ul className="mt-2 space-y-1">
                  <li>Blur variance: <span className="font-mono">{metrics.blurVariance.toFixed(1)}</span></li>
                  <li>Gradient energy: <span className="font-mono">{metrics.gradientEnergy.toFixed(1)}</span></li>
                  <li>Symmetry score: <span className="font-mono">{metrics.symmetryScore.toFixed(3)}</span></li>
                  <li>
                    Spoof risk:
                    <span className={`ml-2 rounded px-2 py-0.5 font-mono ${metrics.spoofRisk > 0.66 ? "bg-red-500/20 text-red-600" : metrics.spoofRisk > 0.33 ? "bg-yellow-500/20 text-yellow-700" : "bg-green-500/20 text-green-700"}`}>
                      {(metrics.spoofRisk * 100).toFixed(0)}%
                    </span>
                  </li>
                </ul>
              ) : (
                <div className="mt-2 text-zinc-500">No metrics yet.</div>
              )}
              <div className="mt-4 flex gap-2">
                <button onClick={onExportJSON} disabled={!metrics} className="rounded bg-zinc-900 px-3 py-2 text-xs text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-black">Export JSON</button>
                <button onClick={onExportImage} disabled={!imageSrc} className="rounded bg-zinc-900 px-3 py-2 text-xs text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-black">Export Image</button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="relative rounded-lg border border-zinc-200 bg-white p-2 dark:border-zinc-800 dark:bg-zinc-900">
              {!imageSrc && (
                <div className="flex aspect-video w-full items-center justify-center text-sm text-zinc-500">Upload an image to begin</div>
              )}
              <div className="relative">
                <canvas ref={canvasRef} className="max-h-[70vh] w-full rounded" />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 rounded" />
                {processing && (
                  <div className="absolute right-2 top-2 rounded bg-black/60 px-2 py-1 text-xs text-white">Processing?</div>
                )}
              </div>
              {faces.length > 0 && (
                <div className="mt-2 text-xs text-zinc-600 dark:text-zinc-400">Detected faces: {faces.length}</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
