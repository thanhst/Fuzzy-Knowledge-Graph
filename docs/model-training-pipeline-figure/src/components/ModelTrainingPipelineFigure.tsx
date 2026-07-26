import type { ReactNode } from "react";

type Tone = "image" | "table" | "training" | "xai" | "neutral";
type EdgeKind = "data" | "model" | "xai" | "rule";
type Side = "top" | "right" | "bottom" | "left";

export type DiagramNode = {
  id: string;
  title: string;
  x: number;
  y: number;
  width: number;
  height: number;
  tone: Tone;
  bullets?: string[];
  cards?: RuleCard[];
  icon?: IconKind;
};

export type DiagramEdge = {
  id: string;
  from: string;
  to: string;
  fromSide: Side;
  toSide: Side;
  kind: EdgeKind;
  label?: string;
  labelX?: number;
  labelY?: number;
  waypoints?: Point[];
};

export type Section = {
  id: string;
  title: string;
  x: number;
  y: number;
  width: number;
  height: number;
  tone: Tone;
  subtitle?: string;
};

type Point = {
  x: number;
  y: number;
};

type RuleCard = {
  id: string;
  label: string;
  condition: string;
};

type IconKind = "image" | "preprocess" | "detection" | "cnn" | "features" | "table" | "rules" | "fusion" | "xai";

type FigureConfig = {
  title: string;
  subtitle: string;
  sections: Section[];
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  description: string;
};

const CANVAS = {
  width: 2300,
  height: 1050,
};

const PALETTE: Record<Tone, { fill: string; stroke: string; accent: string; text: string }> = {
  image: { fill: "#eaf3fb", stroke: "#9abbd7", accent: "#2f6f9f", text: "#16364c" },
  table: { fill: "#fff4df", stroke: "#dfbd79", accent: "#a36d16", text: "#4a3211" },
  training: { fill: "#eaf6f2", stroke: "#8cc8b8", accent: "#237b6c", text: "#163c36" },
  xai: { fill: "#f1ecfb", stroke: "#b8a8df", accent: "#6b55a8", text: "#2f2554" },
  neutral: { fill: "#ffffff", stroke: "#cbd5df", accent: "#54687a", text: "#173042" },
};

const EDGE_STYLE: Record<EdgeKind, { stroke: string; width: number; dash?: string; marker: string }> = {
  data: { stroke: "#4e6f83", width: 2.05, marker: "arrow-data" },
  model: { stroke: "#28786c", width: 2.35, marker: "arrow-model" },
  xai: { stroke: "#7a61b6", width: 1.7, dash: "6 6", marker: "arrow-xai" },
  rule: { stroke: "#9b6a12", width: 1.65, dash: "3 5", marker: "arrow-rule" },
};

const TRAINING_SECTIONS: Section[] = [
  {
    id: "image",
    title: "1. Image Branch: Detection, Deep Features, and Hand-Crafted Features",
    x: 50,
    y: 110,
    width: 1280,
    height: 360,
    tone: "image",
  },
  {
    id: "table",
    title: "2. Table Branch: Clinical and Patient-Level Features",
    x: 50,
    y: 500,
    width: 1280,
    height: 260,
    tone: "table",
  },
  {
    id: "training",
    title: "3. Rule Learning, Late Fusion, and Training Result",
    x: 1400,
    y: 110,
    width: 860,
    height: 570,
    tone: "training",
  },
  {
    id: "xai",
    title: "4. Explainability Outputs",
    x: 1400,
    y: 725,
    width: 860,
    height: 275,
    tone: "xai",
    subtitle: "Derived artefacts only; no feedback into model training.",
  },
];

const TRAINING_NODES: DiagramNode[] = [
  {
    id: "rawImage",
    title: "Raw Dental Image",
    x: 90,
    y: 190,
    width: 190,
    height: 112,
    tone: "image",
    icon: "image",
    bullets: ["Intraoral / X-ray image", "Training sample"],
  },
  {
    id: "preprocessing",
    title: "Preprocessing",
    x: 340,
    y: 190,
    width: 210,
    height: 112,
    tone: "image",
    icon: "preprocess",
    bullets: ["Resize", "Normalise", "Denoise / crop ROI"],
  },
  {
    id: "yoloDetection",
    title: "YOLO Detection Model",
    x: 630,
    y: 176,
    width: 235,
    height: 136,
    tone: "image",
    icon: "detection",
    bullets: ["Tooth / lesion detection", "Bounding boxes", "Confidence scores"],
  },
  {
    id: "cnnClassifier",
    title: "CNN Classifier",
    x: 1040,
    y: 168,
    width: 245,
    height: 132,
    tone: "image",
    icon: "cnn",
    bullets: ["Deep feature extraction", "Class probability", "Logits / confidence"],
  },
  {
    id: "imageFeatures",
    title: "Image Feature Extraction",
    x: 630,
    y: 350,
    width: 245,
    height: 112,
    tone: "image",
    icon: "features",
    bullets: ["HSV descriptors", "GLCM texture features", "Lesion area / shape"],
  },
  {
    id: "imageAggregation",
    title: "Image Aggregation",
    x: 1040,
    y: 350,
    width: 245,
    height: 112,
    tone: "image",
    icon: "features",
    bullets: ["Patient-level pooling", "Top-k lesion features", "Summary vector"],
  },
  {
    id: "rawTable",
    title: "Raw Clinical Table",
    x: 90,
    y: 595,
    width: 220,
    height: 112,
    tone: "table",
    icon: "table",
    bullets: ["Patient records", "Tooth-level metadata"],
  },
  {
    id: "tablePreprocessing",
    title: "Table Preprocessing",
    x: 380,
    y: 595,
    width: 245,
    height: 112,
    tone: "table",
    icon: "preprocess",
    bullets: ["Missing value handling", "Categorical encoding", "Normalisation"],
  },
  {
    id: "featureSelection",
    title: "Feature Selection",
    x: 780,
    y: 575,
    width: 325,
    height: 144,
    tone: "table",
    icon: "features",
    bullets: ["Selected clinical features", "Selected image-derived features", "Patient-level fusion vector"],
  },
  {
    id: "fkgModel",
    title: "FKG Rule Model",
    x: 1455,
    y: 410,
    width: 245,
    height: 140,
    tone: "training",
    icon: "rules",
    bullets: ["Fuzzy knowledge graph", "Candidate rule paths", "Rule confidence scores"],
  },
  {
    id: "cnnOutput",
    title: "CNN Output",
    x: 2060,
    y: 175,
    width: 180,
    height: 110,
    tone: "training",
    icon: "cnn",
    bullets: ["Class probabilities", "Image confidence"],
  },
  {
    id: "candidateRules",
    title: "FKG Candidate Rules",
    x: 1730,
    y: 245,
    width: 310,
    height: 315,
    tone: "training",
    icon: "rules",
    cards: [
      { id: "r1", label: "Rule R1", condition: "image_prob high AND lesion_area high" },
      { id: "r2", label: "Rule R2", condition: "clinical_risk high AND texture abnormal" },
      { id: "r3", label: "Rule R3", condition: "detection_conf high AND table_score high" },
    ],
  },
  {
    id: "lateFusion",
    title: "Late Fusion",
    x: 2060,
    y: 370,
    width: 180,
    height: 155,
    tone: "training",
    icon: "fusion",
    bullets: ["Weighted fusion", "CNN evidence", "FKG / rule evidence", "Final class decision"],
  },
  {
    id: "trainingResult",
    title: "Training Result",
    x: 2060,
    y: 565,
    width: 180,
    height: 110,
    tone: "training",
    icon: "fusion",
    bullets: ["Final label", "Confidence score", "Evidence trace"],
  },
  {
    id: "gradCam",
    title: "Grad-CAM Heatmap",
    x: 1455,
    y: 805,
    width: 250,
    height: 90,
    tone: "xai",
    icon: "xai",
    bullets: ["Visual evidence", "Lesion localisation"],
  },
  {
    id: "fullImageView",
    title: "Full Image View",
    x: 1455,
    y: 910,
    width: 250,
    height: 78,
    tone: "xai",
    icon: "image",
    bullets: ["Global dental context", "Original image reference"],
  },
  {
    id: "ruleRationale",
    title: "Rule Rationale",
    x: 1760,
    y: 805,
    width: 250,
    height: 90,
    tone: "xai",
    icon: "rules",
    bullets: ["Activated rule path", "Clinical explanation"],
  },
  {
    id: "explanationReport",
    title: "Explanation Report",
    x: 2055,
    y: 805,
    width: 185,
    height: 180,
    tone: "xai",
    icon: "xai",
    bullets: ["Heatmap", "ROI image", "Feature contribution", "Rule-based rationale"],
  },
];

const TRAINING_EDGES: DiagramEdge[] = [
  { id: "raw-to-preprocess", from: "rawImage", to: "preprocessing", fromSide: "right", toSide: "left", kind: "data" },
  { id: "preprocess-to-yolo", from: "preprocessing", to: "yoloDetection", fromSide: "right", toSide: "left", kind: "data" },
  { id: "yolo-to-cnn", from: "yoloDetection", to: "cnnClassifier", fromSide: "right", toSide: "left", kind: "data", label: "Detected Regions", labelX: 890, labelY: 184 },
  {
    id: "yolo-to-image-features",
    from: "yoloDetection",
    to: "imageFeatures",
    fromSide: "bottom",
    toSide: "top",
    kind: "data",
    label: "Boxes and Crops",
    labelX: 655,
    labelY: 336,
  },
  { id: "features-to-aggregation", from: "imageFeatures", to: "imageAggregation", fromSide: "right", toSide: "left", kind: "data" },
  { id: "raw-table-to-preprocess", from: "rawTable", to: "tablePreprocessing", fromSide: "right", toSide: "left", kind: "data" },
  { id: "table-preprocess-to-selection", from: "tablePreprocessing", to: "featureSelection", fromSide: "right", toSide: "left", kind: "data" },
  {
    id: "aggregation-to-selection",
    from: "imageAggregation",
    to: "featureSelection",
    fromSide: "bottom",
    toSide: "top",
    kind: "data",
    label: "Image-Derived Features",
    labelX: 1048,
    labelY: 535,
    waypoints: [
      { x: 1162, y: 535 },
      { x: 942, y: 535 },
    ],
  },
  {
    id: "selection-to-fkg",
    from: "featureSelection",
    to: "fkgModel",
    fromSide: "right",
    toSide: "left",
    kind: "model",
    label: "Patient-Level Fusion Vector",
    labelX: 1190,
    labelY: 622,
    waypoints: [
      { x: 1260, y: 647 },
      { x: 1260, y: 480 },
    ],
  },
  { id: "fkg-to-rules", from: "fkgModel", to: "candidateRules", fromSide: "right", toSide: "left", kind: "model", label: "Rule Evidence", labelX: 1705, labelY: 425 },
  { id: "rules-to-fusion", from: "candidateRules", to: "lateFusion", fromSide: "right", toSide: "left", kind: "model" },
  {
    id: "cnn-to-output",
    from: "cnnClassifier",
    to: "cnnOutput",
    fromSide: "right",
    toSide: "left",
    kind: "model",
    label: "Class Evidence",
    labelX: 1422,
    labelY: 224,
    waypoints: [{ x: 1368, y: 234 }],
  },
  { id: "output-to-fusion", from: "cnnOutput", to: "lateFusion", fromSide: "bottom", toSide: "top", kind: "model" },
  { id: "fusion-to-result", from: "lateFusion", to: "trainingResult", fromSide: "bottom", toSide: "top", kind: "model" },
  {
    id: "result-to-gradcam",
    from: "trainingResult",
    to: "gradCam",
    fromSide: "bottom",
    toSide: "top",
    kind: "xai",
    label: "Explanation Artefacts",
    labelX: 1810,
    labelY: 780,
    waypoints: [
      { x: 2150, y: 790 },
      { x: 1580, y: 790 },
    ],
  },
  {
    id: "result-to-full-image",
    from: "trainingResult",
    to: "fullImageView",
    fromSide: "bottom",
    toSide: "left",
    kind: "xai",
    waypoints: [
      { x: 2150, y: 790 },
      { x: 1420, y: 790 },
      { x: 1420, y: 949 },
    ],
  },
  {
    id: "result-to-rule-rationale",
    from: "trainingResult",
    to: "ruleRationale",
    fromSide: "bottom",
    toSide: "top",
    kind: "xai",
    waypoints: [
      { x: 2150, y: 790 },
      { x: 1885, y: 790 },
    ],
  },
  {
    id: "result-to-report",
    from: "trainingResult",
    to: "explanationReport",
    fromSide: "bottom",
    toSide: "top",
    kind: "xai",
    waypoints: [
      { x: 2150, y: 790 },
    ],
  },
  {
    id: "rules-to-rationale",
    from: "candidateRules",
    to: "ruleRationale",
    fromSide: "bottom",
    toSide: "top",
    kind: "rule",
    label: "Rule Trace",
    labelX: 1832,
    labelY: 702,
    waypoints: [{ x: 1885, y: 790 }],
  },
];

const TESTING_SECTIONS: Section[] = [
  {
    id: "image",
    title: "1. Image Branch: Detection and Feature Inference",
    x: 50,
    y: 110,
    width: 1280,
    height: 360,
    tone: "image",
  },
  {
    id: "table",
    title: "2. Table Branch: Clinical Feature Inference",
    x: 50,
    y: 500,
    width: 1280,
    height: 260,
    tone: "table",
  },
  {
    id: "training",
    title: "3. Rule Inference, Late Fusion, and Testing Result",
    x: 1400,
    y: 110,
    width: 860,
    height: 570,
    tone: "training",
  },
  {
    id: "xai",
    title: "4. Explainability Outputs",
    x: 1400,
    y: 725,
    width: 860,
    height: 275,
    tone: "xai",
    subtitle: "Derived artefacts only; no feedback into inference.",
  },
];

const TESTING_NODES: DiagramNode[] = [
  {
    id: "rawImage",
    title: "Raw Dental Image",
    x: 90,
    y: 190,
    width: 190,
    height: 112,
    tone: "image",
    icon: "image",
    bullets: ["Unseen dental sample", "Intraoral / X-ray image"],
  },
  {
    id: "preprocessing",
    title: "Preprocessing",
    x: 340,
    y: 190,
    width: 210,
    height: 112,
    tone: "image",
    icon: "preprocess",
    bullets: ["Resize", "Normalise", "Denoise / crop ROI"],
  },
  {
    id: "yoloDetection",
    title: "Trained YOLO Detection Model",
    x: 630,
    y: 176,
    width: 235,
    height: 136,
    tone: "image",
    icon: "detection",
    bullets: ["Tooth / lesion detection", "Bounding boxes", "Confidence scores"],
  },
  {
    id: "cnnClassifier",
    title: "Trained CNN Classifier",
    x: 1040,
    y: 168,
    width: 245,
    height: 132,
    tone: "image",
    icon: "cnn",
    bullets: ["Deep feature inference", "Class probabilities", "Logits / confidence"],
  },
  {
    id: "imageFeatures",
    title: "Image Feature Extraction",
    x: 630,
    y: 350,
    width: 245,
    height: 112,
    tone: "image",
    icon: "features",
    bullets: ["HSV descriptors", "GLCM texture features", "Lesion area / shape"],
  },
  {
    id: "imageAggregation",
    title: "Image Aggregation",
    x: 1040,
    y: 350,
    width: 245,
    height: 112,
    tone: "image",
    icon: "features",
    bullets: ["Patient-level pooling", "Top-k lesion features", "Testing summary vector"],
  },
  {
    id: "rawTable",
    title: "Raw Clinical Table",
    x: 90,
    y: 595,
    width: 220,
    height: 112,
    tone: "table",
    icon: "table",
    bullets: ["Patient record", "Tooth-level metadata"],
  },
  {
    id: "tablePreprocessing",
    title: "Table Preprocessing",
    x: 380,
    y: 595,
    width: 245,
    height: 112,
    tone: "table",
    icon: "preprocess",
    bullets: ["Missing value handling", "Categorical encoding", "Normalisation"],
  },
  {
    id: "featureSelection",
    title: "Apply Feature Selection",
    x: 780,
    y: 575,
    width: 325,
    height: 144,
    tone: "table",
    icon: "features",
    bullets: ["Use trained feature subset", "Clinical feature vector", "Image-derived summary vector"],
  },
  {
    id: "fkgModel",
    title: "Trained FKG Rule Model",
    x: 1455,
    y: 410,
    width: 245,
    height: 140,
    tone: "training",
    icon: "rules",
    bullets: ["Loaded fuzzy rules", "Rule path inference", "Rule confidence scores"],
  },
  {
    id: "cnnOutput",
    title: "CNN Prediction",
    x: 2060,
    y: 175,
    width: 180,
    height: 110,
    tone: "training",
    icon: "cnn",
    bullets: ["Class probabilities", "Image confidence"],
  },
  {
    id: "candidateRules",
    title: "Activated FKG Rules",
    x: 1730,
    y: 245,
    width: 310,
    height: 315,
    tone: "training",
    icon: "rules",
    cards: [
      { id: "r1", label: "Rule R1", condition: "image_prob high AND lesion_area high" },
      { id: "r2", label: "Rule R2", condition: "clinical_risk high AND texture abnormal" },
      { id: "r3", label: "Rule R3", condition: "detection_conf high AND table_score high" },
    ],
  },
  {
    id: "lateFusion",
    title: "Late Fusion Prediction",
    x: 2060,
    y: 370,
    width: 180,
    height: 155,
    tone: "training",
    icon: "fusion",
    bullets: ["Weighted inference", "CNN evidence", "FKG / rule evidence", "Final class decision"],
  },
  {
    id: "trainingResult",
    title: "Testing Result",
    x: 2060,
    y: 565,
    width: 180,
    height: 110,
    tone: "training",
    icon: "fusion",
    bullets: ["Predicted label", "Confidence score", "Evidence trace"],
  },
  {
    id: "gradCam",
    title: "Grad-CAM Heatmap",
    x: 1455,
    y: 805,
    width: 250,
    height: 90,
    tone: "xai",
    icon: "xai",
    bullets: ["Visual evidence", "Lesion localisation"],
  },
  {
    id: "fullImageView",
    title: "Full Image View",
    x: 1455,
    y: 910,
    width: 250,
    height: 78,
    tone: "xai",
    icon: "image",
    bullets: ["Global dental context", "Original image reference"],
  },
  {
    id: "ruleRationale",
    title: "Rule Rationale",
    x: 1760,
    y: 805,
    width: 250,
    height: 90,
    tone: "xai",
    icon: "rules",
    bullets: ["Activated rule path", "Clinical explanation"],
  },
  {
    id: "explanationReport",
    title: "Explanation Report",
    x: 2055,
    y: 805,
    width: 185,
    height: 180,
    tone: "xai",
    icon: "xai",
    bullets: ["Heatmap", "ROI image", "Feature contribution", "Rule-based rationale"],
  },
];

const TESTING_EDGES: DiagramEdge[] = [
  ...TRAINING_EDGES.map((edge) => ({ ...edge })),
];

TESTING_EDGES.find((edge) => edge.id === "selection-to-fkg")!.label = "Testing Fusion Vector";
TESTING_EDGES.find((edge) => edge.id === "fkg-to-rules")!.label = "Activated Rule Evidence";
TESTING_EDGES.find((edge) => edge.id === "cnn-to-output")!.label = "CNN Prediction";

const TRAINING_CONFIG: FigureConfig = {
  title: "Model Training Pipeline: YOLO Detection, CNN Classification, Feature Learning, and FKG Rules",
  subtitle: "Late Fusion and Explainability Outputs",
  sections: TRAINING_SECTIONS,
  nodes: TRAINING_NODES,
  edges: TRAINING_EDGES,
  description:
    "A publication-quality training pipeline connecting YOLO detection, CNN classification, image and table feature learning, fuzzy knowledge graph rules, late fusion, and explainability outputs.",
};

const TESTING_CONFIG: FigureConfig = {
  title: "Model Testing Pipeline: YOLO Detection, CNN Inference, Feature Vector Construction, and FKG Rule Inference",
  subtitle: "Late Fusion Prediction and Explainability Outputs",
  sections: TESTING_SECTIONS,
  nodes: TESTING_NODES,
  edges: TESTING_EDGES,
  description:
    "A publication-quality testing pipeline using trained YOLO, CNN, selected feature subsets, fuzzy knowledge graph rule inference, late fusion prediction, and explanation outputs.",
};

const LEGEND_ITEMS: Array<{ label: string; kind: EdgeKind; description: string }> = [
  { label: "Data Flow", kind: "data", description: "Samples and extracted feature vectors" },
  { label: "Model Output", kind: "model", description: "Predictions, logits, rule scores, and fusion evidence" },
  { label: "Explainability Output", kind: "xai", description: "Post-training artefacts derived from evidence" },
  { label: "Rule Trace", kind: "rule", description: "Activated fuzzy rule path for rationale" },
];

function wrapText(text: string, maxCharacters: number): string[] {
  const words = text.split(" ");
  const lines: string[] = [];
  let line = "";

  words.forEach((word) => {
    const candidate = line ? `${line} ${word}` : word;
    if (candidate.length > maxCharacters && line) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  });

  if (line) lines.push(line);
  return lines;
}

function anchor(node: DiagramNode, side: Side): Point {
  switch (side) {
    case "top":
      return { x: node.x + node.width / 2, y: node.y };
    case "right":
      return { x: node.x + node.width, y: node.y + node.height / 2 };
    case "bottom":
      return { x: node.x + node.width / 2, y: node.y + node.height };
    case "left":
      return { x: node.x, y: node.y + node.height / 2 };
  }
}

function buildPath(points: Point[]): string {
  const [start, ...rest] = points;
  return `M ${start.x} ${start.y} ${rest.map((point) => `L ${point.x} ${point.y}`).join(" ")}`;
}

function SectionBox({ section }: { section: Section }) {
  const tone = PALETTE[section.tone];
  return (
    <g>
      <rect
        x={section.x}
        y={section.y}
        width={section.width}
        height={section.height}
        rx={18}
        fill={tone.fill}
        stroke={tone.stroke}
        strokeWidth={1.6}
      />
      <rect x={section.x} y={section.y} width={section.width} height={38} rx={18} fill={tone.accent} opacity={0.08} />
      <text x={section.x + 22} y={section.y + 26} fill={tone.text} fontSize={16} fontWeight={760}>
        {section.title}
      </text>
      {section.subtitle ? (
        <text x={section.x + 22} y={section.y + 52} fill={tone.text} fontSize={11.5} fontWeight={560} opacity={0.72}>
          {section.subtitle}
        </text>
      ) : null}
    </g>
  );
}

function ExplanationPanel({ section }: { section: Section }) {
  return (
    <g>
      <SectionBox section={section} />
      <path
        d={`M ${section.x + 22} ${section.y + 66} H ${section.x + section.width - 22}`}
        stroke={PALETTE.xai.stroke}
        strokeWidth={1}
        strokeDasharray="5 6"
        opacity={0.65}
      />
    </g>
  );
}

function Icon({ kind, x, y, tone }: { kind: IconKind; x: number; y: number; tone: Tone }) {
  const color = PALETTE[tone].accent;
  const common = { fill: "none", stroke: color, strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

  const content: Record<IconKind, ReactNode> = {
    image: (
      <>
        <rect x={x} y={y + 2} width={20} height={16} rx={3} {...common} />
        <path d={`M ${x + 4} ${y + 14} L ${x + 9} ${y + 9} L ${x + 13} ${y + 12} L ${x + 17} ${y + 7}`} {...common} />
      </>
    ),
    preprocess: (
      <>
        <path d={`M ${x + 3} ${y + 5} H ${x + 19}`} {...common} />
        <path d={`M ${x + 3} ${y + 12} H ${x + 19}`} {...common} />
        <circle cx={x + 8} cy={y + 5} r={2.2} fill="#ffffff" stroke={color} strokeWidth={1.6} />
        <circle cx={x + 14} cy={y + 12} r={2.2} fill="#ffffff" stroke={color} strokeWidth={1.6} />
      </>
    ),
    detection: (
      <>
        <rect x={x + 2} y={y + 3} width={18} height={14} rx={2} {...common} />
        <rect x={x + 6} y={y + 7} width={8} height={5} rx={1.5} stroke={color} fill="rgba(255,255,255,0.65)" strokeWidth={1.4} />
      </>
    ),
    cnn: (
      <>
        {[0, 1, 2].map((column) => (
          <path key={column} d={`M ${x + 4 + column * 6} ${y + 4} V ${y + 17}`} {...common} />
        ))}
        <path d={`M ${x + 4} ${y + 8} H ${x + 16}`} {...common} />
        <path d={`M ${x + 4} ${y + 13} H ${x + 16}`} {...common} />
      </>
    ),
    features: (
      <>
        <circle cx={x + 5} cy={y + 6} r={3} {...common} />
        <circle cx={x + 16} cy={y + 8} r={3} {...common} />
        <circle cx={x + 10} cy={y + 17} r={3} {...common} />
        <path d={`M ${x + 8} ${y + 7} L ${x + 13} ${y + 8} M ${x + 8} ${y + 14} L ${x + 14} ${y + 10}`} {...common} />
      </>
    ),
    table: (
      <>
        <rect x={x + 2} y={y + 3} width={18} height={16} rx={2} {...common} />
        <path d={`M ${x + 2} ${y + 9} H ${x + 20} M ${x + 8} ${y + 3} V ${y + 19} M ${x + 14} ${y + 3} V ${y + 19}`} {...common} />
      </>
    ),
    rules: (
      <>
        <path d={`M ${x + 3} ${y + 5} H ${x + 19} M ${x + 3} ${y + 11} H ${x + 15} M ${x + 3} ${y + 17} H ${x + 18}`} {...common} />
        <circle cx={x + 18} cy={y + 11} r={2.2} fill="#ffffff" stroke={color} strokeWidth={1.5} />
      </>
    ),
    fusion: (
      <>
        <path d={`M ${x + 3} ${y + 5} C ${x + 9} ${y + 5}, ${x + 10} ${y + 11}, ${x + 16} ${y + 11} H ${x + 20}`} {...common} />
        <path d={`M ${x + 3} ${y + 17} C ${x + 9} ${y + 17}, ${x + 10} ${y + 11}, ${x + 16} ${y + 11}`} {...common} />
      </>
    ),
    xai: (
      <>
        <path d={`M ${x + 3} ${y + 11} C ${x + 7} ${y + 4}, ${x + 15} ${y + 4}, ${x + 19} ${y + 11} C ${x + 15} ${y + 18}, ${x + 7} ${y + 18}, ${x + 3} ${y + 11} Z`} {...common} />
        <circle cx={x + 11} cy={y + 11} r={2.6} fill={color} />
      </>
    ),
  };

  return <g aria-hidden="true">{content[kind]}</g>;
}

function PipelineNode({ node }: { node: DiagramNode }) {
  const tone = PALETTE[node.tone];
  const titleX = node.x + 58;
  const bodyTextX = node.x + 44;
  const bulletDotX = node.x + 30;
  const titleMax = Math.max(12, Math.floor((node.width - 58) / 7.4));
  const bulletMax = Math.max(14, Math.floor((node.width - 58) / 5.9));
  const titleLines = wrapText(node.title, titleMax);
  const bulletStartY = node.y + 40 + titleLines.length * 16;
  const bulletRows =
    node.bullets?.flatMap((bullet, bulletIndex) => {
      const previousLineCount =
        node.bullets
          ?.slice(0, bulletIndex)
          .reduce((total, previousBullet) => total + wrapText(previousBullet, bulletMax).length, 0) ?? 0;
      const lines = wrapText(bullet, bulletMax);
      const groupOffset = bulletIndex * 5;

      return lines.map((line, lineIndex) => ({
        key: `${bullet}-${lineIndex}`,
        text: line,
        showDot: lineIndex === 0,
        y: bulletStartY + (previousLineCount + lineIndex) * 13 + groupOffset,
      }));
    }) ?? [];

  return (
    <g>
      <rect
        x={node.x}
        y={node.y}
        width={node.width}
        height={node.height}
        rx={14}
        fill="#ffffff"
        stroke={tone.stroke}
        strokeWidth={1.45}
        filter="url(#node-shadow)"
      />
      <rect x={node.x} y={node.y} width={node.width} height={8} rx={4} fill={tone.accent} opacity={0.86} />
      <circle cx={node.x + 24} cy={node.y + 30} r={15} fill={tone.fill} stroke={tone.stroke} strokeWidth={1} />
      {node.icon ? <Icon kind={node.icon} x={node.x + 13} y={node.y + 19} tone={node.tone} /> : null}

      <text fill={tone.text} fontWeight={760} fontSize={13.5}>
        {titleLines.map((line, index) => (
          <tspan key={line} x={titleX} y={node.y + 28 + index * 15}>
            {line}
          </tspan>
        ))}
      </text>

      {node.bullets ? (
        <g>
          {bulletRows.map((row) => (
            <g key={row.key}>
              {row.showDot ? <circle cx={bulletDotX} cy={row.y - 4} r={2.3} fill={tone.accent} opacity={0.85} /> : null}
              <text x={bodyTextX} y={row.y} fill="#3d5060" fontSize={11.2} fontWeight={500}>
                {row.text}
              </text>
            </g>
          ))}
        </g>
      ) : null}

      {node.cards ? (
        <g>
          {node.cards.map((card, index) => {
            const cardY = node.y + 57 + index * 50;
            return (
              <g key={card.id}>
                <rect x={node.x + 14} y={cardY} width={node.width - 28} height={42} rx={9} fill={tone.fill} stroke={tone.stroke} strokeWidth={0.8} />
                <text x={node.x + 24} y={cardY + 15} fill={tone.text} fontSize={11.5} fontWeight={760}>
                  {card.label}
                </text>
                {wrapText(card.condition, Math.floor((node.width - 48) / 5.8)).slice(0, 2).map((line, lineIndex) => (
                  <text key={line} x={node.x + 24} y={cardY + 29 + lineIndex * 11} fill="#405464" fontSize={9.4} fontWeight={520}>
                    {line}
                  </text>
                ))}
              </g>
            );
          })}
        </g>
      ) : null}
    </g>
  );
}

function BranchLabel({ x, y, children, kind }: { x: number; y: number; children: ReactNode; kind: EdgeKind }) {
  const edge = EDGE_STYLE[kind];
  const label = typeof children === "string" ? children : "";
  return (
    <g>
      <rect x={x - 8} y={y - 17} width={label.length * 6.5 + 16} height={22} rx={11} fill="#ffffff" stroke={edge.stroke} strokeWidth={0.9} opacity={0.96} />
      <text x={x} y={y - 2} fill={edge.stroke} fontSize={10.5} fontWeight={740}>
        {children}
      </text>
    </g>
  );
}

function Arrow({ edge, nodesById }: { edge: DiagramEdge; nodesById: Map<string, DiagramNode> }) {
  const fromNode = nodesById.get(edge.from);
  const toNode = nodesById.get(edge.to);
  if (!fromNode || !toNode) return null;

  const start = anchor(fromNode, edge.fromSide);
  const end = anchor(toNode, edge.toSide);
  const style = EDGE_STYLE[edge.kind];
  const points = [start, ...(edge.waypoints ?? []), end];

  return (
    <g>
      <path
        d={buildPath(points)}
        fill="none"
        stroke={style.stroke}
        strokeWidth={style.width}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray={style.dash}
        markerEnd={`url(#${style.marker})`}
      />
      {edge.label && edge.labelX && edge.labelY ? (
        <BranchLabel x={edge.labelX} y={edge.labelY} kind={edge.kind}>
          {edge.label}
        </BranchLabel>
      ) : null}
    </g>
  );
}

function LegendItem({ x, y, label, description, kind }: { x: number; y: number; label: string; description: string; kind: EdgeKind }) {
  const style = EDGE_STYLE[kind];
  return (
    <g>
      <path
        d={`M ${x} ${y} H ${x + 46}`}
        fill="none"
        stroke={style.stroke}
        strokeWidth={style.width}
        strokeDasharray={style.dash}
        markerEnd={`url(#${style.marker})`}
        strokeLinecap="round"
      />
      <text x={x + 62} y={y - 4} fill="#223849" fontSize={12.5} fontWeight={760}>
        {label}
      </text>
      <text x={x + 62} y={y + 13} fill="#627181" fontSize={10.5} fontWeight={500}>
        {description}
      </text>
    </g>
  );
}

function PipelineFigure({ svgId, config }: { svgId: string; config: FigureConfig }) {
  const nodesById = new Map(config.nodes.map((node) => [node.id, node]));
  const xaiSection = config.sections.find((section) => section.id === "xai");

  return (
    <svg
      id={svgId}
      viewBox={`0 0 ${CANVAS.width} ${CANVAS.height}`}
      role="img"
      aria-labelledby="figure-title figure-desc"
      className="block h-auto w-full"
    >
      <title id="figure-title">{config.title}</title>
      <desc id="figure-desc">{config.description}</desc>

      <defs>
        <filter id="node-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="8" stdDeviation="8" floodColor="#1c2d3b" floodOpacity="0.12" />
        </filter>
        {Object.entries(EDGE_STYLE).map(([kind, style]) => (
          <marker key={kind} id={style.marker} markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M 1.2 1.2 L 9 5 L 1.2 8.8 z" fill={style.stroke} />
          </marker>
        ))}
      </defs>

      <rect width={CANVAS.width} height={CANVAS.height} fill="#f7f8f4" />
      <text x={40} y={44} fill="#173042" fontSize={24} fontWeight={800}>
        <tspan x={40} y={44}>
          {config.title}
        </tspan>
        <tspan x={40} y={70}>
          {config.subtitle}
        </tspan>
      </text>
      <text x={40} y={89} fill="#5b6b79" fontSize={13.5} fontWeight={520}>
        Late Fusion is a decision stage inside the broader pipeline; explanation artefacts are derived after evidence is produced.
      </text>

      {config.sections.filter((section) => section.id !== "xai").map((section) => (
        <SectionBox key={section.id} section={section} />
      ))}
      {xaiSection ? <ExplanationPanel section={xaiSection} /> : null}

      <g opacity={0.96}>
        {config.edges.map((edge) => (
          <Arrow key={edge.id} edge={edge} nodesById={nodesById} />
        ))}
      </g>

      {config.nodes.map((node) => (
        <PipelineNode key={node.id} node={node} />
      ))}

      <g>
        <rect x={50} y={800} width={1280} height={130} rx={18} fill="#ffffff" stroke="#d3dce3" strokeWidth={1.2} />
        <text x={72} y={832} fill="#173042" fontSize={16} fontWeight={800}>
          Legend
        </text>
        {LEGEND_ITEMS.map((item, index) => (
          <LegendItem key={item.label} x={76 + (index % 2) * 610} y={864 + Math.floor(index / 2) * 40} {...item} />
        ))}
      </g>
    </svg>
  );
}

export function ModelTrainingPipelineFigure({ svgId = "model-training-pipeline-svg" }: { svgId?: string }) {
  return <PipelineFigure svgId={svgId} config={TRAINING_CONFIG} />;
}

export function ModelTestingPipelineFigure({ svgId = "model-testing-pipeline-svg" }: { svgId?: string }) {
  return <PipelineFigure svgId={svgId} config={TESTING_CONFIG} />;
}
