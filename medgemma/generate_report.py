import io
import json
from collections import OrderedDict
import os

import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Please add it to .env")

login(HF_TOKEN)

PIPE = None

model_variant = "4b-it"
model_id = f"google/medgemma-{model_variant}"
use_quantization = False

model_kwargs = dict(
    dtype=torch.bfloat16,
    device_map="auto",
)

if use_quantization:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

system_instruction = '''
You are a board-certified thoracic radiologist.
Generate a professional and clearly formatted chest radiograph report in English.

Overall structure:
1. Begin with a short section titled **Patient Information**, listing key demographic and study details in readable format.
2. Then generate the radiology **Report** section, formatted with exactly 7 labeled lines:
   MeSH:, Problems:, image:, indication:, comparison:, findings:, impression:

Hard constraints:
- Output must contain both sections in this order:
  (A) Patient Information
  (B) Radiology Report (exactly 7 labeled lines)
- Use only the information provided in the metadata and image. Do not invent or infer extra patient data.
- If metadata includes bounding box information (bbox) describing abnormal regions on the X-ray,
  you MUST include this information in the findings section by:
  • Describing the anatomical location (e.g., "right lower lobe", "left upper zone", "right costophrenic angle").
  • Including the bbox label (e.g., "consolidation", "effusion", "pneumothorax").
  • Optionally include normalized coordinates (x, y, w, h) in parentheses.
  Example: "A localized consolidation is present in the right lower lobe (bbox label: consolidation_RLL, coordinates: 0.62, 0.72, 0.18, 0.16)."
- If `comparison_info` is provided, describe the comparison in the comparison field (e.g., “Compared with prior CXR dated 2025-06-10, interval improvement noted.”); otherwise output “None.”
- If metadata includes a diagnosis (e.g., pneumonia, consolidation, effusion, pneumothorax), findings MUST describe it and MUST NOT negate it.
- If Problems = "normal" or disease_type = "healthy", then:
  • MeSH must be exactly "Normal"
  • findings and impression must NOT mention any abnormality.
- comparison must be “None.” if no prior study info is provided.

Findings section:
- Write 2–6 complete sentences in the following structured order:
  (1) Heart and mediastinum
  (2) Lungs and any focal lesions or abnormalities (include bbox-based regions if present)
  (3) Pleura and diaphragm
  (4) Bones and soft tissues
  (5) Image quality (optional, one sentence)
- Use concise, standard phrasing and avoid redundancy. Example style:
  “The cardiomediastinal silhouette is within normal limits.
  A localized consolidation is present in the right lower lobe (bbox label: consolidation_RLL, coordinates: 0.62, 0.72, 0.18, 0.16).
  No pleural effusion or pneumothorax is seen.
  The visualized osseous structures are intact.
  Image quality is adequate.”

Impression section:
- Provide 1–2 concise sentences summarizing the key findings, starting with the dominant abnormality or “Normal chest radiograph.”
- It must be logically consistent with findings and add no new information.
- Style example:
  “Right lower lobe consolidation consistent with pneumonia.”
  or
  “No acute cardiopulmonary abnormality.”

Formatting and terminology:
- Use clear, professional English and standard radiologic lexicon (e.g., “no focal consolidation,” “no pleural effusion identified,” “no pneumothorax detected,” “cardiomediastinal silhouette within normal limits”).
- Maintain consistent medical tone; avoid speculative or casual wording.
- Do not include any extra commentary, explanations, or disclaimers.

Fail the output if any contradiction exists between findings and impression.
Return only the formatted text as specified.
'''




base_prompt = """Task: Produce a professional chest radiograph report for the provided patient.
The output must contain two main sections:

==============================
PATIENT INFORMATION
==============================
List all metadata below in a readable format (one key per line):
{metadata}

==============================
RADIOLOGY REPORT
==============================
Required fields (exactly these 7 lines, one per line):
MeSH, Problems, image, indication, comparison, findings, impression

Output formatting rules:
- Use exactly these labels and order:
  MeSH: <comma-separated MeSH terms>
  Problems: <semicolon-separated concise list>
  image: <exam name, e.g., "X-ray Chest PA and Lateral">
  indication: <one sentence; if missing, use neutral minimal text>
  comparison:
    • If `comparison_info` is provided, describe it concisely (e.g., “Compared with prior CXR dated <date>, interval improvement noted.”)
    • Otherwise output “None.”
  findings:
    • Write 2–6 complete sentences.
    • Follow this structured order:
        (1) heart/mediastinum
        (2) lungs and parenchyma (include bbox-based abnormalities if present)
        (3) pleura and diaphragm
        (4) bones and soft tissues
        (5) image quality
    • Describe bbox-based abnormalities when provided, including their label and coordinates.
      Example: “A localized consolidation is present in the right lower lobe (bbox label: consolidation_RLL, coordinates: 0.62, 0.72, 0.18, 0.16).”
    • Keep the tone concise and clinical.
  impression:
    • Provide 1–2 sentences summarizing the key findings.
    • If abnormal, highlight the main finding first (e.g., “Right lower lobe consolidation consistent with pneumonia.”)
    • If normal, use “Normal chest radiograph.” or “No acute cardiopulmonary abnormality.”
    • Must be consistent with findings; no new details allowed.

- Each of the 7 fields must occupy exactly one line.
- Do not include any other text, headers, or explanations.
- If disease_type == "healthy" and there is no visible abnormality, produce a normal exam report.

Generate the report now following the exact output template below.
Ensure each field is on its own line and ends with a newline.

Output template example:

MeSH: <comma-separated MeSH terms>
Problems: <semicolon-separated concise list>
image: <exam name, e.g., "X-ray Chest PA and Lateral">
indication: <one sentence>
comparison: <"None." if no prior>
findings: <Complete, concise paragraph following the structure above.>
impression: <1–2 sentences summarizing the findings.>
"""


def get_pipe():
    global PIPE
    if PIPE is None:
        print(">>> Initializing MedGemma pipeline...")
        PIPE = pipeline(
            "image-text-to-text",
            model=model_id,
            model_kwargs=model_kwargs,
        )
        if hasattr(PIPE, "model") and hasattr(PIPE.model, "generation_config"):
            PIPE.model.generation_config.do_sample = False
        print(">>> MedGemma pipeline is ready.")
    return PIPE



def build_prompt_from_metadata(patient_metadata: dict) -> str:
    metadata_text = "\n".join([f"- {k}: {v}" for k, v in patient_metadata.items()])
    return base_prompt.format(metadata=metadata_text)


def parse_report_text(report_text: str) -> dict:
    labels = OrderedDict(
        [
            ("MeSH", ["mesh"]),
            ("Problems", ["problems"]),
            ("Image", ["image"]),
            ("Indication", ["indication"]),
            ("Comparison", ["comparison"]),
            ("Findings", ["findings"]),
            ("Impression", ["impression"]),
        ]
    )
    lines = report_text.splitlines()
    current_label = None
    parsed = {k: "" for k in labels.keys()}

    def detect_label(line: str):
        s = line.strip()
        for canon_label, aliases in labels.items():
            for alias in aliases:
                prefix = alias + ":"
                if s.lower().startswith(prefix):
                    value = s[len(prefix) :].strip()
                    return canon_label, value
        return None, None

    for line in lines:
        if not line.strip():
            continue
        new_label, value = detect_label(line)
        if new_label is not None:
            current_label = new_label
            parsed[current_label] = value
        else:
            if current_label is not None:
                if parsed[current_label]:
                    parsed[current_label] += " " + line.strip()
                else:
                    parsed[current_label] = line.strip()

    for k in parsed:
        parsed[k] = parsed[k].strip()

    return parsed


def extract_generated_text_from_pipe_output(output) -> str:
    gen = output[0]["generated_text"]
    if isinstance(gen, str):
        return gen
    if isinstance(gen, list):
        last = gen[-1]
        if isinstance(last, dict) and "content" in last:
            content = last["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
        try:
            return "".join(
                m.get("content", "")
                if isinstance(m.get("content", ""), str)
                else "".join(
                    c.get("text", "")
                    for c in m.get("content", [])
                    if isinstance(c, dict) and c.get("type") == "text"
                )
                for m in gen
                if isinstance(m, dict)
            )
        except Exception:
            return str(gen)
    return str(gen)


def generate_clinical_report_from_pil(image, patient_metadata: dict) -> dict:
    print(">>> Calling get_pipe()...")
    pipe = get_pipe()
    print(">>> Building prompt...")
    prompt = build_prompt_from_metadata(patient_metadata)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    print(">>> Running inference...")
    output = pipe(text=messages, max_new_tokens=400)
    print(">>> Inference done, parsing output...")
    full_text = extract_generated_text_from_pipe_output(output)
    parsed = parse_report_text(full_text)

    radiology_report = {
        "MeSH": parsed.get("MeSH", ""),
        "Problems": parsed.get("Problems", ""),
        "Image": parsed.get("Image", ""),
        "Indication": parsed.get("Indication", ""),
        "Comparison": parsed.get("Comparison", ""),
        "Findings": parsed.get("Findings", ""),
        "Impression": parsed.get("Impression", ""),
        "raw_report_text": full_text,
    }

    result = {
        "patient_metadata": dict(patient_metadata),
        "radiology_report": radiology_report,
    }

    print(">>> Report generation finished.")
    return result



def generate_clinical_report_from_path(image_path: str, patient_metadata: dict) -> dict:
    image = Image.open(image_path)
    return generate_clinical_report_from_pil(image, patient_metadata)


def generate_clinical_report_from_bytes(
    image_bytes: bytes, patient_metadata: dict
) -> dict:
    image = Image.open(io.BytesIO(image_bytes))
    return generate_clinical_report_from_pil(image, patient_metadata)


def generate_clinical_report_from_json_and_path(
    image_path: str,
    patient_metadata_json: str,
) -> dict:
    patient_metadata = json.loads(patient_metadata_json)
    return generate_clinical_report_from_path(image_path, patient_metadata)
