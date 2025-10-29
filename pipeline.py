# -*- coding: utf-8 -*-
"""
Creative Enrichment Pipeline Module (pipeline.py)
- Style analysis on an input image using Vertex AI text model (Gemini 2.5 Pro)
- Web scraping of a landing page (Playwright/Requests)
- Refinement via Vertex AI to extract bullets/CTA
- Prompt construction for image generation
- Image generation via Google Generative AI (Gemini 2.5 Flash Image)
"""

import os, io, re, json, time, asyncio, datetime, random
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Core dependencies
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
import pytesseract
from pytesseract import Output

# Vertex AI (text analysis)
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Google Generative AI (image generation)
from google import genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")
client = genai.Client(api_key=GEMINI_API_KEY)

# Playwright (robust async HTML fetching)
from playwright.async_api import async_playwright

# =========================
# Config / Defaults
# =========================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
MODEL_TEXT     = os.getenv("MODEL_TEXT", "gemini-2.5-pro")

OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "outputs")
DEFAULT_OUTFILE = os.path.join(OUTPUT_DIR, "enhanced_banner.png")
ASPECT_RATIO    = os.getenv("ASPECT_RATIO", "9:16")
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.25"))

# Behavior switches (can be toggled from app.py per run)
ALLOW_TEXT_CHANGES = os.getenv("ALLOW_TEXT_CHANGES", "false").lower() == "true"
LOCK_COPY          = os.getenv("LOCK_COPY", "false").lower() == "true"

# Pricing placeholders (optional)
PRICING_USD = {
    "gemini-2.5-pro": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}
PRICING_IMAGE_PER_IMAGE = 0.00

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# User agents for scraping
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# =========================
# Scraping functions
# =========================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6), reraise=False)
async def _fetch_playwright(url: str, timeout_ms: int = 35000, user_agent: Optional[str] = None) -> Optional[str]:
    ua = user_agent or random.choice(UA_POOL)
    launch_args = ["--disable-http2","--disable-blink-features=AutomationControlled","--no-sandbox","--disable-dev-shm-usage"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=launch_args)
        try:
            ctx = await browser.new_context(user_agent=ua, locale="en-US")
            page = await ctx.new_page()
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            await asyncio.sleep(0.8)
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass
            html = await page.content()
            if "<body" in html.lower():
                return html
            await page.goto(url, timeout=timeout_ms, wait_until="load")
            await page.wait_for_selector("body", timeout=5000)
            return await page.content()
        finally:
            await browser.close()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=False)
def _fetch_requests(url: str, user_agent: Optional[str] = None, timeout: int = 25) -> Optional[str]:
    headers = {"User-Agent": user_agent or random.choice(UA_POOL), "Accept-Language": "en-US,en;q=0.9"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except Exception:
        return None
    if r.ok and r.text:
        return r.text
    return None

async def fetch_html(url: str) -> Optional[str]:
    try:
        html = await _fetch_playwright(url, user_agent=random.choice(UA_POOL))
        if html:
            return html
    except Exception as e:
        print("[Playwright] Error:", e)
    return _fetch_requests(url, user_agent=random.choice(UA_POOL))

# =========================
# Scraping helpers
# =========================
def clean_text(t: Optional[str]) -> str:
    if not t: return ""
    return re.sub(r"\s+", " ", t).strip()

def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])

def collect_highlights(soup: BeautifulSoup, text: str, want: int = 6) -> List[str]:
    seeds = []
    for sel in ['meta[name="description"]','meta[property="og:description"]','meta[name="twitter:description"]']:
        m = soup.select_one(sel)
        if m and m.get("content"): seeds.append(m["content"])
    seeds += [h.get_text(" ", strip=True) for h in soup.select("h2, h3")][:20]
    seeds += [li.get_text(" ", strip=True) for li in soup.select("ul li")][:50]
    lines = [ln.strip() for ln in text.split("\n") if 6 <= len(ln) <= 200]
    seeds += lines[:200]

    out = []
    for s in seeds:
        for piece in re.split(r"[•·\u2022\|\;\.\!\?]\s+", s):
            piece = re.sub(r"\s+", " ", piece).strip(" -–—•·")
            if 8 <= len(piece) <= 160:
                out.append(piece)

    keywords = {"free","offer","discount","save","emi","loan","apply","delivery","return","replacement","warranty","feature","benefit","rating","engine","spec","test ride","cashback","price","no cost emi"}
    scored = []
    for x in out:
        score = sum(1 for k in keywords if k in x.lower())
        scored.append((score, x))
    scored.sort(key=lambda t: (-t[0], len(t[1])))

    seen, bullets = set(), []
    for _, x in scored:
        key = x.lower()
        if key in seen: continue
        seen.add(key); bullets.append(x)
        if len(bullets) >= want: break
    return bullets[:want]

def scrape_landing_struct(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.select_one("h1")
    title = clean_text(h1.get_text()) if h1 else (soup.title.get_text().strip() if soup.title else url)
    text = extract_visible_text(html)
    highlights = collect_highlights(soup, text, want=6)
    headings = [clean_text(h.get_text(" ", strip=True)) for h in soup.select("h1, h2, h3")][:20]
    return {"url": url, "title": title, "headings": headings, "bullets": highlights or [], "text": text[:120_000]}

DEFAULT_FALLBACK_BULLETS = [
    "Save more on trusted medicines",
    "Easy e-prescription uploads",
    "Fast home delivery",
    "Genuine, quality-checked products",
    "Exclusive app-only offers",
    "Hassle-free returns and support",
]

# =========================
# Vertex AI helpers
# =========================
def _strip_code_fences(s: str) -> str:
    return re.sub(r"^```json|^```|```$", "", s.strip(), flags=re.MULTILINE).strip()

def _json_or_empty(s: str) -> dict:
    try: return json.loads(_strip_code_fences(s))
    except Exception: return {}

def analyze_style_with_gemini(image: Image.Image, model_name: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    model = GenerativeModel(model_name)
    prompt_style = ("Analyze this advertising banner. Describe the aesthetic style, color palette, "
                    "typography (serif/sans, weight), layout hierarchy, logo placement, and one primary CTA. "
                    "Respond as ~6 crisp bullet points.")
    prompt_principles = ("Evaluate the banner for design principles: whitespace, symmetry/balance, color/gradients, "
                         "and visual hierarchy. Give ~5 actionable notes to redesign it.")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    img_bytes = buf.getvalue()

    r1 = model.generate_content([Part.from_text(prompt_style), Part.from_data(data=img_bytes, mime_type="image/png")])
    r2 = model.generate_content([Part.from_text(prompt_principles), Part.from_data(data=img_bytes, mime_type="image/png")])

    def tk(u):
        if not u: return (0,0,0)
        return (int(getattr(u, "prompt_token_count", 0) or 0),
                int(getattr(u, "candidates_token_count", 0) or 0),
                int(getattr(u, "total_token_count", 0) or 0))
    in1,out1,tot1 = tk(getattr(r1, "usage_metadata", None))
    in2,out2,tot2 = tk(getattr(r2, "usage_metadata", None))
    style = {"style_summary": (r1.text or "").strip(), "principles_summary": (r2.text or "").strip()}
    tokens = {"input": in1+in2, "output": out1+out2, "total": tot1+tot2}
    return style, tokens

def refine_scrape_with_gemini(scrape: dict, model_name: str) -> Tuple[dict, dict]:
    text = "\n".join([
        scrape.get("title",""),
        " | ".join(scrape.get("headings",[])),
        "\n".join(scrape.get("bullets",[])),
        scrape.get("text","")
    ])[:120_000]
    prompt = f"""
Return ONLY JSON with keys: title, url, bullets, key_values, cta.
- bullets: 5–8 crisp, user-facing promises/benefits (no long sentences).
- key_values: short key:value hints (price, offer %, rating, downloads, etc.) when obvious.
- cta: 2–5 word call to action found or inferred.

URL: {scrape.get('url','')}
PAGE TEXT:
{text}
"""
    model = GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    usage = getattr(resp, "usage_metadata", None) or {}
    toks = {
        "input":  int(getattr(usage, "prompt_token_count", 0) or 0),
        "output": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total":  int(getattr(usage, "total_token_count", 0) or 0),
    }
    obj = _json_or_empty(resp.text or "")
    refined = {
        "title": obj.get("title") or scrape.get("title") or "",
        "url": obj.get("url") or scrape.get("url") or "",
        "bullets": obj.get("bullets") or scrape.get("bullets") or [],
        "key_values": obj.get("key_values") or {},
        "cta": obj.get("cta") or ""
    }
    return refined, toks

def _safe_int(x, default=0):
    try: return int(x)
    except Exception: return default

def overlay_text_regions(base_img_path: str, gen_img_path: str, save_path: str, expand: int = 2) -> str:
    """
    Paste every OCR-detected word region from BASE image onto GENERATED image,
    preserving all original glyphs (useful when locking copy).
    """
    try:
        base = Image.open(base_img_path).convert("RGBA")
        gen  = Image.open(gen_img_path).convert("RGBA")
        data = pytesseract.image_to_data(base, output_type=Output.DICT)
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt: continue
            x = _safe_int(data["left"][i]);  y = _safe_int(data["top"][i])
            w = _safe_int(data["width"][i]); h = _safe_int(data["height"][i])
            x0 = max(0, x - expand); y0 = max(0, y - expand)
            x1 = min(base.width,  x + w + expand); y1 = min(base.height, y + h + expand)
            patch = base.crop((x0, y0, x1, y1))
            gen.paste(patch, (x0, y0))
        gen.convert("RGB").save(save_path)
        return save_path
    except Exception as e:
        print("[WARN] overlay_text_regions failed:", e)
        from shutil import copyfile
        copyfile(gen_img_path, save_path)
        return save_path

# =========================
# Prompt construction
# =========================
def build_image_prompt_from_sections(style_dict: dict, refined: dict, allow_text_changes: bool = False) -> str:
    style_summary = (style_dict.get("style_summary") or "").strip()
    principles_summary = (style_dict.get("principles_summary") or "").strip()
    bullets = refined.get("bullets") or []
    cta = (refined.get("cta") or "").strip()
    key_values = refined.get("key_values") or {}

    lines = []
    lines.append("Enrich the image using the style recognizer, principle notes, and landing page focal points.")
    lines.append("Do NOT change the base layout/composition of the uploaded creative.")
    lines.append("")
    lines.append("=== STYLE RECOGNIZER ===");    lines.append(style_summary);       lines.append("")
    lines.append("=== PRINCIPLES NOTES ===");    lines.append(principles_summary);  lines.append("")
    lines.append("=== LANDING PAGE FOCAL POINTS ===")
    for b in bullets: lines.append(f"- {b}")
    lines.append("")
    if cta:
        lines.append("=== CTA ==="); lines.append(cta); lines.append("")
    if key_values:
        lines.append("=== KEY VALUES ==="); lines.append(json.dumps(key_values, ensure_ascii=False, indent=2)); lines.append("")

    lines.append("=== NON-NEGOTIABLES (STRICT) ===")
    if allow_text_changes:
        lines.append("- You MAY re-typeset, correct spelling/grammar, and replace rasterized text with new text EXACTLY as specified in later sections.")
        lines.append("- Keep brand-safe colors, high readability; do not move the brand logo unless explicitly asked.")
    else:
        lines.append("- Absolutely NO spell-checking, normalization, or rewriting of any existing text in the base image.")
        lines.append("- Preserve all original strings exactly as they appear.")
        lines.append("- Do not alter numerals, brand names, legal text, or fine print.")
    lines.append("- Additions must not cover/obscure existing copy; keep ample whitespace.")
    lines.append("- Maintain brand-safe colors and clear hierarchy.")
    lines.append("")
    lines.append("Rendering constraints: modern, clean, brand-colored; ample whitespace; strong contrast focal area for headline.")
    return "\n".join(lines).strip()

@dataclass
class IterSession:
    base_image_path: Optional[str] = None
    style_dict: Optional[dict] = None
    refined_dict: Optional[dict] = None
    image_prompt_built: Optional[str] = None
    last_output_path: Optional[str] = None
    history: list = field(default_factory=list)

SESSION = IterSession()

def compose_final_prompt(auto_prompt: str, manual_override: str = "", feedback_from_last: str = "", allow_text_changes: bool = False) -> str:
    """Overrides FIRST (top priority), then base prompt, then layout/style feedback."""
    parts = []
    if manual_override.strip():
        parts += [
            "=== MANDATORY OVERRIDES (TOP PRIORITY) ===",
            "If any previous instruction conflicts, FOLLOW THESE OVERRIDES.",
            "Text edits are " + ("ALLOWED" if allow_text_changes else "NOT ALLOWED") + ".",
            manual_override.strip(),
            ""
        ]
    parts += ["=== BASE PROMPT ===", auto_prompt.strip(), ""]
    if feedback_from_last.strip():
        parts += [
            "=== LAYOUT & STYLE FEEDBACK (SECONDARY) ===",
            "Use these to refine layout/legibility while respecting overrides:",
            feedback_from_last.strip(),
            ""
        ]
    parts += [
        "=== GUARANTEES ===",
        "No text truncation; increase text box height as needed.",
        "Ensure strong contrast and clean sans-serif for small copy; no emojis in text."
    ]
    return "\n".join(parts)

def _unique_name(base: str, suffix=""):
    stem, ext = os.path.splitext(base)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stem}-{ts}{('-' + suffix) if suffix else ''}{ext}"

# =========================
# Image generation
# =========================
def _save_inline_image(resp, out_path: str) -> bool:
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content: continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                Image.open(io.BytesIO(inline.data)).save(out_path)
                return True
    return False

def generate_banner_with_gemini_image(creative_prompt: str, base_image: Image.Image, out_path: str,
                                      aspect_ratio: str = "9:16", temperature: float = 0.6) -> str:
    cfg = None
    try:
        from google.genai import types as gtypes
        if hasattr(gtypes, "GenerateContentConfig") and hasattr(gtypes, "ImageConfig"):
            cfg = gtypes.GenerateContentConfig(
                temperature=temperature,
                response_modalities=["IMAGE"],
                image_config=gtypes.ImageConfig(aspect_ratio=aspect_ratio),
            )
    except Exception:
        cfg = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "prompt_sent_to_gemini.txt"), "w", encoding="utf-8") as f:
        f.write(f"# ALLOW_TEXT_CHANGES={ALLOW_TEXT_CHANGES}  LOCK_COPY={LOCK_COPY}\n")
        f.write(f"# aspect_ratio={aspect_ratio}  temperature={temperature}\n\n")
        f.write(creative_prompt)

    if cfg is not None:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[creative_prompt, base_image],
            config=cfg,
        )
    else:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[f"[Frame: {aspect_ratio}] {creative_prompt}", base_image],
            config={"temperature": temperature, "response_modalities": ["IMAGE"]},
        )

    if _save_inline_image(resp, out_path):
        print(f"✅ Image saved → {out_path}")
        return out_path
    else:
        raise RuntimeError("Gemini did not return image bytes.")

# =========================
# Pipeline orchestration
# =========================
def dollars_for_tokens(model_name: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_USD.get(model_name, {"input_per_1k": 0.0, "output_per_1k": 0.0})
    return (in_tokens/1000.0)*p["input_per_1k"] + (out_tokens/1000.0)*p["output_per_1k"]

@dataclass
class PipelineResult:
    step1_tokens: Dict[str,int]
    step2_tokens: Dict[str,int]
    step3_tokens: Dict[str,int]
    total_tokens: Dict[str,int]
    style_summary: str
    principles_summary: str
    refined_bullets: List[str]
    refined_cta: str
    refined_key_values: Dict[str,Any]
    image_prompt: str
    output_path: str
    costs_usd: Dict[str, float]

async def prepare_once_and_cache(creative_image_path: str, landing_url: str):
    """Style analysis + scrape/clean once; cache prompt/session and RETURN token dicts."""
    base_img = Image.open(creative_image_path).convert("RGB")
    style_dict, step1_tokens = analyze_style_with_gemini(base_img, MODEL_TEXT)

    html = await fetch_html(landing_url)
    if not html:
        scraped = {"url": landing_url, "title": "Landing Page", "headings": [], "bullets": DEFAULT_FALLBACK_BULLETS, "text": ""}
    else:
        scraped = scrape_landing_struct(html, landing_url)
        if not scraped.get("bullets"):
            scraped["bullets"] = DEFAULT_FALLBACK_BULLETS

    refined, step2_tokens = refine_scrape_with_gemini(scraped, MODEL_TEXT)
    auto_prompt = build_image_prompt_from_sections(style_dict, refined, allow_text_changes=ALLOW_TEXT_CHANGES)

    SESSION.base_image_path    = creative_image_path
    SESSION.style_dict         = style_dict
    SESSION.refined_dict       = refined
    SESSION.image_prompt_built = auto_prompt

    return step1_tokens, step2_tokens

def run_iteration(manual_override: str = "", feedback_from_last: str = "", aspect_ratio: str = "9:16",
                  temperature: float = 0.25, edit_previous_output: bool = False, outfile_suffix: str = "") -> str:
    """Generate a new image based on prepared prompt (optionally editing last output)."""
    assert SESSION.image_prompt_built, "Prepare the prompt first by calling prepare_once_and_cache."
    base_image_path = SESSION.last_output_path if (edit_previous_output and SESSION.last_output_path) else SESSION.base_image_path
    base_img = Image.open(base_image_path).convert("RGB")

    final_prompt = compose_final_prompt(
        auto_prompt=SESSION.image_prompt_built,
        manual_override=manual_override,
        feedback_from_last=feedback_from_last,
        allow_text_changes=ALLOW_TEXT_CHANGES
    )

    out_path = _unique_name(os.path.splitext(DEFAULT_OUTFILE)[0] + ".png", outfile_suffix or ("edit" if edit_previous_output else "v"))
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(out_path))

    generate_banner_with_gemini_image(
        creative_prompt=final_prompt,
        base_image=base_img,
        out_path=out_path,
        aspect_ratio=aspect_ratio,
        temperature=temperature,
    )

    # Optional: lock copy by re-pasting OCR regions from the base creative
    if LOCK_COPY:
        locked = out_path.replace(".png", "-locked.png")
        overlay_text_regions(base_image_path, out_path, locked, expand=2)
        out_path = locked

    SESSION.last_output_path = out_path
    SESSION.history.append({
        "prompt": final_prompt,
        "base_image_used": base_image_path,
        "out_path": out_path,
        "when": time.time(),
        "aspect_ratio": aspect_ratio,
        "temperature": temperature,
    })
    print(f"[iteration] Saved output: {out_path}")
    return out_path

async def run_pipeline_once_and_generate_first(creative_image_path: str, landing_url: str,
                                               aspect_ratio: str, temperature: float) -> PipelineResult:
    """End-to-end first run: analyze, scrape/refine, build prompt, generate first image."""
    base_img = Image.open(creative_image_path).convert("RGB")
    style_dict, step1_tokens = analyze_style_with_gemini(base_img, MODEL_TEXT)

    html = await fetch_html(landing_url)
    if not html:
        scraped = {"url": landing_url, "title": "Landing Page", "headings": [], "bullets": DEFAULT_FALLBACK_BULLETS, "text": ""}
    else:
        scraped = scrape_landing_struct(html, landing_url)
        if not scraped.get("bullets"):
            scraped["bullets"] = DEFAULT_FALLBACK_BULLETS

    refined, step2_tokens = refine_scrape_with_gemini(scraped, MODEL_TEXT)
    image_prompt = build_image_prompt_from_sections(style_dict, refined, allow_text_changes=ALLOW_TEXT_CHANGES)

    out_path = _unique_name(os.path.join(OUTPUT_DIR, "enhanced_banner.png"), "v1")
    generate_banner_with_gemini_image(
        creative_prompt=image_prompt,
        base_image=base_img,
        out_path=out_path,
        aspect_ratio=aspect_ratio,
        temperature=temperature,
    )

    if LOCK_COPY:
        locked = out_path.replace(".png", "-locked.png")
        overlay_text_regions(creative_image_path, out_path, locked, expand=2)
        out_path = locked

    # Save session state
    SESSION.base_image_path    = creative_image_path
    SESSION.style_dict         = style_dict
    SESSION.refined_dict       = refined
    SESSION.image_prompt_built = image_prompt
    SESSION.last_output_path   = out_path
    SESSION.history.append({"prompt": image_prompt, "out_path": out_path, "when": time.time()})

    total = {"input": step1_tokens["input"] + step2_tokens["input"],
             "output": step1_tokens["output"] + step2_tokens["output"],
             "total": step1_tokens["total"] + step2_tokens["total"]}
    cost_text = dollars_for_tokens(MODEL_TEXT, total["input"], total["output"])
    cost_image = PRICING_IMAGE_PER_IMAGE
    grand_total = cost_text + cost_image

    print("\n===== Token Usage =====")
    print(f"Model (text): {MODEL_TEXT}")
    print(f"Step1 (Image analysis) → in: {step1_tokens['input']}, out: {step1_tokens['output']}")
    print(f"Step2 (Content refine) → in: {step2_tokens['input']}, out: {step2_tokens['output']}")
    print(f"Total tokens used: {total}")
    print("=======================")

    return PipelineResult(
        step1_tokens=step1_tokens,
        step2_tokens=step2_tokens,
        step3_tokens={"input": 0, "output": 0, "total": 0},
        total_tokens=total,
        style_summary=style_dict.get("style_summary", ""),
        principles_summary=style_dict.get("principles_summary", ""),
        refined_bullets=refined.get("bullets", []),
        refined_cta=refined.get("cta", ""),
        refined_key_values=refined.get("key_values", {}),
        image_prompt=image_prompt,
        output_path=out_path,
        costs_usd={
            "model": MODEL_TEXT,
            "text_steps_usd": round(cost_text, 6),
            "image_usd": round(cost_image, 6),
            "grand_total_usd": round(grand_total, 6),
        }
    )

# =========================
# Initialization for Vertex AI
# =========================
def init_vertex():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set. Please configure the .env file.")
    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID not set in .env.")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    print(f"Vertex AI initialized (project={GCP_PROJECT_ID}, location={GCP_LOCATION})")
