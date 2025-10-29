import os, io, csv, asyncio, json
import base64, re
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
from flask import Response

load_dotenv()

import pipeline  # updated pipeline.py

CSV_PATH = os.getenv("CREATIVES_CSV", "Creative Ads Details - Meta.csv")
ALLOWED_EXT = {".png"}

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

pipeline.init_vertex()

def _ext_ok(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def _read_creatives_csv(path: str):
    if not os.path.exists(path): return []
    prefer = lambda row, keys: next((row[k] for k in keys if k in row and row[k]), "")
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title     = prefer(row, ["Creative Title","Creative_Name","Creative Name","Title","Ad Title"])
            brand_url = prefer(row, ["Link url","Brand/Product URL","Brand URL","Ad creative url","URL","Landing URL"])
            image_url = prefer(row, ["Ad creative url","Image URL","Ad Image URL","Creative Image"])
            if not (title or brand_url or image_url): continue
            out.append({"id": str(i), "title": title.strip(), "brand_url": brand_url.strip(), "image_url": image_url.strip()})
    return out

CREATIVES = _read_creatives_csv(CSV_PATH)

def _best_landing_url(provided: str = "") -> str:
    if provided: return provided
    if getattr(pipeline.SESSION, "refined_dict", None):
        url = (pipeline.SESSION.refined_dict or {}).get("url") or ""
        if url: return url
    return ""

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/creatives")
def api_creatives():
    return jsonify({"items": CREATIVES})

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory("outputs", filename, as_attachment=False)

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory("uploads", filename, as_attachment=False)
# @app.get("/proxy_image")
# def proxy_image():
#     """
#     Fetch an external image URL (e.g. CloudFront) and return it
#     as if it's hosted locally. This avoids CORS-tainted canvas.
#     Call it like:
#       /proxy_image?url=https%3A%2F%2Fcdn...%2Fcreative.png
#     """
#     img_url = request.args.get("url", "")
#     if not img_url:
#         return Response("Missing url", status=400)

#     try:
#         upstream = requests.get(img_url, timeout=5)
#     except Exception as e:
#         return Response(f"Fetch failed: {e}", status=502)

#     if upstream.status_code != 200:
#         return Response(f"Upstream {upstream.status_code}", status=502)

#     # Try to preserve the content-type if possible (png)
#     ctype = upstream.headers.get("Content-Type", "image/png")

#     resp = Response(upstream.content, mimetype=ctype)
#     # Allow the browser to use this in <canvas> and still export/download
#     resp.headers["Access-Control-Allow-Origin"] = "*"
#     return resp

# ------------------------
# First generate
# ------------------------
@app.post("/generate")
def generate():
    brand_url = request.form.get("brand_url", "").strip()
    file = request.files.get("image")
    if not file or not file.filename:
        return render_template("index.html", error="Please upload a PNG image.", result_image=None)
    if not _ext_ok(file.filename):
        return render_template("index.html", error="Upload a .png file only.", result_image=None)

    fname = secure_filename(file.filename)
    up_path = os.path.join("uploads", fname)
    file.save(up_path)

    temp = float(os.getenv("IMAGE_TEMPERATURE", "0.10"))

    try:
        result = asyncio.run(
            pipeline.run_pipeline_once_and_generate_first(
                up_path,
                brand_url,
                pipeline.ASPECT_RATIO,
                temp
            )
        )

        meta = {
            "model_text": pipeline.MODEL_TEXT,
            "model_image": "gemini-2.5-flash-image",
            "temperature": temp,
            "aspect_ratio": pipeline.ASPECT_RATIO,
            "project": os.getenv("GCP_PROJECT_ID", ""),
            "location": os.getenv("GCP_LOCATION", ""),
            "tokens": result.total_tokens,
            "tokens_step1": result.step1_tokens,
            "tokens_step2": result.step2_tokens,
            "costs": result.costs_usd,
            "prompt_full": result.image_prompt,
            "base_image_name": os.path.basename(up_path),
            "output_image_name": os.path.basename(result.output_path),
            "landing_url": brand_url,
        }

        return render_template("result.html", **meta)

    except Exception as e:
        return render_template("index.html", error=f"Generation failed: {e}", result_image=None)

# ------------------------
# Variation page
# ------------------------
def _save_data_url_png(data_url: str, out_path: str) -> str:
    m = re.match(r"^data:image/(png|jpeg);base64,(.+)$", data_url)
    if not m:
        raise ValueError("Invalid data URL")
    b = base64.b64decode(m.group(2))
    with open(out_path, "wb") as f:
        f.write(b)
    return out_path

@app.post("/edit/save")
def edit_save():
    """
    Accepts a PNG data URL from the canvas editor and saves it as a new output.
    Optionally promotes it to session base for future variations.
    """
    data_url = request.form.get("img", "")
    set_as_base = request.form.get("set_as_base") in ("on", "true", "1")
    if not data_url:
        return jsonify({"ok": False, "error": "Missing image data"}), 400

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = f"edited_{ts}.png"
    out_path = os.path.join("outputs", out_name)
    try:
        _save_data_url_png(data_url, out_path)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Save failed: {e}"}), 400

    # Update session so the edited image shows up as "current output"
    pipeline.SESSION.last_output_path = out_path
    if set_as_base:
        pipeline.SESSION.base_image_path = out_path

    return jsonify({
        "ok": True,
        "saved": out_name,
        "url": f"/outputs/{out_name}",
        "set_as_base": bool(set_as_base)
    })


@app.get("/variation")
def variation_page():
    last_out = os.path.basename(pipeline.SESSION.last_output_path) if pipeline.SESSION.last_output_path else ""
    base_img = os.path.basename(pipeline.SESSION.base_image_path) if pipeline.SESSION.base_image_path else ""
    landing_url = _best_landing_url()

    return render_template(
        "variation_result.html",
        output_image_name=last_out,
        base_image_name=base_img,
        landing_url=landing_url,
        model_text=pipeline.MODEL_TEXT,
        model_image="gemini-2.5-flash-image",
        temperature=0.2,  # default precise edits
        aspect_ratio=pipeline.ASPECT_RATIO,
        prompt_full=pipeline.SESSION.history[-1]["prompt"] if pipeline.SESSION.history else "",
        allow_text_changes="checked" if pipeline.ALLOW_TEXT_CHANGES else ""
    )

@app.post("/variation")
def variation():
    brand_url = _best_landing_url(request.form.get("brand_url", "").strip())
    manual_override = request.form.get("manual_override", "").strip()
    feedback = request.form.get("feedback", "").strip()
    aspect_ratio = request.form.get("aspect_ratio", pipeline.ASPECT_RATIO).strip() or pipeline.ASPECT_RATIO
    try:
        temp_val = float(request.form.get("temperature", "0.2"))
    except ValueError:
        temp_val = 0.2

    # Checkbox from variation_result.html:
    allow_text_changes = request.form.get("allow_text_changes") == "1"
    setattr(pipeline, "ALLOW_TEXT_CHANGES", allow_text_changes)
    print("[VARIATION] ALLOW_TEXT_CHANGES =", pipeline.ALLOW_TEXT_CHANGES)

    file = request.files.get("image2")
    edit_previous = True
    step1_tokens, step2_tokens = {}, {}

    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext != ".png":
            last_out_name = os.path.basename(pipeline.SESSION.last_output_path) if pipeline.SESSION.last_output_path else ""
            return render_template(
                "variation_result.html",
                error="Upload a .png file only for the new base image.",
                output_image_name=last_out_name,
                base_image_name=os.path.basename(pipeline.SESSION.base_image_path) if pipeline.SESSION.base_image_path else "",
                landing_url=brand_url,
                model_text=pipeline.MODEL_TEXT,
                model_image="gemini-2.5-flash-image",
                temperature=temp_val,
                aspect_ratio=aspect_ratio,
                prompt_full=pipeline.SESSION.history[-1]["prompt"] if pipeline.SESSION.history else "",
                allow_text_changes="checked" if pipeline.ALLOW_TEXT_CHANGES else ""
            )

        # New base image → re-prepare
        from PIL import Image
        fname = secure_filename(file.filename)
        new_base_path = os.path.join("uploads", fname)
        file.save(new_base_path)
        try:
            _ = Image.open(new_base_path).convert("RGB")
            prep_res = asyncio.run(pipeline.prepare_once_and_cache(new_base_path, brand_url))
            if isinstance(prep_res, tuple) and len(prep_res) == 2:
                step1_tokens, step2_tokens = prep_res
            edit_previous = False
        except Exception as e:
            last_out_name = os.path.basename(pipeline.SESSION.last_output_path) if pipeline.SESSION.last_output_path else ""
            return render_template(
                "variation_result.html",
                error=f"Failed to use the new base image: {e}",
                output_image_name=last_out_name,
                base_image_name=os.path.basename(pipeline.SESSION.base_image_path) if pipeline.SESSION.base_image_path else "",
                landing_url=brand_url,
                model_text=pipeline.MODEL_TEXT,
                model_image="gemini-2.5-flash-image",
                temperature=temp_val,
                aspect_ratio=aspect_ratio,
                prompt_full=pipeline.SESSION.history[-1]["prompt"] if pipeline.SESSION.history else "",
                allow_text_changes="checked" if pipeline.ALLOW_TEXT_CHANGES else ""
            )

    print("\n[VARIATION] edit_previous_output =", edit_previous)
    base_will_be = pipeline.SESSION.last_output_path if (edit_previous and pipeline.SESSION.last_output_path) else pipeline.SESSION.base_image_path
    print("[VARIATION] base will be:", base_will_be)
    print("[VARIATION] temp =", temp_val, "AR =", aspect_ratio)
    print("[VARIATION] LOCK_COPY =", getattr(pipeline, "LOCK_COPY", None))
    print("[VARIATION] override_len =", len(manual_override or ""), "feedback_len =", len(feedback or ""))

    # Version suffix
    try:
        next_idx = len(pipeline.SESSION.history) + 1
    except Exception:
        next_idx = 1
    suffix = f"v{next_idx}"

    try:
        out_path = pipeline.run_iteration(
            manual_override=manual_override,
            feedback_from_last=feedback,
            aspect_ratio=aspect_ratio,
            temperature=temp_val,
            edit_previous_output=edit_previous,
            outfile_suffix=suffix
        )
    except Exception as e:
        last_out_name = os.path.basename(pipeline.SESSION.last_output_path) if pipeline.SESSION.last_output_path else ""
        return render_template(
            "variation_result.html",
            error=f"Variation generation failed: {e}",
            output_image_name=last_out_name,
            base_image_name=os.path.basename(pipeline.SESSION.base_image_path) if pipeline.SESSION.base_image_path else "",
            landing_url=brand_url,
            model_text=pipeline.MODEL_TEXT,
            model_image="gemini-2.5-flash-image",
            temperature=temp_val,
            aspect_ratio=aspect_ratio,
            prompt_full=pipeline.SESSION.history[-1]["prompt"] if pipeline.SESSION.history else "",
            allow_text_changes="checked" if pipeline.ALLOW_TEXT_CHANGES else ""
        )

    output_filename = os.path.basename(out_path)
    total_tokens, costs = {}, {}
    if step1_tokens and step2_tokens:
        total_tokens = {
            "input":  step1_tokens.get("input",0) + step2_tokens.get("input",0),
            "output": step1_tokens.get("output",0) + step2_tokens.get("output",0),
            "total":  step1_tokens.get("total",0) + step2_tokens.get("total",0)
        }
        costs = {
            "model": pipeline.MODEL_TEXT,
            "text_steps_usd": round(pipeline.dollars_for_tokens(pipeline.MODEL_TEXT, total_tokens["input"], total_tokens["output"]), 6),
            "image_usd": round(pipeline.PRICING_IMAGE_PER_IMAGE, 6),
            "grand_total_usd": round(
                pipeline.dollars_for_tokens(pipeline.MODEL_TEXT, total_tokens["input"], total_tokens["output"]) + pipeline.PRICING_IMAGE_PER_IMAGE, 6
            )
        }

    return render_template(
        "variation_result.html",
        output_image_name=output_filename,
        base_image_name=os.path.basename(pipeline.SESSION.base_image_path) if pipeline.SESSION.base_image_path else "",
        landing_url=brand_url,
        model_text=pipeline.MODEL_TEXT,
        model_image="gemini-2.5-flash-image",
        temperature=temp_val,
        aspect_ratio=aspect_ratio,
        tokens=total_tokens,
        costs=costs,
        prompt_full=pipeline.SESSION.history[-1]["prompt"] if pipeline.SESSION.history else "",
        allow_text_changes="checked" if pipeline.ALLOW_TEXT_CHANGES else ""
    )

# ------------------------
# Diagnostics
# ------------------------
@app.get("/diag")
def diag():
    d = {
        "ALLOW_TEXT_CHANGES": getattr(pipeline, "ALLOW_TEXT_CHANGES", None),
        "LOCK_COPY": getattr(pipeline, "LOCK_COPY", None),
        "base_image_path": getattr(pipeline.SESSION, "base_image_path", None),
        "last_output_path": getattr(pipeline.SESSION, "last_output_path", None),
        "history_len": len(getattr(pipeline.SESSION, "history", [])),
        "landing_url": _best_landing_url(),
        "last_prompt_excerpt": (pipeline.SESSION.history[-1]["prompt"][:500] + "…") if pipeline.SESSION.history else "",
    }
    return app.response_class(json.dumps(d, ensure_ascii=False, indent=2), mimetype="application/json")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
