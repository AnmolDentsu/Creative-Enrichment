# Creative-Enrichment
Upload an ad creative and landing url, and the backend uses an LLM api calls to image model and scrapping url pipeline to analyze the style, details, and generate refined prompt based on it . You can guide and customize variation accordingly using prompt, temperatures, manual canvas toolbox.

Project Overflow:
Creactive Enrichment/
├─ app.py                 # Flask app / routes
├─ pipeline.py            # Creative enrichment + model calls
├─ outputs/               # Generated & edited images land here
├─ uploads/               # User-uploaded base creatives
├─ static/
│   ├─ editor.js          # Canvas editor logic
│   └─ style.css          # UI styling
├─ templates/
│   ├─ index.html         # Upload + variation controls form
│   ├─ variation_result.html
│   └─ result.html / etc. # View current output + editor
└─ .env                   # API keys / model config (not committed)

Input:
![westelm](https://github.com/user-attachments/assets/7b2e0d4a-5191-4e51-9260-b7ca650667c5)

output:
<img width="768" height="1344" alt="enhanced_banner-20251022-110739-v1" src="https://github.com/user-attachments/assets/5e7d374f-240a-418e-8ff0-60c3fbf55196" />


