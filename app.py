<<<<<<< HEAD
import os
import re
import uuid
import subprocess
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils.data_fetch import fetch_and_add_papers
from flask_cors import CORS  
from graph.peer_review_graph import build_graph

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "data/results"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
CORS(app)

agent_graph = build_graph()

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_review(text):

    sections = {}

    pattern = r"\*\*(\d+)\.\s*([^\*]+)\*\*"

    matches = list(re.finditer(pattern, text))

    for i, match in enumerate(matches):

        section_num = match.group(1)
        title = match.group(2).strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        sections[f"{section_num}. {title}"] = content

    return sections

@app.route("/api/review", methods=["POST"])
def review():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(pdf_path)

        deep_search = request.form.get("deep_search")
        topic = request.form.get("topic")
        if deep_search and topic:
            print(f"[INFO] Running deep search for topic: {topic}")
            fetch_and_add_papers(topic, max_papers=15)
            subprocess.run(["python", "utils/faiss_index.py"])

        run_id = str(uuid.uuid4())[:8]
        run_dir = os.path.join(RESULTS_FOLDER, run_id)
        os.makedirs(run_dir, exist_ok=True)

        state = {
            "pdf_path": pdf_path,
            "topic": topic if topic else "general",
            "run_dir": run_dir
        }

        result = agent_graph.invoke(state)

        review_text = result.get("final_review_text", "")
        review_file = os.path.join(run_dir, "review.txt")

        with open(review_file, "w", encoding="utf-8") as f:
            f.write(review_text)

        if not review_text or len(review_text.strip()) < 20:
            if os.path.exists(review_file):
                with open(review_file, "r", encoding="utf-8") as f:
                    review_text = f.read()
            else:
                review_text = "Review generation failed."

        sections = parse_review(review_text)

        return jsonify({"review": sections})

    return jsonify({"error": "Invalid file type. Only PDF allowed."}), 400

if __name__ == "__main__":
=======
import os
import re
import uuid
import subprocess
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils.data_fetch import fetch_and_add_papers

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "data/results"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_review(text):
    """Parse the review.txt into sections."""
    sections = {}
    matches = re.split(r"\*\*(\d+\..*?)\*\*", text)
    for i in range(1, len(matches), 2):
        title = matches[i].strip()
        content = matches[i + 1].strip() if i + 1 < len(matches) else ""
        sections[title] = content
    return sections


@app.route("/", methods=["GET", "POST"])
def index():
    sections, error = {}, None

    if request.method == "POST":
        # Check if file is in request
        if "file" not in request.files:
            error = "No file part"
            return render_template("index.html", error=error, sections=sections)

        file = request.files["file"]
        if file.filename == "":
            error = "No file selected"
            return render_template("index.html", error=error, sections=sections)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(pdf_path)

            # Handle Deep Search
            deep_search = request.form.get("deep_search")
            topic = request.form.get("topic")
            if deep_search and topic:
                print(f"[INFO] Running deep search for topic: {topic}")
                fetch_and_add_papers(topic, max_papers=15)   # <--- limit to 15
                subprocess.run(["python", "utils/faiss_index.py"])

            # Create unique output folder per run
            run_id = str(uuid.uuid4())[:8]
            run_dir = os.path.join(RESULTS_FOLDER, run_id)
            os.makedirs(run_dir, exist_ok=True)

            # Run pipeline
            os.system(f"python utils/run_pipeline.py --pdf_path {pdf_path} --out_dir {run_dir}")

            review_file = os.path.join(run_dir, "review.txt")

            if not os.path.exists(review_file):
                with open(review_file, "w", encoding="utf-8") as f:
                    f.write(
                        "**1. Summary of the Paper**\nThis is a mock summary.\n\n"
                        "**2. Strengths**\n- Example strength.\n\n"
                        "**3. Weaknesses**\n- Example weakness.\n\n"
                        "**9. Final Recommendation**\n📌 Reject."
                    )

            with open(review_file, "r", encoding="utf-8") as f:
                review_text = f.read()
            sections = parse_review(review_text)
        else:
            error = "Invalid file type. Only PDF allowed."

    return render_template("index.html", error=error, sections=sections)


if __name__ == "__main__":
>>>>>>> 207b69cc470569cd2f15ae6043c87e1cbac834d0
    app.run(debug=True, host="0.0.0.0", port=5000)