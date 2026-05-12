from flask import Flask, render_template, request, jsonify
import os

from ingest import process_pdfs
from rag import ask

app = Flask(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# =========================
# Upload Page
# =========================
@app.route("/")
@app.route("/upload")
def upload_page():
    return render_template("upload.html")


# =========================
# Upload API
# =========================
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        if file.filename == "":
            return "Empty filename", 400

        file_path = os.path.join(DATA_DIR, file.filename)
        file.save(file_path)

        print("📄 File saved:", file.filename)

        process_pdfs()

        return "✅ PDF uploaded and indexed!"

    except Exception as e:
        print("❌ Upload Error:", e)
        return "Upload failed", 500


# =========================
# Chat Page
# =========================
@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


# =========================
# Chat API
# =========================
@app.route("/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"answer": "Invalid request"}), 400

        query = data.get("query", "")

        if not query:
            return jsonify({"answer": "Empty query"}), 400

        print("🧑 Query:", query)

        answer = ask(query)

        print("🤖 Answer:", answer)

        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ Chat Error:", e)
        return jsonify({"answer": "Server error"}), 500





@app.route("/ask", methods=["POST"])
def ask_api():
    data = request.get_json()
    query = data.get("query")

    answer = ask(query)

    return jsonify({
        "answer": answer
    })
# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
