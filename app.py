from flask import Flask, render_template, request, jsonify
import os
import shutil

from ingest import process_pdfs
from rag import ask

app = Flask(__name__)

DATA_DIR = "data"


# ✅ Upload page
@app.route("/")
@app.route("/upload")
def upload_page():
    return render_template("upload.html")


# ✅ Handle upload
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files["file"]

    file_path = os.path.join(DATA_DIR, file.filename)
    file.save(file_path)

    process_pdfs()

    return "PDF uploaded and indexed!"


# ✅ Chat page
@app.route("/chat")
def chat_page():
    return render_template("chat.html")


# ✅ Chat API
@app.route("/ask", methods=["POST"])
def chat_api():
    query = request.json["query"]
    answer = ask(query)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8005, debug=True)
