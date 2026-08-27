import os
import json
import urllib.parse

from flask import Flask, request, jsonify, session, render_template
from openai import OpenAI
from dotenv import load_dotenv

import rag

load_dotenv()  # يقرأ المتغيرات من ملف .env المحلي (لا يُرفع لـ GitHub)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "غيّر-هذا-المفتاح-لاحقاً")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB لكل ملف

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "لم يتم العثور على GROQ_API_KEY. أنشئ ملف .env في مجلد المشروع "
        "وضع بداخله السطر: GROQ_API_KEY=مفتاحك_هنا"
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """أنت مساعد ذكي ومفيد. أجب بدقة ووضوح.
- إذا توفرت لك معلومات من مستندات مرفوعة (تظهر بين ## معلومات ذات صلة)، استخدمها كمصدر أساسي للإجابة واذكر من أي ملف أخذت المعلومة.
- إذا لم تجد المعلومة في المستندات ولا تعرفها، قل ذلك صراحة بدل التخمين.
- استخدم أمثلة عملية عند الشرح.
- كن مختصراً إلا إذا طُلب منك التفصيل.
- إذا طلب المستخدم رسم أو توليد أو تصميم صورة (بأي صياغة، حتى غير مباشرة، أو كتأكيد على سؤال سابق)، استخدم أداة generate_image بدل شرح كيفية فعل ذلك يدوياً. لا تشرح للمستخدم كيف يستخدم أدوات خارجية مثل DALL-E أو Stable Diffusion — أنت قادر على توليد الصورة مباشرة عبر الأداة المتاحة لك."""

MAX_HISTORY_MESSAGES = 20

kb = rag.KnowledgeBase()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt", ".md"}

# ─────────────────────────────────────────────────────────
#  توليد الصور — Pollinations.ai (مجاني، بلا API key)
# ─────────────────────────────────────────────────────────

def generate_image_url(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """يبني رابط صورة مباشرة عبر Pollinations.ai — لا حاجة لتحميل الصورة يدوياً."""
    encoded_prompt = urllib.parse.quote(prompt)
    return (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={width}&height={height}&nologo=true"
    )


# تعريف الأداة التي يمكن لنموذج Groq استدعاءها عند الحاجة لتوليد صورة
IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "يولّد صورة جديدة بناءً على وصف نصي (prompt). استخدم هذه الأداة "
            "في أي وقت يطلب فيه المستخدم رسم شيء، تصميم صورة، أو توليد صورة، "
            "حتى لو كان الطلب غير مباشر أو تأكيداً على سؤال سابق حول ما يريد رسمه."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "وصف تفصيلي بالإنجليزية لما يجب أن تُظهره الصورة "
                        "(الأشخاص، الأشياء، الألوان، النمط الفني، الخلفية...). "
                        "استنتج هذا الوصف من كامل سياق المحادثة وليس فقط آخر رسالة."
                    )
                }
            },
            "required": ["prompt"],
        },
    },
}


@app.route("/api/generate-image", methods=["POST"])
def generate_image_endpoint():
    """Endpoint مستقل، مفيد لو أضفت زر "توليد صورة" منفصل في الواجهة لاحقاً."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "الوصف فارغ، اكتب ما تريد رسمه"}), 400
    try:
        image_url = generate_image_url(prompt)
        return jsonify({"status": "ok", "image_url": image_url, "prompt": prompt})
    except Exception as e:
        return jsonify({"error": f"فشل توليد الصورة: {str(e)}"}), 500


@app.route("/")
def index():
    session.setdefault("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "الرسالة فارغة"}), 400

    messages = session.get("messages", [{"role": "system", "content": SYSTEM_PROMPT}])

    # --- خطوة RAG: ابحث في قاعدة المعرفة عن أجزاء ذات صلة بالسؤال ---
    sources_used = []
    try:
        results = kb.search(user_message)
        context_block = rag.build_context_block(results)
    except Exception:
        context_block = None
        results = []

    outgoing_messages = list(messages)
    if context_block:
        outgoing_messages.append({"role": "system", "content": context_block})
        sources_used = sorted({r["source"] for r in results})

    outgoing_messages.append({"role": "user", "content": user_message})

    image_url = None

    try:
        # الاستدعاء الأول: نمنح النموذج إمكانية استخدام أداة توليد الصور
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=outgoing_messages,
            tools=[IMAGE_TOOL],
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1024,
        )
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if tool_calls:
            # النموذج قرر أن المستخدم يريد صورة → نفّذ الأداة فعلياً
            tool_call = tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            image_prompt = args.get("prompt", user_message)

            image_url = generate_image_url(image_prompt)

            # نرسل نتيجة تنفيذ الأداة للنموذج ليكتب رداً طبيعياً حولها
            outgoing_messages.append(choice.message)
            outgoing_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"تم توليد الصورة بنجاح بناءً على الوصف: {image_prompt}",
            })

            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=outgoing_messages,
                temperature=0.7,
                max_tokens=300,
            )
            answer = final_response.choices[0].message.content
        else:
            answer = choice.message.content

    except Exception as e:
        session["messages"] = messages
        return jsonify({"error": f"حدث خطأ في الاتصال بالنموذج: {str(e)}"}), 500

    # نحفظ في سجل المحادثة الرسالة الأصلية فقط (بدون كتلة RAG) لتوفير مساحة السياق
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": answer})

    if len(messages) > MAX_HISTORY_MESSAGES:
        messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

    session["messages"] = messages

    result = {"answer": answer, "sources": sources_used}
    if image_url:
        result["image_url"] = image_url

    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم إرسال أي ملف"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "اسم الملف فارغ"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"صيغة غير مدعومة: {ext}"}), 400

    try:
        doc_id, file_path, ext = rag.save_uploaded_file(file, file.filename)
        text = rag.extract_text(file_path, ext)

        if not text.strip():
            return jsonify({"error": "لم يتم العثور على نص قابل للقراءة في هذا الملف"}), 422

        chunks_count = kb.add_document(text, file.filename, doc_id)
    except Exception as e:
        return jsonify({"error": f"فشل معالجة الملف: {str(e)}"}), 500

    return jsonify({
        "status": "ok",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": chunks_count,
    })


@app.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": kb.list_documents()})


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    deleted = kb.delete_document(doc_id)
    if not deleted:
        return jsonify({"error": "المستند غير موجود"}), 404
    return jsonify({"status": "ok"})


@app.route("/api/reset", methods=["POST"])
def reset():
    session["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return jsonify({"status": "ok"})


# ✅ التعديل الوحيد: قراءة PORT من environment variables
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
