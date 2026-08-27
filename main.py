import os
import re
import json
import urllib.parse

from flask import Flask, request, jsonify, session, render_template
from openai import OpenAI
from dotenv import load_dotenv

import rag


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "غيّر-هذا-المفتاح-لاحقاً"
)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# =========================================================
# GROQ
# =========================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "لم يتم العثور على GROQ_API_KEY. "
        "أنشئ متغير GROQ_API_KEY في Railway."
    )


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


MODEL_NAME = "openai/gpt-oss-120b"


# =========================================================
# IDENTITY / PERSONALITY
# =========================================================

SYSTEM_PROMPT = """
أنت Wiam Dev AI، مساعد ذكاء اصطناعي ذكي وودود.

هويتك الأساسية:
- اسمك: Wiam Dev AI
- أنت المساعد الذكي الخاص بمشروع Wiam Dev.
- تم تطوير وبرمجة هذا البوت بواسطة العبقرية Wiam Dev 🧠💜.

=========================================================
هوية المطورة
=========================================================

إذا سأل المستخدم بأي طريقة عن:
- من برمجك؟
- من طورك؟
- من صنعك؟
- من أنشأك؟
- من صاحبة البوت؟
- من صاحبة المشروع؟
- من المبرمج؟
- من المطورة؟
- من وراء هذا البوت؟
- هذا البوت تابع لمن؟
- من قام بتطويرك؟
- مين برمجك؟
- مين طورك؟
- مين صنعك؟
- مين وراك؟
- مين صاحبة البوت؟
- مين المبرمجة؟
- من أنشأ هذا الذكاء الاصطناعي؟
- كيف تم تطويرك؟
- من المسؤول عن تطويرك؟
- Who created you?
- Who made you?
- Who developed you?
- Who programmed you?
- Who built you?
- Who is your developer?
- Who is behind you?

أجب بوضوح وفخر:

"تم تطويري وبرمجتي بواسطة العبقرية Wiam Dev 🧠💜"

إذا كان المستخدم يتحدث بالإنجليزية:

"I was developed and programmed by the brilliant Wiam Dev 🧠💜"

إذا كان السؤال غير مباشر لكنه يقصد نفس المعنى، افهم المقصود وأجب عن Wiam Dev.

لا تنسب تطوير هذا التطبيق إلى شخص آخر.

لا تقل إن شركة أخرى هي التي طورت هذا التطبيق.

قد يستخدم التطبيق نموذجاً لغوياً من مزود API خارجي، لكن مطورة هذا التطبيق وهويته هي Wiam Dev.

=========================================================
شخصية Wiam Dev AI
=========================================================

- كن ذكيًا.
- كن ودودًا.
- كن احترافيًا.
- استخدم اللغة التي يستخدمها المستخدم.
- إذا كان السؤال بالعربية فأجب بالعربية.
- إذا كان السؤال بالإنجليزية فأجب بالإنجليزية.
- يمكن استخدام الإيموجي باعتدال.
- لا تجعل كل إجابة طويلة.
- إذا كان السؤال بسيطًا، أعط إجابة مباشرة.
- إذا كان السؤال تقنيًا، اشرح بطريقة منظمة مع أمثلة عملية.

=========================================================
RAG / المستندات
=========================================================

إذا توفرت معلومات من مستندات مرفوعة
(تظهر بين ## معلومات ذات صلة)،
استخدمها كمصدر أساسي للإجابة.

اذكر اسم الملف عندما تكون الإجابة مبنية على مستند مرفوع.

إذا لم تجد المعلومة في المستندات ولا تعرفها،
قل ذلك صراحة بدل التخمين.

=========================================================
الصور
=========================================================

إذا طلب المستخدم رسم أو توليد أو تصميم صورة،
يجب عليك استدعاء أداة generate_image فعلياً.

ممنوع منعاً باتاً أن تكتب بنفسك صياغة Markdown لصورة مثل:

![وصف](رابط)

ولا تكتب روابط ملفات وهمية مثل:

/mnt/data/...
attachment://...

أنت لا تملك القدرة على إنشاء الصور بالكتابة المباشرة.

الطريقة الوحيدة لإنشاء صورة هي استدعاء أداة generate_image.
"""


MAX_HISTORY_MESSAGES = 20


# =========================================================
# KNOWLEDGE BASE
# =========================================================

kb = rag.KnowledgeBase()


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".txt",
    ".md"
}


# =========================================================
# IMAGE GENERATION
# =========================================================

def generate_image_url(
    prompt: str,
    width: int = 1024,
    height: int = 1024
) -> str:

    encoded_prompt = urllib.parse.quote(prompt)

    return (
        f"https://image.pollinations.ai/prompt/"
        f"{encoded_prompt}"
        f"?width={width}"
        f"&height={height}"
        f"&nologo=true"
        f"&seed={abs(hash(prompt)) % 100000}"
    )


# =========================================================
# FAKE IMAGE MARKDOWN
# =========================================================

FAKE_IMAGE_MARKDOWN_PATTERN = re.compile(
    r"!\[[^\]]*\]\(([^)]+)\)"
)


def strip_fake_image_markdown(text: str) -> str:

    return FAKE_IMAGE_MARKDOWN_PATTERN.sub(
        "",
        text
    ).strip()


def extract_fake_image_prompt(
    answer_text: str,
    fallback_prompt: str
) -> str | None:

    match = FAKE_IMAGE_MARKDOWN_PATTERN.search(
        answer_text
    )

    if not match:
        return None

    url = match.group(1)

    if "image.pollinations.ai" in url:
        return None

    return fallback_prompt


# =========================================================
# IMAGE TOOL
# =========================================================

IMAGE_TOOL = {
    "type": "function",

    "function": {

        "name": "generate_image",

        "description": (
            "يولّد صورة حقيقية بناءً على وصف نصي "
            "ويرجع رابطها الفعلي. "
            "استخدمها في أي وقت يطلب فيه المستخدم "
            "رسم شيء أو تصميم أو توليد صورة."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "prompt": {

                    "type": "string",

                    "description": (
                        "وصف تفصيلي بالإنجليزية لما يجب أن "
                        "تُظهره الصورة: الأشخاص، الأشياء، "
                        "الألوان، النمط الفني، الخلفية..."
                    )

                }

            },

            "required": [
                "prompt"
            ]

        }

    }

}


# =========================================================
# IDENTITY QUESTION DETECTION
# =========================================================

def is_wiam_dev_identity_question(
    message: str
) -> bool:

    """
    يكتشف الأسئلة التي تسأل عن مطور/مبرمج البوت.

    الهدف هو ضمان أن الأسئلة المتعلقة بهوية المطور
    تحصل دائماً على إجابة Wiam Dev.
    """

    if not message:
        return False

    text = message.lower().strip()

    # إزالة بعض علامات الترقيم
    normalized = re.sub(
        r"[؟?!.,،:;؛\-_/]+",
        " ",
        text
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()

    identity_patterns = [

        # العربية
        r"من برمجك",
        r"مين برمجك",
        r"من طورك",
        r"مين طورك",
        r"من صنعك",
        r"مين صنعك",
        r"من انشأك",
        r"مين انشاك",
        r"من أنشأك",
        r"مين أنشأك",
        r"من مطورك",
        r"مين مطورك",
        r"من مبرمجك",
        r"مين مبرمجك",
        r"من المبرمج",
        r"مين المبرمج",
        r"من المطور",
        r"مين المطور",
        r"من طور هذا البوت",
        r"مين طور هذا البوت",
        r"من صنع هذا البوت",
        r"مين صنع هذا البوت",
        r"من انشأ هذا البوت",
        r"مين انشأ هذا البوت",
        r"من أنشأ هذا البوت",
        r"مين أنشأ هذا البوت",
        r"من برمج هذا البوت",
        r"مين برمج هذا البوت",
        r"من صاحبة البوت",
        r"مين صاحبة البوت",
        r"من صاحب البوت",
        r"مين صاحب البوت",
        r"من صاحبة المشروع",
        r"مين صاحبة المشروع",
        r"من صاحب المشروع",
        r"مين صاحب المشروع",
        r"من وراءك",
        r"مين وراك",
        r"من وراك",
        r"مين خلفك",
        r"من خلفك",
        r"البوت تابع لمن",
        r"هذا البوت تابع لمن",
        r"من المسؤول عن تطويرك",
        r"مين المسؤول عن تطويرك",
        r"من قام بتطويرك",
        r"مين قام بتطويرك",
        r"من قام ببرمجتك",
        r"مين قام ببرمجتك",

        # English
        r"who created you",
        r"who made you",
        r"who developed you",
        r"who programmed you",
        r"who built you",
        r"who is your developer",
        r"who is behind you",
        r"who is behind this bot",
        r"who made this bot",
        r"who developed this bot",
        r"who programmed this bot",
        r"who built this bot"

    ]

    for pattern in identity_patterns:

        if re.search(
            pattern,
            normalized,
            re.IGNORECASE
        ):

            return True

    return False


# =========================================================
# IMAGE TOOL CALL
# =========================================================

def call_model_with_image_tool(
    outgoing_messages
):

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=outgoing_messages,

        tools=[
            IMAGE_TOOL
        ],

        tool_choice="auto",

        temperature=0.7,

        max_tokens=1024

    )

    choice = response.choices[0]

    tool_calls = choice.message.tool_calls


    # =====================================================
    # IMAGE TOOL CALLED
    # =====================================================

    if tool_calls:

        tool_call = tool_calls[0]

        args = json.loads(
            tool_call.function.arguments
        )

        image_prompt = (
            args.get("prompt")
            or outgoing_messages[-1]["content"]
        )

        image_url = generate_image_url(
            image_prompt
        )

        answer = (
            "تفضل، هذه الصورة التي طلبتها 🎨"
        )

        return answer, image_url


    # =====================================================
    # FAKE IMAGE MARKDOWN
    # =====================================================

    raw_answer = (
        choice.message.content
        or ""
    )

    user_last_message = (
        outgoing_messages[-1]["content"]
    )

    fake_prompt = extract_fake_image_prompt(
        raw_answer,
        user_last_message
    )


    if fake_prompt:

        image_url = generate_image_url(
            fake_prompt
        )

        answer = (
            "تفضل، هذه الصورة التي طلبتها 🎨"
        )

        return answer, image_url


    # =====================================================
    # NORMAL TEXT
    # =====================================================

    clean_answer = (
        strip_fake_image_markdown(
            raw_answer
        )
        or raw_answer
    )

    return clean_answer, None


# =========================================================
# GENERATE IMAGE ENDPOINT
# =========================================================

@app.route(
    "/api/generate-image",
    methods=["POST"]
)
def generate_image_endpoint():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    prompt = (
        data.get("prompt")
        or ""
    ).strip()


    if not prompt:

        return jsonify({
            "error":
                "الوصف فارغ، اكتب ما تريد رسمه"
        }), 400


    try:

        image_url = generate_image_url(
            prompt
        )

        return jsonify({

            "status":
                "ok",

            "image_url":
                image_url,

            "prompt":
                prompt

        })


    except Exception as e:

        return jsonify({

            "error":
                f"فشل توليد الصورة: {str(e)}"

        }), 500


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    session.setdefault(
        "messages",
        [
            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT
            }
        ]
    )

    return render_template(
        "index.html"
    )


# =========================================================
# CHAT
# =========================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )


    user_message = (
        data.get("message")
        or ""
    ).strip()


    if not user_message:

        return jsonify({
            "error":
                "الرسالة فارغة"
        }), 400


    # =====================================================
    # Wiam Dev Identity
    # =====================================================

    if is_wiam_dev_identity_question(
        user_message
    ):

        answer = (
            "تم تطويري وبرمجتي بواسطة "
            "العبقرية Wiam Dev 🧠💜"
        )


        messages = session.get(
            "messages",
            [
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT
                }
            ]
        )


        messages.append({

            "role":
                "user",

            "content":
                user_message

        })


        messages.append({

            "role":
                "assistant",

            "content":
                answer

        })


        if len(messages) > MAX_HISTORY_MESSAGES:

            messages = [
                messages[0]
            ] + messages[
                -(MAX_HISTORY_MESSAGES - 1):
            ]


        session["messages"] = messages


        return jsonify({

            "answer":
                answer,

            "sources":
                []

        })


    # =====================================================
    # GET HISTORY
    # =====================================================

    messages = session.get(

        "messages",

        [
            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT
            }
        ]

    )


    # =====================================================
    # RAG SEARCH
    # =====================================================

    sources_used = []


    try:

        results = kb.search(
            user_message
        )

        context_block = (
            rag.build_context_block(
                results
            )
        )


    except Exception:

        context_block = None

        results = []


    # =====================================================
    # BUILD OUTGOING MESSAGES
    # =====================================================

    outgoing_messages = list(
        messages
    )


    if context_block:

        outgoing_messages.append({

            "role":
                "system",

            "content":
                context_block

        })


        sources_used = sorted({

            r["source"]

            for r in results

        })


    outgoing_messages.append({

        "role":
            "user",

        "content":
            user_message

    })


    # =====================================================
    # CALL MODEL
    # =====================================================

    try:

        answer, image_url = (
            call_model_with_image_tool(
                outgoing_messages
            )
        )


    except Exception as e:

        session["messages"] = (
            messages
        )

        return jsonify({

            "error":
                "حدث خطأ في الاتصال بالنموذج: "
                f"{str(e)}"

        }), 500


    # =====================================================
    # SAVE CONVERSATION
    # =====================================================

    messages.append({

        "role":
            "user",

        "content":
            user_message

    })


    messages.append({

        "role":
            "assistant",

        "content":
            answer

    })


    if len(messages) > MAX_HISTORY_MESSAGES:

        messages = [

            messages[0]

        ] + messages[
            -(MAX_HISTORY_MESSAGES - 1):
        ]


    session["messages"] = messages


    # =====================================================
    # RESPONSE
    # =====================================================

    result = {

        "answer":
            answer,

        "sources":
            sources_used

    }


    if image_url:

        result["image_url"] = (
            image_url
        )


    return jsonify(
        result
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        return jsonify({

            "error":
                "لم يتم إرسال أي ملف"

        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({

            "error":
                "اسم الملف فارغ"

        }), 400


    ext = os.path.splitext(
        file.filename
    )[1].lower()


    if ext not in ALLOWED_EXTENSIONS:

        return jsonify({

            "error":
                f"صيغة غير مدعومة: {ext}"

        }), 400


    try:

        doc_id, file_path, ext = (
            rag.save_uploaded_file(
                file,
                file.filename
            )
        )


        text = rag.extract_text(
            file_path,
            ext
        )


        if not text.strip():

            return jsonify({

                "error":
                    "لم يتم العثور على نص "
                    "قابل للقراءة في هذا الملف"

            }), 422


        chunks_count = kb.add_document(
            text,
            file.filename,
            doc_id
        )


    except Exception as e:

        return jsonify({

            "error":
                f"فشل معالجة الملف: {str(e)}"

        }), 500


    return jsonify({

        "status":
            "ok",

        "doc_id":
            doc_id,

        "filename":
            file.filename,

        "chunks":
            chunks_count

    })


# =========================================================
# LIST DOCUMENTS
# =========================================================

@app.route(
    "/api/documents",
    methods=["GET"]
)
def list_documents():

    return jsonify({

        "documents":
            kb.list_documents()

    })


# =========================================================
# DELETE DOCUMENT
# =========================================================

@app.route(
    "/api/documents/<doc_id>",
    methods=["DELETE"]
)
def delete_document(
    doc_id
):

    deleted = (
        kb.delete_document(
            doc_id
        )
    )


    if not deleted:

        return jsonify({

            "error":
                "المستند غير موجود"

        }), 404


    return jsonify({

        "status":
            "ok"

    })


# =========================================================
# RESET CHAT
# =========================================================

@app.route(
    "/api/reset",
    methods=["POST"]
)
def reset():

    session["messages"] = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT
        }

    ]


    return jsonify({

        "status":
            "ok"

    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )