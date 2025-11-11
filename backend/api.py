from dotenv import load_dotenv

load_dotenv()

import os
import asyncio
import tempfile
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from backend.main import RunAzureRagPipeline
from backend.auth import (
    generate_access_token,
    verify_access_token,
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")
    CORS(
        app,
        resources={r"/*": {"origins": os.environ.get("CORS_ORIGINS", "*")}},
        supports_credentials=False,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "OPTIONS", "DELETE"],
        max_age=600,
    )

    try:
        app.rag_pipeline = RunAzureRagPipeline()
        print("✅ RAG pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("❌ Failed to initialize RAG pipeline")
        app.rag_pipeline = None

    def require_auth(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1].strip()
            if not token:
                return jsonify({"error": "Missing bearer token"}), 401
            ok, payload = verify_access_token(token)
            if not ok:
                return jsonify({"error": "Invalid or expired token"}), 401
            request.jwt = payload
            return f(*args, **kwargs)

        return _wrapper

    @app.get("/health")
    def health():
        return jsonify(
            {"status": "healthy", "pipeline_initialized": app.rag_pipeline is not None}
        )

    @app.post("/auth/login")
    def login():
        """
        Simple demo login:
        Accepts JSON {email, password}; validates against env or demo users.
        Issues a JWT on success.
        """
        try:
            data = request.get_json() or {}
            email = (data.get("email") or "").strip()
            password = (data.get("password") or "").strip()

            # Demo accounts; replace with proper user store in production
            valid_users = {
                os.environ.get("ADMIN_EMAIL", "admin@xyz.com"): os.environ.get(
                    "ADMIN_PASSWORD", "admin"
                ),
                os.environ.get("USER1_EMAIL", "user1@xyz.com"): os.environ.get(
                    "USER1_PASSWORD", "user1"
                ),
            }
            if email in valid_users and password == valid_users[email]:
                is_admin = "admin" in email
                token = generate_access_token(
                    subject=email, email=email, is_admin=is_admin, expires_in_minutes=120
                )
                return jsonify({"access_token": token, "token_type": "Bearer"})
            return jsonify({"error": "Invalid credentials"}), 401
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/chat")
    @require_auth
    def chat():
        try:
            if app.rag_pipeline is None:
                return jsonify({"error": "RAG pipeline not initialized"}), 500
            data = request.get_json() or {}
            question = (data.get("question") or "").strip()
            user_id = (data.get("user_id") or "").strip()
            conversation_id = (data.get("conversation_id") or "").strip()
            session_id = (data.get("session_id") or "").strip()
            file_names = data.get("file_names") or []
            if not question:
                return jsonify({"error": "Please provide a question"}), 400

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    app.rag_pipeline.query(
                        question,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        file_names=file_names,
                        top_k=8,
                    )
                )
                # persist
                if user_id and conversation_id and session_id:
                    timestamp = datetime.now().isoformat()
                    app.rag_pipeline.save_cosmo_chat_message(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        question=question,
                        answer=response.get("answer", ""),
                        timestamp=timestamp,
                        rephrased_question=response.get("rephrased_question", ""),
                        retrieved_documents=response.get("source_documents", []),
                        source_documents=response.get("source_documents", []),
                    )
                return jsonify(
                    {
                        "answer": response.get("answer", ""),
                        "question": question,
                        "timestamp": response.get("timestamp", ""),
                        "source_documents": response.get("source_documents", []),
                        "references": response.get("references", ""),
                    }
                )
            finally:
                loop.close()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/available_files")
    @require_auth
    def available_files():
        try:
            if app.rag_pipeline is None:
                return jsonify({"error": "RAG pipeline not initialized"}), 500
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                files = loop.run_until_complete(app.rag_pipeline.get_available_files())
                return jsonify({"files": files})
            finally:
                loop.close()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/view_highlights")
    @require_auth
    def view_highlights():
        """
        For now, return the original PDF without server-side highlighting to keep API surface minimal.
        Frontend can do client-side highlighting if needed.
        """
        source = request.get_json() or {}
        filename = source.get("filename")
        if not filename:
            return jsonify({"error": "filename is required"}), 400
        try:
            blob_name = filename
            blob_name = blob_name.replace("@", "/")
            blob_data = app.rag_pipeline.get_pdf_content_from_blob(blob_name=blob_name)
            response = Response(
                blob_data,
                mimetype="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{blob_name}"'},
            )
            # best-effort page number header pass-through
            if isinstance(source.get("page_number"), list) and source.get("page_number"):
                response.headers["X-Page-Number"] = str(min(source["page_number"]))
            return response
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/upload_pdf")
    @require_auth
    def upload_pdf():
        if "pdfs" not in request.files:
            return jsonify({"error": "No PDF files provided."}), 400
        files = request.files.getlist("pdfs")
        if len(files) == 0 or len(files) > 1:
            return jsonify({"error": "You must upload only 1 files."}), 400

        blob_kwargs = {
            "from_ui": True,
            "meta_data": {
                "filename": request.form.get("field1", ""),
                "project_code": request.form.get("field2", ""),
                "label_tag": request.form.get("field3", ""),
            },
        }

        results = []
        for file in files:
            if file.filename == "":
                results.append({"filename": "", "status": "No filename"})
                continue
            if not file.filename.lower().endswith(".pdf"):
                results.append({"filename": file.filename, "status": "Not a PDF"})
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp)
                tmp_path = tmp.name
            try:
                blob_name = file.filename
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    app.rag_pipeline.run(
                        upload_to_blob=True,
                        pdf_path=tmp_path,
                        blob_name=blob_name,
                        index_document=True,
                        blob_kwargs=blob_kwargs,
                    )
                )
                loop.close()
                results.append({"filename": file.filename, "status": "Uploaded and indexed"})
                status_code = 200
            except Exception as e:
                status_code = 400
                results.append({"filename": file.filename, "status": f"Error: {str(e)}"})
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return jsonify({"results": results, "metadata": blob_kwargs["meta_data"]}), status_code

    @app.get("/view_pdf/<blob_name>")
    @require_auth
    def view_pdf(blob_name):
        try:
            if app.rag_pipeline is None:
                return jsonify({"error": "RAG pipeline not initialized"}), 500
            blob_name = blob_name.replace("@", "/")
            blob_data = app.rag_pipeline.get_pdf_content_from_blob(blob_name=blob_name)
            response = Response(
                blob_data,
                mimetype="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{blob_name}"'},
            )
            return response
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/chat_history")
    @require_auth
    def chat_history():
        try:
            user_email = request.jwt.get("email", "")
            user_id = app.rag_pipeline.generate_user_id(user_email) if hasattr(app.rag_pipeline, "generate_user_id") else user_email
            history = app.rag_pipeline.get_cosmo_user_chat_history(user_id)
            return jsonify({"history": history})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/user_sessions")
    @require_auth
    def user_sessions():
        try:
            user_email = request.jwt.get("email", "")
            user_id = app.rag_pipeline.generate_user_id(user_email) if hasattr(app.rag_pipeline, "generate_user_id") else user_email
            sessions = app.rag_pipeline.get_cosmo_user_sessions(user_id)
            return jsonify({"sessions": sessions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/session_messages")
    @require_auth
    def session_messages():
        try:
            user_email = request.jwt.get("email", "")
            user_id = app.rag_pipeline.generate_user_id(user_email) if hasattr(app.rag_pipeline, "generate_user_id") else user_email
            session_id = request.args.get("session_id")
            if not session_id:
                return jsonify({"error": "Missing session_id"}), 400
            items = app.rag_pipeline.get_cosmo_user_sessions_message(
                user_id=user_id, session_id=session_id
            )
            return jsonify({"messages": items})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/delete_session")
    @require_auth
    def delete_session():
        try:
            user_email = request.jwt.get("email", "")
            user_id = app.rag_pipeline.generate_user_id(user_email) if hasattr(app.rag_pipeline, "generate_user_id") else user_email
            data = request.get_json() or {}
            session_id = data.get("session_id")
            if not session_id:
                return jsonify({"error": "Missing session_id"}), 400
            status = app.rag_pipeline.delete_cosmo_chat_message(
                user_id=user_id, session_id=session_id
            )
            if status:
                return jsonify({"success": True})
            return jsonify({"error": "Error deleting chat message"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/speech_token")
    @require_auth
    def speech_token():
        try:
            import requests

            speech_key = os.environ.get("AZURE_SPEECH_KEY")
            speech_region = os.environ.get("AZURE_SPEECH_REGION")
            if not speech_key or not speech_region:
                return jsonify({"error": "Speech key/region not configured on server"}), 500
            token_url = f"https://{speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
            headers = {"Ocp-Apim-Subscription-Key": speech_key, "Content-Length": "0"}
            resp = requests.post(token_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return jsonify({"error": "Failed to acquire speech token", "detail": resp.text}), 502
            return jsonify({"token": resp.text, "region": speech_region})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


