import time
import gradio as gr
# Ensure 'generation.py' is in the same directory and has these functions
from generation import generate_rag_response, generate_with_gemini_streaming


# =====================================================================
#   MAIN PROCESSING FUNCTIONS
# =====================================================================
def process_query(query, search_type, model_type, stream, top_k):
    """Process and stream or return the final response."""
    if not query.strip():
        yield "Please enter a question."
        return

    try:
        if stream:
            full_response = ""
            for chunk in generate_rag_response_with_streaming(query, search_type, model_type, top_k):
                full_response += chunk
                time.sleep(0.005) # Slight delay for UI smoothness
                yield full_response
        else:
            result = generate_rag_response(query, search_type=search_type, top_k=top_k)
            yield result

    except Exception as e:
        yield f"❌ Error: {str(e)}"


def generate_rag_response_with_streaming(query, search_type, model_type, top_k):
    """Generates streaming RAG output from Gemini."""
    try:
        print("\n" + "="*80)
        print("STREAMING RAG PIPELINE")
        print("="*80)

        # Import inside function to avoid circular imports or initialization issues
        # Ensure retrieval.py exists with these functions
        try:
            from retrieval import hybrid_search, keyword_search, semantic_search
            
            # Retrieval step
            if search_type == "keyword":
                results = keyword_search(query, top_k=top_k)
            elif search_type == "semantic":
                results = semantic_search(query, top_k=top_k)
            else:
                results = hybrid_search(query, top_k=top_k)
                
        except ImportError:
             # Fallback for testing if retrieval module is missing
            print("⚠ 'retrieval' module not found. Using mock data.")
            results = [{
                "_source": {
                    "content": "Retrieval module missing. This is mock content.",
                    "file_name": "mock.pdf",
                    "content_type": "text",
                },
                "_score": 1.0
            }]

        if not results:
            yield "⚠ No relevant documents found."
            return

        # Build context block
        contexts = []
        for i, hit in enumerate(results, 1):
            src = hit.get("_source", {})
            content = src.get("content", "")
            file = src.get("file_name", "unknown")
            ctype = src.get("content_type", "text")
            score = hit.get("_score", 0)

            contexts.append(
                f"""
[Document {i}]
Type: {ctype}
File: {file}
Relevance Score: {score:.4f}
Content:
{content}
"""
            )

        context_text = "\n---\n".join(contexts)

        # Prompt
        prompt_text = f"""
You are an expert AI assistant specializing in Retrieval-Augmented Generation.

Use the following retrieved documents to answer the user's question.
If the documents lack information, you must acknowledge it.

RETRIEVED DOCUMENTS:
{context_text}

USER QUESTION:
{query}

YOUR ANSWER:
"""

        print("Prompt ready. Streaming model response...")

        # Stream from Gemini
        for chunk in generate_with_gemini_streaming(prompt_text, model_name=model_type):
            yield chunk

    except Exception as e:
        yield f"✗ Error in streaming pipeline: {e}"


# =====================================================================
#   UI COMPONENTS
# =====================================================================
def create_ui():
    css = """
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
        font-weight: bold;
    }
    .model-card:hover {
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    """

    # Removed css=css from Blocks constructor to fix TypeError
    with gr.Blocks(title="LocalRAG Q&A System") as demo:
        
        # Inject CSS directly via HTML component
        gr.HTML(f"<style>{css}</style>")

        # Header
        gr.HTML("""
        <div style="text-align:center;margin-bottom:30px;">
            <h1 class="header-title">📚 LocalRAG Q&A System</h1>
            <p style="color:#666;font-size:1.1em;">
                Ask questions about Retrieval-Augmented Generation and get accurate AI answers.
            </p>
        </div>
        """)

        with gr.Row():
            # LEFT SIDE — INPUTS
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Your Question")
                query_input = gr.Textbox(
                    placeholder="Ask anything about RAG...",
                    lines=4,
                    container=False,
                    show_label=False
                )

                gr.Markdown("### ⚙️ Configuration")
                with gr.Row():
                    with gr.Column():
                        search_type = gr.Radio(
                            ["keyword", "semantic", "hybrid"],
                            value="hybrid",
                            label="🔍 Search Method"
                        )
                    with gr.Column():
                        model_type = gr.Radio(
                            [
                                ("Gemini 2.0 Flash", "gemini-2.0-flash"),
                                ("Gemini 2.0 Pro", "gemini-2.0-pro"),
                                ("Gemini 1.5 Pro", "gemini-1.5-pro"),
                            ],
                            value="gemini-2.0-flash",
                            label="🤖 Model"
                        )

                with gr.Accordion("⚙️ Advanced Options", open=False):
                    # FIX: Explicitly use 'label' for the text, as positional arg 1 is 'value'
                    stream_checkbox = gr.Checkbox(label="🔄 Stream Response", value=True)
                    top_k_slider = gr.Slider(1, 10, value=5, step=1, label="📄 Documents to Retrieve")

                submit_btn = gr.Button("✨ Generate Answer", variant="primary", scale=2)
                clear_btn = gr.Button("🔄 Clear")

            # RIGHT SIDE — OUTPUT
            with gr.Column(scale=2):
                gr.Markdown("### 💡 Answer")
                output = gr.Textbox(
                    lines=25,
                    container=False,
                    interactive=False,
                    show_label=False
                )

                with gr.Row():
                    status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                    token_text = gr.Textbox(label="Characters", value="0", interactive=False)

        # Submit Logic
        def on_submit(query, search_type, model_type, stream, top_k):
            if not query.strip():
                yield "⚠️ Please enter a question.", "Error", "0"
                return
            
            # Yield initial status
            yield "", "Processing...", "0"
            
            full_text = ""
            for chunk in process_query(query, search_type, model_type, stream, top_k):
                full_text = chunk
                # Yield update
                yield full_text, "Generating...", str(len(full_text))
            
            # Final yield
            yield full_text, "Done", str(len(full_text))

        # Clear Logic
        def clear_all():
            return "", "", "Ready", "0"

        submit_btn.click(
            on_submit,
            inputs=[query_input, search_type, model_type, stream_checkbox, top_k_slider],
            outputs=[output, status_text, token_text]
        )

        clear_btn.click(
            clear_all,
            outputs=[query_input, output, status_text, token_text]
        )

        # Example Questions
        gr.Markdown("### 📝 Example Questions")
        gr.Examples(
            examples=[
                ["How does RAG work?", "hybrid", "gemini-2.0-flash"],
                ["Advantages of RAG vs fine-tuning?", "semantic", "gemini-2.0-pro"],
                ["Explain retrieval steps in RAG", "keyword", "gemini-2.0-flash"]
            ],
            inputs=[query_input, search_type, model_type]
        )

    return demo


# =====================================================================
#   RUN APP
# =====================================================================
if __name__ == "__main__":
    app = create_ui()
    app.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )