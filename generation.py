import json
import os
import ollama
from langchain_core.prompts import PromptTemplate

# =====================================================================
#  CONFIG & SETUP
# =====================================================================

# ---------------------------------------------------------------------
#  CRITICAL: LOCAL MODEL CONFIGURATION
# ---------------------------------------------------------------------
# We ignore whatever model name comes from the frontend 
# and force the use of your local downloaded model.
LOCAL_MODEL_NAME = "qwen2:0.5b"

# Import retrieval functions
# We assume 'retrieval.py' exists and connects to OpenSearch as you mentioned.
try:
    from retrieval import hybrid_search, keyword_search, semantic_search
except ImportError:
    print("⚠ Warning: 'retrieval' module not found. Please ensure retrieval.py is present.")
    # Fallback mocks to prevent crash if file is missing during testing
    def hybrid_search(*args, **kwargs): return []
    def keyword_search(*args, **kwargs): return []
    def semantic_search(*args, **kwargs): return []

# Define RAG prompt template
RAG_PROMPT_TEMPLATE = """You are an expert AI assistant specialized in Retrieval-Augmented Generation (RAG).

Use the following retrieved documents to answer the user's question accurately.
If the documents don't contain relevant information, explicitly state that you don't have enough information.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Be accurate and cite the source documents when relevant
- Keep the answer Explainable  but helpful
- Do not make up information not present in the documents

YOUR ANSWER:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

# =====================================================================
#  GENERATION FUNCTIONS (OLLAMA BACKEND)
# =====================================================================

def generate_with_gemini_streaming(prompt_text, model_name="ignored"):
    """
    Generate response using Local Ollama (Replacing Gemini).
    
    Args:
        prompt_text (str): The formatted prompt text
        model_name (str): Accepted for compatibility with app.py, but ignored.
    
    Returns:
        generator: Streaming text chunks
    """
    try:
        print(f"✓ Streaming with Local Ollama ({LOCAL_MODEL_NAME})...")
        
        # Stream from Ollama
        stream = ollama.chat(
            model=LOCAL_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt_text}],
            stream=True,
        )

        for chunk in stream:
            # Ollama returns chunks like {'message': {'content': 'text...'}}
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content
                        
    except Exception as e:
        error_msg = f"✗ Error with Local Ollama: {str(e)}"
        print(error_msg)
        yield error_msg


def generate_with_gemini(prompt_text, model_name="ignored"):
    """
    Generate response using Local Ollama (Non-streaming).
    
    Args:
        prompt_text (str): The formatted prompt text
        model_name (str): Accepted for compatibility with app.py, but ignored.
        
    Returns:
        str: Response text
    """
    try:
        print(f"✓ Generating with Local Ollama ({LOCAL_MODEL_NAME})...")

        response = ollama.chat(
            model=LOCAL_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt_text}],
            stream=False,
        )

        # Extract content
        return response['message']['content']

    except Exception as e:
        error_msg = f"✗ Error with Local Ollama: {str(e)}"
        print(error_msg)
        return error_msg


def generate_rag_response(query, search_type="hybrid", top_k=5, stream=False):
    """
    Generate RAG response using retrieved chunks from OpenSearch and Local Ollama.

    Args:
        query (str): User's question
        search_type (str): Type of search - 'keyword', 'semantic', or 'hybrid'
        top_k (int): Number of documents to retrieve
        stream (bool): Whether to stream the response

    Returns:
        str or generator: Generated response or streaming generator
    """
    try:
        print(f"\n{'='*80}")
        print(f"LOCAL RAG PIPELINE ({LOCAL_MODEL_NAME})")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Search Type: {search_type}")
        print(f"Top K: {top_k}")
        print(f"Streaming: {stream}\n")

        # Step 1: Retrieve relevant documents (from retrieval.py/OpenSearch)
        print("Step 1: Retrieving relevant documents...")
        if search_type == "keyword":
            results = keyword_search(query, top_k=top_k)
        elif search_type == "semantic":
            results = semantic_search(query, top_k=top_k)
        else:  # hybrid (default)
            results = hybrid_search(query, top_k=top_k)

        if not results:
            message = "⚠ No relevant documents found. Please try a different search type or refine your question."
            print(message)
            return message

        print(f"✓ Retrieved {len(results)} documents\n")

        # Step 2: Format retrieved contexts
        print("Step 2: Formatting retrieved contexts...")
        contexts = []
        for i, hit in enumerate(results, 1):
            source = hit.get("_source", {})
            content = source.get("content", "")
            content_type = source.get("content_type", "text")
            file_name = source.get("file_name", "unknown")
            score = hit.get("_score", 0)

            # Format context entry with metadata
            context_entry = f"""
[Document {i}]
Type: {content_type}
File: {file_name}
Relevance Score: {score:.4f}
Content:
{content}
"""
            contexts.append(context_entry)

        # Step 3: Combine contexts
        context_text = "\n---\n".join(contexts)
        print(f"✓ Formatted {len(contexts)} contexts\n")

        # Step 4: Format the prompt
        print(f"Step 3: Formatting prompt for {LOCAL_MODEL_NAME}...")
        prompt_text = prompt_template.format(context=context_text, question=query)
        print(f"✓ Prompt size: {len(prompt_text)} characters\n")

        # Step 5: Generate response
        print(f"Step 4: Generating response with {LOCAL_MODEL_NAME}...\n")
        print("="*80)
        
        # We pass "ignored" for model_name because we hardcoded LOCAL_MODEL_NAME above
        if stream:
            return generate_with_gemini_streaming(prompt_text, model_name="ignored")
        else:
            result = generate_with_gemini(prompt_text, model_name="ignored")
            print("="*80)
            return result

    except Exception as e:
        error_message = f"✗ Error in RAG process: {str(e)}"
        print(error_message)
        return error_message


def interactive_rag():
    """
    Interactive RAG chat interface.
    """
    print("\n" + "="*80)
    print("INTERACTIVE RAG CHATBOT (LOCAL)")
    print("="*80)
    print("\nCommands:")
    print("  'exit' or 'quit' - Exit the chatbot")
    print("  'help' - Show help information")
    print("\nSearch types: keyword, semantic, hybrid (default)\n")

    while True:
        try:
            query = input("\n📝 Enter your question (or 'exit'): ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                print("✓ Goodbye!")
                break
            
            if query.lower() == "help":
                print("\nRAG Chatbot Help:")
                print("- Ask questions about RAG and related topics")
                print(f"- The system retrieves relevant documents and uses {LOCAL_MODEL_NAME} to answer")
                continue
            
            if not query:
                print("⚠ Please enter a question")
                continue

            # Get search type preference
            search_type = input("Search type (keyword/semantic/hybrid) [default: hybrid]: ").strip().lower()
            if search_type not in ["keyword", "semantic", "hybrid"]:
                search_type = "hybrid"

            # Get top_k preference
            try:
                top_k = int(input("Number of documents to retrieve [default: 5]: ") or "5")
            except ValueError:
                top_k = 5

            # Generate response
            print("\n🔄 Processing...\n")
            response = generate_rag_response(query, search_type=search_type, top_k=top_k, stream=False)
            
            print("\n📚 RESPONSE:")
            print("-" * 80)
            print(response)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n✓ Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    # Example query
    query = "What is Retrieval-Augmented Generation and how does it work?"
    
    print("\n" + "="*80)
    print("EXAMPLE: RAG Response (Non-Streaming)")
    print("="*80)
    
    # Non-streaming response
    print("\n📚 GENERATING RESPONSE...\n")
    
    response = generate_rag_response(query, search_type="hybrid", top_k=1, stream=False)
    
    print("FINAL RESPONSE:")
    print("="*80)
    print(response)
    print("="*80)