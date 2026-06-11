from dotenv import load_dotenv
import dotenv
from utils.audio_processor import download_youtube_audio, convert_to_wav, process_input
from core.transcriber import load_model, transcribe_all
from core.summarize import summarize, generate_title, split_transcript
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question, format_docs, get_llm
from core.vector_store import create_vector_store, load_vector_store, get_retriever

load_dotenv()

def run_pipeline(input_source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")

    chunks = process_input(input_source)
    transcript = transcribe_all(chunks, language)
    print("Raw Transcript:", transcript)

    title = generate_title(transcript)
    print("Generated Title:", title)

    summary = summarize(transcript)
    
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"🎯 Title: {result['title']}")
    print(f"\n📝 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['questions']}")
    print("=" * 60)

    # Phase 2 - Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"🤖 Assistant: {answer}\n")