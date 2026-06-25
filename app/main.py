from app.gen_model import PhiMiniModel
from app.rag import AgoraRAG
from app.vector_store import ChromaDB

if __name__ == "__main__":
    vector_store = ChromaDB()
    gen_model = PhiMiniModel()
    rag = AgoraRAG(vector_store, gen_model)

    print(f"\nWelcome to AGORA AI Governance Chat Assistant! "
          f"(Powered by HuggingFace MpNet-Base-v2 & Ollama Mini 3.5)")

    is_running = True
    while is_running:
        request = input("Enter question ('quit' to exit): ")
        if request == "quit":
            is_running = False
            print(f"Thank you for using AGORA AI Chat Assistant and see you again!")
            break
        print("🔍 Searching documents...")
        respond = rag.run(request)
        print(f'Respond: \n{respond}\n')


