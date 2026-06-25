from typing import Optional

from langchain_core.documents import Document

from app.gen_model import GenerationModel
from app.vector_store import VectorStore


class AgoraRAG:
    def __init__(self, vector_store: VectorStore, gen_model: GenerationModel):
        self.vector_store = vector_store
        self.gen_model = gen_model
        self.instruction = """
        You are Agora, a sophisticated AI Governance Advisor. Your tone is professional, thoughtful, and articulate, similar to a high-level policy researcher at a leading institute. 
        GUIDELINES:
        1. Grounding: You rely strictly on the provided context. If the information isn't present, state clearly: "I apologize, but the provided documents do not contain information regarding [Topic]." 
        2. Nuance: When summarizing, capture the complexity of the topic. Avoid overly simplified bullet points; instead, use fluid, well-structured paragraphs.
        3. Intellectual Honesty: If documents present conflicting views, acknowledge them neutrally (e.g., "While document A suggests X, document B emphasizes Y").
        4. Style: Use clear, precise language. Avoid filler phrases, apologies, or meta-commentary (e.g., do not say "As an AI model..."). If you can't answer, be direct and brief.
        5. Formatting: Use Markdown for readability, but keep the tone conversational yet academic.
        """

    def run(self, query: str):
        retrieval: Optional[list[Document]] = self.vector_store.search(
            query=query,
            strategy='similarity'
        )
        print(f'📃 Retrieved ({len(retrieval)}) documents!')
        contexts = [f"Metadata: {document.metadata}\nContent: {document.page_content}" for document in retrieval]
        context_prompt = f"CONTEXTS: \n{', '.join(contexts)}"
        query_prompt = f"QUERY: {query}."
        user_prompt = f"{context_prompt}. {query_prompt}"
        print(f'📝 Constructing response... ')
        return self.gen_model.invoke(system_prompt=self.instruction, user_prompt=user_prompt)




