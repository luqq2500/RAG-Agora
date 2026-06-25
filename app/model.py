from abc import ABC, abstractmethod

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

class GenerationModel(ABC):
    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str)->str:
        pass
    @abstractmethod
    def invoke_stream(self, system_prompt: str, user_prompt: str):
        pass

class OllamaModel(GenerationModel):
    def __init__(self, model: str='phi3.5:latest', temperature: float=0.3):
        try:
            self.model = ChatOllama(model=model, temperature=temperature) # maximize context, minimize creative
            self.model.invoke("hi")
        except Exception as e:
            raise RuntimeError(f"Ollama model {model} is not reachable: {e}")

    def invoke(self, system_prompt: str, user_prompt: str, stream: bool=True) -> str :
        prompts = [SystemMessage(system_prompt),HumanMessage(user_prompt)]
        return self.model.invoke(prompts).content

    def invoke_stream(self, system_prompt: str, user_prompt: str):
        prompts = [SystemMessage(system_prompt),HumanMessage(user_prompt)]
        for chunk in self.model.stream(prompts):
            yield chunk.content



