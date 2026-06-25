from abc import ABC, abstractmethod

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

class GenerationModel(ABC):
    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str)->str:
        pass

class PhiMiniModel(GenerationModel):
    def __init__(self, model: str='phi3.5', temperature: float=0.3):
        self.model = ChatOllama(model=model, temperature=temperature) # maximize context, minimize creative
    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        prompts = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt),
        ]
        return self.model.invoke(
            prompts
        ).content




