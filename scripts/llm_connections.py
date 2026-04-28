from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class LLMClient:
    """
    A unified client for connecting to and querying multiple LLM providers
    using LangChain chat models.

    Supported providers:
        - OpenAI (ChatOpenAI)
        - Anthropic (ChatAnthropic)
        - Grok (X.AI via ChatXAI)
        - Gemini (Google via ChatGoogleGenerativeAI)

    Notes:
        - All interactions are synchronous.
        - API keys are expected in environment variables by default.
        - Default sampling temperature is low (0.05) for more deterministic outputs.
    """

    DEFAULT_TEMPERATURE: float = 0.05

    def __init__(self) -> None:
        self.openai_model: Optional[BaseChatModel] = None
        self.anthropic_model: Optional[BaseChatModel] = None
        self.grok_model: Optional[BaseChatModel] = None
        self.gemini_model: Optional[BaseChatModel] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_non_empty_message(message: str) -> None:
        if not isinstance(message, str):
            raise TypeError(f"message must be str, got {type(message).__name__}")
        if not message.strip():
            raise ValueError("message must be a non-empty string")

    @staticmethod
    def _prepare_messages(message: str) -> List[BaseMessage]:
        """
        Prepare a single-turn conversation consisting of one human message.
        """
        return [HumanMessage(content=message)]

    @staticmethod
    def _check_env_var(name: str) -> None:
        if not os.getenv(name):
            raise RuntimeError(
                f"Environment variable '{name}' is not set. "
                f"Set it or pass an explicit API key via **kwargs to the connect method."
            )

    @staticmethod
    def _invoke_model(model: BaseChatModel, message: str) -> str:
        """
        Invoke a LangChain chat model with a single human message.
        """
        try:
            payload = LLMClient._prepare_messages(message)
            response: BaseMessage = model.invoke(payload, max_tokens=8000)
            # LangChain BaseMessage guarantees 'content', but we guard defensively.
            content = getattr(response, "content", None)
            if isinstance(content, str):
                return content
            return str(response)
        except:
            logger.error(
                f"Failed to invoke model '{model}' with message '{message}'", )

    @staticmethod
    def _connect_model(
        *,
        model_cls: type[BaseChatModel],
        model_name: str,
        temperature: Optional[float],
        env_var: Optional[str] = None,
        log_provider: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Shared connection helper to reduce duplication across providers.
        """
        try:
            if env_var is not None:
                LLMClient._check_env_var(env_var)

            effective_temperature = (
                temperature if temperature is not None else LLMClient.DEFAULT_TEMPERATURE
            )

            model = model_cls(
                model=model_name,
                temperature=effective_temperature,
                **kwargs,
            )
            logger.info(
                "Connected to %s model '%s' with temperature=%s",
                log_provider,
                model_name,
                effective_temperature,
            )
            logging.info(f"Successfully connected the model: {model}, temperature: {temperature}")
            return model
        except Exception as ex:
            logging.error(f"Failed to connect to model: {model} with temperature: {temperature}", )

    # ------------------------------------------------------------------
    # Connection methods
    # ------------------------------------------------------------------
    def connect_openai(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Connect to OpenAI using LangChain's ChatOpenAI.

        Args:
            model: OpenAI model name. Defaults to "gpt-4o-mini".
            temperature: Override sampling temperature. Defaults to 0.05.
            **kwargs: Additional keyword arguments passed to ChatOpenAI.
        """
        self.openai_model = self._connect_model(
            model_cls=ChatOpenAI,
            model_name=model,
            temperature=temperature,
            env_var="OPENAI_API_KEY",
            log_provider="OpenAI",
            **kwargs,
        )

    def connect_anthropic(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Connect to Anthropic using LangChain's ChatAnthropic.

        Args:
            model: Anthropic model name. Defaults to "claude-sonnet-4-5-20250929".
            temperature: Override sampling temperature. Defaults to 0.05.
            **kwargs: Additional keyword arguments passed to ChatAnthropic.
        """
        self.anthropic_model = self._connect_model(
            model_cls=ChatAnthropic,
            model_name=model,
            temperature=temperature,
            max_tokens=8000,
            env_var="ANTHROPIC_API_KEY",
            log_provider="Anthropic",
            **kwargs,
        )

    def connect_grok(
        self,
        model: str = "grok-code-fast-1",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Connect to Grok (X.AI) using LangChain's ChatXAI.

        Args:
            model: Grok model name. Defaults to "grok-code-fast-1".
            temperature: Override sampling temperature. Defaults to 0.05.
            **kwargs: Additional keyword arguments passed to ChatXAI.
        """
        self.grok_model = self._connect_model(
            model_cls=ChatXAI,
            model_name=model,
            temperature=temperature,
            env_var="XAI_API_KEY",
            log_provider="Grok (X.AI)",
            **kwargs,
        )

    def connect_gemini(
        self,
        model: str = "gemini-2.5-pro",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Connect to Gemini using LangChain's ChatGoogleGenerativeAI.

        Args:
            model: Gemini model name. Defaults to "gemini-2.5-pro".
            temperature: Override sampling temperature. Defaults to 0.05.
            **kwargs: Additional keyword arguments passed to ChatGoogleGenerativeAI.
        """
        self.gemini_model = self._connect_model(
            model_cls=ChatGoogleGenerativeAI,
            model_name=model,
            temperature=temperature,
            env_var="GOOGLE_API_KEY",
            log_provider="Gemini",
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------
    def query_openai(self, message: str) -> str:
        """
        Query the connected OpenAI model with a single user message.
        """
        if self.openai_model is None:
            raise RuntimeError(
                "OpenAI model is not connected. Call connect_openai() first."
            )
        self._ensure_non_empty_message(message)
        return self._invoke_model(self.openai_model, message)

    def query_anthropic(self, message: str) -> str:
        """
        Query the connected Anthropic model with a single user message.
        """
        if self.anthropic_model is None:
            raise RuntimeError(
                "Anthropic model is not connected. Call connect_anthropic() first."
            )
        self._ensure_non_empty_message(message)
        return self._invoke_model(self.anthropic_model, message)

    def query_grok(self, message: str) -> str:
        """
        Query the connected Grok (X.AI) model with a single user message.
        """
        if self.grok_model is None:
            raise RuntimeError(
                "Grok model is not connected. Call connect_grok() first."
            )
        self._ensure_non_empty_message(message)
        return self._invoke_model(self.grok_model, message)

    def query_gemini(self, message: str) -> str:
        """
        Query the connected Gemini model with a single user message.
        """
        if self.gemini_model is None:
            raise RuntimeError(
                "Gemini model is not connected. Call connect_gemini() first."
            )
        self._ensure_non_empty_message(message)
        return self._invoke_model(self.gemini_model, message)