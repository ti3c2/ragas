from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI

from ragas.cache import CacheInterface, cacher
from ragas.exceptions import LLMDidNotFinishException
from ragas.integrations.helicone import helicone_config
from ragas.run_config import RunConfig, add_async_retry

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue
    from llama_index.core.base.llms.base import BaseLLM
    from pydantic import BaseModel


logger = logging.getLogger(__name__)

MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]


def is_multiple_completion_supported(llm: BaseLanguageModel) -> bool:
    """Return whether the given LLM supports n-completion."""
    for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, llm_type):
            return True
    return False


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig = field(default_factory=RunConfig, repr=False)
    multiple_completion_supported: bool = field(default=False, repr=False)
    cache: t.Optional[CacheInterface] = field(default=None, repr=False)

    def __post_init__(self):
        # If a cache_backend is provided, wrap the implementation methods at construction time.
        if self.cache is not None:
            self.generate_text = cacher(cache_backend=self.cache)(self.generate_text)
            self.agenerate_text = cacher(cache_backend=self.cache)(self.agenerate_text)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 1e-8

    def is_finished(self, response: LLMResult) -> bool:
        logger.warning(
            f"is_finished not implemented for {self.__class__.__name__}. Will default to True."
        )
        return True

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    def supports_structured_output(self, output_model: "type[BaseModel]") -> bool:
        """Return True if this LLM wrapper can produce structured outputs for the given model.

        Default is False. Concrete implementations may override.
        """
        return False

    async def agenerate_prompt_structured(
        self,
        prompt: "PromptValue",
        output_model: "type[BaseModel]",
        n: int = 1,
        temperature: t.Optional[float] = None,
        callbacks: "Callbacks" = None,
    ) -> t.List[t.Any]:
        """Generate structured outputs based on the provided Pydantic model.

        Implementations should return a list of parsed objects (e.g., pydantic model instances or dicts),
        one per completion, preserving order.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement structured output generation"
        )

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Generate text using the given event loop with retry logic for LLM finish failures."""

        async def _attempt_generation():
            """Single attempt at generation with finish check."""
            temp = temperature if temperature is not None else self.get_temperature(n)
            result = await self.agenerate_text(
                prompt=prompt,
                n=n,
                temperature=temp,
                stop=stop,
                callbacks=callbacks,
            )
            # Check there are no max_token issues
            if not self.is_finished(result):
                raise LLMDidNotFinishException()
            return result

        # Wrap the ENTIRE generation + finish check in retry logic
        generate_with_retry = add_async_retry(_attempt_generation, self.run_config)

        try:
            return await generate_with_retry()
        except LLMDidNotFinishException as e:
            # Add more context to the error after all retries exhausted
            logger.error(
                f"LLM failed to complete generation after {self.run_config.max_retries} retries. "
                f"This may indicate GPU memory pressure, rate limits, or model issues. "
                f"Consider reducing max_workers or increasing max_tokens."
            )
            raise e


class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    def __init__(
        self,
        langchain_llm: BaseLanguageModel[BaseMessage],
        run_config: t.Optional[RunConfig] = None,
        is_finished_parser: t.Optional[t.Callable[[LLMResult], bool]] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        self.is_finished_parser = is_finished_parser

    def is_finished(self, response: LLMResult) -> bool:
        """
        Parse the response to check if the LLM finished by checking the finish_reason
        or stop_reason. Supports OpenAI and Vertex AI models.
        """
        if self.is_finished_parser is not None:
            return self.is_finished_parser(response)
        # if no parser is provided default to our own

        is_finished_list = []
        for g in response.flatten():
            resp = g.generations[0][0]
            if resp.generation_info is not None:
                # generation_info is provided - so we parse that
                finish_reason = resp.generation_info.get("finish_reason")
                if finish_reason is not None:
                    # OpenAI uses "stop"
                    # Vertex AI uses "STOP" or "MAX_TOKENS"
                    # WatsonX AI uses "eos_token"
                    is_finished_list.append(
                        finish_reason in ["stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )

                # provied more conditions here
                # https://github.com/explodinggradients/ragas/issues/1548

            # if generation_info is empty, we parse the response_metadata
            # this is less reliable

            elif (
                isinstance(resp, ChatGeneration)
                and t.cast(ChatGeneration, resp).message is not None
            ):
                resp_message: BaseMessage = t.cast(ChatGeneration, resp).message
                if resp_message.response_metadata.get("finish_reason") is not None:
                    finish_reason = resp_message.response_metadata.get("finish_reason")
                    is_finished_list.append(
                        finish_reason in ["stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )
                elif resp_message.response_metadata.get("stop_reason") is not None:
                    stop_reason = resp_message.response_metadata.get("stop_reason")
                    is_finished_list.append(
                        stop_reason
                        in ["end_turn", "stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )
            # default to True
            else:
                is_finished_list.append(True)
        return all(is_finished_list)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # figure out the temperature to set
        old_temperature: float | None = None
        if temperature is None:
            temperature = self.get_temperature(n=n)
        if hasattr(self.langchain_llm, "temperature"):
            self.langchain_llm.temperature = temperature  # type: ignore
            old_temperature = temperature

        if is_multiple_completion_supported(self.langchain_llm):
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

        # reset the temperature to the original value
        if old_temperature is not None:
            self.langchain_llm.temperature = old_temperature  # type: ignore

        return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # handle temperature
        old_temperature: float | None = None
        if temperature is None:
            temperature = self.get_temperature(n=n)
        if hasattr(self.langchain_llm, "temperature"):
            self.langchain_llm.temperature = temperature  # type: ignore
            old_temperature = temperature

        # handle n
        if hasattr(self.langchain_llm, "n"):
            self.langchain_llm.n = n  # type: ignore
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

        # reset the temperature to the original value
        if old_temperature is not None:
            self.langchain_llm.temperature = old_temperature  # type: ignore

        return result

    def supports_structured_output(self, output_model: "type[BaseModel]") -> bool:
        # Prefer LangChain's with_structured_output, available on many chat models
        return hasattr(self.langchain_llm, "with_structured_output")

    async def agenerate_prompt_structured(
        self,
        prompt: "PromptValue",
        output_model: "type[BaseModel]",
        n: int = 1,
        temperature: t.Optional[float] = None,
        adjust_temperature: bool = False,
        callbacks: "Callbacks" = None,
    ) -> t.List[t.Any]:
        if not self.supports_structured_output(output_model):
            raise NotImplementedError(
                "Structured outputs are not supported by the underlying LangChain LLM."
            )

        async def _attempt_structured_generation():
            """Single attempt at structured generation."""
            # Adjust temperature if supported
            old_temperature: float | None = None
            temp = temperature if temperature is not None else self.get_temperature(n=n)
            if hasattr(self.langchain_llm, "temperature"):
                old_temperature = self.langchain_llm.temperature
                if adjust_temperature:
                    self.langchain_llm.temperature = temp

            structured_llm = self.langchain_llm.with_structured_output(output_model)

            # Prepare the input for invoke/ainvoke: prefer messages for chat, else string
            invoke_input: t.Any
            if hasattr(prompt, "to_messages"):
                try:
                    invoke_input = prompt.to_messages()  # type: ignore[attr-defined]
                except Exception:
                    # Fallback to string
                    invoke_input = prompt.to_string()
            else:
                # StringPromptValue exposes .to_string()/.text
                invoke_input = getattr(prompt, "text", None) or prompt.to_string()

            results: t.List[t.Any]
            try:
                # Prefer batch invocation if available; pass callbacks via config to avoid duplication
                if hasattr(structured_llm, "abatch") and n > 1:
                    batch_inputs = [invoke_input] * n
                    if callbacks is not None:
                        results = await structured_llm.abatch(batch_inputs, config={"callbacks": callbacks})
                    else:
                        results = await structured_llm.abatch(batch_inputs)
                else:
                    if n == 1:
                        if callbacks is not None:
                            single = await structured_llm.ainvoke(invoke_input, config={"callbacks": callbacks})
                        else:
                            single = await structured_llm.ainvoke(invoke_input)
                        results = [single]
                    else:
                        # Fallback: parallel single invocations
                        coros = []
                        for _ in range(n):
                            if callbacks is not None:
                                coros.append(structured_llm.ainvoke(invoke_input, config={"callbacks": callbacks}))  # type: ignore[attr-defined]
                            else:
                                coros.append(structured_llm.ainvoke(invoke_input))
                        results = list(await asyncio.gather(*coros))
            finally:
                # reset the temperature to the original value
                if old_temperature is not None and hasattr(self.langchain_llm, "temperature"):
                    self.langchain_llm.temperature = old_temperature

            return list(results)

        # Apply retry logic to structured generation
        generate_structured_with_retry = add_async_retry(_attempt_structured_generation, self.run_config)
        return await generate_structured_with_retry()

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # configure if using OpenAI API
        if isinstance(self.langchain_llm, BaseOpenAI) or isinstance(
            self.langchain_llm, ChatOpenAI
        ):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.langchain_llm.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(langchain_llm={self.langchain_llm.__class__.__name__}(...))"


class LlamaIndexLLMWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs
    """

    def __init__(
        self,
        llm: BaseLLM,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.llm = llm

        try:
            self._signature = type(self.llm).__name__.lower()
        except AttributeError:
            self._signature = ""

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def check_args(
        self,
        n: int,
        temperature: float,
        stop: t.Optional[t.List[str]],
        callbacks: Callbacks,
    ) -> dict[str, t.Any]:
        if n != 1:
            logger.warning("n values greater than 1 not support for LlamaIndex LLMs")
        if temperature != 1e-8:
            logger.info("temperature kwarg passed to LlamaIndex LLM")
        if stop is not None:
            logger.info("stop kwarg passed to LlamaIndex LLM")
        if callbacks is not None:
            logger.info(
                "callbacks not supported for LlamaIndex LLMs, ignoring callbacks"
            )
        if self._signature in ["anthropic", "bedrock"]:
            return {"temperature": temperature}
        else:
            return {
                "n": n,
                "temperature": temperature,
                "stop": stop,
            }

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = self.llm.complete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)

        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = await self.llm.acomplete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm.__class__.__name__}(...))"


def llm_factory(
    model: str = "gpt-4o-mini",
    run_config: t.Optional[RunConfig] = None,
    default_headers: t.Optional[t.Dict[str, str]] = None,
    base_url: t.Optional[str] = None,
) -> BaseRagasLLM:
    """
    Create and return a BaseRagasLLM instance. Used for running default LLMs used
    in Ragas (OpenAI).

    Parameters
    ----------
    model : str, optional
        The name of the model to use, by default "gpt-4o-mini".
    run_config : RunConfig, optional
        Configuration for the run, by default None.
    default_headers : dict of str, optional
        Default headers to be used in API requests, by default None.
    base_url : str, optional
        Base URL for the API, by default None.

    Returns
    -------
    BaseRagasLLM
        An instance of BaseRagasLLM configured with the specified parameters.
    """
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout

    # if helicone is enabled, use the helicone
    if helicone_config.is_enabled:
        default_headers = helicone_config.default_headers()
        base_url = helicone_config.base_url

    openai_model = ChatOpenAI(
        model=model, timeout=timeout, default_headers=default_headers, base_url=base_url
    )
    return LangchainLLMWrapper(openai_model, run_config)
