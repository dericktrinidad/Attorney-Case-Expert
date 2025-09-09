from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Optional, Sequence

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from openinference.semconv.trace import SpanAttributes

def instrument(
    *,
    kind: str,
    name: Optional[str] = None,
    attrs: Optional[Dict[str, Any]] = None,
    input_getter: Optional[Callable[..., Any]] = None,
    output_getter: Optional[Callable[[Any], Any]] = None,
    docs_getter: Optional[Callable[[Any], Sequence[Dict[str, Any]]]] = None,
    tracer_name: str = "cinemaRAG",
    max_doc_events: int = 5,
    ):
    """
    Decorate a function to create an OpenTelemetry span annotated with OpenInference attrs.
    Works with Arize Phoenix when your OTLP exporter points at Phoenix.

    Example:
        @instrument(
            kind="RETRIEVER",
            name="WeaviateRetriever.retrieve",
            input_getter=lambda self, query, **_: query,
            docs_getter=lambda result: result,  # expects list[dict|str]
        )
        def retrieve(self, query: str) -> list[dict]: ...
    """
    def deco(func: Callable):
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(tracer_name)
            span_attrs = {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind}
            if attrs:
                span_attrs.update(attrs)
            
            with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:
                try:
                    #Record Output
                    if input_getter:
                        try:
                            inp = input_getter(*args, **kwargs)
                            if inp is not None:
                                span.set_attribute(SpanAttributes.INPUT_VALUE, str(inp))
                        except Exception as e:
                            span.add_event("input_getter_error", {"error": repr(e)})
                    
                    result = func(*args, **kwargs)
                    #Log Docs
                    if docs_getter:
                        try:
                            docs = docs_getter(result) or []
                            for i, doc in enumerate(docs[:max_doc_events]):
                                doc_id = doc.get("id", f"doc_{i}")
                                text = doc.get("text") or doc.get("content") or ""
                            else:
                                doc_id = f"doc_{i}"
                                text = str(doc)
                            span.add_event(
                                "retrieved_document",
                                    {
                                        "document.id": str(doc_id),
                                        "document.content": str(text)[:500],
                                        "document.rank": i,
                                    },
                            )                            
                        except Exception as e:
                            span.add_event("docs_getter_error", {"error": repr(e)})
                    return result
                
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
        return wrapper
    return deco

def instrument_retriever(**kwargs):
    """Shortcut: @instrument_retriever(name=..., input_getter=..., docs_getter=...)"""
    return instrument(kind="RETRIEVER", **kwargs)


def instrument_llm(**kwargs):
    """Shortcut: @instrument_llm(name=..., input_getter=..., output_getter=...)"""
    return instrument(kind="LLM", **kwargs)


def instrument_reranker(**kwargs):
    return instrument(kind="RERANKER", **kwargs)