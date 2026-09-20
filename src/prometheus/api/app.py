"""
FastAPI application exposing Prometheus RAG query and ingestion endpoints.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from prometheus.llm.base import LLMError
from prometheus.models.gateway import get_model_gateway
from prometheus.models.types import ModelGatewayError
from prometheus.rag.models import RagQueryRequest, RagResult
from prometheus.rag.pipeline import RagPipelineError, query_rag
from prometheus.retrieval.ingestion import ingest_document
from prometheus.retrieval.parsers import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Prometheus RAG API",
    description="Minimal REST API for grounded research and document question-answering in Prometheus",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/rag/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "prometheus-rag"}


@app.get("/api/usage/summary")
def usage_summary() -> dict:
    """Return aggregate model usage without exposing prompts or provider secrets."""
    return asdict(get_model_gateway().usage_tracker.get_usage_summary())


@app.get("/api/usage/models")
def usage_models() -> list[dict]:
    """Return configured pricing entries, if any."""
    return [asdict(item) for item in get_model_gateway().pricing_registry.models()]


@app.post("/api/rag/query", response_model=RagResult)
def rag_query_endpoint(request: RagQueryRequest) -> RagResult:
    """Query the RAG pipeline with a user question and optional document filter."""
    try:
        result = query_rag(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
            include_academic_evidence=request.include_academic_evidence,
            openalex_max_results=request.openalex_max_results,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except LLMError as llm_err:
        logger.exception("RAG LLM generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAG answer generation failed.",
        ) from llm_err
    except ModelGatewayError as gateway_err:
        logger.exception("RAG model gateway failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAG answer generation failed.",
        ) from gateway_err
    except RagPipelineError as pipeline_err:
        logger.exception("RAG evidence processing failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(pipeline_err),
        ) from pipeline_err
    except Exception as exc:
        logger.exception("Unexpected RAG query failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG query processing failed.",
        ) from exc


@app.post("/api/rag/ingest")
async def ingest_file_endpoint(
    file: UploadFile = File(...),  # noqa: B008
    document_id: str | None = Form(None),
) -> dict:
    """Upload and ingest a document file (.pdf, .docx, .pptx, .txt) into the RAG knowledge store."""
    filename = file.filename or "uploaded_doc"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Save to a temporary file for parser processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Document ingestion started", extra={"document_filename": filename, "file_type": ext.lstrip(".")})
        doc_id = ingest_document(
            path=str(tmp_path),
            document_id=document_id,
            metadata={"original_filename": filename},
        )
        logger.info("Document ingestion completed", extra={"document_id": doc_id, "document_filename": filename})
        return {
            "status": "success",
            "document_id": doc_id,
            "filename": filename,
            "file_type": ext.lstrip("."),
        }
    except (FileNotFoundError, OSError, ValueError) as ingest_err:
        logger.warning("Document ingestion rejected", extra={"document_filename": filename})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document could not be ingested: {ingest_err}",
        ) from ingest_err
    except Exception as exc:
        logger.exception("Unexpected document ingestion failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from exc
    finally:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
