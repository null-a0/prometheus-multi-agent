"""
Findings DB repository -- read/write access used by:
  - Retriever Agent (query past findings as a data store)
  - Meta-Orchestrator (persist final report + reasoning chain on approval)
  - MCP tools (search_papers / save_report expose a subset of this)

Phase 1 provides basic save/search (plain substring match). Phase 4 upgrades
search_findings to the same BM25 + dense hybrid machinery as
prometheus.retrieval.hybrid_search, once report volume justifies it.
"""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from prometheus.config import settings
from prometheus.findings.models import Base, Claim, Objection, ReasoningStep, Report

engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_report(report: dict) -> str:
    """report: {question, summary, status?, claims?: [{statement, confidence_level, supporting_evidence_ids}]}"""
    async with AsyncSessionLocal() as session:
        row = Report(
            question=report["question"],
            summary=report.get("summary", ""),
            status=report.get("status", "draft"),
        )
        for claim in report.get("claims", []):
            row.claims.append(
                Claim(
                    statement=claim["statement"],
                    confidence_level=claim.get("confidence_level", 0.0),
                    supporting_evidence_ids=",".join(claim.get("supporting_evidence_ids", [])),
                )
            )
        session.add(row)
        await session.commit()
        return row.id


async def get_report(report_id: str) -> Report | None:
    async with AsyncSessionLocal() as session:
        return session.get(Report, report_id)


async def search_findings(query: str, top_k: int = 10) -> list[Report]:
    like = f"%{query}%"
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Report)
            .filter(or_(Report.question.ilike(like), Report.summary.ilike(like)))
            .order_by(Report.created_at.desc())
            .limit(top_k)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


async def save_reasoning_chain(report_id: str, steps: list[dict]) -> None:
    """steps: [{agent_role, input, output}]"""
    async with AsyncSessionLocal() as session:
        for step in steps:
            session.add(
                ReasoningStep(
                    report_id=report_id,
                    agent_role=step["agent_role"],
                    input=step["input"],
                    output=step["output"],
                )
            )
        await session.commit()


async def save_objection(report_id: str, objection: dict) -> str:
    """objection: {issue_type, explanation, claim_id?, resolved?}"""
    async with AsyncSessionLocal() as session:
        row = Objection(
            report_id=report_id,
            claim_id=objection.get("claim_id"),
            issue_type=objection["issue_type"],
            explanation=objection["explanation"],
            resolved=objection.get("resolved", False),
        )
        session.add(row)
        await session.commit()
        return row.id
