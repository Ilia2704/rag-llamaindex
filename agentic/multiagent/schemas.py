from typing import Annotated, Literal
from pydantic import Field
from agentic.toolcalling.demo import Contract, FinalAnswer

Role = Literal["equipment", "projects", "security"]


class Task(Contract):
    role: Role
    question: Annotated[str, Field(min_length=3, max_length=600)]


class Plan(Contract):
    tasks: Annotated[list[Task], Field(min_length=1, max_length=3)]


class SearchArgs(Contract):
    query: Annotated[str, Field(min_length=2, max_length=500)]
    category: Literal["Projects", "IT_Security"]
    year: Literal[2026] = 2026


class ReadArgs(Contract):
    document_id: Annotated[str, Field(min_length=1, max_length=150)]


class Evidence(Contract):
    document_id: str
    quote: Annotated[str, Field(min_length=10, max_length=1500)]


class Finding(Contract):
    role: Literal["projects", "security"]
    status: Literal["ok", "insufficient"]
    evidence: Annotated[list[Evidence], Field(max_length=6)]


class Review(Contract):
    approved: bool
    issues: Annotated[list[str], Field(max_length=5)]
    rerun_roles: Annotated[list[Role], Field(max_length=3)]


class OnboardingAnswer(Contract):
    equipment: FinalAnswer | None
    findings: list[Finding]
    access_verified: Literal[False] = False
