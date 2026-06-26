from __future__ import annotations

import abc
import re
import shlex
import uuid
from dataclasses import dataclass, field

from cairn.dispatcher.config import WorkerConfig


@dataclass(slots=True)
class DriverResult:
    argv: list[str]
    session: str | None = None


@dataclass(slots=True)
class TrajectoryStep:
    step_id: int
    action: str
    observation: str | None = None
    tool_type: str | None = None
    thinking: str | None = None


class WorkerDriver(abc.ABC):
    type_name: str

    def supports_conclude(self) -> bool:
        return True

    def prepare_session(self) -> str | None:
        return None

    @staticmethod
    def model_args(worker: WorkerConfig) -> list[str]:
        if worker.model:
            return ["--model", worker.model]
        return []

    def build_startup_healthcheck(self, worker: WorkerConfig) -> list[str]:
        return self.build_healthcheck(worker)

    def describe_startup_healthcheck(self, worker: WorkerConfig) -> str:
        return shlex.join(self.build_startup_healthcheck(worker))

    @abc.abstractmethod
    def build_healthcheck(self, worker: WorkerConfig) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def build_execute(self, worker: WorkerConfig, prompt: str, session: str | None) -> DriverResult:
        raise NotImplementedError

    @abc.abstractmethod
    def build_conclude(self, worker: WorkerConfig, prompt: str, session: str) -> list[str]:
        raise NotImplementedError

    def extract_session(self, session: str | None, stdout: str, stderr: str) -> str | None:
        return session

    def extract_response_text(self, stdout: str, stderr: str) -> str:
        return stdout

    def extract_trajectory(self, session_data: str) -> list[TrajectoryStep]:
        """Parse raw session log data into a list of TrajectorySteps.

        Each driver implements its own parsing for its log format.
        ``session_data`` is the raw text content of the session log file
        (JSONL for pi/claudecode, or stdout capture for others).
        Returns an empty list if parsing fails or is not supported.
        """
        return []


class SeedSessionDriver(WorkerDriver):
    def prepare_session(self) -> str | None:
        return str(uuid.uuid4())


class RegexSessionDriver(WorkerDriver):
    session_pattern = re.compile(r"session id:\s*([0-9a-fA-F-]+)")

    def extract_session(self, session: str | None, stdout: str, stderr: str) -> str | None:
        if session:
            return session
        match = self.session_pattern.search(stderr)
        if match:
            return match.group(1)
        return None
