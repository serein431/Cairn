from __future__ import annotations

import json
from typing import Any

from cairn.dispatcher.config import WorkerConfig
from cairn.dispatcher.workers.adapters._curl import build_verbose_curl_healthcheck, expand_env, render_curl_command
from cairn.dispatcher.workers.base import DriverResult, RegexSessionDriver, TrajectoryStep


class CodexDriver(RegexSessionDriver):
    type_name = "codex"

    def build_healthcheck(self, worker: WorkerConfig) -> list[str]:
        return [
            "curl",
            "-sS",
            "--fail",
            "-o",
            "/dev/null",
            self._healthcheck_url(worker),
            *self._healthcheck_headers(worker),
            "-d",
            self._healthcheck_payload(worker),
        ]

    def build_startup_healthcheck(self, worker: WorkerConfig) -> list[str]:
        return build_verbose_curl_healthcheck(
            self._healthcheck_url(worker),
            headers=self._healthcheck_headers(worker),
            payload=self._healthcheck_payload(worker),
        )

    def describe_startup_healthcheck(self, worker: WorkerConfig) -> str:
        return render_curl_command(
            self._healthcheck_url(worker),
            headers=[
                "-H",
                expand_env("Authorization: Bearer $OPENAI_API_KEY"),
                "-H",
                "content-type: application/json",
            ],
            payload=self._healthcheck_payload(worker),
        )

    def _model_and_provider_args(self, worker: WorkerConfig) -> list[str]:
        model_args = self.model_args(worker)
        if not model_args and worker.env.get("CODEX_MODEL"):
            model_args = ["--model", worker.env["CODEX_MODEL"]]
        provider_args: list[str] = []
        if worker.env.get("CODEX_BASE_URL"):
            provider_args = [
                "-c",
                'model_provider="cairn"',
                "-c",
                'model_providers.cairn.name="cairn"',
                "-c",
                'model_providers.cairn.wire_api="responses"',
                "-c",
                'model_reasoning_effort="high"',
                "-c",
                f'model_providers.cairn.base_url="{worker.env["CODEX_BASE_URL"]}"',
                "-c",
                'model_providers.cairn.env_key="OPENAI_API_KEY"',
            ]
        return [*model_args, *provider_args]

    def build_execute(self, worker: WorkerConfig, prompt: str, session: str | None) -> DriverResult:
        return DriverResult(
            argv=[
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                *self._model_and_provider_args(worker),
                "--",
                prompt,
            ]
        )

    def build_conclude(self, worker: WorkerConfig, prompt: str, session: str) -> list[str]:
        return [
            "codex",
            "exec",
            "resume",
            session,
            "--dangerously-bypass-approvals-and-sandbox",
            *self._model_and_provider_args(worker),
            "--",
            prompt,
        ]

    @staticmethod
    def _healthcheck_url(worker: WorkerConfig) -> str:
        return f"{worker.env['CODEX_BASE_URL']}/responses"

    @staticmethod
    def _healthcheck_headers(worker: WorkerConfig) -> list[str]:
        return [
            "-H",
            f"Authorization: Bearer {worker.env['OPENAI_API_KEY']}",
            "-H",
            "content-type: application/json",
        ]

    @staticmethod
    def _healthcheck_payload(worker: WorkerConfig) -> str:
        return (
            '{"input":[{"content":"ping","role":"user"}],'
            '"model":"'
            + worker.env["CODEX_MODEL"]
            + '","stream":false}'
        )

    def extract_trajectory(self, session_data: str) -> list[TrajectoryStep]:
        """Parse Codex session JSONL into TrajectorySteps.

        Codex emits JSONL events with function_call / function results.
        """
        steps: list[TrajectoryStep] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        step_id = 0

        for line in session_data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            msg = event.get("message", event)
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                tool_calls = msg.get("tool_calls") or msg.get("function_call")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        call_id = tc.get("id", "")
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args_raw = fn.get("arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except json.JSONDecodeError:
                            args = {"raw": args_raw}
                        action = args.get("command", "") or args.get("content", "") or json.dumps(args, ensure_ascii=False)[:2000]
                        pending_calls[call_id] = {"name": name, "action": action}

            elif role in ("tool", "function"):
                call_id = msg.get("tool_call_id", "")
                observation = (content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))[:8000]
                call_info = pending_calls.pop(call_id, None)
                if call_info:
                    step_id += 1
                    steps.append(TrajectoryStep(
                        step_id=step_id,
                        action=call_info["action"],
                        observation=observation,
                        tool_type=call_info["name"],
                    ))

        return steps
