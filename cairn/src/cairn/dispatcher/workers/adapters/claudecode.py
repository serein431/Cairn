from __future__ import annotations

import json
from typing import Any

from cairn.dispatcher.config import WorkerConfig
from cairn.dispatcher.workers.adapters._curl import build_verbose_curl_healthcheck, expand_env, render_curl_command
from cairn.dispatcher.workers.base import DriverResult, SeedSessionDriver, TrajectoryStep


ANTHROPIC_VERSION = "2023-06-01"


class ClaudeCodeDriver(SeedSessionDriver):
    type_name = "claudecode"

    def build_healthcheck(self, worker: WorkerConfig) -> list[str]:
        env = worker.env
        return [
            "curl",
            "-sS",
            "--fail",
            "-o",
            "/dev/null",
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            "-H",
            f"Authorization: Bearer {env['ANTHROPIC_AUTH_TOKEN']}",
            "-H",
            f"anthropic-version: {ANTHROPIC_VERSION}",
            "-H",
            "content-type: application/json",
            "-d",
            (
                '{"model":"'
                + env["ANTHROPIC_MODEL"]
                + '","max_tokens":10,"messages":[{"role":"user","content":"ping"}]}'
            ),
        ]

    def build_startup_healthcheck(self, worker: WorkerConfig) -> list[str]:
        env = worker.env
        return build_verbose_curl_healthcheck(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            headers=[
                "-H",
                f"Authorization: Bearer {env['ANTHROPIC_AUTH_TOKEN']}",
                "-H",
                f"anthropic-version: {ANTHROPIC_VERSION}",
                "-H",
                "content-type: application/json",
            ],
            payload=(
                '{"model":"'
                + env["ANTHROPIC_MODEL"]
                + '","max_tokens":10,"messages":[{"role":"user","content":"ping"}]}'
            ),
        )

    def describe_startup_healthcheck(self, worker: WorkerConfig) -> str:
        env = worker.env
        return render_curl_command(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            headers=[
                "-H",
                expand_env("Authorization: Bearer $ANTHROPIC_AUTH_TOKEN"),
                "-H",
                f"anthropic-version: {ANTHROPIC_VERSION}",
                "-H",
                "content-type: application/json",
            ],
            payload=(
                '{"model":"'
                + env["ANTHROPIC_MODEL"]
                + '","max_tokens":10,"messages":[{"role":"user","content":"ping"}]}'
            ),
        )

    def build_execute(self, worker: WorkerConfig, prompt: str, session: str | None) -> DriverResult:
        assert session is not None
        return DriverResult(
            argv=[
                "claude",
                *self.model_args(worker),
                "--session-id",
                session,
                "--dangerously-skip-permissions",
                "-p",
                "--",
                prompt,
            ],
            session=session,
        )

    def build_conclude(self, worker: WorkerConfig, prompt: str, session: str) -> list[str]:
        return [
            "claude",
            *self.model_args(worker),
            "-r",
            session,
            "--dangerously-skip-permissions",
            "-p",
            "--",
            prompt,
        ]

    def extract_trajectory(self, session_data: str) -> list[TrajectoryStep]:
        """Parse Claude Code JSONL session into TrajectorySteps.

        Claude Code session JSONL has messages with type: assistant (tool_use),
        tool_result. Format is similar to Pi but with slight field differences.
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

            if role == "assistant" and isinstance(content, list):
                thinking_text = ""
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type", "")
                    if item_type == "thinking":
                        thinking_text = item.get("thinking", "")
                    elif item_type == "text":
                        thinking_text = thinking_text or item.get("text", "")
                    elif item_type == "tool_use":
                        call_id = item.get("id", "")
                        tool_name = item.get("name", "")
                        tool_input = item.get("input", {})
                        if isinstance(tool_input, dict):
                            action = tool_input.get("command", "") or tool_input.get("content", "") or json.dumps(tool_input, ensure_ascii=False)[:2000]
                        else:
                            action = str(tool_input)[:2000]
                        pending_calls[call_id] = {
                            "name": tool_name,
                            "action": action,
                            "thinking": thinking_text,
                        }
                        thinking_text = ""

            elif role in ("tool_result", "tool") and isinstance(content, (list, str)):
                call_id = msg.get("tool_use_id", "") or msg.get("toolCallId", "")
                if isinstance(content, str):
                    observation = content[:8000]
                else:
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    observation = "\n".join(parts)[:8000]

                call_info = pending_calls.pop(call_id, None)
                if call_info:
                    step_id += 1
                    steps.append(TrajectoryStep(
                        step_id=step_id,
                        action=call_info["action"],
                        observation=observation,
                        tool_type=call_info["name"],
                        thinking=call_info.get("thinking") or None,
                    ))

        return steps
