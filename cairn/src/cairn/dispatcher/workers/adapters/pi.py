from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from cairn.dispatcher.config import WorkerConfig
from cairn.dispatcher.workers.base import DriverResult, TrajectoryStep, WorkerDriver


class PiDriver(WorkerDriver):
    type_name = "pi"

    def build_healthcheck(self, worker: WorkerConfig) -> list[str]:
        env = worker.env
        return self._wrap_with_models(
            worker,
            [
                "--provider",
                "cairn",
                "--model",
                env["PI_MODEL"],
                "--mode",
                "json",
                "--session-dir",
                self._session_dir(worker),
                "--no-session",
                "--no-tools",
                "-p",
                "Reply with exactly pong.",
            ],
            enable_tools=False,
        )

    def build_execute(self, worker: WorkerConfig, prompt: str, session: str | None) -> DriverResult:
        env = worker.env
        argv = [
            "--provider",
            "cairn",
            "--model",
            env["PI_MODEL"],
            "--mode",
            "json",
            "--session-dir",
            self._session_dir(worker),
        ]
        if session:
            argv.extend(["--session", session])
        argv.extend(["-p", prompt])
        return DriverResult(argv=self._wrap_with_models(worker, argv), session=session)

    def build_conclude(self, worker: WorkerConfig, prompt: str, session: str) -> list[str]:
        env = worker.env
        argv = [
            "--provider",
            "cairn",
            "--model",
            env["PI_MODEL"],
            "--mode",
            "json",
            "--session-dir",
            self._session_dir(worker),
            "--session",
            session,
            "-p",
            prompt,
        ]
        return self._wrap_with_models(worker, argv)

    def extract_session(self, session: str | None, stdout: str, stderr: str) -> str | None:
        if session:
            return session
        for event in self._iter_events(stdout):
            if event.get("type") != "session":
                continue
            session_id = event.get("id")
            if isinstance(session_id, str) and session_id:
                return session_id
        return None

    def extract_response_text(self, stdout: str, stderr: str) -> str:
        assistant_message: dict[str, Any] | None = None
        for event in self._iter_events(stdout):
            event_type = event.get("type")
            if event_type == "turn_end":
                message = event.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    assistant_message = message
            elif event_type == "agent_end":
                messages = event.get("messages")
                if isinstance(messages, list):
                    for message in reversed(messages):
                        if isinstance(message, dict) and message.get("role") == "assistant":
                            assistant_message = message
                            break
        if assistant_message is None:
            return stdout
        content = assistant_message.get("content")
        if not isinstance(content, list):
            return stdout
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts).strip() or stdout

    def _wrap_with_models(self, worker: WorkerConfig, pi_argv: list[str], *, enable_tools: bool = True) -> list[str]:
        script = (
            'agent_dir="$1"\n'
            'models_json="$2"\n'
            "shift 2\n"
            'mkdir -p "$agent_dir"\n'
            'mkdir -p "$agent_dir/sessions"\n'
            'printf "%s" "$models_json" > "$agent_dir/models.json"\n'
            'exec env PI_CODING_AGENT_DIR="$agent_dir" pi "$@"\n'
        )
        argv = [
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
        ]
        if enable_tools:
            argv.extend(["--tools", "read,write,edit,bash,grep,find,ls"])
        return [
            "/bin/sh",
            "-lc",
            script,
            "--",
            self._agent_dir(worker),
            self._models_json(worker),
            *argv,
            *pi_argv,
        ]

    @staticmethod
    def _agent_dir(worker: WorkerConfig) -> str:
        return str(PurePosixPath("/tmp/cairn-pi") / worker.name)

    @staticmethod
    def _session_dir(worker: WorkerConfig) -> str:
        return str(PurePosixPath(PiDriver._agent_dir(worker)) / "sessions")

    @staticmethod
    def _iter_events(stdout: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def extract_trajectory(self, session_data: str) -> list[TrajectoryStep]:
        """Parse Pi agent JSONL session into TrajectorySteps.

        Pi session JSONL has messages with roles: assistant (tool_use), toolResult.
        We pair each tool_use with its corresponding toolResult by toolCallId.
        """
        events = self._iter_events(session_data)
        # collect tool calls and results
        pending_calls: dict[str, dict[str, Any]] = {}  # toolCallId -> {name, input}
        steps: list[TrajectoryStep] = []
        step_id = 0

        for event in events:
            msg = event.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, list):
                thinking_text = ""
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "thinking":
                        thinking_text = item.get("thinking", "")
                    elif item.get("type") == "text":
                        thinking_text = thinking_text or item.get("text", "")
                    elif item.get("type") in ("tool_use", "toolCall"):
                        call_id = item.get("toolCallId") or item.get("id", "")
                        tool_name = item.get("name", "")
                        tool_input = item.get("input") or item.get("arguments") or {}
                        if isinstance(tool_input, dict):
                            action = tool_input.get("command", "") or tool_input.get("content", "") or tool_input.get("path", "") or json.dumps(tool_input, ensure_ascii=False)[:2000]
                        else:
                            action = str(tool_input)[:2000]
                        pending_calls[call_id] = {
                            "name": tool_name,
                            "action": action,
                            "thinking": thinking_text,
                        }
                        thinking_text = ""

            elif role == "toolResult" and isinstance(content, list):
                call_id = msg.get("toolCallId", "")
                observation_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        observation_parts.append(item.get("text", ""))
                observation = "\n".join(observation_parts)[:8000]

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
                else:
                    step_id += 1
                    steps.append(TrajectoryStep(
                        step_id=step_id,
                        action=f"[unknown tool call {call_id}]",
                        observation=observation,
                    ))

        return steps

    @staticmethod
    def _models_json(worker: WorkerConfig) -> str:
        env = worker.env
        model: dict[str, Any] = {
            "id": env["PI_MODEL"],
            "name": env["PI_MODEL"],
        }
        context_window = env.get("PI_MODEL_CONTEXT_WINDOW")
        if context_window:
            model["contextWindow"] = int(context_window)

        provider: dict[str, Any] = {
            "baseUrl": env["PI_BASE_URL"],
            "api": env["PI_PROVIDER_API"],
            "apiKey": env["PI_API_KEY"],
            "models": [model],
        }
        payload = {"providers": {"cairn": provider}}
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
