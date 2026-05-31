from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.recovery.conversation_repair import RepairReport, ensure_tool_result_pairing
from agent_core.types import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock


NO_RESPONSE_REQUESTED = "[No response requested]"
CONTINUE_FROM_INTERRUPTION = "Continue from where you left off."


@dataclass(slots=True)
class ResumeReport:
    repair: RepairReport = field(default_factory=RepairReport)
    removed_orphan_thinking: int = 0
    removed_blank_assistant: int = 0
    inserted_continuation: bool = False
    inserted_sentinel: bool = False

    @property
    def changed(self) -> bool:
        return (
            self.repair.repaired
            or self.removed_orphan_thinking > 0
            or self.removed_blank_assistant > 0
            or self.inserted_continuation
            or self.inserted_sentinel
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "repair": self.repair.to_dict(),
            "removed_orphan_thinking": self.removed_orphan_thinking,
            "removed_blank_assistant": self.removed_blank_assistant,
            "inserted_continuation": self.inserted_continuation,
            "inserted_sentinel": self.inserted_sentinel,
        }


def _assistant_has_tool_use_without_following_result(messages: list[Message], index: int) -> bool:
    msg = messages[index]
    tool_ids = [block.id for block in msg.content if isinstance(block, ToolUseBlock)]
    if not tool_ids:
        return False
    if index + 1 >= len(messages) or messages[index + 1].role != "user":
        return True
    result_ids = {block.tool_use_id for block in messages[index + 1].content if isinstance(block, ToolResultBlock)}
    return any(tool_id not in result_ids for tool_id in tool_ids)


def recover_messages_for_resume(messages: list[Message]) -> tuple[list[Message], ResumeReport]:
    repaired, repair_report = ensure_tool_result_pairing(messages)
    report = ResumeReport(repair=repair_report)
    filtered: list[Message] = []
    for idx, msg in enumerate(repaired):
        if msg.role == "assistant":
            has_text = any(isinstance(block, TextBlock) and block.text.strip() for block in msg.content)
            has_tool_use = any(isinstance(block, ToolUseBlock) for block in msg.content)
            has_tool_result = any(isinstance(block, ToolResultBlock) for block in msg.content)
            only_thinking = msg.content and all(isinstance(block, ThinkingBlock) for block in msg.content)
            if only_thinking:
                report.removed_orphan_thinking += 1
                continue
            if not has_tool_use and not has_tool_result and not has_text:
                report.removed_blank_assistant += 1
                continue
        filtered.append(msg)

    last_relevant_idx = next((i for i in range(len(filtered) - 1, -1, -1) if filtered[i].role in {"user", "assistant"}), -1)
    if last_relevant_idx != -1:
        last = filtered[last_relevant_idx]
        if last.role == "assistant" and _assistant_has_tool_use_without_following_result(filtered, last_relevant_idx):
            filtered.append(Message.user(CONTINUE_FROM_INTERRUPTION))
            report.inserted_continuation = True
            last = filtered[-1]
        if last.role == "user":
            filtered.insert(last_relevant_idx + 1, Message.assistant(NO_RESPONSE_REQUESTED))
            report.inserted_sentinel = True
    return filtered, report
