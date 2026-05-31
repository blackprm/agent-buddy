from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


SYNTHETIC_TOOL_RESULT_PLACEHOLDER = "[Tool result missing due to internal error]"


@dataclass(slots=True)
class RepairReport:
    repaired: bool = False
    inserted_missing_results: list[str] = field(default_factory=list)
    removed_orphan_results: list[str] = field(default_factory=list)
    removed_duplicate_tool_uses: list[str] = field(default_factory=list)
    removed_duplicate_results: list[str] = field(default_factory=list)
    inserted_placeholders: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "repaired": self.repaired,
            "inserted_missing_results": self.inserted_missing_results,
            "removed_orphan_results": self.removed_orphan_results,
            "removed_duplicate_tool_uses": self.removed_duplicate_tool_uses,
            "removed_duplicate_results": self.removed_duplicate_results,
            "inserted_placeholders": self.inserted_placeholders,
        }


def _tool_use_ids(message: Message) -> list[str]:
    return [block.id for block in message.content if isinstance(block, ToolUseBlock)]


def _tool_result_ids(message: Message) -> list[str]:
    return [block.tool_use_id for block in message.content if isinstance(block, ToolResultBlock)]


def ensure_tool_result_pairing(messages: list[Message]) -> tuple[list[Message], RepairReport]:
    """Repair Claude-style assistant tool_use ↔ user tool_result pairing.

    This mirrors Claude Code's API-boundary repair: missing results are filled
    with synthetic error tool_result blocks; orphan/duplicate results are
    stripped so the next model request does not get stuck on provider 400s.
    """
    report = RepairReport()
    repaired: list[Message] = []
    seen_tool_use_ids: set[str] = set()
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role != "assistant":
            if msg.role == "user" and (not repaired or repaired[-1].role != "assistant"):
                new_content = []
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        report.removed_orphan_results.append(block.tool_use_id)
                        report.repaired = True
                    else:
                        new_content.append(block)
                if not new_content and len(new_content) != len(msg.content):
                    new_content = [TextBlock(text="[Orphaned tool result removed due to conversation resume]")]
                    report.inserted_placeholders += 1
                repaired.append(Message(role=msg.role, content=new_content, metadata=dict(msg.metadata)))
            else:
                repaired.append(msg)
            i += 1
            continue

        assistant_content = []
        current_tool_ids: list[str] = []
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                if block.id in seen_tool_use_ids or block.id in current_tool_ids:
                    report.removed_duplicate_tool_uses.append(block.id)
                    report.repaired = True
                    continue
                seen_tool_use_ids.add(block.id)
                current_tool_ids.append(block.id)
            assistant_content.append(block)
        if not assistant_content:
            assistant_content = [TextBlock(text="[Tool use interrupted]")]
            report.inserted_placeholders += 1
            report.repaired = True
        assistant_msg = Message(role="assistant", content=assistant_content, metadata=dict(msg.metadata))
        repaired.append(assistant_msg)

        if not current_tool_ids:
            i += 1
            continue

        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        existing: set[str] = set()
        duplicate_results: set[str] = set()
        if next_msg and next_msg.role == "user":
            for result_id in _tool_result_ids(next_msg):
                if result_id in existing:
                    duplicate_results.add(result_id)
                existing.add(result_id)
        missing = [tool_id for tool_id in current_tool_ids if tool_id not in existing]
        orphaned = [result_id for result_id in existing if result_id not in set(current_tool_ids)]

        if not missing and not orphaned and not duplicate_results:
            i += 1
            continue

        report.repaired = True
        report.inserted_missing_results.extend(missing)
        report.removed_orphan_results.extend(orphaned)
        report.removed_duplicate_results.extend(sorted(duplicate_results))
        synthetic = [
            ToolResultBlock(tool_use_id=tool_id, content=SYNTHETIC_TOOL_RESULT_PLACEHOLDER, is_error=True)
            for tool_id in missing
        ]
        if next_msg and next_msg.role == "user":
            seen_results: set[str] = set()
            patched_content = []
            for block in next_msg.content:
                if isinstance(block, ToolResultBlock):
                    if block.tool_use_id in orphaned or block.tool_use_id in seen_results:
                        continue
                    seen_results.add(block.tool_use_id)
                patched_content.append(block)
            patched_content = synthetic + patched_content
            repaired.append(Message(role="user", content=patched_content, metadata=dict(next_msg.metadata)))
            i += 2
        else:
            repaired.append(Message(role="user", content=synthetic, metadata={"synthetic": True, "repair": "missing_tool_result"}))
            i += 1
    return repaired, report
