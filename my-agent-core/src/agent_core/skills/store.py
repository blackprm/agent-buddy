"""Claude Code style local skill discovery and loading.

Supported locations intentionally mirror the high-value subset of Claude Code:
- project: ``.claude/skills/<skill-name>/SKILL.md`` under the current cwd
- user: ``~/.claude/skills/<skill-name>/SKILL.md``

The model sees a compact skill listing in system prompt.  Full skill content is
loaded only when the model invokes the ``Skill`` tool.
"""
from __future__ import annotations

import os
import re
import shlex
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


MAX_LISTING_DESC_CHARS = 250
DEFAULT_CHAR_BUDGET = 8_000


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path
    base_dir: Path
    source: str
    when_to_use: str | None = None
    aliases: list[str] = field(default_factory=list)
    argument_hint: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    version: str | None = None
    model: str | None = None
    execution_context: str | None = None
    agent: str | None = None
    effort: str | None = None
    shell: str | None = None
    paths: list[str] | None = None
    argument_names: list[str] = field(default_factory=list)
    disable_model_invocation: bool = False
    user_invocable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_description(self) -> str:
        text = f"{self.description} - {self.when_to_use}" if self.when_to_use else self.description
        text = " ".join(text.split())
        if len(text) > MAX_LISTING_DESC_CHARS:
            return text[: MAX_LISTING_DESC_CHARS - 1] + "…"
        return text


class SkillStore:
    def __init__(self, *, cwd: str | Path | None = None, extra_dirs: list[str | Path] | None = None) -> None:
        self.cwd = Path(cwd or os.getcwd()).expanduser().resolve()
        self.extra_dirs = [Path(p).expanduser().resolve() for p in (extra_dirs or [])]
        self._cache: dict[str, Skill] | None = None
        self._all_cache: dict[str, Skill] | None = None
        self._activated_conditional_names: set[str] = set()

    def clear_cache(self) -> None:
        self._cache = None
        self._all_cache = None

    def skill_roots(self) -> list[tuple[str, Path]]:
        roots: list[tuple[str, Path]] = []
        roots.append(("project", self.cwd / ".claude" / "skills"))
        for extra in self.extra_dirs:
            roots.append(("project", extra / ".claude" / "skills"))
        managed = os.getenv("AGENT_MANAGED_SKILLS_DIR")
        if managed:
            roots.append(("managed", Path(managed).expanduser().resolve()))
        roots.append(("user", Path.home() / ".claude" / "skills"))
        return roots

    def command_roots(self) -> list[tuple[str, Path]]:
        roots: list[tuple[str, Path]] = [("project", self.cwd / ".claude" / "commands")]
        for extra in self.extra_dirs:
            roots.append(("project", extra / ".claude" / "commands"))
        roots.append(("user", Path.home() / ".claude" / "commands"))
        return roots

    def list_skills(self, *, refresh: bool = False) -> list[Skill]:
        if refresh or self._cache is None:
            skills = self._load_all_skills(refresh=refresh)
            visible: dict[str, Skill] = {}
            for key, skill in skills.items():
                if skill.paths and skill.name not in self._activated_conditional_names:
                    continue
                visible.setdefault(key, skill)
            self._cache = visible
        unique: dict[Path, Skill] = {}
        for skill in self._cache.values():
            unique.setdefault(skill.path.resolve(), skill)
        return sorted(unique.values(), key=lambda s: (s.source, s.name))

    def list_all_skills(self, *, refresh: bool = False) -> list[Skill]:
        skills = self._load_all_skills(refresh=refresh)
        unique: dict[Path, Skill] = {}
        for skill in skills.values():
            unique.setdefault(skill.path.resolve(), skill)
        return sorted(unique.values(), key=lambda s: (s.source, s.name))

    def _load_all_skills(self, *, refresh: bool = False) -> dict[str, Skill]:
        if refresh or self._all_cache is None:
            skills: dict[str, Skill] = {}
            seen_paths: set[Path] = set()
            for source, root in self.skill_roots():
                for skill in self._load_root(root, source):
                    resolved = skill.path.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    # First wins: project entries are loaded before user entries.
                    skills.setdefault(skill.name, skill)
                    for alias in skill.aliases:
                        skills.setdefault(alias, skill)
            for source, root in self.command_roots():
                for skill in self._load_commands_root(root, source):
                    resolved = skill.path.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    skills.setdefault(skill.name, skill)
                    for alias in skill.aliases:
                        skills.setdefault(alias, skill)
            self._all_cache = skills
        return self._all_cache

    def get_skill(self, name: str) -> Skill | None:
        if self._cache is None:
            self.list_skills()
        normalized = name[1:] if name.startswith("/") else name
        assert self._cache is not None
        skill = self._cache.get(normalized)
        if skill is not None:
            return skill
        return self._load_all_skills().get(normalized)

    def activate_for_paths(self, paths: list[str | Path]) -> list[Skill]:
        activated: list[Skill] = []
        all_skills = self.list_all_skills()
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            try:
                rel = str(path.resolve().relative_to(self.cwd))
            except Exception:
                rel = str(path)
            for skill in all_skills:
                if not skill.paths:
                    continue
                if any(_path_matches(rel, pattern) for pattern in skill.paths):
                    if skill.name not in self._activated_conditional_names:
                        self._activated_conditional_names.add(skill.name)
                        activated.append(skill)
        if activated:
            self._cache = None
        return activated

    def render_skill(self, name: str, args: str | None = None, *, session_id: str | None = None) -> str:
        skill = self.get_skill(name)
        if skill is None:
            available = ", ".join(s.name for s in self.list_skills()) or "none"
            raise FileNotFoundError(f"Skill '{name}' not found. Available skills: {available}")
        text = f"<command-name>{skill.name}</command-name>\nBase directory for this skill: {skill.base_dir}\n\n{skill.content}"
        text = text.replace("${CLAUDE_SKILL_DIR}", str(skill.base_dir))
        text = text.replace("${CLAUDE_SESSION_ID}", session_id or "")
        text = substitute_arguments(text, args, append_if_no_placeholder=True, argument_names=skill.argument_names)
        return text

    def build_listing(self, *, char_budget: int = DEFAULT_CHAR_BUDGET) -> str:
        skills = [s for s in self.list_skills() if s.user_invocable]
        if not skills:
            return ""
        lines = [f"- {s.name}: {s.display_description}" for s in skills]
        output: list[str] = []
        used = 0
        for line in lines:
            add = len(line) + (1 if output else 0)
            if output and used + add > char_budget:
                remaining = len(lines) - len(output)
                output.append(f"- ... ({remaining} more skills omitted due to budget)")
                break
            output.append(line)
            used += add
        return "\n".join(output)

    def _load_root(self, root: Path, source: str) -> list[Skill]:
        if not root.exists() or not root.is_dir():
            return []
        skills: list[Skill] = []
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                skills.append(_parse_skill_file(skill_file, source=source, fallback_name=entry.name))
            except Exception:
                # Bad skills should not break startup or prompt construction.
                continue
        return skills

    def _load_commands_root(self, root: Path, source: str) -> list[Skill]:
        if not root.exists() or not root.is_dir():
            return []
        files: list[Path] = []
        for path in sorted(root.rglob("*.md"), key=lambda p: str(p).lower()):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            # If a directory has SKILL.md, load only that file for that directory.
            if path.name.lower() != "skill.md" and (path.parent / "SKILL.md").exists():
                continue
            files.append(path)
        skills: list[Skill] = []
        for file_path in files:
            try:
                if file_path.name.lower() == "skill.md":
                    rel_parent = file_path.parent.relative_to(root)
                    fallback_name = ":".join(rel_parent.parts)
                    base_dir = file_path.parent
                else:
                    rel = file_path.relative_to(root).with_suffix("")
                    fallback_name = ":".join(rel.parts)
                    base_dir = None
                skill = _parse_skill_file(file_path, source=source, fallback_name=fallback_name, loaded_from="commands_DEPRECATED")
                if base_dir is None:
                    skill.base_dir = file_path.parent
                skills.append(skill)
            except Exception:
                continue
        return skills


def build_skill_prompt_section(store: SkillStore) -> str:
    listing = store.build_listing()
    if not listing:
        return (
            "# Skills\n"
            "No local skills found. Create skills in .claude/skills/<name>/SKILL.md or ~/.claude/skills/<name>/SKILL.md."
        )
    return f"""# Skills
Skills provide specialized capabilities and domain knowledge.

CRITICAL REQUIREMENT:
- When the user's request clearly matches an available skill, invoke the `Skill` tool BEFORE producing any other response about the task.
- Never mention a skill without actually calling the `Skill` tool.
- Treat slash-command-like requests such as `/review` or `/commit` as skill invocations when a matching skill exists.
- If a `<command-name>` tag or loaded skill instructions are already present in the current turn, follow those instructions instead of invoking the same skill again.
- Do not invoke `Skill` when a dedicated built-in/core tool directly handles the request. For example, use `FeishuApi` for Feishu/Lark OpenAPI operations such as sending messages instead of loading a Feishu/Lark skill.

Available skills:
{listing}
""".strip()


def _parse_skill_file(path: Path, *, source: str, fallback_name: str, loaded_from: str = "skills") -> Skill:
    raw = path.read_text(encoding="utf-8")
    frontmatter, content = _split_frontmatter(raw)
    data: dict[str, Any] = {}
    if frontmatter.strip():
        parsed = yaml.safe_load(frontmatter) or {}
        if isinstance(parsed, dict):
            data = parsed
    name = _safe_name(str(data.get("name") or fallback_name))
    description = str(data.get("description") or _extract_description(content) or f"Skill {name}")
    aliases = data.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
    allowed_tools = data.get("allowed-tools") or data.get("allowed_tools") or []
    if isinstance(allowed_tools, str):
        allowed_tools = [a.strip() for a in allowed_tools.split(",") if a.strip()]
    argument_names = _parse_argument_names(data.get("arguments"))
    paths = _parse_paths(data.get("paths"))
    return Skill(
        name=name,
        description=description,
        content=content.strip(),
        path=path,
        base_dir=path.parent,
        source=source,
        when_to_use=str(data.get("when_to_use") or data.get("when-to-use") or "") or None,
        aliases=[_safe_name(str(a)) for a in aliases],
        argument_hint=str(data.get("argument-hint") or data.get("argument_hint") or "") or None,
        allowed_tools=[str(t) for t in allowed_tools],
        version=str(data.get("version") or "") or None,
        model=None if data.get("model") in (None, "inherit") else str(data.get("model")),
        execution_context="fork" if data.get("context") == "fork" else "inline",
        agent=str(data.get("agent") or "") or None,
        effort=str(data.get("effort") or "") or None,
        shell=str(data.get("shell") or "") or None,
        paths=paths,
        argument_names=argument_names,
        disable_model_invocation=_parse_bool(data.get("disable-model-invocation", data.get("disable_model_invocation", False))),
        user_invocable=_parse_bool(data.get("user-invocable", data.get("user_invocable", True))),
        metadata={**data, "loaded_from": loaded_from},
    )


def parse_arguments(args: str | None) -> list[str]:
    if not args or not args.strip():
        return []
    try:
        return shlex.split(args)
    except ValueError:
        return args.split()


def substitute_arguments(content: str, args: str | None, *, append_if_no_placeholder: bool = True, argument_names: list[str] | None = None) -> str:
    if args is None:
        return content
    parsed = parse_arguments(args)
    original = content
    for i, name in enumerate(argument_names or []):
        if name:
            content = re.sub(rf"\${re.escape(name)}(?![\[\w])", parsed[i] if i < len(parsed) else "", content)
    content = re.sub(r"\$ARGUMENTS\[(\d+)\]", lambda m: parsed[int(m.group(1))] if int(m.group(1)) < len(parsed) else "", content)
    content = re.sub(r"\$(\d+)(?!\w)", lambda m: parsed[int(m.group(1))] if int(m.group(1)) < len(parsed) else "", content)
    content = content.replace("$ARGUMENTS", args).replace("{{args}}", args)
    if content == original and append_if_no_placeholder and args:
        content += f"\n\nARGUMENTS: {args}"
    return content


def _split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---\n") or text.startswith("---\r\n"):
        match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", text, flags=re.DOTALL)
        if match:
            return match.group(1), match.group(2)
    return "", text


def _extract_description(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:200]
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip()).strip("-") or "skill"


def _parse_argument_names(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = []
        for item in value:
            if isinstance(item, dict):
                raw.append(str(item.get("name") or item.get("argument") or item.get("id") or ""))
            else:
                raw.append(str(item))
    elif isinstance(value, dict):
        raw = [str(k) for k in value.keys()]
    elif isinstance(value, str):
        raw = [p.strip() for p in re.split(r"[,\s]+", value) if p.strip()]
    else:
        raw = []
    return [_safe_name(v.strip()) for v in raw if v.strip() and not v.strip().isdigit()]


def _parse_paths(value: Any) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        patterns = [str(v).strip() for v in value]
    else:
        patterns = [p.strip() for p in re.split(r"[,\n]", str(value))]
    patterns = [p[:-3] if p.endswith("/**") else p for p in patterns if p and p != "**"]
    return patterns or None


def _path_matches(path: str, pattern: str) -> bool:
    normalized = path.replace(os.sep, "/")
    pat = pattern.replace(os.sep, "/")
    if fnmatch.fnmatch(normalized, pat) or normalized.startswith(pat.rstrip("/") + "/"):
        return True
    if "/**/" in pat:
        zero_depth = pat.replace("/**/", "/")
        return fnmatch.fnmatch(normalized, zero_depth)
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
