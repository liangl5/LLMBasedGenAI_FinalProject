"""The one toolset every agent in every system receives."""

MAX_TOOL_OUTPUT = 8000
MAX_READ_LINES = 400


def _schema(name, description, required=(), **props):
    return {"name": name, "description": description,
            "parameters": {"type": "OBJECT", "properties": props, "required": list(required)}}


_STR, _INT, _BOOL = {"type": "STRING"}, {"type": "INTEGER"}, {"type": "BOOLEAN"}

TOOL_SCHEMAS = [
    _schema("read_file", "Read a repository file with line numbers (up to 400 lines).", ["path"],
            path=_STR, start_line=_INT),
    _schema("search", "Search Python files with a regular expression.", ["pattern"], pattern=_STR, path=_STR),
    _schema("edit_file", "Replace one exact, unique string. With empty old_text, create a new file.",
            ["path", "old_text", "new_text"], path=_STR, old_text=_STR, new_text=_STR),
    _schema("run_command", "Run a shell command in the repository root.", ["command"], command=_STR),
    _schema("finish", "End your turn. `summary` is what the next agent receives. Testers set `passed`.",
            ["summary"], summary=_STR, passed=_BOOL),
]


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[{len(text) - limit} chars omitted]...\n{text[-half:]}"


def _read(sandbox, path, start_line=1):
    lines = sandbox.read(path).splitlines()
    start = max(int(start_line), 1)
    shown = lines[start - 1:start - 1 + MAX_READ_LINES]
    body = "\n".join(f"{start + i}: {line}" for i, line in enumerate(shown))
    return f"{body}\n[{len(lines)} lines total]"


def _edit(sandbox, path, old_text, new_text):
    if not old_text:
        try:
            sandbox.read(path)
        except (FileNotFoundError, IsADirectoryError):
            sandbox.write(path, new_text)
            return f"created {path}"
        raise ValueError(f"{path} already exists; provide old_text")
    text = sandbox.read(path)
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text matched {count} times in {path}; it must match exactly once")
    sandbox.write(path, text.replace(old_text, new_text))
    return f"edited {path}"


def _run(sandbox, command):
    code, output = sandbox.run(command)
    return f"exit code {code}\n{output}"


_HANDLERS = {"read_file": _read, "search": lambda s, pattern, path=".": s.search(pattern, path) or "no matches",
             "edit_file": _edit, "run_command": _run}


def execute_tool(sandbox, name: str, args: dict) -> str:
    """Tool failures go back to the model as text so one bad call cannot end a run."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"ERROR: unknown tool {name}"
    try:
        return truncate(handler(sandbox, **args))
    except (TypeError, ValueError, OSError, UnicodeError) as e:
        return f"ERROR: {type(e).__name__}: {e}"
    except Exception as e:  # re.error and subprocess failures must not kill a paid run
        return f"ERROR: {type(e).__name__}: {e}"
