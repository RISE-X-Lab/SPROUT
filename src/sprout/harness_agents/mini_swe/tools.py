"""The mini bash contract shared by actor prompts, requests and reviewers."""
from copy import deepcopy
import json
import shlex

#: Seconds one agent command may run before the runtime kills it, unless the run
#: names another (``--command-timeout``, the mini environment's ``timeout``). The
#: model's bash tool has no timeout argument, so every command runs under this.
DEFAULT_COMMAND_TIMEOUT_S = 600


def validate_mini_tools(tools: list[dict]) -> list[dict]:
    """Check the interface our mini executor implements, without rewriting history."""
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
        raise ValueError("Mini requires exactly one recorded bash tool")
    tool = tools[0]
    function = tool.get("function")
    if tool.get("type") != "function" or not isinstance(function, dict) or function.get("name") != "bash":
        raise ValueError("Recorded actor tool must be the bash function")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(parameters.get("properties"), dict):
        raise ValueError("Recorded bash tool requires an object parameter schema")
    properties = parameters["properties"]
    if (parameters.get("type") != "object" or set(properties) != {"command"}
            or not isinstance(properties["command"], dict) or properties["command"].get("type") != "string"
            or parameters.get("required") != ["command"]):
        raise ValueError("Recorded bash tool must require exactly one string command")
    return deepcopy(tools)


def mini_tool_schema(upstream: dict | None = None) -> list[dict]:
    # The pinned mini2.4.6 contract. Rendering a task must not import mini and
    # load its host-global .env before the slot establishes its isolated config.
    tools = [{"type": "function", "function": {
        "name": "bash", "description": "Execute a bash command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                       "required": ["command"], "additionalProperties": False},
    }}]
    if upstream is not None:
        actual = validate_mini_tools([upstream])
        # Our executor already rejects extra arguments. Advertise that limit.
        actual[0]["function"]["parameters"]["additionalProperties"] = False
        if actual != tools:
            raise ValueError("Installed mini tool definition differs from the pinned actor contract")
    return tools


def mini_tool_primer(workdir: str) -> str:
    tools = json.dumps(mini_tool_schema(), indent=2)
    example = json.dumps({"command": f"cd {shlex.quote(workdir)} && git status --short"})
    return (
        "## Your tool\n\n"
        "You have one function tool named `bash`. Its exact definition is:\n\n"
        f"```json\n{tools}\n```\n\n"
        "Arguments must contain only the string `command`. Use bash commands to "
        "inspect, edit and test files. Each call starts a fresh process; put cd "
        "and environment assignments inside the command. Execution timeouts are "
        "configured by the runner, not supplied as function arguments.\n"
        f"The default working directory is {workdir}.\n\n"
        f"Example bash arguments:\n```json\n{example}\n```\n"
    )
