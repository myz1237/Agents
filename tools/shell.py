import subprocess

from consts import ALLOWED_COMMANDS, SANDBOX_DIR


def parse_and_validate_command(tool_input: dict) -> dict:
    command = tool_input.get("command")
    if not command.strip():
        return {"content": "Command is empty, please provide a valid shell command to run.", "is_error": True}

    command_name = command.split()[0]
    if command_name not in ALLOWED_COMMANDS:
        return {
            "content": f"Command '{command_name}' is not allowed. Allowed commands are: {', '.join(ALLOWED_COMMANDS)}.",
            "is_error": True,
        }
    return {"content": command.split(" "), "is_error": False}


def run_shell_in_sandbox(command: str, timeout: int = 10) -> dict:
    parsed_command = parse_and_validate_command(command)
    if parsed_command["is_error"]:
        return parsed_command

    command: list = parsed_command["content"]

    try:
        result = subprocess.run(command, shell=True, cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip() if result.stdout else result.stderr.strip()

        if len(output) > 500:
            return {"content": f"Command output is too long to display: {output[:500]}...", "is_error": False}
        return {"content": f"stdout: {output}", "is_error": False}
    except subprocess.TimeoutExpired:
        return {"content": f"Command timed out after {timeout} seconds.", "is_error": True}
    except Exception as e:
        print(f"Error running shell command: {command}: {e}")
        return {"content": f"Error running shell command: {str(e)}", "is_error": True}
