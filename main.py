from openai import OpenAI
import os
from pathlib import Path
import json

MAX_MODEL_CALLS = 10
MAX_READ_LINES = 300
MAX_CONTENT_BYTES = 20 * 1024

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "name": "list_directory",
        "description": "列出本地指定目录中的文件和子目录，不读取文件内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录路径，例如 C:/Users/19151/Documents",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "按行读取 UTF-8 文本文件，行号从 1 开始。"
            "通常每次请求 100 行，最多 300 行，"
            "返回内容最多 20 KiB。"
            "如果 truncated 为 true，"
            "使用 next_start_line 作为下一次的 start_line 继续读取。"
            "如果报告单行过长，不要使用相同参数反复重试。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "起始行号，从 1 开始；首次读取传 1。",
                },
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "最多读取多少行，通常传 100。",
                },
            },
            "required": ["path", "start_line", "max_lines"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# def resolve_path(path: str):


def list_directory(path: str):
    return os.listdir(path)


def skip_one_line(f):
    while True:
        chunk = f.readline(MAX_CONTENT_BYTES + 1)

        if not chunk:
            return False

        if chunk.endswith("\n"):
            return True


def read_file(path: str, start_line: int = 1, max_lines: int = 100):
    if type(start_line) is not int or start_line < 1:
        raise ValueError("start_line 必须是大于等于 1 的整数。")

    if type(max_lines) is not int or not 1 <= max_lines <= MAX_READ_LINES:
        raise ValueError("max_line 必须是 1~300 的整数。")

    lines = []
    used_bytes = 0

    with open(path, "r", encoding="utf-8") as f:
        for _ in range(start_line - 1):
            if not skip_one_line(f):
                break
        else:
            while len(lines) < max_lines:
                line_number = start_line + len(lines)

                line = f.readline(MAX_CONTENT_BYTES + 1)

                if not line:
                    break

                line_bytes = len(line.encode("utf-8"))

                if line_bytes > MAX_CONTENT_BYTES:
                    raise ValueError(
                        f"第{line_number} 行超过 20 KiB，"
                        "此工具无法返回该行；"
                        "请先拆分该行，或使用支持行内分块的工具。"
                    )

                if used_bytes + line_bytes > MAX_CONTENT_BYTES:
                    return {
                        "path": path,
                        "start_line": start_line,
                        "end_line": line_number - 1,
                        "truncated": True,
                        "next_start_line": line_number,
                        "content": "".join(lines),
                    }
                lines.append(line)
                used_bytes += line_bytes
        has_more = bool(f.read(1))

    return {
        "path": path,
        "start_line": start_line,
        "end_line": start_line + len(lines) - 1 if lines else None,
        "truncated": has_more,
        "next_start_line": start_line + len(lines) if has_more else None,
        "content": "".join(lines),
    }


def run_agent(messages):

    response = client.responses.create(
        model="deepseek-flash", tools=tools, input=messages
    )

    return response


tool_handlers = {
    "list_directory": list_directory,
    "read_file": read_file,
}


def main():
    messages = [{"role": "system", "content": "你是一个严厉的导师。"}]

    while True:
        user_text = input("你：").strip()

        if user_text.lower() in {"exit", "quit"}:
            break

        if not user_text:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )
        # while True:
        for _ in range(MAX_MODEL_CALLS):
            response = run_agent(messages=messages)

            messages.extend(response.output)

            calls = [item for item in response.output if item.type == "function_call"]

            if response.output_text:
                print("Agent:", response.output_text)

            if not calls:
                print("response status: ", response.status)
                if response.usage:
                    print(
                        f"[本次输入：{response.usage.input_tokens} tokens,"
                        f"本次输出：{response.usage.output_tokens} tokens]"
                    )
                break

            for call in calls:
                try:
                    handler = tool_handlers.get(call.name)
                    if handler is None:
                        raise ValueError(f"未知工具：{call.name}")

                    arguments = json.loads(call.arguments)
                    result = handler(**arguments)

                    output = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False)
                    )
                except Exception as exc:
                    output = json.dumps(
                        {
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )

                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )
        else:
            print(f"Agent: 本轮已达到 {MAX_MODEL_CALLS} 次模型请求上限。")


if __name__ == "__main__":
    main()
