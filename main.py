from openai import OpenAI
import os
from pathlib import Path
import json

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
        "description": "读取 UTF-8 文本文件内容，例如代码、Markdown、TXT、JSON。",
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
]


# def resolve_path(path: str):


def list_directory(path: str):
    return os.listdir(path)


def read_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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
        while True:
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


if __name__ == "__main__":
    main()
