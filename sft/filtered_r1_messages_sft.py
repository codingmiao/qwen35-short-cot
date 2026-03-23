import json
import re

input_file = "distill_r1_110k_sft.jsonl"
output_file = "filtered_r1_messages.jsonl"

def repo_match(repo_name: str) -> bool:
    """判断 repo_name 是否包含 stem 或 math（忽略大小写）"""
    if not repo_name:
        return False
    repo_name = repo_name.lower()
    return ("stem" in repo_name) or ("math" in repo_name)

def token_ok(data):
    """检查token长度"""
    total = (
        data.get("prompt_tokens_len", 0)
        + data.get("reasoning_content_tokens_len", 0)
        + data.get("content_tokens_len", 0)
    )
    return total <= 1536

def score_ok(data):
    """检查score"""
    return data.get("score", 0) > 9


def build_user_text(instruction, input_text):
    """构造user内容"""
    instruction = instruction or ""
    input_text = input_text or ""

    if input_text.strip():
        return instruction.strip() + "\n" + input_text.strip()
    else:
        return instruction.strip()


def transform_record(data):
    """转换为messages格式"""

    user_text = build_user_text(
        data.get("instruction", ""),
        data.get("input", "")
    )

    assistant_text = data.get("output", "")

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
    }


count_total = 0
count_keep = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        count_total += 1

        try:
            data = json.loads(line)
        except Exception:
            continue

        # ----------- 过滤条件 -----------
        if not score_ok(data):
            continue

        if not token_ok(data):
            continue

        if not repo_match(data.get("repo_name", "")):
            continue
        # --------------------------------

        new_record = transform_record(data)

        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

        count_keep += 1


print("处理完成")
print("原始数据:", count_total)
print("过滤后:", count_keep)
print("输出文件:", output_file)
