#!/usr/bin/env python3
"""
流式评估脚本
"""

import json
import time
import re
import threading
import queue
import os
import signal
import sys
from urllib import request, error
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 配置加载 ====================
def load_config(config_file: str = "config.json") -> Dict:
    """加载配置文件"""
    default_config = {
        "sample_size": 5000,
        "max_retries": 3,
        "timeout": 180,
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_workers": 16,
        "data_file": "filtered_r1_messages_test.jsonl",
        "output_file": "evaluation_results.jsonl",
    }

    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        default_config.update(file_config)

    return default_config

# ==================== 全局配置 ====================
CONFIG = load_config()

# 提示词模板（使用 %s 格式化，避免与题目中的 {} 冲突）
PROMPT_TEMPLATE = "%s\n\n请一步步推理，并把最终答案放到 \\boxed{}。"

# 裁判模型提示词（使用 %s 格式化，避免与题目中的 {} 冲突）
JUDGE_PROMPT_TEMPLATE = """请判断以下两个数学答案是否等价：

题目：%s
标准答案：%s
模型答案：%s

请结合题目背景判断两个答案是否等价或表示相同的意思。

判断标准：

1. **结合题目语境理解**
   - 如果是判断题，"×"、"错误"、"不对"、"不正确"等都表示否定，应视为等价
   - 如果是选择题，不同选项的表示形式（A/选项A/第一个选项）视为等价
   - 如果是填空题，等价的表达式或数值应视为相同

2. **LaTeX格式等价性（重要）**
   - e^{\\frac{2}{e}} 和 e^{2/e} ✅ 等价（LaTeX写法不同）
   - \\frac{1}{2} 和 1/2 ✅ 等价（都是二分之一）
   - \\sin \\frac{\\theta}{2} 和 \\sin\\left(\\frac{\\theta}{2}\\right) ✅ 等价（括号风格差异）
   - \\cos(x) 和 \\cos x ✅ 等价（括号可选）
   - 任何只是LaTeX括号风格差异（如 \\left( \\right) vs 直接括号）都视为等价

3. **方程的等价性**
   - x - y + 1 = 0 和 y = x + 1 ✅ 等价（方程变形）
   - 2x + 3 = 7 和 x = 2 ✅ 等价（方程和它的解）

4. **数值和表达式的等价性**
   - 1/2 和 0.5 ✅ 等价
   - sin²x + cos²x 和 1 ✅ 等价（三角恒等式）

输出格式（严格按照此格式）：
CORRECT [简短理由，说明为什么等价]
或
INCORRECT [简短理由，说明为什么不等价]

**重要提醒**：
- 如果两个答案在数学上等价或表示相同含义，必须输出 CORRECT
- 不要因为LaTeX格式、括号风格、书写形式等表面差异而判断为INCORRECT
- 重点关注数学本质和语义，而非形式差异
- 推理过程和最终判断必须保持一致"""

# ==================== 工具函数 ====================

def load_jsonl(file_path: str, sample_size: Optional[int] = None) -> List[Dict]:
    """加载JSONL文件到内存"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if sample_size and sample_size < len(data):
        print(f"  ℹ️  使用前 {sample_size} 条数据（固定顺序）")
        return data[:sample_size]
    return data


def load_completed_tasks(output_file: str) -> set:
    """从JSONL结果文件中加载已完成的任务"""
    if not os.path.exists(output_file):
        return set()

    completed = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        q_id = result.get("question_id")
                        api_name = result.get("api_name")
                        if q_id and api_name:
                            completed.add((q_id, api_name))
                    except json.JSONDecodeError:
                        continue  # 跳过损坏的行
    except Exception as e:
        print(f"  ⚠️  读取已完成任务时出错: {e}")
        return set()

    return completed


def extract_boxed_answer(text: str) -> Optional[str]:
    """提取最后一个\\boxed{}的内容，使用栈匹配嵌套大括号"""
    boxed_pattern = r'\\boxed\{'
    matches = []

    # 找到所有 \boxed{ 的位置
    start_positions = []
    for match in re.finditer(boxed_pattern, text):
        start_positions.append(match.start())

    # 对每个起始位置，用栈匹配对应的 }
    for start in start_positions:
        stack = []
        i = start + len(r'\boxed{')  # 跳过 \boxed{ 本身

        while i < len(text):
            char = text[i]

            if char == '{':
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                else:
                    # 找到匹配的 }
                    content = text[start + len(r'\boxed{'):i]
                    matches.append(content.strip())
                    break

            i += 1

    if matches:
        return matches[-1]
    return None


def extract_ground_truth(messages: List[Dict]) -> Optional[str]:
    """从messages中提取标准答案（assistant的最后一个\boxed{}）"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            answer = extract_boxed_answer(msg.get("content", ""))
            if answer:
                return answer
    return None


def normalize_for_compare(s: str) -> str:
    """规范化字符串用于比较"""
    s = re.sub(r'\s+', '', s)  # 移除所有空白
    s = re.sub(r'\\[a-zA-Z]+', '', s)  # 移除LaTeX命令
    s = s.replace('{', '').replace('}', '')
    return s.lower()


def compare_answers(ground_truth: str, model_answer: str, judge_api: Dict, question: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """比较答案，返回是否正确及匹配方式"""
    if not model_answer:
        return {
            "is_correct": False,
            "match_method": "no_boxed",
            "reason": "模型未输出\\boxed{}包裹的答案",
            "judge_output": None,
        }

    # 方式1：字符串比较（规范化后）
    gt_normalized = normalize_for_compare(ground_truth)
    ma_normalized = normalize_for_compare(model_answer)

    if gt_normalized == ma_normalized:
        return {
            "is_correct": True,
            "match_method": "direct_string",
            "reason": "字符串匹配（规范化后）",
            "judge_output": None,
        }

    # 方式2：裁判模型判断（传入题目，结合语境）
    prompt = JUDGE_PROMPT_TEMPLATE % (question, ground_truth, model_answer)

    result = call_api_with_retry(
        judge_api["url"],
        judge_api["model"],
        prompt,
        temperature=0.0,
        max_tokens=1024,  # 增加到1024，给裁判模型更多输出空间
        max_retries=max_retries,
        timeout=60
    )

    if result.get("error"):
        return {
            "is_correct": False,
            "match_method": "judge_error",
            "reason": f"裁判模型调用失败: {result['error']}",
            "judge_output": None,
        }

    judge_output = result["content"].strip()

    # 解析裁判输出（支持多种格式）
    # 裁判模型可能输出详细的推理过程，最终判断在最后一行
    # 可能的格式：CORRECT, **CORRECT**, INCORRECT, **INCORRECT**
    decision = None
    reason = ""

    import re

    lines = judge_output.split('\n')

    # 从后往前找最后一个 CORRECT 或 INCORRECT（因为裁判会输出推理，最后一行才是最终判断）
    decision_line_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        line_upper = lines[i].upper()
        if re.search(r'\*\*(CORRECT|INCORRECT)\*\*', line_upper):
            decision_line_idx = i
            break
        elif re.search(r'\b(CORRECT|INCORRECT)\b', line_upper):
            # 只有在没有 ** 格式时才使用普通格式（避免误匹配）
            decision_line_idx = i
            break

    if decision_line_idx >= 0:
        line = lines[decision_line_idx]  # 使用原始大小写，保留中文内容
        line_upper = line.upper()

        # 优先匹配 **CORRECT** 或 **INCORRECT**
        match = re.search(r'\*\*(CORRECT|INCORRECT)\*\*', line_upper, re.IGNORECASE)
        if not match:
            match = re.search(r'\b(CORRECT|INCORRECT)\b', line_upper, re.IGNORECASE)

        if match:
            decision = match.group(1).upper()

            # 提取同一行判断词之后的内容作为理由
            match_start = match.end()
            after_decision = line[match_start:].strip()

            if after_decision:
                reason = after_decision
            else:
                # 如果同一行没有理由，检查后续行
                reason_lines = []
                for j in range(decision_line_idx + 1, min(decision_line_idx + 6, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        break  # 遇到空行停止
                    elif next_line.startswith("**理由**") or next_line.startswith("**理由**："):
                        # 提取冒号后的内容
                        colon_pos = next_line.find("：")
                        if colon_pos >= 0:
                            content = next_line[colon_pos + 1:].strip()
                            if content:
                                reason_lines.append(content)
                        elif ": " in next_line:
                            # 英文冒号
                            content = next_line.split(": ", 1)[1].strip()
                            if content:
                                reason_lines.append(content)
                    elif not next_line.startswith("**"):
                        # 普通文本行
                        reason_lines.append(next_line)

                if reason_lines:
                    reason = " ".join(reason_lines).strip()

    # 默认判断
    if not decision:
        decision = "INCORRECT"
        reason = "无法解析裁判输出"

    if decision == "CORRECT":
        return {
            "is_correct": True,
            "match_method": "judge_model",
            "reason": reason,
            "judge_output": judge_output,
        }
    else:
        return {
            "is_correct": False,
            "match_method": "judge_model",
            "reason": reason,
            "judge_output": judge_output,
        }


def call_api(api_url: str, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, timeout: int = 180) -> Dict[str, Any]:
    """调用API（无密钥）"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
    }

    req = request.Request(
        api_url,
        data=data,
        headers=headers,
        method='POST'
    )

    with request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode('utf-8'))

    # 提取token信息
    usage = result.get("usage", {})
    return {
        "content": result["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def call_api_with_retry(api_url: str, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, max_retries: int = 3, timeout: int = 180) -> Dict[str, Any]:
    """带重试的API调用"""
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_api(api_url, model, prompt, temperature, max_tokens, timeout)
        except error.HTTPError as e:
            last_error = f"HTTP Error: {e.code} - {e.reason}"
        except error.URLError as e:
            last_error = f"URL Error: {e.reason}"
        except Exception as e:
            last_error = f"Unexpected Error: {str(e)}"

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 指数退避

    return {
        "error": last_error,
        "content": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


# ==================== 结果写入线程 ====================

def result_writer_worker(output_file: str, results_queue: queue.Queue, stop_event: threading.Event):
    """
    专门的结果写入线程
    从队列获取结果，追加写入JSONL文件
    """
    written_count = 0
    last_flush_time = time.time()

    with open(output_file, 'a', encoding='utf-8') as f:
        while not stop_event.is_set():
            try:
                # 等待结果，超时0.1秒检查stop_event
                result = results_queue.get(timeout=0.1)

                if result is None:  # 结束信号
                    break

                # 写入JSONL
                try:
                    json_str = json.dumps(result, ensure_ascii=False)
                    f.write(json_str + '\n')
                    f.flush()  # 立即刷盘

                    written_count += 1

                    # 每10秒打印一次进度
                    current_time = time.time()
                    if current_time - last_flush_time >= 10:
                        print(f"    💾 已写入 {written_count} 条结果")
                        last_flush_time = current_time

                except Exception as e:
                    print(f"    ❌ 写入失败: {e}")
                    # 写入错误记录
                    error_record = {
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "result": str(result)[:500]  # 只保存前500字符
                    }
                    f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                    f.flush()

            except queue.Empty:
                continue

    print(f"    💾 写入线程结束，共写入 {written_count} 条结果")


# ==================== 主评估流程 ====================

def evaluate_single_question(question_data: Dict, api_config: Dict, judge_api: Dict, question_idx: int, temperature: float, max_tokens: int, max_retries: int = 3, timeout: int = 180) -> Dict[str, Any]:
    """评估单个API对单道题的表现"""
    # 提取问题和标准答案
    messages = question_data.get("messages", [])
    question = ""
    for msg in messages:
        if msg.get("role") == "user":
            question = msg.get("content", "")
            break

    ground_truth = extract_ground_truth(messages)

    # 构造提示词
    prompt = PROMPT_TEMPLATE % question

    # 调用API
    start_time = time.time()
    api_result = call_api_with_retry(api_config["url"], api_config["model"], prompt, temperature, max_tokens, max_retries, timeout)
    response_time = time.time() - start_time

    question_id = str(question_idx)

    # 检查是否调用失败
    if api_result.get("error"):
        return {
            "question_id": question_id,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "ground_truth": ground_truth,
            "model_answer": None,
            "is_correct": False,
            "match_method": "api_error",
            "reason": api_result["error"],
            "judge_reasoning": None,
            "response_time": response_time,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_name": api_config["name"],
            "error_type": "api_error",
            "raw_output": None,
            "timestamp": datetime.now().isoformat(),
        }

    # 提取模型答案
    model_output = api_result["content"]
    model_answer = extract_boxed_answer(model_output)

    # 比较答案（传入题目，结合语境判断）
    comparison = compare_answers(
        ground_truth or "无标准答案",
        model_answer or "",
        judge_api,
        question,  # 传入题目，让裁判模型理解语境
        max_retries
    )

    return {
        "question_id": question_id,
        "question": question[:200] + "..." if len(question) > 200 else question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "is_correct": comparison["is_correct"],
        "match_method": comparison["match_method"],
        "reason": comparison.get("reason", ""),
        "judge_reasoning": comparison.get("judge_output") if comparison.get("match_method") == "judge_model" else None,
        "response_time": response_time,
        "prompt_tokens": api_result["prompt_tokens"],
        "completion_tokens": api_result["completion_tokens"],
        "total_tokens": api_result["total_tokens"],
        "api_name": api_config["name"],
        "error_type": None if comparison["is_correct"] else comparison.get("match_method"),
        "raw_output": model_output,
        "timestamp": datetime.now().isoformat(),
    }


def run_evaluation(config: Dict) -> None:
    """运行完整评估流程（流式写入JSONL）"""
    print(f"{'='*60}")
    print("数学推理模型评估（流式版本 - JSONL输出）")
    print(f"{'='*60}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据文件: {config['data_file']}")
    print(f"抽样题数: {config['sample_size']}")
    print(f"测试API: {len(config['test_apis'])}个")
    print(f"最大并发: {config.get('max_workers', 16)}")
    print(f"输出文件: {config['output_file']}")
    print(f"{'='*60}\n")

    # 检查输出文件是否已存在
    if os.path.exists(config["output_file"]):
        # 统计已有记录数
        try:
            with open(config["output_file"], 'r') as f:
                existing_count = sum(1 for line in f if line.strip())
            print(f"💡 发现已有结果文件: {config['output_file']}")
            print(f"   已完成记录: {existing_count} 条")
            print(f"   将继续执行剩余任务...\n")
        except Exception as e:
            print(f"⚠️  输出文件已存在但无法读取: {e}")
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                print("   已取消")
                return

    # [1/3] 加载数据到内存
    print(f"[1/3] 加载测试数据...")
    questions = load_jsonl(config["data_file"], config["sample_size"])
    print(f"  ✓ 已加载 {len(questions)} 道题到内存")
    print(f"  内存占用: ~{sum(len(json.dumps(q)) for q in questions) / 1024 / 1024:.1f} MB\n")

    # [2/3] 生成任务索引（不存储question_data，节省内存）
    print(f"[2/3] 生成任务索引...")

    # 检查已完成的任务
    completed_tasks = load_completed_tasks(config["output_file"])
    if completed_tasks:
        print(f"  ℹ️  发现已完成任务: {len(completed_tasks)} 个")
        print(f"  ℹ️  将跳过这些任务，继续执行剩余任务\n")

    all_tasks = []
    skipped_count = 0
    for question_idx in range(len(questions)):
        question_id = str(question_idx)
        for api_config in config["test_apis"]:
            task_key = (question_id, api_config["name"])
            if task_key not in completed_tasks:
                all_tasks.append((question_idx, api_config))
            else:
                skipped_count += 1

    total_tasks = len(all_tasks)
    print(f"  ✓ 总任务数: {len(questions) * len(config['test_apis'])} 个（{len(questions)} 题 × {len(config['test_apis'])} API）")
    print(f"  ✓ 已完成: {skipped_count} 个")
    print(f"  ✓ 待执行: {total_tasks} 个")
    print(f"  内存占用: ~{total_tasks * 100 / 1024:.1f} KB（仅索引）\n")

    if total_tasks == 0:
        print(f"\n✓ 所有任务已完成！")
        print(f"💡 使用统计脚本查看结果:")
        print(f"   python3 analyze_results.py {config['output_file']}\n")
        return

    # 创建线程安全的结果队列
    results_queue = queue.Queue(maxsize=1000)  # 限制队列大小，防止内存爆炸
    stop_event = threading.Event()

    # 启动写入线程
    writer_thread = threading.Thread(
        target=result_writer_worker,
        args=(config["output_file"], results_queue, stop_event),
        daemon=True  # 设为守护线程
    )
    writer_thread.start()
    print(f"  ✓ 写入线程已启动\n")

    # [3/3] 并发评估
    print(f"[3/3] 开始并发评估...")
    print(f"  并发模式: 所有任务统一调度")
    print(f"  最大并发: {config.get('max_workers', 16)} 个线程")
    print(f"  写入模式: 实时追加到JSONL\n")

    max_workers = config.get("max_workers", 16)
    completed_count = 0
    error_count = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {}
            for question_idx, api_config in all_tasks:
                future = executor.submit(
                    evaluate_single_question,
                    questions[question_idx],  # 按需获取question_data
                    api_config,
                    config["judge_api"],
                    question_idx,
                    config["temperature"],
                    config["max_tokens"],
                    config["max_retries"],
                    config["timeout"]
                )
                futures[future] = (question_idx, api_config["name"])

            # 收集结果并放入队列
            for future in as_completed(futures):
                question_idx, api_name = futures[future]

                try:
                    result = future.result()

                    # 实时输出进度
                    completed_count += 1
                    status = "✓" if result["is_correct"] else "✗"
                    if result["error_type"] == "api_error":
                        status = "⚠"
                        error_count += 1

                    print(f"  [{completed_count}/{total_tasks}] 题{question_idx+1} {api_name}: {status} ({result['response_time']:.2f}s)")

                    # 放入队列，由写入线程处理
                    results_queue.put(result)

                except Exception as e:
                    print(f"  [{completed_count+1}/{total_tasks}] 题{question_idx+1} {api_name}: ❌ 异常 - {e}")
                    error_count += 1

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在保存已完成的任务...")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
    finally:
        # 通知写入线程结束
        print(f"\n  正在关闭写入线程...")
        stop_event.set()
        results_queue.put(None)  # 发送结束信号
        writer_thread.join(timeout=5)

    # 打印统计
    print(f"\n{'='*60}")
    print(f"✓ 评估完成！")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总任务数: {total_tasks}")
    print(f"已完成: {completed_count}")
    print(f"错误数: {error_count}")
    print(f"结果文件: {config['output_file']}")
    print(f"{'='*60}\n")

    print(f"💡 提示：使用统计脚本查看详细结果")
    print(f"   python3 analyze_results.py {config['output_file']}\n")


# ==================== 入口点 ====================

if __name__ == "__main__":
    # 设置信号处理，优雅退出
    def signal_handler(sig, frame):
        print("\n\n⚠️  收到中断信号，正在清理...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行评估
    run_evaluation(CONFIG)
