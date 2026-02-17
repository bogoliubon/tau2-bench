# Create a reusable script that prints colorized conversations from your results file.
from textwrap import fill, indent
import json
import argparse
import os
import sys
import textwrap
from typing import Optional, List, Dict

import openai

ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "user": "\033[92m",       # green
    "assistant": "\033[96m",  # cyan
    "tool": "\033[95m",       # magenta
    "meta": "\033[90m",       # gray
    "error": "\033[91m",      # red
}

def _supports_color(force_color: bool) -> bool:
    if force_color:
        return True
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # On Windows 10+ most terminals support ANSI, but detect via env when possible
        return True
    return True

def _color(text: str, code: str, enable: bool) -> str:
    return f"{ANSI[code]}{text}{ANSI['reset']}" if enable else text

def _wrap(text: str, width: int) -> str:
    if width <= 0:
        return text
    # Preserve code blocks and lists minimally by line
    lines = []
    for line in text.splitlines():
        if line.strip().startswith(("```", "> ", "- ", "* ")):
            lines.append(line)
        else:
            lines.append(fill(line, width=width))
    return "\n".join(lines)

def _compact_args(arg_str: str, max_len: int = 200) -> str:
    """
    Turn a function.arguments JSON string into a compact key=value, comma-separated
    representation, truncating long values.
    """
    if not arg_str:
        return ""
    try:
        args = json.loads(arg_str)
    except Exception:
        s = arg_str.replace("\n", " ")
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    parts = []
    def trunc(v):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= 80 else s[:79] + "…"

    if isinstance(args, dict):
        for k, v in args.items():
            parts.append(f"{k}={trunc(v)}")
        out = ", ".join(parts)
    else:
        out = trunc(args)

    return out if len(out) <= max_len else out[: max_len - 1] + "…"


def print_conversation(
    file_path: str,
    task_id: int,
    trial: int,
    width: int = 100,
    color: bool = True,
):
    if not os.path.exists(file_path):
        print(_color(f"[error] File not found: {file_path}", "error", color))
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(_color(f"[error] Failed to parse JSON: {e}", "error", color))
        return

    # Find the entry with matching task_id and trial
    match = None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("task_id") == task_id and entry.get("trial") == trial:
                match = entry
                break
    else:
        print(_color("[error] Expected a list at JSON top level.", "error", color))
        return

    if match is None:
        print(_color(f"[error] No record found for task_id={task_id}, trial={trial}.", "error", color))
        return

    traj = match.get("traj", [])
    if not traj:
        print(_color("[error] No 'traj' found in the selected record.", "error", color))
        return

    # Pretty header
    header = f"Conversation — task_id={task_id}, trial={trial}"
    print(_color(header, "bold", color))
    print(_color("─" * len(header), "meta", color))

    # Instruction (if present)
    instr = None
    try:
        instr = match["info"]["task"]["instruction"]
    except Exception:
        instr = None
    if instr:
        print(_color("[instruction]", "bold", color))
        print(indent(_wrap(instr, width), "  "))
        print(_color("─" * len(header), "meta", color))

    # Build tool_call_id -> (func_name, compact_args)
    id2call = {}
    for turn in traj:
        if turn.get("role") == "assistant":
            for tc in (turn.get("tool_calls") or []):
                func = (tc or {}).get("function") or {}
                fname = func.get("name", "")
                fargs = func.get("arguments", "")
                comp = _compact_args(fargs)
                call_id = tc.get("id") or tc.get("tool_call_id")
                if call_id:
                    id2call[call_id] = (fname, comp)

    # Start from first user turn
    start_idx = 0
    for i, turn in enumerate(traj):
        if isinstance(turn, dict) and turn.get("role") == "user":
            start_idx = i
            break

    # Iterate and print
    for turn in traj[start_idx:]:
        role = turn.get("role", "assistant")
        raw_content = turn.get("content")
        tool_name = turn.get("name")                # for role == "tool"
        tool_call_id = turn.get("tool_call_id")     # for role == "tool"

        if role == "assistant":
            role_tag = _color("[assistant]", "assistant", color)
            if raw_content is None:
                # Assistant made tool call(s) only — print compact summary with args
                tool_calls = turn.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        func = (tc or {}).get("function") or {}
                        fname = func.get("name", "<?>")
                        comp = _compact_args(func.get("arguments", ""))
                        parts.append(f"{fname}({comp})" if comp else f"{fname}()")
                    msg = "→ tool call(s): " + "; ".join(parts)
                else:
                    msg = ""
            else:
                msg = raw_content

        elif role == "user":
            role_tag = _color("[user]", "user", color)
            msg = raw_content or ""

        elif role == "tool":
            # Show tool name with compact args from the originating assistant call
            fname, comp = ("", "")
            if tool_call_id and tool_call_id in id2call:
                fname, comp = id2call[tool_call_id]
            display = tool_name or fname or ""
            if display and comp:
                role_tag = _color(f"[tool:{display}({comp})]", "tool", color)
            elif display:
                role_tag = _color(f"[tool:{display}]", "tool", color)
            else:
                role_tag = _color("[tool]", "tool", color)
            msg = raw_content or ""

        elif role == "system":
            role_tag = _color("[system]", "meta", color)
            msg = raw_content or ""

        else:
            role_tag = _color(f"[{role}]", "meta", color)
            msg = raw_content or ""

        # Wrap and indent content
        msg_wrapped = _wrap(msg, width)
        print(f"{role_tag} ")
        if msg_wrapped.strip():
            print(indent(msg_wrapped, "  "))
        else:
            print(indent(_color("(no content)", "dim", color), "  "))

def filter_tasks_by_tool(
    file_path: str,
    tool_name: str,
) -> list[int]:
    """
    Filter tasks by ground truth tool calls.
    
    Args:
        file_path: Path to results.json
        tool_name: Tool name to filter by (e.g., "exchange_delivered_order_items")
    
    Returns:
        List of task_ids that match the filter criteria
        Successful ids on this result file 
    """
    if not os.path.exists(file_path):
        print(f"[error] File not found: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to parse JSON: {e}")
        return []
    
    if not isinstance(data, list):
        print("[error] Expected a list at JSON top level.")
        return []
    
    matching_task_ids = []
    successful_task_ids = []
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        task_id = entry.get("task_id")
        if task_id is None:
            continue
        
        # Extract ground truth tool calls
        try:
            gt_tools = entry["info"]["task"]["actions"]
        except (KeyError, TypeError):
            continue
        
        if not isinstance(gt_tools, list):
            continue
        
        # Check if this task matches our filter
        # At least one ground truth tool must be the target tool
        for tool_call in gt_tools:
            gt_tool_name = tool_call.get("name", "")
            if gt_tool_name == tool_name and task_id not in matching_task_ids:
                matching_task_ids.append(task_id)
                if entry.get("reward", 0.0) > 0.0:
                    successful_task_ids.append(task_id)

    return sorted(matching_task_ids), sorted(successful_task_ids)

def extract_conversation_text(
    file_path: str,
    task_id: int,
    trial: int,
    include_instruction: bool = False,
    include_system: bool = False,
    width: int = None,
) -> str:
    """
    Extract conversation as clean formatted text for feeding to LLM.
    
    Args:
        file_path: Path to results.json
        task_id: Task ID to extract
        trial: Trial number to extract
        include_instruction: Whether to include the task instruction at the top
        include_system: Whether to include system messages
        width: Line width for wrapping (None for no wrapping)
    
    Returns:
        Formatted conversation string, or empty string if not found
    """
    if not os.path.exists(file_path):
        return ""
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    
    # Find the entry
    match = None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("task_id") == task_id and entry.get("trial") == trial:
                match = entry
                break
    
    if match is None:
        return ""
    
    traj = match.get("traj", [])
    if not traj:
        return ""
    
    lines = []
    
    # Add instruction if requested
    if include_instruction:
        try:
            instr = match["info"]["task"]["instruction"]
            if instr:
                lines.append("[instruction]")
                if width:
                    lines.append(textwrap.fill(instr, width=width))
                else:
                    lines.append(instr)
                lines.append("---")
        except Exception:
            pass
    
    # Build tool_call_id -> (func_name, compact_args) mapping
    id2call = {}
    for turn in traj:
        if turn.get("role") == "assistant":
            for tc in (turn.get("tool_calls") or []):
                func = (tc or {}).get("function") or {}
                fname = func.get("name", "")
                fargs = func.get("arguments", "")
                comp = _compact_args(fargs)
                call_id = tc.get("id") or tc.get("tool_call_id")
                if call_id:
                    id2call[call_id] = (fname, comp)
    
    # Find first user turn
    start_idx = 0
    for i, turn in enumerate(traj):
        if isinstance(turn, dict) and turn.get("role") == "user":
            start_idx = i
            break
    
    # Process turns
    for turn in traj[start_idx:]:
        role = turn.get("role", "assistant")
        raw_content = turn.get("content")
        tool_name = turn.get("name")
        tool_call_id = turn.get("tool_call_id")
        
        if role == "assistant":
            if raw_content is None:
                # Tool call(s) only
                tool_calls = turn.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        func = (tc or {}).get("function") or {}
                        fname = func.get("name", "<?>")
                        comp = _compact_args(func.get("arguments", ""))
                        parts.append(f"{fname}({comp})" if comp else f"{fname}()")
                    msg = "→ tool call(s): " + "; ".join(parts)
                else:
                    msg = ""
            else:
                msg = raw_content
            
            lines.append("[assistant]")
            if msg:
                if width:
                    lines.append(textwrap.fill(msg, width=width))
                else:
                    lines.append(msg)
        
        elif role == "user":
            msg = raw_content or ""
            lines.append("[user]")
            if msg:
                if width:
                    lines.append(textwrap.fill(msg, width=width))
                else:
                    lines.append(msg)
        
        elif role == "tool":
            fname, comp = ("", "")
            if tool_call_id and tool_call_id in id2call:
                fname, comp = id2call[tool_call_id]
            display = tool_name or fname or ""
            if display and comp:
                role_tag = f"[tool:{display}({comp})]"
            elif display:
                role_tag = f"[tool:{display}]"
            else:
                role_tag = "[tool]"
            
            msg = raw_content or ""
            lines.append(role_tag)
            if msg:
                if width:
                    lines.append(textwrap.fill(msg, width=width))
                else:
                    lines.append(msg)
        
        elif role == "system":
            if include_system:
                msg = raw_content or ""
                lines.append("[system]")
                if msg:
                    if width:
                        lines.append(textwrap.fill(msg, width=width))
                    else:
                        lines.append(msg)
    
    return "\n".join(lines)


def filter_tasks_by_single_critical_tool(
    file_path: str,
    target_tool: str,
    critical_tools: Optional[List[str]] = None,
) -> List[int]:
    """
    Filter tasks that involve ONLY the target tool/flow from the critical tool list.
    
    Tasks can use other non-critical tools (like get_order_details), but cannot
    use multiple critical tools together.
    
    Special handling for "modify" - groups all modify_pending_order_* tools together.
    
    Args:
        file_path: Path to results.json
        target_tool: The critical tool/flow to filter for. 
                     Use 'modify' to match any modify_pending_order_* tool.
                     Examples: 'exchange_delivered_order_items', 'modify'
        critical_tools: List of critical tools. If None, uses default list.
    
    Returns:
        List of task_ids where ground truth contains target_tool and no other critical tools
    """
    if critical_tools is None:
        # Default critical tools list
        critical_tools = [
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "cancel_pending_order",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
        ]
    
    # Define modify tools
    modify_tools = [
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
    ]
    
    if not os.path.exists(file_path):
        print(f"[error] File not found: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to parse JSON: {e}")
        return []
    
    if not isinstance(data, list):
        print("[error] Expected a list at JSON top level.")
        return []
    
    matching_task_ids = []
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        task_id = entry.get("task_id")
        if task_id is None:
            continue
        
        # Extract ground truth tools
        try:
            gt_tools = entry["info"]["task"]["actions"]
        except (KeyError, TypeError):
            continue
        
        if not isinstance(gt_tools, list):
            continue
        
        # Check which critical tools are present
        critical_tools_present = [tool['name'] for tool in gt_tools if tool['name'] in critical_tools]
        
        # Special handling for "modify" target
        if target_tool == "modify":
            # Check if task contains ONLY modify tools (and no other critical tools)
            has_modify = any(tool in modify_tools for tool in critical_tools_present)
            has_non_modify_critical = any(tool not in modify_tools for tool in critical_tools_present)
            
            if has_modify and not has_non_modify_critical:
                if task_id not in matching_task_ids:
                    matching_task_ids.append(task_id)
        else:
            # Standard case: task must contain ONLY the target tool
            if critical_tools_present == [target_tool]:
                if task_id not in matching_task_ids:
                    matching_task_ids.append(task_id)
    
    return sorted(matching_task_ids)

def call_llm(
    prompt: str,
    model_name: str = "gpt-4o") -> str:
    """
    Call LLM API to generate response.
    
    Args:
        prompt: The prompt to send
        model_name: Specific model name (e.g., "claude-sonnet-4-5-20250929")
    
    Returns:
        Generated text response
    """
    if model_name.startswith("claude"):
        client = anthropic.Anthropic()
        model_name = model_name or "claude-sonnet-4-5-20250929"
        
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    elif model_name.startswith("gpt"):
        client = openai.OpenAI()
        model_name = model_name or "gpt-4o"
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif model_name.startswith("llama"):
        # TODO: Add llama implementation (via together.ai, replicate, or local)
        raise NotImplementedError("Llama support coming soon")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def calculate_acc_on_tasks(
    file_path: str,
    task_ids: List[int],
) -> float:
    """
    Calculate accuracy over specified task IDs.
    
    Args:
        file_path: Path to results.json
        task_ids: List of task IDs to calculate accuracy on
    
    Returns:
        Accuracy as a float (0.0 to 1.0)
    """
    if not os.path.exists(file_path):
        print(f"[error] File not found: {file_path}")
        return 0.0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to parse JSON: {e}")
        return 0.0
    
    if not isinstance(data, list):
        print("[error] Expected a list at JSON top level.")
        return 0.0
    
    id_set = set(task_ids)
    total = 0
    correct = 0
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        task_id = entry.get("task_id")
        if task_id is None or task_id not in id_set:
            continue
        
        total += 1
        if entry.get("reward", 0.0) > 0.0:
            correct += 1
    
    if total == 0:
        return 0.0
    
    return correct / total


def summarize_wiki(
    wikis: List[str],
    model_name: Optional[str] = None,
) -> str:
    """
    Merge multiple flow-specific policies into one unified policy.
    
    Args:
        wikis: List of policy strings to merge
        model: Model family to use ("claude", "gpt", "llama")
        model_name: Specific model name
    
    Returns:
        Merged policy string
    """
    if len(wikis) == 0:
        return ""
    
    if len(wikis) == 1:
        return wikis[0]
    
    # Build the merge prompt
    policies_text = ""
    for i, wiki in enumerate(wikis, 1):
        policies_text += f"=== Wiki {i} ===\n{wiki}\n\n"
    
    MERGE_PROMPT = f"""You are given multiple agent policies, each focused on a specific customer service workflow (e.g., exchanges, returns, cancellations, modifications).

Your task is to combine these into ONE document while preserving ALL workflow-specific details.

{policies_text}

Please organize these policies into a single document that:
1. Preserves ALL specific rules, constraints, and details for each workflow type
2. Removes only redundant information that appears across multiple policies (e.g., general communication guidelines)
3. Uses clear section headers to separate different workflow types
4. Maintains the same level of detail and specificity as the original policies

CRITICAL: Do NOT generalize or simplify the workflow-specific rules. Every detail about how to handle exchanges, returns, cancellations, and modifications must be preserved exactly.

Provide the complete organized policy document."""
    
    # Call LLM
    print(f"Merging {len(wikis)} policies using 'gpt-5...")
    merged_policy = call_llm(MERGE_PROMPT, "gpt-5")
    
    return merged_policy

def tool_name_to_description(tool_name: str) -> str:
    """Convert tool name to human-readable description."""
    return tool_name.replace("_", " ")


def extract_ground_truth_actions(results_path: str, task_id: int, trial: int = 0) -> str:
    """
    Extract ground truth actions from the trajectory's info field.
    
    Args:
        results_path: Path to results.json
        task_id: Task ID
        trial: Trial number
        
    Returns:
        Formatted string of ground truth actions, or empty string if not found
    """
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Find the trajectory
    for result in data:
        if result["task_id"] == task_id and result["trial"] == trial:
            try:
                actions = result["info"]["task"]["actions"]
                if not actions:
                    return ""
                
                # Format actions nicely
                formatted_actions = []
                for i, action in enumerate(actions, 1):
                    name = action.get("name", "unknown_action")
                    kwargs = action.get("kwargs", {})
                    kwargs_str = json.dumps(kwargs, indent=2)
                    formatted_actions.append(f"{i}. {name}({kwargs_str})")
                
                return "\n".join(formatted_actions)
            except (KeyError, TypeError):
                return ""
    
    return ""


def prepare_trajectory_with_ground_truth(
    results_path: str,
    task_id: int,
    trial: int,
    include_instruction: bool = False
) -> tuple[str, bool]:
    """
    Prepare trajectory text, adding ground truth for failed trajectories.
    
    Args:
        results_path: Path to results.json
        task_id: Task ID
        trial: Trial number
        include_instruction: Whether to include instruction
        
    Returns:
        Tuple of (prepared_trajectory_text, is_success)
    """
    # Get conversation text
    conversation = extract_conversation_text(
        results_path,
        task_id,
        trial,
        include_instruction=include_instruction
    )
    
    if not conversation:
        return "", False
    
    # Check if successful
    with open(results_path, "r") as f:
        data = json.load(f)
    
    is_success = False
    for result in data:
        if result["task_id"] == task_id and result["trial"] == trial:
            is_success = (result.get("reward", 0.0) == 1.0)
            break
    
    # For successful trajectories, return as-is
    if is_success:
        prepared = f"=== SUCCESSFUL TRAJECTORY ===\n\n{conversation}\n"
        return prepared, True
    
    # For failed trajectories, add ground truth
    ground_truth = extract_ground_truth_actions(results_path, task_id, trial)
    
    if ground_truth:
        prepared = f"=== FAILED TRAJECTORY ===\n\n{conversation}\n\n"
        prepared += f"--- GROUND TRUTH (What should have been done) ---\n{ground_truth}\n"
    else:
        prepared = f"=== FAILED TRAJECTORY ===\n\n{conversation}\n\n"
        prepared += f"--- GROUND TRUTH NOT AVAILABLE ---\n"
    
    return prepared, False

def tool_name_to_description(tool_name: str) -> str:
    """Convert tool name to human-readable description."""
    return tool_name.replace("_", " ")

def extract_ground_truth_actions(results_path: str, task_id: int, trial: int = 0) -> str:
    """
    Extract ground truth actions from the trajectory's info field.
    
    Args:
        results_path: Path to results.json
        task_id: Task ID
        trial: Trial number
        
    Returns:
        Formatted string of ground truth actions, or empty string if not found
    """
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Find the trajectory
    for result in data:
        if result["task_id"] == task_id and result["trial"] == trial:
            try:
                actions = result["info"]["task"]["actions"]
                if not actions:
                    return ""
                
                # Format actions nicely
                formatted_actions = []
                for i, action in enumerate(actions, 1):
                    name = action.get("name", "unknown_action")
                    kwargs = action.get("kwargs", {})
                    kwargs_str = json.dumps(kwargs, indent=2)
                    formatted_actions.append(f"{i}. {name}({kwargs_str})")
                
                return "\n".join(formatted_actions)
            except (KeyError, TypeError):
                return ""
    
    return ""


def prepare_trajectory_with_ground_truth(
    results_path: str,
    task_id: int,
    trial: int,
    include_instruction: bool = False
) -> tuple[str, bool]:
    """
    Prepare trajectory text, adding ground truth for failed trajectories.
    
    Args:
        results_path: Path to results.json
        task_id: Task ID
        trial: Trial number
        include_instruction: Whether to include instruction
        
    Returns:
        Tuple of (prepared_trajectory_text, is_success)
    """
    # Get conversation text
    conversation = extract_conversation_text(
        results_path,
        task_id,
        trial,
        include_instruction=include_instruction
    )
    
    if not conversation:
        return "", False
    
    # Check if successful
    with open(results_path, "r") as f:
        data = json.load(f)
    
    is_success = False
    for result in data:
        if result["task_id"] == task_id and result["trial"] == trial:
            is_success = (result.get("reward", 0.0) == 1.0)
            break
    
    # For successful trajectories, return as-is
    if is_success:
        prepared = f"=== SUCCESSFUL TRAJECTORY ===\n\n{conversation}\n"
        return prepared, True
    
    # For failed trajectories, add ground truth
    ground_truth = extract_ground_truth_actions(results_path, task_id, trial)
    
    if ground_truth:
        prepared = f"=== FAILED TRAJECTORY ===\n\n{conversation}\n\n"
        prepared += f"--- GROUND TRUTH (What should have been done) ---\n{ground_truth}\n"
    else:
        prepared = f"=== FAILED TRAJECTORY ===\n\n{conversation}\n\n"
        prepared += f"--- GROUND TRUTH NOT AVAILABLE ---\n"
    
    return prepared, False