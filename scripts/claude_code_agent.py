"""
Long-running interactive agent for Claude Code to play as the retail customer service agent.

Runs as a persistent process. Reads commands from a FIFO pipe, writes observations to a file.
Saves trajectory and results to an output directory for later analysis.

Usage:
    # Terminal 1: Start the agent server for a task
    python scripts/claude_code_agent.py --task_id 5

    # Terminal 2 (Claude Code): Send responses
    echo "I'd be happy to help. Could you provide your email?" > /tmp/tau2_cmd.txt
    cat /tmp/tau2_obs.txt

    # Or send tool calls
    echo "find_user_id_by_email(email='john@email.com')" > /tmp/tau2_cmd.txt
"""

import json
import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Suppress all logging for clean output
logging.getLogger().setLevel(logging.CRITICAL)
for name in list(logging.root.manager.loggerDict):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).disabled = True

from loguru import logger
logger.remove()

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tau2.gym.gym_agent import AgentGymEnv

CMD_FILE = "/tmp/tau2_cmd.txt"
OBS_FILE = "/tmp/tau2_obs.txt"
INFO_FILE = "/tmp/tau2_info.json"
READY_FILE = "/tmp/tau2_ready"
DEFAULT_OUTPUT_DIR = str(Path(__file__).parent.parent / "claude_code_trajectories")


def write_obs(text):
    with open(OBS_FILE, "w") as f:
        f.write(text)


def write_info(data):
    with open(INFO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_trajectory(task_id, reward, simulation_run_json, output_dir):
    """Save the trajectory and update progress."""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual trajectory
    traj_file = os.path.join(output_dir, f"task_{task_id}.json")
    traj_data = {
        "task_id": str(task_id),
        "reward": reward,
        "simulation_run": json.loads(simulation_run_json) if isinstance(simulation_run_json, str) else simulation_run_json,
    }
    with open(traj_file, "w") as f:
        json.dump(traj_data, f, indent=2)

    # Update progress file
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {"completed": {}, "failed": {}}

    if reward > 0:
        progress["completed"][str(task_id)] = reward
        progress["failed"].pop(str(task_id), None)
    else:
        progress["failed"][str(task_id)] = reward

    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"  Trajectory saved to {traj_file}")


def run_task(task_id, domain="retail", user_llm="gpt-4.1", output_dir=DEFAULT_OUTPUT_DIR):
    """Run a single task interactively."""

    # Clean up old files
    for f in [CMD_FILE, OBS_FILE, INFO_FILE, READY_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Starting task {task_id}...")

    env = AgentGymEnv(
        domain=domain,
        task_id=str(task_id),
        user_llm=user_llm,
        user_llm_args={"temperature": 0.0},
    )

    observation, info = env.reset()

    # Build tools description
    tools = info.get("tools", [])
    tools_text = ""
    for tool in tools:
        name = tool.name if hasattr(tool, 'name') else str(tool)
        desc = ""
        if hasattr(tool, 'short_desc') and tool.short_desc:
            desc = tool.short_desc
        params = ""
        if hasattr(tool, 'params') and tool.params:
            try:
                schema = tool.params.model_json_schema()
                if "properties" in schema:
                    params = ", ".join(f"{k}: {v.get('type','')}" for k, v in schema["properties"].items())
            except:
                pass
        tools_text += f"  {name}({params}) - {desc}\n"

    # Get policy
    policy = info.get("policy", "")

    # Write initial observation
    obs_text = f"=== TASK {task_id} STARTED ===\n\n"
    obs_text += f"=== POLICY ===\n{policy}\n\n"
    obs_text += f"=== TOOLS ===\n{tools_text}\n"
    obs_text += f"=== OBSERVATION ===\n{observation}\n"
    obs_text += f"\n=== WAITING FOR YOUR RESPONSE ===\n"
    obs_text += f"Write your response to {CMD_FILE}\n"
    write_obs(obs_text)

    info_data = {"task_id": task_id, "step": 0, "done": False, "reward": None}
    write_info(info_data)

    # Signal ready
    Path(READY_FILE).touch()
    print(f"Ready. Observation written to {OBS_FILE}")
    print(f"Send responses by writing to {CMD_FILE}")
    print("Waiting for commands...")

    # Main loop - poll for commands
    step = 0
    reward = 0.0
    while True:
        # Wait for command file to appear
        while not os.path.exists(CMD_FILE):
            time.sleep(0.3)

        # Small delay to ensure file is fully written
        time.sleep(0.1)

        # Read command
        with open(CMD_FILE, "r") as f:
            cmd = f.read().strip()

        # Remove command file
        os.remove(CMD_FILE)

        if cmd.lower() == "quit":
            print("Quitting.")
            break

        if not cmd:
            continue

        step += 1
        print(f"[Step {step}] Action: {cmd[:100]}...")

        # Step the environment
        try:
            observation, reward, terminated, truncated, info = env.step(cmd)
        except Exception as e:
            obs_text = f"=== STEP {step} ERROR ===\n"
            obs_text += f"Your action: {cmd}\n"
            obs_text += f"Error: {e}\n\n"
            obs_text += f"=== WAITING FOR YOUR RESPONSE ===\n"
            write_obs(obs_text)
            print(f"  Error: {e}")
            continue

        obs_text = f"=== STEP {step} ===\n"
        obs_text += f"Your action: {cmd}\n\n"

        if terminated:
            obs_text += f"=== TASK TERMINATED ===\n"
            obs_text += f"Reward: {reward}\n\n"
            obs_text += f"=== FINAL OBSERVATION ===\n{observation}\n"
            write_obs(obs_text)
            info_data = {"task_id": task_id, "step": step, "done": True, "reward": reward}
            write_info(info_data)
            # Save trajectory
            sim_run = info.get("simulation_run", "{}")
            save_trajectory(task_id, reward, sim_run, output_dir)
            print(f"  TERMINATED. Reward: {reward}")
            break
        elif truncated:
            obs_text += f"=== TASK TRUNCATED ===\n"
            obs_text += f"Reward: {reward}\n\n"
            obs_text += f"=== FINAL OBSERVATION ===\n{observation}\n"
            write_obs(obs_text)
            info_data = {"task_id": task_id, "step": step, "done": True, "reward": reward}
            write_info(info_data)
            # Save trajectory
            sim_run = info.get("simulation_run", "{}")
            save_trajectory(task_id, reward, sim_run, output_dir)
            print(f"  TRUNCATED. Reward: {reward}")
            break
        else:
            obs_text += f"=== OBSERVATION ===\n{observation}\n"
            obs_text += f"\n=== WAITING FOR YOUR RESPONSE ===\n"
            write_obs(obs_text)
            info_data = {"task_id": task_id, "step": step, "done": False, "reward": None}
            write_info(info_data)
            print(f"  Observation updated.")

    # Cleanup
    for f in [CMD_FILE, READY_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nTask {task_id} finished. Final reward: {info_data.get('reward')}")


def main():
    parser = argparse.ArgumentParser(description="Claude Code interactive agent for tau2")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to run")
    parser.add_argument("--domain", type=str, default="retail", help="Domain (default: retail)")
    parser.add_argument("--user_llm", type=str, default="gpt-4.1", help="User simulator LLM")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for trajectories")

    args = parser.parse_args()
    run_task(args.task_id, args.domain, args.user_llm, args.output_dir)


if __name__ == "__main__":
    main()
