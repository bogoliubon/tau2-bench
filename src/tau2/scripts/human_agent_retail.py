#!/usr/bin/env python3
"""
Interactive interface for humans to act as the agent in tau2-bench retail environment.

This script allows you to:
- Type messages to respond to customers
- Call API tools interactively
- See the conversation history and environment state
- Practice being a customer service agent
"""

import json
import logging
from typing import Optional

from loguru import logger
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from tau2.gym.gym_agent import AgentGymEnv
from tau2.run import load_tasks
from tau2.utils.tools import is_functional_tool_call, parse_functional_tool_call

# Initialize Rich console
console = Console()


def disable_logging():
    """Disable all logging for cleaner CLI output."""
    logger.remove()
    logger.add(lambda msg: None, level="CRITICAL")
    logging.getLogger().setLevel(logging.CRITICAL)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).disabled = True


def display_welcome():
    """Display welcome message."""
    welcome_text = Text()
    welcome_text.append("🛍️  ", style="bold blue")
    welcome_text.append("Retail Customer Service Agent Simulator", style="bold green")

    welcome_panel = Panel(
        """Welcome! You will act as a customer service agent in a retail environment.

[bold]Your Goal:[/bold]
• Help customers with their orders, returns, exchanges, and account questions
• Use the available API tools to query and modify the database
• Follow the agent policy guidelines
• Provide excellent customer service

[bold]How It Works:[/bold]
• You'll practice with 74 training scenarios
• You'll see customer messages and can respond with text or tool calls
• Type messages naturally to communicate with customers
• Use tool calls like: [cyan]get_user_details(user_id='U123')[/cyan]
• Follow the policy to ensure you're following company guidelines

[bold cyan]Ready to help some customers?[/bold cyan] 🚀""",
        title=welcome_text,
        border_style="blue",
        box=box.DOUBLE,
    )
    console.print(welcome_panel)


def display_tasks(domain: str = "retail", task_split_set: Optional[str] = "train"):
    """Display available retail tasks and let user choose one.

    Defaults to 'train' split for practice scenarios.
    """
    try:
        tasks = load_tasks(domain, task_split_set)
    except Exception as e:
        console.print(f"[red]Error loading tasks: {e}[/red]")
        raise

    # Create a table for tasks
    split_name = f" ({task_split_set} split)" if task_split_set else ""
    table = Table(title=f"📋 Available Customer Service Scenarios{split_name}", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center", width=8)
    table.add_column("Task ID", style="green", justify="left", width=10)
    table.add_column("Scenario", style="white", justify="left")

    for i, task in enumerate(tasks, 1):
        # Get description safely
        try:
            if hasattr(task, "description") and task.description:
                if isinstance(task.description, str):
                    description = task.description
                elif hasattr(task.description, "purpose"):
                    description = task.description.purpose
                else:
                    description = str(task.description)
            else:
                description = "Customer service task"
        except Exception:
            description = "Customer service task"

        # Truncate long descriptions
        if len(description) > 80:
            description = description[:77] + "..."

        table.add_row(str(i), task.id, description)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a scenario[/bold blue] (1-{len(tasks)})",
                default="1",
            )
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(tasks):
                return tasks[choice_idx]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(tasks)}[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def display_policy(policy: str):
    """Display the agent policy."""
    if not policy:
        console.print(Panel("No policy available.", style="yellow"))
        return

    # Parse as markdown for better formatting
    policy_md = Markdown(policy)
    policy_panel = Panel(
        policy_md,
        title="📋 Agent Policy & Guidelines",
        border_style="yellow",
        box=box.ROUNDED,
    )
    console.print(policy_panel)


def display_tools(tools):
    """Display available API tools."""
    if not tools:
        console.print(Panel("No tools available.", style="red"))
        return

    # Separate tools into categories
    read_tools = []
    write_tools = []
    other_tools = []

    for tool in tools:
        tool_name = tool.name.lower()
        if any(
            x in tool_name
            for x in ["find", "get", "list", "search", "query", "check", "view"]
        ):
            read_tools.append(tool)
        elif any(
            x in tool_name
            for x in [
                "cancel",
                "modify",
                "update",
                "change",
                "exchange",
                "return",
                "create",
                "delete",
            ]
        ):
            write_tools.append(tool)
        else:
            other_tools.append(tool)

    # Create table for READ tools
    if read_tools:
        console.print("\n[bold cyan]📖 READ TOOLS[/bold cyan] (Query information)")
        table = Table(box=box.SIMPLE)
        table.add_column("Tool Name", style="cyan", justify="left")
        table.add_column("Description", style="white", justify="left")

        for tool in read_tools:
            desc = (
                tool.short_desc
                if hasattr(tool, "short_desc") and tool.short_desc
                else "No description"
            )
            table.add_row(tool.name, desc)

        console.print(table)

    # Create table for WRITE tools
    if write_tools:
        console.print("\n[bold yellow]✏️  WRITE TOOLS[/bold yellow] (Modify database)")
        table = Table(box=box.SIMPLE)
        table.add_column("Tool Name", style="yellow", justify="left")
        table.add_column("Description", style="white", justify="left")

        for tool in write_tools:
            desc = (
                tool.short_desc
                if hasattr(tool, "short_desc") and tool.short_desc
                else "No description"
            )
            table.add_row(tool.name, desc)

        console.print(table)

    # Create table for OTHER tools
    if other_tools:
        console.print("\n[bold green]🔧 OTHER TOOLS[/bold green]")
        table = Table(box=box.SIMPLE)
        table.add_column("Tool Name", style="green", justify="left")
        table.add_column("Description", style="white", justify="left")

        for tool in other_tools:
            desc = (
                tool.short_desc
                if hasattr(tool, "short_desc") and tool.short_desc
                else "No description"
            )
            table.add_row(tool.name, desc)

        console.print(table)


def display_tool_details(tools):
    """Display detailed tool information including parameters."""
    if not tools:
        console.print(Panel("No tools available.", style="red"))
        return

    for tool in tools:
        console.print(f"\n[bold cyan]📌 {tool.name}[/bold cyan]")

        # Description
        desc = (
            tool.long_desc
            if hasattr(tool, "long_desc") and tool.long_desc
            else tool.short_desc
            if hasattr(tool, "short_desc") and tool.short_desc
            else "No description available"
        )
        console.print(f"  [dim]{desc}[/dim]")

        # Parameters
        if hasattr(tool, "params") and tool.params:
            try:
                params_schema = tool.params.model_json_schema()
                if "properties" in params_schema:
                    console.print(f"  [bold]Parameters:[/bold]")
                    for param_name, param_info in params_schema["properties"].items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        required = (
                            "required"
                            if param_name in params_schema.get("required", [])
                            else "optional"
                        )
                        console.print(
                            f"    • [yellow]{param_name}[/yellow] ([green]{param_type}[/green], {required}): {param_desc}"
                        )
            except Exception:
                console.print(f"  [dim]Parameters: Unable to display[/dim]")

        # Example usage
        console.print(f"  [bold]Example:[/bold]")
        if tool.name == "find_user_id_by_email":
            console.print(f"    [cyan]{tool.name}(email='john@example.com')[/cyan]")
        elif tool.name == "get_user_details":
            console.print(f"    [cyan]{tool.name}(user_id='U123')[/cyan]")
        elif tool.name == "get_order_details":
            console.print(f"    [cyan]{tool.name}(order_id='#W0000001')[/cyan]")
        elif tool.name == "cancel_pending_order":
            console.print(
                f"    [cyan]{tool.name}(order_id='#W0000001', reason='no longer needed')[/cyan]"
            )
        else:
            # Generic example
            if hasattr(tool, "params") and tool.params:
                try:
                    params_schema = tool.params.model_json_schema()
                    if "properties" in params_schema:
                        example_params = []
                        for param_name in list(params_schema["properties"].keys())[:2]:
                            example_params.append(f"{param_name}='...'")
                        console.print(
                            f"    [cyan]{tool.name}({', '.join(example_params)})[/cyan]"
                        )
                except Exception:
                    console.print(f"    [cyan]{tool.name}(...)[/cyan]")


def format_observation(observation: str, step_count: int):
    """Format and display the current observation."""
    if not observation.strip():
        return

    title = f"💬 STEP {step_count} - CONVERSATION"

    # Split by lines and format each message
    formatted_lines = []
    lines = observation.strip().split("\n")

    for line in lines:
        if line.strip():
            if line.startswith("user:"):
                formatted_lines.append(
                    f"[bold blue]👤 CUSTOMER:[/bold blue] {line[5:].strip()}"
                )
            elif line.startswith("assistant:"):
                formatted_lines.append(
                    f"[bold green]🤖 YOU (AGENT):[/bold green] {line[10:].strip()}"
                )
            elif line.startswith("system:"):
                formatted_lines.append(
                    f"[bold yellow]⚙️  SYSTEM:[/bold yellow] {line[7:].strip()}"
                )
            else:
                formatted_lines.append(f"[white]{line.strip()}[/white]")

    content = "\n".join(formatted_lines)
    panel = Panel(content, title=title, border_style="blue", box=box.ROUNDED)
    console.print(panel)


def display_help():
    """Display help information."""
    help_content = """[bold]📋 Available Commands:[/bold]

• [cyan]help[/cyan] - Show this help message
• [cyan]tools[/cyan] - Show available API tools (summary)
• [cyan]tools-detail[/cyan] - Show detailed tool information with parameters
• [cyan]policy[/cyan] - Show agent policy and guidelines
• [cyan]quit[/cyan] or [cyan]exit[/cyan] - Exit the simulation

[bold]💡 How to Respond:[/bold]

[bold]1. Send a text message:[/bold]
   Just type your message naturally:
   [green]I'd be happy to help! Could you provide your email address?[/green]

[bold]2. Make a tool call:[/bold]
   Use function syntax with the tool name and parameters:
   [cyan]find_user_id_by_email(email='john@example.com')[/cyan]
   [cyan]get_order_details(order_id='#W0000001')[/cyan]
   [cyan]cancel_pending_order(order_id='#W0000001', reason='no longer needed')[/cyan]

[bold]📌 Important Tips:[/bold]

• Always authenticate the user before helping them (use find_user_id_by_email or find_user_id_by_name_zip)
• Get explicit confirmation before making database changes
• Only make one tool call at a time
• Follow the policy guidelines (type 'policy' to view)
• Use READ tools (get_*, find_*, list_*) to query information
• Use WRITE tools (cancel_*, modify_*, exchange_*, return_*) to make changes
• Some actions can only be done once per order (modify/exchange items)

[bold]🎯 Best Practices:[/bold]

• Be polite and professional
• Verify user identity before accessing their account
• Explain what you're doing and why
• Confirm actions before executing WRITE operations
• Follow the refund policy (gift card = immediate, others = 5-7 days)
"""
    help_panel = Panel(
        help_content,
        title="🆘 Help & Guide",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(help_panel)


def get_user_action(step_count: int, tools, policy: str) -> Optional[str]:
    """Get the next action from the user."""
    console.print(
        f"\n[bold cyan]━━━ STEP {step_count} - YOUR TURN ━━━[/bold cyan]"
    )
    console.print(
        "[dim]Type a message, make a tool call, or enter a command (type 'help' for options)[/dim]"
    )

    while True:
        action = Prompt.ask("[bold green]Your action[/bold green]")

        # Handle commands
        if action.lower() in ["quit", "exit"]:
            return None
        elif action.lower() == "help":
            display_help()
            continue
        elif action.lower() == "tools":
            display_tools(tools)
            continue
        elif action.lower() == "tools-detail":
            display_tool_details(tools)
            continue
        elif action.lower() == "policy":
            display_policy(policy)
            continue
        elif action.strip() == "":
            console.print("[yellow]Please enter an action or command[/yellow]")
            continue

        # Check if it's a tool call
        if is_functional_tool_call(action):
            try:
                tool_call = parse_functional_tool_call(action)
                console.print(
                    f"[green]🔧 Tool call parsed:[/green] [cyan]{tool_call.name}[/cyan]"
                )
                console.print(f"[dim]Arguments: {json.dumps(tool_call.arguments, indent=2)}[/dim]")
                return action
            except (ValueError, SyntaxError) as e:
                console.print(f"[red]❌ Error parsing tool call: {e}[/red]")
                console.print(
                    "[yellow]Format: function_name(arg1='value1', arg2='value2')[/yellow]"
                )
                console.print(
                    "[yellow]Example: get_user_details(user_id='U123')[/yellow]"
                )
                continue

        # It's a regular text message
        return action


def main():
    """Main function for the human agent retail interface."""
    disable_logging()

    display_welcome()

    try:
        # Step 1: Choose a task
        console.print("\n[bold]Step 1: Choose a Customer Service Scenario[/bold]")
        task = display_tasks(domain="retail")
        console.print(f"\n[green]✅ Selected:[/green] Task {task.id}")

        # Display task description if available
        if hasattr(task, "description") and task.description:
            try:
                if hasattr(task.description, "purpose"):
                    console.print(
                        f"[dim]📝 Purpose: {task.description.purpose}[/dim]"
                    )
                    if hasattr(task.description, "relevant_policies"):
                        console.print(
                            f"[dim]📋 Relevant policies: {task.description.relevant_policies}[/dim]"
                        )
                elif isinstance(task.description, str):
                    console.print(f"[dim]📝 Description: {task.description}[/dim]")
            except Exception:
                pass

        # Step 2: Create environment
        console.print("\n[bold]Step 2: Initialize Environment[/bold]")
        with console.status("[bold green]Loading environment...", spinner="dots"):
            env = AgentGymEnv(domain="retail", task_id=task.id, solo_mode=False)

        # Step 3: Reset and start
        console.print("\n[bold green]🚀 Starting simulation...[/bold green]")
        observation, info = env.reset()

        # Get tools and policy
        tools = info.get("tools", [])
        policy = info.get("policy", "")

        # Display initial info
        console.print("\n" + "=" * 80)
        console.print("[bold yellow]📚 Reference Information[/bold yellow]")
        console.print("=" * 80)

        console.print("\n[bold cyan]Available API Tools:[/bold cyan]")
        display_tools(tools)

        console.print(
            "\n[dim]💡 Tip: Type 'tools-detail' anytime to see detailed parameter information[/dim]"
        )
        console.print(
            "[dim]💡 Tip: Type 'policy' anytime to review the agent guidelines[/dim]"
        )
        console.print("[dim]💡 Tip: Type 'help' for commands and usage tips[/dim]")

        console.print("\n" + "=" * 80)
        console.print("[bold green]🎬 Simulation Started[/bold green]")
        console.print("=" * 80)

        # Main interaction loop
        step_count = 0
        while True:
            step_count += 1

            # Display current observation
            format_observation(observation, step_count)

            # Get user action
            action = get_user_action(step_count, tools, policy)
            if action is None:
                console.print("\n[yellow]👋 Exiting simulation...[/yellow]")
                break

            # Step the environment
            try:
                with console.status(
                    "[bold green]Processing action...", spinner="dots"
                ):
                    observation, reward, terminated, truncated, info = env.step(action)

                # Update tools and policy
                tools = info.get("tools", tools)
                policy = info.get("policy", policy)

                # Check if done
                if terminated:
                    format_observation(observation, step_count + 1)
                    console.print(
                        Panel(
                            f"[bold green]🏆 Simulation Completed![/bold green]\n\n"
                            f"Final Reward: [bold yellow]{reward}[/bold yellow]\n\n"
                            f"{'[green]✅ Success! You handled the customer request correctly.[/green]' if reward > 0 else '[yellow]⚠️  The solution may not be optimal. Review the policy and try again.[/yellow]'}",
                            title="🏁 Simulation Complete",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                    )
                    break
                elif truncated:
                    console.print(
                        Panel(
                            "[bold yellow]⏰ Simulation truncated (time/step limit reached)[/bold yellow]",
                            title="Simulation Truncated",
                            border_style="yellow",
                            box=box.ROUNDED,
                        )
                    )
                    break

            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                console.print("[yellow]🔄 Continuing...[/yellow]")
                continue

        # End message
        console.print(
            Panel(
                "[bold green]🎉 Thank you for practicing! Try another scenario to improve your skills.[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    except KeyboardInterrupt:
        console.print("\n\n[red]⏹️  Interrupted by user.[/red]")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]❌ Error: {e}[/bold red]\n\n"
                "Please check the logs or try again.",
                title="Error",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        raise


if __name__ == "__main__":
    main()
