#!/usr/bin/env python3
"""
Interactive interface for humans to act as the agent in tau2-bench airline environment.

This script allows you to:
- Type messages to respond to customers
- Call API tools interactively
- See the conversation history and environment state
- Practice being a customer service agent
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

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

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Trajectory save location
TRAJECTORY_DIR = Path("data/tau2/human_trajectories/airline")
PROGRESS_FILE = TRAJECTORY_DIR / "progress.json"


def load_progress() -> Set[str]:
    """Load the set of completed task IDs."""
    if not PROGRESS_FILE.exists():
        return set()

    try:
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get("completed_tasks", []))
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load progress file: {e}[/yellow]")
        return set()


def save_progress(completed_tasks: Set[str]):
    """Save the set of completed task IDs."""
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data = {
            "completed_tasks": sorted(list(completed_tasks)),
            "last_updated": datetime.now().isoformat(),
            "total_completed": len(completed_tasks)
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        console.print(f"[red]Error saving progress: {e}[/red]")


def save_trajectory(task_id: str, env: AgentGymEnv, reward: float):
    """Save the human trajectory for a task."""
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Get the simulation run from the environment
        if hasattr(env, '_simulation_run') and env._simulation_run:
            simulation_run = env._simulation_run

            # Create trajectory file
            trajectory_file = TRAJECTORY_DIR / f"task_{task_id}_human.json"

            # Save as JSON
            with open(trajectory_file, 'w') as f:
                f.write(simulation_run.model_dump_json(indent=2))

            console.print(f"[green]✅ Trajectory saved to:[/green] [cyan]{trajectory_file}[/cyan]")
            return True
        else:
            console.print("[yellow]⚠️  Could not save trajectory: simulation data not available[/yellow]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error saving trajectory: {e}[/red]")
        return False


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
    welcome_text.append("✈️  ", style="bold blue")
    welcome_text.append("Airline Customer Service Agent Simulator", style="bold green")

    welcome_panel = Panel(
        """Welcome! You will act as a customer service agent in an airline environment.

[bold]Your Goal:[/bold]
• Help customers with their reservations, bookings, cancellations, and travel questions
• Use the available API tools to query and modify the database
• Follow the agent policy guidelines
• Provide excellent customer service

[bold]How It Works:[/bold]
• You'll practice with 30 training scenarios
• You'll see customer messages and can respond with text or tool calls
• Type messages naturally to communicate with customers
• Use tool calls like: [cyan]get_reservation_details(reservation_id='ABC123')[/cyan]
• Follow the policy to ensure you're following company guidelines

[bold cyan]Ready to help some travelers?[/bold cyan] 🚀""",
        title=welcome_text,
        border_style="blue",
        box=box.DOUBLE,
    )
    console.print(welcome_panel)


def display_tasks(domain: str = "airline", task_split_set: Optional[str] = "train", completed_tasks: Optional[Set[str]] = None):
    """Display available airline tasks and let user choose one.

    Defaults to 'train' split for practice scenarios.
    Shows completion status if completed_tasks is provided.
    """
    try:
        tasks = load_tasks(domain, task_split_set)
    except Exception as e:
        console.print(f"[red]Error loading tasks: {e}[/red]")
        raise

    if completed_tasks is None:
        completed_tasks = set()

    # Show progress summary
    if completed_tasks:
        console.print(f"\n[bold]Progress:[/bold] {len(completed_tasks)}/{len(tasks)} tasks completed")

    # Create a table for tasks
    split_name = f" ({task_split_set} split)" if task_split_set else ""
    table = Table(title=f"📋 Available Customer Service Scenarios{split_name}", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center", width=8)
    table.add_column("Status", style="white", justify="center", width=10)
    table.add_column("Task ID", style="green", justify="left", width=10)
    table.add_column("Scenario", style="white", justify="left")

    for i, task in enumerate(tasks, 1):
        # Get description safely
        try:
            if hasattr(task, "description") and task.description:
                if isinstance(task.description, str):
                    description = task.description
                elif hasattr(task.description, "purpose"):
                    description = task.description.purpose if task.description.purpose else "Customer service task"
                else:
                    description = str(task.description)
            else:
                description = "Customer service task"
        except Exception:
            description = "Customer service task"

        # Ensure description is not None
        if description is None:
            description = "Customer service task"

        # Truncate long descriptions
        if len(description) > 70:
            description = description[:67] + "..."

        # Determine status
        is_completed = task.id in completed_tasks
        status = "[green]✓ Done[/green]" if is_completed else ""

        table.add_row(str(i), status, task.id, description)

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
    """Display available API tools with their parameters."""
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
                "book",
                "send",
            ]
        ):
            write_tools.append(tool)
        else:
            other_tools.append(tool)

    # Helper function to get parameter string
    def get_params_string(tool):
        """Get a formatted string of tool parameters."""
        if not hasattr(tool, "params") or not tool.params:
            return "none"

        try:
            params_schema = tool.params.model_json_schema()
            if "properties" not in params_schema:
                return "none"

            required_params = params_schema.get("required", [])
            param_list = []

            for param_name in params_schema["properties"].keys():
                if param_name in required_params:
                    param_list.append(f"[red]{param_name}*[/red]")
                else:
                    param_list.append(f"[dim]{param_name}[/dim]")

            return ", ".join(param_list)
        except Exception:
            return "unknown"

    # Create table for READ tools
    if read_tools:
        console.print("\n[bold cyan]📖 READ TOOLS[/bold cyan] (Query information)")
        table = Table(box=box.SIMPLE)
        table.add_column("Tool Name", style="cyan", justify="left", width=30)
        table.add_column("Parameters", style="white", justify="left", width=40)
        table.add_column("Description", style="dim", justify="left")

        for tool in read_tools:
            desc = (
                tool.short_desc
                if hasattr(tool, "short_desc") and tool.short_desc
                else "No description"
            )
            params = get_params_string(tool)
            table.add_row(tool.name, params, desc)

        console.print(table)
        console.print("[dim]  * = required parameter[/dim]")

    # Create table for WRITE tools
    if write_tools:
        console.print("\n[bold yellow]✏️  WRITE TOOLS[/bold yellow] (Modify database)")
        table = Table(box=box.SIMPLE)
        table.add_column("Tool Name", style="yellow", justify="left", width=30)
        table.add_column("Parameters", style="white", justify="left", width=40)
        table.add_column("Description", style="dim", justify="left")

        for tool in write_tools:
            desc = (
                tool.short_desc
                if hasattr(tool, "short_desc") and tool.short_desc
                else "No description"
            )
            params = get_params_string(tool)
            table.add_row(tool.name, params, desc)

        console.print(table)
        console.print("[dim]  * = required parameter[/dim]")

    # Create table for OTHER tools
    if other_tools:
        console.print("\n[bold green]🔧 OTHER TOOLS[/bold green]")
        table = Table(box=box.SIMPLE)
        table.add_column("Tool Name", style="green", justify="left", width=30)
        table.add_column("Parameters", style="white", justify="left", width=40)
        table.add_column("Description", style="dim", justify="left")

        for tool in other_tools:
            desc = (
                tool.short_desc
                if hasattr(tool, "short_desc") and tool.short_desc
                else "No description"
            )
            params = get_params_string(tool)
            table.add_row(tool.name, params, desc)

        console.print(table)
        console.print("[dim]  * = required parameter[/dim]")


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
        if tool.name == "get_reservation_details":
            console.print(f"    [cyan]{tool.name}(reservation_id='ABC123')[/cyan]")
        elif tool.name == "get_user_details":
            console.print(f"    [cyan]{tool.name}(user_id='sara_doe_496')[/cyan]")
        elif tool.name == "cancel_reservation":
            console.print(f"    [cyan]{tool.name}(reservation_id='ABC123')[/cyan]")
        elif tool.name == "search_direct_flight":
            console.print(f"    [cyan]{tool.name}(origin='JFK', destination='LAX', date='2024-05-15')[/cyan]")
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


def format_tool_output(tool_output: str) -> str:
    """Format tool output JSON into readable tables and structures."""
    try:
        # Try to parse as JSON
        data = json.loads(tool_output)

        # Create formatted output
        formatted_parts = []

        # Handle list of flights (search results)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            # Airport list
            if isinstance(first, dict) and "iata" in first:
                formatted_parts.append("[bold cyan]Available Airports:[/bold cyan]")
                table = Table(box=box.SIMPLE, show_header=True)
                table.add_column("IATA", style="cyan", width=8)
                table.add_column("City", style="white", width=20)
                for airport in data:
                    table.add_row(airport.get("iata", ""), airport.get("city", ""))
                from io import StringIO
                from rich.console import Console as RichConsole
                string_console = RichConsole(file=StringIO(), width=100)
                string_console.print(table)
                formatted_parts.append(string_console.file.getvalue())

            # Direct flight results
            elif isinstance(first, dict) and "flight_number" in first:
                from io import StringIO
                from rich.console import Console as RichConsole
                formatted_parts.append(f"[bold cyan]Available Flights[/bold cyan] [dim]({len(data)} found)[/dim]")
                table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold", pad_edge=False)
                table.add_column("#", style="dim", width=3, justify="right")
                table.add_column("Flight", style="bold cyan", width=9)
                table.add_column("Route", style="white", width=10)
                table.add_column("Date", style="white", width=12)
                table.add_column("Dep → Arr (EST)", style="green", width=20)
                table.add_column("Business", style="yellow", width=16, justify="right")
                table.add_column("Economy", style="yellow", width=16, justify="right")
                table.add_column("Basic Eco", style="yellow", width=16, justify="right")

                def fmt_cabin(prices, seats, cabin):
                    p = prices.get(cabin, "")
                    s = seats.get(cabin, "?")
                    return f"${p} ({s} seats)" if p != "" else "—"

                for i, flight in enumerate(data, 1):
                    prices = flight.get("prices", {})
                    seats = flight.get("available_seats", {})
                    dep = flight.get("scheduled_departure_time_est", "")
                    arr = flight.get("scheduled_arrival_time_est", "")
                    table.add_row(
                        str(i),
                        flight.get("flight_number", ""),
                        f"{flight.get('origin', '')}→{flight.get('destination', '')}",
                        flight.get("date", "") or "—",
                        f"{dep} → {arr}" if dep or arr else "—",
                        fmt_cabin(prices, seats, "business"),
                        fmt_cabin(prices, seats, "economy"),
                        fmt_cabin(prices, seats, "basic_economy"),
                    )
                string_console = RichConsole(file=StringIO(), width=120)
                string_console.print(table)
                formatted_parts.append(string_console.file.getvalue())

            # One-stop flight results (list of pairs)
            elif isinstance(first, list) and len(first) == 2:
                from io import StringIO
                from rich.console import Console as RichConsole
                formatted_parts.append(f"[bold cyan]One-Stop Flights[/bold cyan] [dim]({len(data)} options)[/dim]")

                def fmt_cabin(prices, seats, cabin):
                    p = prices.get(cabin, "")
                    s = seats.get(cabin, "?")
                    return f"${p} ({s} seats)" if p != "" else "—"

                for i, (leg1, leg2) in enumerate(data, 1):
                    formatted_parts.append(f"\n  [bold]Option {i}[/bold]")
                    table = Table(box=box.SIMPLE, show_header=True, header_style="bold", show_edge=False, pad_edge=False)
                    table.add_column("Leg", style="dim", width=5)
                    table.add_column("Flight", style="bold cyan", width=9)
                    table.add_column("Route", style="white", width=10)
                    table.add_column("Date", style="white", width=12)
                    table.add_column("Dep → Arr (EST)", style="green", width=20)
                    table.add_column("Business", style="yellow", width=16, justify="right")
                    table.add_column("Economy", style="yellow", width=16, justify="right")
                    table.add_column("Basic Eco", style="yellow", width=16, justify="right")
                    for leg_num, leg in [(1, leg1), (2, leg2)]:
                        prices = leg.get("prices", {})
                        seats = leg.get("available_seats", {})
                        dep = leg.get("scheduled_departure_time_est", "")
                        arr = leg.get("scheduled_arrival_time_est", "")
                        table.add_row(
                            f"✈ {leg_num}",
                            leg.get("flight_number", ""),
                            f"{leg.get('origin', '')}→{leg.get('destination', '')}",
                            leg.get("date", "") or "—",
                            f"{dep} → {arr}" if dep or arr else "—",
                            fmt_cabin(prices, seats, "business"),
                            fmt_cabin(prices, seats, "economy"),
                            fmt_cabin(prices, seats, "basic_economy"),
                        )
                    string_console = RichConsole(file=StringIO(), width=120)
                    string_console.print(table)
                    formatted_parts.append(string_console.file.getvalue())
            else:
                formatted_parts.append(json.dumps(data, indent=2))

        # Handle reservation details
        elif isinstance(data, dict) and "reservation_id" in data:
            status = data.get("status", "active")
            status_color = "red" if status == "cancelled" else "green"
            formatted_parts.append(f"[bold cyan]Reservation Details:[/bold cyan]")
            formatted_parts.append(f"  Reservation ID: [yellow]{data.get('reservation_id')}[/yellow]")
            formatted_parts.append(f"  Status: [{status_color}]{status or 'active'}[/{status_color}]")
            formatted_parts.append(f"  User ID: {data.get('user_id')}")
            formatted_parts.append(f"  Trip: {data.get('origin', '')} → {data.get('destination', '')} ({data.get('flight_type', '')})")
            formatted_parts.append(f"  Cabin: {data.get('cabin', '')}")
            formatted_parts.append(f"  Insurance: {data.get('insurance', '')}")
            formatted_parts.append(f"  Baggages: {data.get('total_baggages', 0)} total, {data.get('nonfree_baggages', 0)} paid")

            if "flights" in data and data.get("flights"):
                formatted_parts.append(f"\n  [bold]Flights:[/bold]")
                for flight in data.get("flights", []):
                    formatted_parts.append(
                        f"    • [cyan]{flight.get('flight_number', '')}[/cyan] "
                        f"{flight.get('origin', '')} → {flight.get('destination', '')} "
                        f"on {flight.get('date', '')} — ${flight.get('price', 0)}/person"
                    )

            if "passengers" in data and data.get("passengers"):
                formatted_parts.append(f"\n  [bold]Passengers:[/bold]")
                for pax in data.get("passengers", []):
                    formatted_parts.append(
                        f"    • {pax.get('first_name', '')} {pax.get('last_name', '')} (DOB: {pax.get('dob', '')})"
                    )

            if "payment_history" in data and data.get("payment_history"):
                formatted_parts.append(f"\n  [bold]Payment History:[/bold]")
                total = 0
                for payment in data.get("payment_history", []):
                    amount = payment.get("amount", 0)
                    payment_id = payment.get("payment_id", "N/A")
                    total += amount
                    color = "red" if amount < 0 else "green"
                    label = "Refund" if amount < 0 else "Payment"
                    formatted_parts.append(f"    • [{color}]{label}:[/{color}] ${abs(amount)} ({payment_id})")
                formatted_parts.append(f"\n  [bold]Net Total:[/bold] [green]${total}[/green]")

            formatted_parts.append(f"\n  Created: {data.get('created_at', '')}")

        # Handle user details
        elif isinstance(data, dict) and "user_id" in data:
            formatted_parts.append(f"[bold cyan]User Details:[/bold cyan]")
            formatted_parts.append(f"  User ID: [yellow]{data.get('user_id')}[/yellow]")
            formatted_parts.append(f"  Membership: [bold]{data.get('membership', 'regular')}[/bold]")

            if "name" in data:
                name = data.get("name", {})
                formatted_parts.append(f"  Name: {name.get('first_name', '')} {name.get('last_name', '')}")

            if "email" in data:
                formatted_parts.append(f"  Email: {data.get('email')}")

            if "dob" in data:
                formatted_parts.append(f"  Date of Birth: {data.get('dob')}")

            if "address" in data:
                addr = data.get("address", {})
                formatted_parts.append(f"  Address:")
                formatted_parts.append(f"    {addr.get('address1', '')}")
                if addr.get('address2'):
                    formatted_parts.append(f"    {addr.get('address2')}")
                formatted_parts.append(f"    {addr.get('city', '')}, {addr.get('state', '')} {addr.get('zip', '')}")
                if addr.get('country'):
                    formatted_parts.append(f"    {addr.get('country')}")

            if "payment_methods" in data:
                formatted_parts.append(f"\n  [bold]Payment Methods:[/bold]")
                for pm_id, pm in data.get("payment_methods", {}).items():
                    pm_type = pm.get("source", "unknown")
                    if pm_type == "credit_card":
                        formatted_parts.append(f"    • [yellow]{pm_id}[/yellow] - Credit Card: {pm.get('brand', '')} ****{pm.get('last_four', '')}")
                    elif pm_type == "gift_card":
                        formatted_parts.append(f"    • [yellow]{pm_id}[/yellow] - Gift Card (Balance: ${pm.get('amount', 0)})")
                    elif pm_type == "certificate":
                        formatted_parts.append(f"    • [yellow]{pm_id}[/yellow] - Certificate (Value: ${pm.get('amount', 0)})")
                    else:
                        formatted_parts.append(f"    • [yellow]{pm_id}[/yellow] - {pm_type}")

            if "saved_passengers" in data and data.get("saved_passengers"):
                formatted_parts.append(f"\n  [bold]Saved Passengers:[/bold]")
                for pax in data.get("saved_passengers", []):
                    formatted_parts.append(
                        f"    • {pax.get('first_name', '')} {pax.get('last_name', '')} (DOB: {pax.get('dob', '')})"
                    )

            if "reservations" in data:
                formatted_parts.append(f"\n  [bold]Reservations:[/bold] {', '.join(data.get('reservations', []))}")

        # Handle simple string response
        elif isinstance(data, str):
            formatted_parts.append(f"[green]{data}[/green]")

        # Handle dict with simple structure
        elif isinstance(data, dict):
            for key, value in data.items():
                formatted_parts.append(f"  {key}: [yellow]{value}[/yellow]")

        # Fallback to pretty JSON
        else:
            formatted_parts.append(json.dumps(data, indent=2))

        return "\n".join(formatted_parts)

    except json.JSONDecodeError:
        # Not JSON, return as-is
        return tool_output


def format_observation(observation: str, step_count: int):
    """Format and display the current observation."""
    if not observation.strip():
        return

    title = f"💬 STEP {step_count} - CONVERSATION"

    # Split by lines and format each message
    formatted_lines = []
    lines = observation.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip():
            if line.startswith("user:"):
                formatted_lines.append(
                    f"[bold blue]👤 CUSTOMER:[/bold blue] {line[5:].strip()}"
                )
            elif line.startswith("assistant:"):
                formatted_lines.append(
                    f"[bold green]🤖 YOU (AGENT):[/bold green] {line[10:].strip()}"
                )
            elif line.startswith("tool:"):
                # Direct tool output (without system: prefix)
                tool_msg = line[5:].strip()
                tool_start = tool_msg.find("{")
                if tool_start == -1:
                    tool_start = tool_msg.find("[")
                if tool_start != -1:
                    # Find the complete JSON (might span multiple lines)
                    json_str = tool_msg[tool_start:]

                    # Check if JSON continues on next lines
                    open_char = json_str[0]
                    close_char = "}" if open_char == "{" else "]"
                    count = json_str.count(open_char) - json_str.count(close_char)
                    j = i + 1
                    while count > 0 and j < len(lines):
                        json_str += "\n" + lines[j]
                        count = json_str.count(open_char) - json_str.count(close_char)
                        j += 1

                    # Format the tool output
                    formatted_output = format_tool_output(json_str)
                    formatted_lines.append(f"[bold yellow]⚙️  SYSTEM (Tool Output):[/bold yellow]")
                    formatted_lines.append(formatted_output)

                    # Skip the lines we consumed
                    i = j - 1
                else:
                    formatted_lines.append(f"[bold yellow]⚙️  TOOL:[/bold yellow] {tool_msg}")
            elif line.startswith("system:"):
                # Check if this is a tool output
                system_msg = line[7:].strip()

                # Look for tool output pattern
                if "tool:" in system_msg:
                    # Extract tool output JSON
                    tool_start = system_msg.find("{")
                    if tool_start == -1:
                        tool_start = system_msg.find("[")
                    if tool_start != -1:
                        # Find the complete JSON (might span multiple lines)
                        json_str = system_msg[tool_start:]

                        open_char = json_str[0]
                        close_char = "}" if open_char == "{" else "]"
                        count = json_str.count(open_char) - json_str.count(close_char)
                        j = i + 1
                        while count > 0 and j < len(lines):
                            json_str += "\n" + lines[j]
                            count = json_str.count(open_char) - json_str.count(close_char)
                            j += 1

                        # Format the tool output
                        formatted_output = format_tool_output(json_str)
                        formatted_lines.append(f"[bold yellow]⚙️  SYSTEM (Tool Output):[/bold yellow]")
                        formatted_lines.append(formatted_output)

                        # Skip the lines we consumed
                        i = j - 1
                    else:
                        formatted_lines.append(
                            f"[bold yellow]⚙️  SYSTEM:[/bold yellow] {system_msg}"
                        )
                else:
                    formatted_lines.append(
                        f"[bold yellow]⚙️  SYSTEM:[/bold yellow] {system_msg}"
                    )
            else:
                formatted_lines.append(f"[white]{line.strip()}[/white]")
        i += 1

    content = "\n".join(formatted_lines)
    panel = Panel(content, title=title, border_style="blue", box=box.ROUNDED)
    console.print(panel)


def interactive_tool_call(tools) -> Optional[str]:
    """Interactive tool selection and parameter filling.

    Returns the formatted tool call string, or None if cancelled.
    """
    console.print("\n[bold cyan]🔧 Interactive Tool Call Builder[/bold cyan]")
    console.print("[dim]Select a tool and fill in the parameters step by step[/dim]\n")

    # Helper function to get parameter string
    def get_params_display(tool):
        """Get a formatted string of tool parameters for display."""
        if not hasattr(tool, "params") or not tool.params:
            return ""

        try:
            params_schema = tool.params.model_json_schema()
            if "properties" not in params_schema:
                return ""

            required_params = params_schema.get("required", [])
            param_list = []

            for param_name in params_schema["properties"].keys():
                if param_name in required_params:
                    param_list.append(f"{param_name}*")
                else:
                    param_list.append(f"{param_name}")

            return f"({', '.join(param_list)})"
        except Exception:
            return ""

    # Create a numbered list of tools
    tool_list = []
    for i, tool in enumerate(tools, 1):
        desc = tool.short_desc if hasattr(tool, "short_desc") and tool.short_desc else "No description"
        params_display = get_params_display(tool)
        tool_list.append((tool, desc))
        console.print(f"  [cyan]{i:2d}.[/cyan] [yellow]{tool.name}[/yellow]{params_display}")
        console.print(f"      [dim]{desc}[/dim]")

    console.print(f"\n  [cyan] 0.[/cyan] [dim]Cancel and go back[/dim]")
    console.print(f"  [dim]* = required parameter[/dim]")

    # Get tool selection
    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a tool[/bold blue] (0-{len(tools)})",
                default="0"
            )
            choice_idx = int(choice)
            if choice_idx == 0:
                return None
            if 1 <= choice_idx <= len(tools):
                selected_tool = tools[choice_idx - 1]
                break
            else:
                console.print(f"[red]Please enter a number between 0 and {len(tools)}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

    console.print(f"\n[green]✓ Selected:[/green] [bold]{selected_tool.name}[/bold]")

    # Get tool parameters
    params_dict = {}
    if hasattr(selected_tool, "params") and selected_tool.params:
        try:
            params_schema = selected_tool.params.model_json_schema()
            if "properties" in params_schema:
                required_params = params_schema.get("required", [])

                console.print(f"\n[bold cyan]Fill in the parameters:[/bold cyan]")
                console.print("[dim]Tip: You can copy-paste values from the conversation above[/dim]\n")

                for param_name, param_info in params_schema["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required_params

                    # Display parameter info
                    req_label = "[red]*required[/red]" if is_required else "[dim]optional[/dim]"
                    console.print(f"[yellow]{param_name}[/yellow] ({param_type}, {req_label})")
                    if param_desc:
                        console.print(f"  [dim]{param_desc}[/dim]")

                    # Prompt for value
                    while True:
                        value = Prompt.ask(
                            f"  Value",
                            default="" if not is_required else None
                        )

                        # Skip if optional and empty
                        if not value and not is_required:
                            break

                        # Validate and convert type
                        if value:
                            try:
                                # Handle different types
                                if param_type == "array":
                                    # Try to parse as JSON array
                                    if value.startswith("["):
                                        import json as _json
                                        params_dict[param_name] = _json.loads(value)
                                    else:
                                        # Assume comma-separated values
                                        params_dict[param_name] = [v.strip().strip("'\"") for v in value.split(",")]
                                elif param_type == "integer":
                                    params_dict[param_name] = int(value)
                                elif param_type == "number":
                                    params_dict[param_name] = float(value)
                                elif param_type == "boolean":
                                    params_dict[param_name] = value.lower() in ["true", "yes", "1"]
                                else:
                                    # String - remove quotes if present
                                    params_dict[param_name] = value.strip("'\"")
                                break
                            except (ValueError, json.JSONDecodeError) as e:
                                console.print(f"[red]Invalid value for type {param_type}: {e}[/red]")
                                console.print("[yellow]Please try again[/yellow]")
                        elif is_required:
                            console.print("[red]This parameter is required[/red]")
        except Exception as e:
            console.print(f"[red]Error processing parameters: {e}[/red]")
            return None

    # Build the tool call string
    if params_dict:
        # Format parameters as key=value pairs
        param_strs = []
        for key, value in params_dict.items():
            if isinstance(value, str):
                param_strs.append(f"{key}='{value}'")
            elif isinstance(value, list):
                # Format list with proper quotes
                formatted_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in value]
                param_strs.append(f"{key}=[{', '.join(formatted_items)}]")
            else:
                param_strs.append(f"{key}={value}")

        tool_call = f"{selected_tool.name}({', '.join(param_strs)})"
    else:
        tool_call = f"{selected_tool.name}()"

    # Show preview
    console.print(f"\n[bold green]✓ Tool call built:[/bold green]")
    console.print(f"  [cyan]{tool_call}[/cyan]")

    # Confirm
    if Prompt.ask("\n[bold blue]Execute this tool call?[/bold blue] (y/n)", choices=["y", "n"], default="y") == "y":
        return tool_call
    else:
        console.print("[yellow]Cancelled[/yellow]")
        return None


def display_help():
    """Display help information."""
    help_content = """[bold]📋 Available Commands:[/bold]

• [cyan]help[/cyan] - Show this help message
• [cyan]tools[/cyan] - Show available API tools (summary)
• [cyan]tools-detail[/cyan] - Show detailed tool information with parameters
• [cyan]policy[/cyan] - Show agent policy and guidelines
• [cyan]call[/cyan] or [cyan]api[/cyan] - Interactive tool call builder (recommended!)
• [cyan]quit[/cyan] or [cyan]exit[/cyan] - Exit the simulation

[bold]💡 How to Respond:[/bold]

[bold]1. Send a text message:[/bold]
   Just type your message naturally:
   [green]I'd be happy to help! Could you provide your reservation ID?[/green]

[bold]2. Make a tool call (Interactive - Easy!):[/bold]
   Type [cyan]call[/cyan] or [cyan]api[/cyan] to open the interactive tool builder
   • Select tool from the numbered list
   • Fill in parameters one by one
   • Copy-paste values from conversation
   • Get confirmation before executing

[bold]3. Make a tool call (Direct - Advanced):[/bold]
   Type the function call directly:
   [cyan]get_reservation_details(reservation_id='ABC123')[/cyan]
   [cyan]get_user_details(user_id='sara_doe_496')[/cyan]
   [cyan]cancel_reservation(reservation_id='ABC123')[/cyan]

[bold]📌 Important Tips:[/bold]

• Always authenticate the user before helping them (ask for user ID)
• Get explicit confirmation before making database changes
• Only make one tool call at a time
• Follow the policy guidelines (type 'policy' to view)
• Use READ tools (get_*, list_*, search_*) to query information
• Use WRITE tools (cancel_*, update_*, book_*, send_*) to make changes

[bold]🎯 Key Policy Rules:[/bold]

• Verify the customer's identity using their user ID before accessing account info
• Always confirm before cancelling reservations or making changes
• Baggages: first bag free for economy/business, none free for basic economy
• Insurance: $30 per passenger
• Non-free baggage: $50 each
• Refunds go back to the original payment method
• Certificates ($100) can be sent as compensation to gold/silver members
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
        "[dim]Type a message, 'call' for interactive tool builder, or 'help' for all options[/dim]"
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
        elif action.lower() in ["call", "api"]:
            # Interactive tool call builder
            tool_call = interactive_tool_call(tools)
            if tool_call:
                return tool_call
            else:
                console.print("[yellow]Tool call cancelled, try again[/yellow]")
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
                    "[yellow]Example: get_reservation_details(reservation_id='ABC123')[/yellow]"
                )
                continue

        # It's a regular text message
        return action


def main():
    """Main function for the human agent airline interface."""
    disable_logging()

    display_welcome()

    try:
        # Load progress
        completed_tasks = load_progress()

        # Step 1: Choose a task
        console.print("\n[bold]Step 1: Choose a Customer Service Scenario[/bold]")
        task = display_tasks(domain="airline", completed_tasks=completed_tasks)
        console.print(f"\n[green]✅ Selected:[/green] Task {task.id}")

        # Check if task is already completed
        if task.id in completed_tasks:
            console.print("[yellow]⚠️  This task was previously completed. You can redo it if you like.[/yellow]")

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
            env = AgentGymEnv(domain="airline", task_id=task.id, solo_mode=False)

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

                    # Ask if user wants to save trajectory
                    console.print("\n[bold]Would you like to save this conversation?[/bold]")
                    console.print("[dim]Only save trajectories where you performed well.[/dim]")
                    save_choice = Prompt.ask(
                        "[bold blue]Save this trajectory?[/bold blue]",
                        choices=["y", "n"],
                        default="y"
                    )

                    if save_choice.lower() == "y":
                        console.print("\n[bold]Saving your conversation...[/bold]")
                        if save_trajectory(task.id, env, reward):
                            # Mark task as completed
                            completed_tasks.add(task.id)
                            save_progress(completed_tasks)
                            total = 30  # training set size
                            console.print(f"[green]✅ Progress updated: {len(completed_tasks)}/{total} tasks completed[/green]")
                    else:
                        console.print("[yellow]⏭️  Trajectory not saved. You can redo this task later.[/yellow]")

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
