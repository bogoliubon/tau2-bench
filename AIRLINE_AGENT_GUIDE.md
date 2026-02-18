# Human-as-Agent Airline Interface Guide

This guide explains how to use the interactive interface to act as a customer service agent in the tau2-bench airline environment.

## Overview

The airline agent interface allows you to:
- **Practice being a customer service agent** - Interact with simulated customers
- **Use real API tools** - Query and modify a realistic airline reservation database
- **Learn the policy** - Follow company guidelines for customer service
- **Get feedback** - See if you solved the customer's problem correctly

## Quick Start

### Option 1: Using the CLI Command (Recommended)

```bash
tau2 airline-agent
```

### Option 2: Running the Script Directly

```bash
python -m tau2.scripts.human_agent_airline
```

## How It Works

### 1. Choose a Scenario
When you start, you'll see a list of customer service scenarios to choose from. Each scenario represents a different type of customer request (e.g., reservation cancellation, flight changes, baggage updates, compensation).

### 2. Interact with the Customer
The simulation will show you messages from the customer. You respond by:
- **Typing text messages** - Communicate with the customer naturally
- **Making tool calls** - Use API functions to query or modify the database
  - **Interactive mode** (recommended): Type `call` or `api` to use the menu-driven tool builder
  - **Direct mode** (advanced): Type the function call directly like `get_reservation_details(reservation_id='ABC123')`

### 3. Use Available Tools
You have access to API tools organized into categories:

**READ TOOLS** (Query information - no database changes):
- `get_reservation_details(reservation_id='...')` - Get full reservation info
- `get_user_details(user_id='...')` - Get user account details
- `list_all_airports()` - List all available airports with IATA codes
- `search_direct_flight(origin='...', destination='...', date='...')` - Search for direct flights
- `search_onestop_flight(origin='...', destination='...', date='...')` - Search for one-stop flights
- `get_flight_status(flight_number='...', date='...')` - Get status of a specific flight

**WRITE TOOLS** (Modify the database):
- `book_reservation(user_id='...', ...)` - Book a new reservation
- `cancel_reservation(reservation_id='...')` - Cancel a reservation
- `update_reservation_baggages(reservation_id='...', ...)` - Update baggage count
- `update_reservation_flights(reservation_id='...', ...)` - Change flights on a reservation
- `update_reservation_passengers(reservation_id='...', ...)` - Update passenger info
- `send_certificate(user_id='...', amount=...)` - Send a compensation certificate

**OTHER TOOLS**:
- `calculate(expression='...')` - Perform calculations
- `transfer_to_human_agents(summary='...')` - Escalate to human agent

## Commands During Simulation

While interacting with customers, you can use these commands:

- `help` - Show help and usage guide
- `tools` - Display available API tools (summary)
- `tools-detail` - Show detailed tool information with parameters
- `policy` - View the agent policy and guidelines
- `call` or `api` - **Open interactive tool call builder** (recommended!)
- `quit` or `exit` - Exit the simulation

## Interactive Tool Call Builder (Recommended!)

The easiest way to make API calls is using the **interactive tool builder**. Just type `call` or `api` and you'll get:

### How It Works:
1. **Select a tool** - Choose from a numbered list of all tools
2. **Fill parameters** - The system prompts you for each parameter one by one
3. **Copy-paste friendly** - Easily copy values from the conversation history
4. **Type validation** - Automatic validation for strings, numbers, arrays, etc.
5. **Preview & confirm** - See the final tool call before executing

### Example Flow:
```
Your action: call

🔧 Interactive Tool Call Builder
Select a tool and fill in the parameters step by step

   1. get_reservation_details
       Get the details of a reservation
   2. get_user_details
       Get the details of a user
  ...

Select a tool (0-14): 1
✓ Selected: get_reservation_details

Fill in the parameters:
Tip: You can copy-paste values from the conversation above

reservation_id (string, *required)
  The reservation ID, such as '8JX2WO'
  Value: 8JX2WO

✓ Tool call built:
  get_reservation_details(reservation_id='8JX2WO')

Execute this tool call? (y/n) [y]: y
```

## Example Interaction

Here's a typical interaction flow:

```
👤 CUSTOMER: Hi, I need to cancel my flight reservation

🤖 YOU: I'd be happy to help you cancel your reservation. Could you please provide your user ID and reservation ID?

👤 CUSTOMER: My user ID is sara_doe_496 and reservation ID is 8JX2WO

🤖 YOU: [Types 'call' to use interactive tool builder]
        [Selects: get_user_details]
        [Enters user_id: sara_doe_496]
        [Confirms and executes]

⚙️ SYSTEM: User details returned - confirms sara_doe_496 exists

🤖 YOU: [Types 'call' again]
        [Selects: get_reservation_details]
        [Enters reservation_id: 8JX2WO]

⚙️ SYSTEM: Reservation 8JX2WO returned - SFO → JFK, economy, 2 passengers

🤖 YOU: I can see your reservation 8JX2WO for a flight from SFO to JFK.
        Would you like me to cancel this reservation? The refund will go back to your original payment method.

👤 CUSTOMER: Yes, please cancel it

🤖 YOU: [Types 'call']
        [Selects: cancel_reservation]
        [Enters reservation_id: 8JX2WO]

⚙️ SYSTEM: Reservation cancelled successfully

🤖 YOU: Your reservation 8JX2WO has been successfully cancelled.
        The refund will be processed to your original payment method. Is there anything else I can help you with?
```

## Key Policy Guidelines

When acting as an agent, remember to:

1. **Authenticate users first** - Always verify identity using user ID before helping
2. **Get confirmation for changes** - Ask "yes/no" before modifying the database
3. **One tool call at a time** - Make only one API call per response
4. **Know the refund rules**:
   - Cancellations: refund to original payment method
   - Gift card/certificate refunds: processed immediately
   - Credit card refunds: may take a few business days
5. **Understand baggage policy**:
   - Basic economy: no free bags
   - Economy/Business: 1 free checked bag
   - Extra bags: $50 each
6. **Insurance**: $30 per passenger, added at booking
7. **Certificates**: $100 compensation certificates for eligible issues (gold/silver members)
8. **Don't make up information** - Only provide facts from the database

## Tips for Success

- **Read the policy** - Type `policy` to view the full guidelines
- **Check tool parameters** - Type `tools-detail` to see what each tool needs
- **Be conversational** - Mix tool calls with friendly messages
- **Verify information** - Always look up details before making changes
- **Collect all info first** - For complex operations like `book_reservation`, gather all required info (flights, passengers, payment) before calling

## Practice Scenarios

The interface uses the **training set with 30 tasks** for practice. The full tau2-bench airline environment has:
- **Training set**: 30 tasks (used by this interface)
- **Test set**: 20 tasks (reserved for agent evaluation)
- **Base set**: All 50 tasks combined

The 30 training scenarios cover:
- Reservation cancellations
- Flight changes (different dates, routes, cabin class)
- Baggage updates
- Passenger information corrections
- Compensation certificates
- New reservation bookings
- Policy inquiries and edge cases

Try different scenarios to practice various airline customer service situations!

## Trajectory Saving & Progress Tracking

The interface automatically saves your conversations and tracks your progress:

### Auto-Save
- Each completed task is saved to `data/tau2/human_trajectories/airline/task_{id}_human.json`
- Trajectories are saved in tau2-bench's standard format
- Compatible with the evaluation system

### Progress Tracking
- Your progress is tracked in `data/tau2/human_trajectories/airline/progress.json`
- Shows which tasks you've completed (e.g., "10/30 tasks completed")
- Task list displays **✓ Done** next to completed tasks
- You can still redo completed tasks if you want

### Resume Anytime
- **Pause and resume** - Complete some tasks, take a break, come back later
- Progress persists across sessions
- No need to complete everything in one sitting!

### Example Task List
```
Progress: 5/30 tasks completed

┌────────┬──────────┬──────────┬─────────────────────────────────┐
│ Number │ Status   │ Task ID  │ Scenario                        │
├────────┼──────────┼──────────┼─────────────────────────────────┤
│   1    │ ✓ Done   │ 0        │ Cancel reservation              │
│   2    │ ✓ Done   │ 1        │ Update flight to different date │
│   3    │          │ 2        │ Add extra baggage               │
│   4    │ ✓ Done   │ 3        │ Book new reservation            │
└────────┴──────────┴──────────┴─────────────────────────────────┘
```

## Troubleshooting

**Tool call syntax errors?**
- Use the format: `function_name(param1='value1', param2='value2')`
- Strings need quotes: `'example'` or `"example"`
- Lists use brackets: `['item1', 'item2']`
- Use the interactive builder (`call`) to avoid syntax errors

**Can't find user?**
- User IDs are in format like `sara_doe_496`
- Ask the customer to provide their exact user ID

**Can't find flight?**
- Use `search_direct_flight` or `search_onestop_flight` to find available options
- Use `list_all_airports` to see valid IATA codes
- Dates must be in YYYY-MM-DD format (e.g., `2024-05-15`)

**Booking failed?**
- Check that payment amounts sum to the exact total price
- Verify there are enough available seats in the selected cabin class
- Make sure the user has the specified payment method IDs

## Need Help?

- Type `help` during any simulation for quick reference
- Type `tools-detail` to see detailed API documentation
- Type `policy` to review the agent guidelines
- Review the example interaction above for guidance

## Feedback & Scoring

At the end of each scenario, you'll receive:
- A reward score (0.0 to 1.0)
- Feedback on whether you solved the problem correctly
- Information about what was expected

Use this feedback to improve your customer service skills!

---

Happy customer service practice! ✈️
