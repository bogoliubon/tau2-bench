# Human-as-Agent Retail Interface Guide

This guide explains how to use the interactive interface to act as a customer service agent in the tau2-bench retail environment.

## Overview

The retail agent interface allows you to:
- **Practice being a customer service agent** - Interact with simulated customers
- **Use real API tools** - Query and modify a realistic retail database
- **Learn the policy** - Follow company guidelines for customer service
- **Get feedback** - See if you solved the customer's problem correctly

## Quick Start

### Option 1: Using the CLI Command (Recommended)

```bash
tau2 retail-agent
```

### Option 2: Running the Script Directly

```bash
python -m tau2.scripts.human_agent_retail
```

## How It Works

### 1. Choose a Scenario
When you start, you'll see a list of customer service scenarios to choose from. Each scenario represents a different type of customer request (e.g., order cancellation, returns, exchanges).

### 2. Interact with the Customer
The simulation will show you messages from the customer. You respond by:
- **Typing text messages** - Communicate with the customer naturally
- **Making tool calls** - Use API functions to query or modify the database

### 3. Use Available Tools
You have access to 17 API tools organized into categories:

**READ TOOLS** (Query information - no database changes):
- `find_user_id_by_email(email='...')` - Find user by email
- `find_user_id_by_name_zip(first_name='...', last_name='...', zip='...')` - Find user by name and zip
- `get_user_details(user_id='...')` - Get user account details
- `get_order_details(order_id='...')` - Get order information
- `get_product_details(product_id='...')` - Get product variants
- `list_all_product_types()` - List all product categories

**WRITE TOOLS** (Modify the database):
- `cancel_pending_order(order_id='...', reason='...')` - Cancel an order
- `modify_pending_order_address(order_id='...', address1='...', ...)` - Update shipping address
- `modify_pending_order_items(order_id='...', item_ids=['...'], new_item_ids=['...'], payment_method_id='...')` - Change order items
- `modify_pending_order_payment(order_id='...', payment_method_id='...')` - Change payment method
- `modify_user_address(user_id='...', ...)` - Update user's default address
- `exchange_delivered_order_items(...)` - Exchange items from delivered order
- `return_delivered_order_items(...)` - Return items
- `transfer_to_human_agents(summary='...')` - Escalate to human agent

**OTHER TOOLS**:
- `calculate(expression='...')` - Perform calculations

## Commands During Simulation

While interacting with customers, you can use these commands:

- `help` - Show help and usage guide
- `tools` - Display available API tools (summary)
- `tools-detail` - Show detailed tool information with parameters
- `policy` - View the agent policy and guidelines
- `quit` or `exit` - Exit the simulation

## Example Interaction

Here's a typical interaction flow:

```
👤 CUSTOMER: Hi, I need to cancel my order

🤖 YOU: I'd be happy to help you cancel your order. To get started, could you please provide your email address?

👤 CUSTOMER: Sure, it's john.doe@example.com

🤖 YOU: [Makes tool call]
find_user_id_by_email(email='john.doe@example.com')

⚙️ SYSTEM: Found user with ID: U12345

🤖 YOU: [Makes tool call]
get_user_details(user_id='U12345')

⚙️ SYSTEM: [Returns user details with order IDs]

🤖 YOU: I can see you have order #W0000123 for a Red T-Shirt. Is that the order you'd like to cancel?

👤 CUSTOMER: Yes, that's the one

🤖 YOU: I can cancel that for you. Just to confirm - would you like to cancel order #W0000123?

👤 CUSTOMER: Yes please

🤖 YOU: [Makes tool call]
cancel_pending_order(order_id='#W0000123', reason='no longer needed')

⚙️ SYSTEM: Order cancelled successfully

🤖 YOU: Your order #W0000123 has been successfully cancelled. The refund will be processed to your original payment method within 5-7 business days. Is there anything else I can help you with?
```

## Key Policy Guidelines

When acting as an agent, remember to:

1. **Authenticate users first** - Always verify identity using email or name+zip before helping
2. **Get confirmation for changes** - Ask "yes/no" before modifying the database
3. **One tool call at a time** - Make only one API call per response
4. **Follow refund rules**:
   - Gift card refunds: Immediate
   - Other payment methods: 5-7 business days
5. **Know order status transitions**:
   - Pending orders can be cancelled or modified
   - Delivered orders can be returned or exchanged
6. **Modify/exchange limits** - Can only modify or exchange items ONCE per order
7. **Don't make up information** - Only provide facts from the database

## Tips for Success

- **Read the policy** - Type `policy` to view the full guidelines
- **Check tool parameters** - Type `tools-detail` to see what each tool needs
- **Be conversational** - Mix tool calls with friendly messages
- **Verify information** - Always look up details before making changes
- **Collect all info first** - For exchanges/modifications, gather all item IDs before calling the tool (remember: one-time only!)

## Practice Scenarios

The retail environment includes 114 different scenarios covering:
- Order cancellations
- Returns and exchanges
- Shipping address updates
- Payment method changes
- Product information requests
- Account management
- Complex multi-step issues

Try different scenarios to practice various customer service situations!

## Troubleshooting

**Tool call syntax errors?**
- Use the format: `function_name(param1='value1', param2='value2')`
- Strings need quotes: `'example'` or `"example"`
- Lists use brackets: `['item1', 'item2']`

**Can't find user?**
- Check email spelling
- Try name + zip code if email fails
- Ask customer to provide correct information

**Can't modify order?**
- Check order status (only pending orders can be modified most ways)
- Delivered orders use different tools (return/exchange)
- Already cancelled orders cannot be modified

**Modification already done?**
- Remember: `modify_pending_order_items` and `exchange_delivered_order_items` can only be called ONCE per order
- If you need to make changes, collect ALL items upfront before calling

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

Happy customer service practice! 🛍️
