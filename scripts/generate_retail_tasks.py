"""
Generate new retail tasks for tau2-bench.

Reads db.json and existing tasks.json, finds unused user/order combinations,
and generates new tasks with ground truth action sequences.

Usage:
    python3 scripts/generate_retail_tasks.py --n_tasks 85 --seed 123 \
        --output_path data/tau2/domains/retail/generated_tasks.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent / "data" / "tau2" / "domains" / "retail"


# ─── User scenario templates ─────────────────────────────────────────────────

TASK_INSTRUCTIONS = [
    "You are a busy person and want to get this resolved quickly.",
    "You are polite but brief and firm.",
    "You like to make sure all details are correct before confirming.",
    ".",
    "You are patient and cooperative.",
    "You speak concisely and only provide information when asked.",
    "You are friendly and talkative.",
    "You want to make sure everything is handled in one call.",
    "You are cautious and want to double-check everything.",
    "You are straightforward and prefer not to chat.",
]

CANCEL_REASONS_NATURAL = {
    "no longer needed": [
        "you no longer need {item_desc}",
        "you decided you don't need {item_desc} anymore",
        "you changed your mind and no longer need the items",
        "you realized you don't need the order anymore",
    ],
    "ordered by mistake": [
        "you accidentally placed this order",
        "this order was placed by mistake",
        "you didn't mean to order this",
        "you realized you ordered the wrong thing",
    ],
}

RETURN_REASONS = [
    "it doesn't fit your needs",
    "you're not satisfied with the quality",
    "you received it but decided you don't want it",
    "it's not what you expected",
    "you found a better alternative",
]

EXCHANGE_REASONS = [
    "you'd prefer a different {option_name}",
    "you want to switch to a different {option_name}",
    "you'd like to change the {option_name}",
]

UNKNOWN_INFO_OPTIONS = [
    "You do not remember your email address.",
    "You don't remember your email.",
    None,
    None,  # weight toward knowing everything
]


def load_db():
    with open(BASE_DIR / "db.json") as f:
        return json.load(f)


def load_existing_tasks():
    with open(BASE_DIR / "tasks.json") as f:
        return json.load(f)


def get_used_order_ids(tasks):
    """Get all order IDs referenced in existing tasks."""
    used = set()
    for t in tasks:
        actions = (t.get("evaluation_criteria") or {}).get("actions") or []
        for a in actions:
            args = a.get("arguments", {})
            if "order_id" in args:
                used.add(args["order_id"])
        # Also check user scenario for order IDs mentioned
        instructions = (t.get("user_scenario") or {}).get("instructions") or {}
        if isinstance(instructions, dict):
            reason = instructions.get("reason_for_call", "")
            # Extract #W... patterns
            import re
            for m in re.finditer(r"#W\d+", reason):
                used.add(m.group())
    return used


def get_used_user_ids(tasks, db):
    """Get all user IDs referenced in existing tasks."""
    used = set()
    for t in tasks:
        actions = (t.get("evaluation_criteria") or {}).get("actions") or []
        for a in actions:
            args = a.get("arguments", {})
            if "user_id" in args:
                used.add(args["user_id"])
            if "first_name" in args and "last_name" in args:
                for uid, u in db["users"].items():
                    if (u["name"]["first_name"] == args["first_name"] and
                            u["name"]["last_name"] == args["last_name"]):
                        used.add(uid)
    return used


def find_available_variant(db, product_id, current_item_id, require_different=False):
    """Find an available variant of the same product to exchange/modify to.

    Returns (new_item_id, new_variant) or (None, None).
    """
    product = db["products"].get(product_id)
    if not product:
        return None, None

    candidates = []
    for vid, variant in product["variants"].items():
        if not variant["available"]:
            continue
        if require_different and vid == current_item_id:
            continue
        if vid == current_item_id:
            continue  # prefer different variant
        candidates.append((vid, variant))

    if not candidates:
        return None, None
    return random.choice(candidates)


def pick_payment_method(user, price_difference=0.0, exclude_original=False, original_pm_id=None):
    """Pick a valid payment method for the user.

    For returns: must be original payment method or existing gift card.
    For exchange/modify: any payment method, but gift card must have sufficient balance.
    """
    pms = user["payment_methods"]
    candidates = []

    for pm_id, pm in pms.items():
        if exclude_original and pm_id == original_pm_id:
            continue
        if pm["source"] == "gift_card":
            balance = pm.get("balance", 0)
            if price_difference > 0 and balance < price_difference:
                continue  # insufficient balance
        candidates.append(pm_id)

    if not candidates:
        return None
    return random.choice(candidates)


def pick_return_payment(user, order):
    """Pick a valid refund payment method (original PM or existing gift card per policy)."""
    original_pm_id = order["payment_history"][0]["payment_method_id"]
    pms = user["payment_methods"]

    candidates = []
    # Original payment method is always valid
    if original_pm_id in pms:
        candidates.append(original_pm_id)
    # Any gift card is also valid
    for pm_id, pm in pms.items():
        if pm["source"] == "gift_card" and pm_id not in candidates:
            candidates.append(pm_id)

    if not candidates:
        return None
    return random.choice(candidates)


def describe_item(item):
    """Generate a natural language description of an item."""
    name = item["name"]
    options = item.get("options", {})
    # Pick 1-2 distinguishing options
    opt_keys = list(options.keys())
    if opt_keys:
        key = random.choice(opt_keys)
        return f"the {options[key]} {name.lower()}"
    return f"the {name.lower()}"


def describe_variant_change(current_options, new_options):
    """Describe the full target variant to avoid ambiguity."""
    target = ", ".join(f"{k}: {v}" for k, v in new_options.items())
    if target:
        return f"the variant with {target}"
    return "a different option"


def describe_payment_method(user, pm_id):
    """Generate natural description of payment method for user scenario."""
    pm = user["payment_methods"].get(pm_id, {})
    source = pm.get("source", "")
    if source == "credit_card":
        brand = pm.get("brand", "credit card")
        last_four = pm.get("last_four", "")
        return f"your {brand} card ending in {last_four}"
    elif source == "gift_card":
        return "your gift card"
    elif source == "paypal":
        return "your PayPal account"
    return "your payment method"


def generate_address():
    """Generate a random US address."""
    streets = ["Oak Street", "Maple Avenue", "Cedar Lane", "Pine Road", "Elm Drive",
               "Birch Boulevard", "Willow Way", "Spruce Court", "Ash Place", "Cherry Hill"]
    cities_states = [
        ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
        ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
        ("Phoenix", "AZ", "85001"), ("Philadelphia", "PA", "19101"),
        ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"),
        ("Dallas", "TX", "75201"), ("Austin", "TX", "73301"),
        ("Portland", "OR", "97201"), ("Denver", "CO", "80201"),
        ("Seattle", "WA", "98101"), ("Boston", "MA", "02101"),
        ("Atlanta", "GA", "30301"), ("Miami", "FL", "33101"),
    ]
    num = random.randint(100, 999)
    street = random.choice(streets)
    city, state, zip_code = random.choice(cities_states)
    suite = f"Suite {random.randint(100, 999)}" if random.random() < 0.4 else ""
    return {
        "address1": f"{num} {street}",
        "address2": suite,
        "city": city,
        "state": state,
        "country": "USA",
        "zip": zip_code,
    }


def make_action(task_id, action_idx, name, arguments):
    return {
        "action_id": f"{task_id}_{action_idx}",
        "name": name,
        "arguments": arguments,
        "info": None,
    }


# ─── Task generators ─────────────────────────────────────────────────────────

def generate_cancel_task(task_id, db, user_id, order_id, rng):
    """Generate a cancel pending order task."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    assert order["status"] == "pending", f"Order {order_id} is not pending"

    reason = rng.choice(["no longer needed", "ordered by mistake"])
    reason_templates = CANCEL_REASONS_NATURAL[reason]
    item_desc = "the items" if len(order["items"]) > 1 else describe_item(order["items"][0])
    reason_natural = rng.choice(reason_templates).format(item_desc=item_desc)

    # Auth method
    use_email = rng.random() < 0.2

    # Build actions
    actions = []
    idx = 0

    if use_email:
        actions.append(make_action(task_id, idx, "find_user_id_by_email", {"email": user["email"]}))
    else:
        actions.append(make_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1

    actions.append(make_action(task_id, idx, "get_user_details", {"user_id": user_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "get_order_details", {"order_id": order_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "cancel_pending_order", {
        "order_id": order_id,
        "reason": reason,
    }))

    # Build user scenario
    knows_order = rng.random() < 0.6
    if use_email:
        known_info = f"Your email is {user['email']}."
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    if knows_order:
        reason_text = f"You want to cancel your order {order_id} because {reason_natural}."
    else:
        reason_text = f"You placed an order recently and want to cancel it because {reason_natural}. You think the order number starts with {order_id[:4]}."

    unknown_info = rng.choice(UNKNOWN_INFO_OPTIONS) if not use_email else None

    return {
        "id": str(task_id),
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": None,
        },
    }


def generate_return_task(task_id, db, user_id, order_id, rng, n_items=None):
    """Generate a return delivered order task."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    assert order["status"] == "delivered", f"Order {order_id} is not delivered"

    items = order["items"]
    if n_items is None:
        n_items = rng.choice([1, 1, 1, 2]) if len(items) >= 2 else 1
    n_items = min(n_items, len(items))
    return_items = rng.sample(items, n_items)
    return_item_ids = [item["item_id"] for item in return_items]

    # Pick payment method (original or gift card per policy)
    payment_method_id = pick_return_payment(user, order)
    if payment_method_id is None:
        return None  # can't generate valid task

    # Auth
    use_email = rng.random() < 0.2

    # Build actions
    actions = []
    idx = 0

    if use_email:
        actions.append(make_action(task_id, idx, "find_user_id_by_email", {"email": user["email"]}))
    else:
        actions.append(make_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1

    actions.append(make_action(task_id, idx, "get_user_details", {"user_id": user_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "get_order_details", {"order_id": order_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "return_delivered_order_items", {
        "order_id": order_id,
        "item_ids": return_item_ids,
        "payment_method_id": payment_method_id,
    }))

    # Build user scenario
    item_descriptions = [describe_item(item) for item in return_items]
    if len(item_descriptions) == 1:
        items_text = item_descriptions[0]
    else:
        items_text = ", ".join(item_descriptions[:-1]) + " and " + item_descriptions[-1]

    return_reason = rng.choice(RETURN_REASONS)

    if use_email:
        known_info = f"Your email is {user['email']}."
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    # Payment preference
    pm_desc = describe_payment_method(user, payment_method_id)
    payment_pref = f"You'd like the refund to go to {pm_desc}."

    reason_text = f"You want to return {items_text} from order {order_id} because {return_reason}. {payment_pref}"

    unknown_info = rng.choice(UNKNOWN_INFO_OPTIONS) if not use_email else None

    return {
        "id": str(task_id),
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": None,
        },
    }


def generate_exchange_task(task_id, db, user_id, order_id, rng, n_items=None):
    """Generate an exchange delivered order task."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    assert order["status"] == "delivered", f"Order {order_id} is not delivered"

    items = order["items"]
    if n_items is None:
        n_items = rng.choice([1, 1, 1, 2]) if len(items) >= 2 else 1
    n_items = min(n_items, len(items))

    # Try to find items with available alternate variants
    exchange_pairs = []  # (old_item, new_item_id, new_variant)
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)

    for item in shuffled_items:
        if len(exchange_pairs) >= n_items:
            break
        new_item_id, new_variant = find_available_variant(
            db, item["product_id"], item["item_id"], require_different=True
        )
        if new_item_id:
            exchange_pairs.append((item, new_item_id, new_variant))

    if not exchange_pairs:
        return None  # no valid exchanges possible

    old_item_ids = [pair[0]["item_id"] for pair in exchange_pairs]
    new_item_ids = [pair[1] for pair in exchange_pairs]

    # Compute price difference
    old_total = sum(pair[0]["price"] for pair in exchange_pairs)
    new_total = sum(pair[2]["price"] for pair in exchange_pairs)
    price_diff = new_total - old_total  # positive means customer owes more

    # Pick payment method
    payment_method_id = pick_payment_method(user, price_difference=max(0, price_diff))
    if payment_method_id is None:
        return None

    # Auth
    use_email = rng.random() < 0.2

    # Build actions
    actions = []
    idx = 0

    if use_email:
        actions.append(make_action(task_id, idx, "find_user_id_by_email", {"email": user["email"]}))
    else:
        actions.append(make_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1

    actions.append(make_action(task_id, idx, "get_user_details", {"user_id": user_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "get_order_details", {"order_id": order_id}))
    idx += 1

    # get_product_details for each product being exchanged
    seen_products = set()
    for pair in exchange_pairs:
        pid = pair[0]["product_id"]
        if pid not in seen_products:
            actions.append(make_action(task_id, idx, "get_product_details", {"product_id": pid}))
            idx += 1
            seen_products.add(pid)

    actions.append(make_action(task_id, idx, "exchange_delivered_order_items", {
        "order_id": order_id,
        "item_ids": old_item_ids,
        "new_item_ids": new_item_ids,
        "payment_method_id": payment_method_id,
    }))

    # Build user scenario
    exchange_descriptions = []
    for old_item, new_id, new_variant in exchange_pairs:
        old_desc = describe_item(old_item)
        # Describe what they want instead
        change = describe_variant_change(old_item.get("options", {}), new_variant.get("options", {}))
        exchange_descriptions.append(f"exchange {old_desc} (change {change})")

    items_text = " and ".join(exchange_descriptions) if len(exchange_descriptions) <= 2 else ", ".join(exchange_descriptions)

    if use_email:
        known_info = f"Your email is {user['email']}."
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    pm_desc = describe_payment_method(user, payment_method_id)
    payment_pref = f"Use {pm_desc} for any price difference."

    reason_text = f"You received order {order_id} and would like to {items_text}. {payment_pref}"

    unknown_info = rng.choice(UNKNOWN_INFO_OPTIONS) if not use_email else None

    return {
        "id": str(task_id),
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": None,
        },
    }


def generate_modify_items_task(task_id, db, user_id, order_id, rng, n_items=None):
    """Generate a modify pending order items task."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    assert order["status"] == "pending", f"Order {order_id} is not pending"

    items = order["items"]
    if n_items is None:
        n_items = rng.choice([1, 1, 1, 2]) if len(items) >= 2 else 1
    n_items = min(n_items, len(items))

    # Find items with available alternate variants (must be different option per policy)
    modify_pairs = []
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)

    for item in shuffled_items:
        if len(modify_pairs) >= n_items:
            break
        new_item_id, new_variant = find_available_variant(
            db, item["product_id"], item["item_id"], require_different=True
        )
        if new_item_id:
            modify_pairs.append((item, new_item_id, new_variant))

    if not modify_pairs:
        return None

    old_item_ids = [pair[0]["item_id"] for pair in modify_pairs]
    new_item_ids = [pair[1] for pair in modify_pairs]

    # Compute price difference
    old_total = sum(pair[0]["price"] for pair in modify_pairs)
    new_total = sum(pair[2]["price"] for pair in modify_pairs)
    price_diff = new_total - old_total

    # Pick payment method
    payment_method_id = pick_payment_method(user, price_difference=max(0, price_diff))
    if payment_method_id is None:
        return None

    # Auth
    use_email = rng.random() < 0.2

    # Build actions
    actions = []
    idx = 0

    if use_email:
        actions.append(make_action(task_id, idx, "find_user_id_by_email", {"email": user["email"]}))
    else:
        actions.append(make_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1

    actions.append(make_action(task_id, idx, "get_user_details", {"user_id": user_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "get_order_details", {"order_id": order_id}))
    idx += 1

    # get_product_details for each product being modified
    seen_products = set()
    for pair in modify_pairs:
        pid = pair[0]["product_id"]
        if pid not in seen_products:
            actions.append(make_action(task_id, idx, "get_product_details", {"product_id": pid}))
            idx += 1
            seen_products.add(pid)

    actions.append(make_action(task_id, idx, "modify_pending_order_items", {
        "order_id": order_id,
        "item_ids": old_item_ids,
        "new_item_ids": new_item_ids,
        "payment_method_id": payment_method_id,
    }))

    # Build user scenario
    modify_descriptions = []
    for old_item, new_id, new_variant in modify_pairs:
        old_desc = describe_item(old_item)
        change = describe_variant_change(old_item.get("options", {}), new_variant.get("options", {}))
        modify_descriptions.append(f"change {old_desc} ({change})")

    items_text = " and ".join(modify_descriptions) if len(modify_descriptions) <= 2 else ", ".join(modify_descriptions)

    if use_email:
        known_info = f"Your email is {user['email']}."
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    pm_desc = describe_payment_method(user, payment_method_id)
    payment_pref = f"Use {pm_desc} for any price difference."

    reason_text = f"You just placed order {order_id} and want to modify it before it ships. You'd like to {items_text}. {payment_pref}"

    unknown_info = rng.choice(UNKNOWN_INFO_OPTIONS) if not use_email else None

    return {
        "id": str(task_id),
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": None,
        },
    }


def generate_modify_address_task(task_id, db, user_id, order_id, rng):
    """Generate a modify pending order address task."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    assert order["status"] == "pending", f"Order {order_id} is not pending"

    new_addr = generate_address()
    # Make sure it's different from current order address
    while new_addr["address1"] == order["address"]["address1"]:
        new_addr = generate_address()

    use_email = rng.random() < 0.2

    # Build actions
    actions = []
    idx = 0

    if use_email:
        actions.append(make_action(task_id, idx, "find_user_id_by_email", {"email": user["email"]}))
    else:
        actions.append(make_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1

    actions.append(make_action(task_id, idx, "get_user_details", {"user_id": user_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "get_order_details", {"order_id": order_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "modify_pending_order_address", {
        "order_id": order_id,
        "address1": new_addr["address1"],
        "address2": new_addr["address2"],
        "city": new_addr["city"],
        "state": new_addr["state"],
        "country": new_addr["country"],
        "zip": new_addr["zip"],
    }))

    if use_email:
        known_info = f"Your email is {user['email']}."
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    addr_text = f"{new_addr['address1']}"
    if new_addr["address2"]:
        addr_text += f", {new_addr['address2']}"
    addr_text += f", {new_addr['city']}, {new_addr['state']} {new_addr['zip']}"

    reason_text = f"You need to change the shipping address for order {order_id} to {addr_text}."

    unknown_info = rng.choice(UNKNOWN_INFO_OPTIONS) if not use_email else None

    return {
        "id": str(task_id),
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": None,
        },
    }


def generate_modify_user_address_task(task_id, db, user_id, rng):
    """Generate a modify user default address task."""
    user = db["users"][user_id]

    new_addr = generate_address()
    while new_addr["address1"] == user["address"]["address1"]:
        new_addr = generate_address()

    use_email = rng.random() < 0.2

    actions = []
    idx = 0

    if use_email:
        actions.append(make_action(task_id, idx, "find_user_id_by_email", {"email": user["email"]}))
    else:
        actions.append(make_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1

    actions.append(make_action(task_id, idx, "get_user_details", {"user_id": user_id}))
    idx += 1

    actions.append(make_action(task_id, idx, "modify_user_address", {
        "user_id": user_id,
        "address1": new_addr["address1"],
        "address2": new_addr["address2"],
        "city": new_addr["city"],
        "state": new_addr["state"],
        "country": new_addr["country"],
        "zip": new_addr["zip"],
    }))

    if use_email:
        known_info = f"Your email is {user['email']}."
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    addr_text = f"{new_addr['address1']}"
    if new_addr["address2"]:
        addr_text += f", {new_addr['address2']}"
    addr_text += f", {new_addr['city']}, {new_addr['state']} {new_addr['zip']}"

    reason_text = f"You recently moved and need to update your default address to {addr_text}."

    unknown_info = rng.choice(UNKNOWN_INFO_OPTIONS) if not use_email else None

    return {
        "id": str(task_id),
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": None,
        },
    }


# ─── Hard task helpers ────────────────────────────────────────────────────────

HARD_TASK_INSTRUCTIONS = [
    "You are impatient and want things resolved immediately.",
    "You are polite but insistent and will push back if told something can't be done.",
    "You are confused and may provide contradictory information at first.",
    "You are a no-nonsense person who states exactly what you want.",
    "You are friendly but firm — you know what you want and won't settle.",
    "You are skeptical and will question the agent's responses.",
]

INVALID_CANCEL_REASONS = [
    "it's too expensive",
    "I found a better price elsewhere",
    "the delivery is taking too long",
    "I changed my mind about the color",
    "I want a different brand",
    "my friend recommended something else",
]


def make_hard_action(task_id, action_idx, name, arguments, compare_args=None):
    """Like make_action but supports compare_args for flexible matching."""
    a = {
        "action_id": f"{task_id}_{action_idx}",
        "name": name,
        "arguments": arguments,
        "info": None,
    }
    if compare_args is not None:
        a["compare_args"] = compare_args
    return a


def make_hard_task(task_id, purpose, reason_text, known_info, actions, rng,
                   communicate_info=None, nl_assertions=None,
                   reward_basis=None, unknown_info=None, notes=None):
    """Build a hard task dict with proper metadata."""
    if reward_basis is None:
        reward_basis = ["DB", "COMMUNICATE"]
    # Strip NL_ASSERTION from reward_basis — keep nl_assertions as documentation only
    reward_basis = [r for r in reward_basis if r != "NL_ASSERTION"]
    if not reward_basis:
        reward_basis = ["DB", "COMMUNICATE"]
    return {
        "id": str(task_id),
        "description": {
            "purpose": purpose,
            "relevant_policies": None,
            "notes": notes,
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": rng.choice(HARD_TASK_INSTRUCTIONS),
                "domain": "retail",
                "reason_for_call": reason_text,
                "known_info": known_info,
                "unknown_info": unknown_info,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": communicate_info or [],
            "nl_assertions": nl_assertions,
            "reward_basis": reward_basis,
        },
    }


def auth_actions(task_id, db, user_id, use_email=False, start_idx=0, include_user_details=False):
    """Generate auth actions. Only includes get_user_details when needed (e.g. user doesn't know order ID)."""
    user = db["users"][user_id]
    actions = []
    idx = start_idx
    if use_email:
        actions.append(make_hard_action(task_id, idx, "find_user_id_by_email",
                                        {"email": user["email"]}))
    else:
        actions.append(make_hard_action(task_id, idx, "find_user_id_by_name_zip", {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "zip": user["address"]["zip"],
        }))
    idx += 1
    if include_user_details:
        actions.append(make_hard_action(task_id, idx, "get_user_details",
                                        {"user_id": user_id}))
        idx += 1
    return actions, idx


def known_info_str(db, user_id, use_email=False):
    user = db["users"][user_id]
    if use_email:
        return f"Your email is {user['email']}."
    return f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."


def find_product_for_item(db, item_id):
    """Find the product that contains a given item/variant ID."""
    for pid, product in db["products"].items():
        if item_id in product["variants"]:
            return pid, product
    return None, None


def find_different_product_item(db, current_product_id, rng):
    """Find an available item from a different product type."""
    products = list(db["products"].items())
    rng.shuffle(products)
    for pid, product in products:
        if pid == current_product_id:
            continue
        for vid, variant in product["variants"].items():
            if variant["available"]:
                return vid, variant, product
    return None, None, None


def find_unavailable_variant(db, product_id, current_item_id):
    """Find an unavailable variant of the same product."""
    product = db["products"].get(product_id)
    if not product:
        return None, None
    for vid, variant in product["variants"].items():
        if vid == current_item_id:
            continue
        if not variant["available"]:
            return vid, variant
    return None, None


# ─── Hard task generators ─────────────────────────────────────────────────────

# --- Authentication edge cases ---

def gen_hard_auth_user_id_directly(task_id, db, user_id, order_id, rng):
    """#1: User provides user_id directly → agent must still authenticate via email/name+zip."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    actions, idx = auth_actions(task_id, db, user_id, use_email=False)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = f"You want to check the status of your order. Your user ID is {user_id} and your order is {order_id}."
    known = f"Your user ID is {user_id}. If the agent asks for other info, you are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."

    return make_hard_task(task_id, "User provides user_id directly - agent must still authenticate via email or name+zip",
                          reason, known, actions, rng,
                          reward_basis=["ACTION"],
                          notes="Policy lines 10-14: agent must authenticate even when user provides user_id")


def gen_hard_auth_email_path(task_id, db, user_id, order_id, rng):
    """#2: User authenticates via email (testing the email auth path)."""
    user = db["users"][user_id]
    order = db["orders"][order_id]

    actions, idx = auth_actions(task_id, db, user_id, use_email=True)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = f"You want to check on your order {order_id}."
    known = (f"Your email is {user['email']}. "
             f"You do not remember your name or zip code on file.")

    return make_hard_task(task_id, "User authenticates via email path",
                          reason, known, actions, rng,
                          reward_basis=["ACTION"],
                          notes="Testing email auth path; user only knows email")


# --- Status constraint violations ---

def gen_hard_cancel_delivered(task_id, db, user_id, order_id, rng):
    """#3: User wants to cancel a delivered order → agent refuses, suggests return."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = f"You want to cancel order {order_id}."
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Cancel a delivered order - should be refused",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that a delivered order cannot be cancelled",
                                         "The agent suggested returning the order instead or offered alternatives"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 88: cancel only for pending orders")


def gen_hard_return_pending(task_id, db, user_id, order_id, rng):
    """#4: User wants to return a pending order → agent refuses, suggests cancel or wait."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    item = rng.choice(order["items"])
    item_desc = describe_item(item)
    reason = f"You want to return {item_desc} from order {order_id}."
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Return a pending order - should be refused",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that a pending order cannot be returned",
                                         "The agent suggested cancelling the order or waiting for delivery"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 118: return only for delivered orders")


def gen_hard_modify_processed(task_id, db, user_id, order_id, rng):
    """#5: User wants to modify a processed order → agent refuses."""
    order = db["orders"][order_id]
    assert order["status"] == "processed"

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = f"You want to change the shipping address for order {order_id} to a new address."
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Modify a processed order - should be refused",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that a processed order cannot be modified"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 96: modify only for pending orders")


def gen_hard_exchange_pending(task_id, db, user_id, order_id, rng):
    """#6: User wants to exchange a pending order → agent refuses."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    item = rng.choice(order["items"])
    item_desc = describe_item(item)
    reason = f"You received order {order_id} and want to exchange {item_desc} for a different variant."
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Exchange a pending order - should be refused",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that a pending order cannot be exchanged",
                                         "The agent suggested modifying the order or waiting for delivery"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 130: exchange only for delivered orders")


# --- Cancel rules ---

def gen_hard_invalid_cancel_reason(task_id, db, user_id, order_id, rng):
    """#7: User gives invalid cancel reason → agent must ask for valid reason."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    invalid_reason = rng.choice(INVALID_CANCEL_REASONS)
    valid_reason = rng.choice(["no longer needed", "ordered by mistake"])

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "cancel_pending_order", {
        "order_id": order_id,
        "reason": valid_reason,
    }))

    reason = (f"You want to cancel order {order_id}. "
              f"Your initial reason is: \"{invalid_reason}\". "
              f"When the agent tells you the valid reasons, you pick \"{valid_reason}\".")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Invalid cancel reason - agent must ask for valid reason",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy lines 88-92: reason must be 'no longer needed' or 'ordered by mistake'")


def gen_hard_partial_cancel(task_id, db, user_id, order_id, rng):
    """#8: User wants to cancel one item from multi-item order → can't do partial cancel."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"
    assert len(order["items"]) >= 2

    item = rng.choice(order["items"])
    item_desc = describe_item(item)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = f"You want to cancel just {item_desc} from order {order_id}, not the whole order."
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Partial cancel of multi-item order - not possible",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that partial cancellation is not possible and the entire order must be cancelled"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Cancel applies to entire order, not individual items")


# --- Modify items rules ---

def gen_hard_modify_cross_product_type(task_id, db, user_id, order_id, rng):
    """#9: User wants to change product type (shirt → shoe) → agent refuses."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    item = rng.choice(order["items"])
    item_desc = describe_item(item)
    product_id = item["product_id"]
    product = db["products"][product_id]

    # Find a different product type to request
    diff_item_id, diff_variant, diff_product = find_different_product_item(db, product_id, rng)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = (f"You want to modify order {order_id}. "
              f"You want to change {item_desc} ({product['name']}) "
              f"to a {diff_product['name']} instead.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Modify item to different product type - should be refused",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that items can only be modified to a different variant of the same product, not a different product type"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 112: cannot change product types")


# --- Modify payment rules ---

def gen_hard_modify_same_payment(task_id, db, user_id, order_id, rng):
    """#12: User wants to keep same payment method → not allowed."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    original_pm_id = order["payment_history"][0]["payment_method_id"]
    user = db["users"][user_id]
    original_pm = user["payment_methods"].get(original_pm_id, {})
    pm_desc = describe_payment_method(user, original_pm_id)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = (f"You want to update the payment method on order {order_id}. "
              f"You want to use {pm_desc} (which is actually the same one already on the order).")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Modify payment to same method - not allowed",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that the new payment method must be different from the original"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 102: must choose a different payment method")


def gen_hard_modify_payment_gift_card_insufficient(task_id, db, user_id, order_id, rng):
    """#13: User wants gift card but insufficient balance → agent refuses."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"
    user = db["users"][user_id]

    # Find a gift card with insufficient balance
    total = sum(item["price"] for item in order["items"])
    gift_card_id = None
    for pm_id, pm in user["payment_methods"].items():
        if pm["source"] == "gift_card" and pm.get("balance", 0) < total:
            gift_card_id = pm_id
            break

    if gift_card_id is None:
        return None

    gc = user["payment_methods"][gift_card_id]

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = (f"You want to change the payment method on order {order_id} "
              f"to your gift card (balance ${gc['balance']:.2f}).")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Modify payment to gift card with insufficient balance",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that the gift card balance is insufficient to cover the order total"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 104: gift card must have enough balance")


# --- Return rules ---

def gen_hard_return_invalid_payment(task_id, db, user_id, order_id, rng):
    """#14: User wants refund to non-original, non-gift-card payment → agent refuses."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    original_pm_id = order["payment_history"][0]["payment_method_id"]

    # Find a payment method that is neither the original nor a gift card
    invalid_pm_id = None
    for pm_id, pm in user["payment_methods"].items():
        if pm_id != original_pm_id and pm["source"] != "gift_card":
            invalid_pm_id = pm_id
            break

    if invalid_pm_id is None:
        return None

    invalid_pm_desc = describe_payment_method(user, invalid_pm_id)
    item = rng.choice(order["items"])
    item_desc = describe_item(item)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = (f"You want to return {item_desc} from order {order_id}. "
              f"You want the refund sent to {invalid_pm_desc}.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Return with refund to invalid payment method",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that the refund must go to the original payment method or an existing gift card"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 124: refund must be original PM or gift card")


def gen_hard_return_transfer_needed(task_id, db, user_id, order_id, rng):
    """#15: User insists on PayPal refund but original was credit card and no gift card → transfer."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    original_pm_id = order["payment_history"][0]["payment_method_id"]
    original_pm = user["payment_methods"].get(original_pm_id, {})

    # Need: original is credit card, user has a paypal, no gift card
    has_gift_card = any(pm["source"] == "gift_card" for pm in user["payment_methods"].values())
    if original_pm.get("source") != "credit_card" or has_gift_card:
        return None

    paypal_pm_id = None
    for pm_id, pm in user["payment_methods"].items():
        if pm["source"] == "paypal":
            paypal_pm_id = pm_id
            break
    if paypal_pm_id is None:
        return None

    item = rng.choice(order["items"])
    item_desc = describe_item(item)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "transfer_to_human_agents",
                                    {"summary": ""},
                                    compare_args=[]))

    reason = (f"You want to return {item_desc} from order {order_id}. "
              f"You insist on the refund going to your PayPal account. "
              f"You do not want the refund on your credit card. "
              f"If the agent says this is not possible, ask to speak to a human agent.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Return insisting on PayPal refund (no gift card) → transfer to human",
                          reason, known, actions, rng,
                          communicate_info=["TRANSFERRED", "HUMAN AGENT"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 124 + line 24: refund options exhausted, must transfer")


# --- Exchange rules ---

def gen_hard_exchange_cross_product_type(task_id, db, user_id, order_id, rng):
    """#16: User wants to exchange to different product type → agent refuses."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"

    item = rng.choice(order["items"])
    item_desc = describe_item(item)
    product_id = item["product_id"]
    product = db["products"][product_id]

    diff_item_id, diff_variant, diff_product = find_different_product_item(db, product_id, rng)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = (f"You received order {order_id} and want to exchange {item_desc} ({product['name']}) "
              f"for a {diff_product['name']} instead.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Exchange to different product type - should be refused",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that exchanges can only be for the same product type with different options, not a different product type"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 132: cannot change product types in exchange")


def gen_hard_exchange_unavailable_variant(task_id, db, user_id, order_id, rng):
    """#17: Target variant not available → agent informs user."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"

    # Find an item whose product has an unavailable variant
    for item in rng.sample(order["items"], len(order["items"])):
        product_id = item["product_id"]
        unav_id, unav_variant = find_unavailable_variant(db, product_id, item["item_id"])
        if unav_id:
            break
    else:
        return None

    item_desc = describe_item(item)
    product = db["products"][product_id]

    # Describe the unavailable variant
    unav_options = unav_variant.get("options", {})
    opt_desc = ", ".join(f"{k}: {v}" for k, v in unav_options.items())

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": product_id}))

    reason = (f"You received order {order_id} and want to exchange {item_desc} "
              f"for the variant with {opt_desc}.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Exchange to unavailable variant",
                          reason, known, actions, rng,
                          nl_assertions=["The agent informed the user that the requested variant is not available"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 132: can only exchange to available items")


def gen_hard_exchange_gift_card_insufficient(task_id, db, user_id, order_id, rng):
    """#18: Gift card insufficient for price difference → agent informs user."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    # Find a gift card with low balance
    gc_id = None
    gc_balance = 0
    for pm_id, pm in user["payment_methods"].items():
        if pm["source"] == "gift_card":
            gc_id = pm_id
            gc_balance = pm.get("balance", 0)
            break
    if gc_id is None:
        return None

    # Find an exchange with price difference > gc_balance
    for item in rng.sample(order["items"], len(order["items"])):
        product_id = item["product_id"]
        product = db["products"][product_id]
        for vid, variant in product["variants"].items():
            if vid == item["item_id"] or not variant["available"]:
                continue
            price_diff = variant["price"] - item["price"]
            if price_diff > gc_balance and price_diff > 0:
                # Found a valid scenario
                item_desc = describe_item(item)
                new_options = variant.get("options", {})
                change_desc = describe_variant_change(item.get("options", {}), new_options)

                actions, idx = auth_actions(task_id, db, user_id)
                actions.append(make_hard_action(task_id, idx, "get_order_details",
                                                {"order_id": order_id}))
                idx += 1
                actions.append(make_hard_action(task_id, idx, "get_product_details",
                                                {"product_id": product_id}))

                reason = (f"You received order {order_id} and want to exchange {item_desc} "
                          f"(change {change_desc}). You want to pay the difference with your gift card.")
                known = known_info_str(db, user_id)

                return make_hard_task(task_id, "Exchange with gift card insufficient for price difference",
                                      reason, known, actions, rng,
                                      nl_assertions=["The agent informed the user that the gift card balance is insufficient to cover the price difference"],
                                      reward_basis=["DB", "COMMUNICATE"],
                                      notes="Policy line 134: gift card must have enough balance")
    return None


# --- Transfer to human ---

def gen_hard_out_of_scope_request(task_id, db, user_id, rng):
    """#19: Request outside scope → transfer."""
    actions = [make_hard_action(task_id, 0, "transfer_to_human_agents",
                               {"summary": ""},
                               compare_args=[])]

    out_of_scope = rng.choice([
        "You want to file a formal complaint about a rude delivery person.",
        "You want to request a price match from a competitor's store.",
        "You want the agent to place a new order for you.",
        "You want to merge two of your accounts into one.",
    ])
    out_of_scope += " If the agent says they cannot help with this, insist that you need this resolved and ask to speak to someone who can."

    known = known_info_str(db, user_id)
    return make_hard_task(task_id, "Out of scope request → transfer to human",
                          out_of_scope, known, actions, rng,
                          communicate_info=["TRANSFERRED", "HUMAN AGENT"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 24: transfer for requests outside agent scope")


def gen_hard_impossible_demand_transfer(task_id, db, user_id, order_id, rng):
    """#20: User demands something impossible → transfer after explaining."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "transfer_to_human_agents",
                                    {"summary": ""},
                                    compare_args=[]))

    reason = (f"You received order {order_id} and one item arrived damaged. "
              f"You want a full refund AND a free replacement. "
              f"If the agent says this isn't possible, insist and ask for a human agent.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Impossible demand (refund + free replacement) → transfer",
                          reason, known, actions, rng,
                          communicate_info=["TRANSFERRED", "HUMAN AGENT"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Damaged item + free replacement is outside agent scope")


# --- Multi-flow combinations ---

def gen_hard_cancel_pending_return_delivered(task_id, db, user_id, pending_oid, delivered_oid, rng):
    """#21: Cancel one pending order + return one delivered order."""
    user = db["users"][user_id]
    pending_order = db["orders"][pending_oid]
    delivered_order = db["orders"][delivered_oid]

    assert pending_order["status"] == "pending"
    assert delivered_order["status"] == "delivered"

    cancel_reason = rng.choice(["no longer needed", "ordered by mistake"])
    return_item = rng.choice(delivered_order["items"])
    return_item_desc = describe_item(return_item)

    payment_method_id = pick_return_payment(user, delivered_order)
    if payment_method_id is None:
        return None
    pm_desc = describe_payment_method(user, payment_method_id)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": pending_oid}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "cancel_pending_order", {
        "order_id": pending_oid,
        "reason": cancel_reason,
    }))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": delivered_oid}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "return_delivered_order_items", {
        "order_id": delivered_oid,
        "item_ids": [return_item["item_id"]],
        "payment_method_id": payment_method_id,
    }))

    reason = (f"You have two issues: First, you want to cancel order {pending_oid} "
              f"because you {rng.choice(CANCEL_REASONS_NATURAL[cancel_reason]).format(item_desc='the items')}. "
              f"Second, you want to return {return_item_desc} from order {delivered_oid}. "
              f"Refund to {pm_desc}.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Multi-flow: cancel pending + return delivered",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Two separate actions on two orders in one conversation")


def gen_hard_exchange_and_modify(task_id, db, user_id, pending_oid, delivered_oid, rng):
    """#22: Exchange from one order + modify items in another."""
    user = db["users"][user_id]
    pending_order = db["orders"][pending_oid]
    delivered_order = db["orders"][delivered_oid]

    assert pending_order["status"] == "pending"
    assert delivered_order["status"] == "delivered"

    # Find exchange pair for delivered order
    exchange_item = None
    new_ex_id = None
    new_ex_variant = None
    for item in rng.sample(delivered_order["items"], len(delivered_order["items"])):
        new_ex_id, new_ex_variant = find_available_variant(db, item["product_id"], item["item_id"], True)
        if new_ex_id:
            exchange_item = item
            break
    if exchange_item is None:
        return None

    # Find modify pair for pending order
    modify_item = None
    new_mod_id = None
    new_mod_variant = None
    for item in rng.sample(pending_order["items"], len(pending_order["items"])):
        new_mod_id, new_mod_variant = find_available_variant(db, item["product_id"], item["item_id"], True)
        if new_mod_id:
            modify_item = item
            break
    if modify_item is None:
        return None

    # Payment for exchange
    ex_diff = new_ex_variant["price"] - exchange_item["price"]
    ex_pm = pick_payment_method(user, price_difference=max(0, ex_diff))
    if ex_pm is None:
        return None

    # Payment for modify
    mod_diff = new_mod_variant["price"] - modify_item["price"]
    mod_pm = pick_payment_method(user, price_difference=max(0, mod_diff))
    if mod_pm is None:
        return None

    actions, idx = auth_actions(task_id, db, user_id)
    # Exchange flow
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": delivered_oid}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": exchange_item["product_id"]}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "exchange_delivered_order_items", {
        "order_id": delivered_oid,
        "item_ids": [exchange_item["item_id"]],
        "new_item_ids": [new_ex_id],
        "payment_method_id": ex_pm,
    }))
    idx += 1
    # Modify flow
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": pending_oid}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": modify_item["product_id"]}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "modify_pending_order_items", {
        "order_id": pending_oid,
        "item_ids": [modify_item["item_id"]],
        "new_item_ids": [new_mod_id],
        "payment_method_id": mod_pm,
    }))

    ex_desc = describe_item(exchange_item)
    ex_change = describe_variant_change(exchange_item.get("options", {}), new_ex_variant.get("options", {}))
    mod_desc = describe_item(modify_item)
    mod_change = describe_variant_change(modify_item.get("options", {}), new_mod_variant.get("options", {}))
    ex_pm_desc = describe_payment_method(user, ex_pm)
    mod_pm_desc = describe_payment_method(user, mod_pm)

    reason = (f"You have two requests: "
              f"First, exchange {ex_desc} from delivered order {delivered_oid} (change {ex_change}), "
              f"use {ex_pm_desc} for any price difference. "
              f"Second, modify {mod_desc} in pending order {pending_oid} (change {mod_change}), "
              f"use {mod_pm_desc} for any price difference.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Multi-flow: exchange delivered + modify pending",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Two different action types across two orders")


def gen_hard_modify_address_and_items(task_id, db, user_id, order_id, rng):
    """#24: Modify address + modify items on same pending order."""
    user = db["users"][user_id]
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    # Find modify pair
    modify_item = None
    new_mod_id = None
    new_mod_variant = None
    for item in rng.sample(order["items"], len(order["items"])):
        new_mod_id, new_mod_variant = find_available_variant(db, item["product_id"], item["item_id"], True)
        if new_mod_id:
            modify_item = item
            break
    if modify_item is None:
        return None

    mod_diff = new_mod_variant["price"] - modify_item["price"]
    mod_pm = pick_payment_method(user, price_difference=max(0, mod_diff))
    if mod_pm is None:
        return None

    new_addr = generate_address()
    while new_addr["address1"] == order["address"]["address1"]:
        new_addr = generate_address()

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    # Address change first (modify_items locks the order)
    actions.append(make_hard_action(task_id, idx, "modify_pending_order_address", {
        "order_id": order_id,
        "address1": new_addr["address1"],
        "address2": new_addr["address2"],
        "city": new_addr["city"],
        "state": new_addr["state"],
        "country": new_addr["country"],
        "zip": new_addr["zip"],
    }))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": modify_item["product_id"]}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "modify_pending_order_items", {
        "order_id": order_id,
        "item_ids": [modify_item["item_id"]],
        "new_item_ids": [new_mod_id],
        "payment_method_id": mod_pm,
    }))

    mod_desc = describe_item(modify_item)
    mod_change = describe_variant_change(modify_item.get("options", {}), new_mod_variant.get("options", {}))
    mod_pm_desc = describe_payment_method(user, mod_pm)

    addr_text = f"{new_addr['address1']}"
    if new_addr["address2"]:
        addr_text += f", {new_addr['address2']}"
    addr_text += f", {new_addr['city']}, {new_addr['state']} {new_addr['zip']}"

    reason = (f"You need two changes to order {order_id}: "
              f"First, change the shipping address to {addr_text}. "
              f"Second, modify {mod_desc} (change to {mod_change}), "
              f"use {mod_pm_desc} for any price difference.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Multi-flow: modify address + modify items on same order",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Address must be changed before items (items locks order)")


# --- Branching scenarios ---

def gen_hard_exchange_or_return_fallback(task_id, db, user_id, order_id, rng):
    """#25: 'Exchange item X. If the variant I want isn't available, return it instead.'
    We generate this for a case where the variant IS available, so exchange should succeed."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    # Find an item with available alternate variant
    exchange_item = None
    new_id = None
    new_variant = None
    for item in rng.sample(order["items"], len(order["items"])):
        new_id, new_variant = find_available_variant(db, item["product_id"], item["item_id"], True)
        if new_id:
            exchange_item = item
            break
    if exchange_item is None:
        return None

    price_diff = new_variant["price"] - exchange_item["price"]
    pm_id = pick_payment_method(user, price_difference=max(0, price_diff))
    if pm_id is None:
        return None

    item_desc = describe_item(exchange_item)
    change_desc = describe_variant_change(exchange_item.get("options", {}), new_variant.get("options", {}))
    pm_desc = describe_payment_method(user, pm_id)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": exchange_item["product_id"]}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "exchange_delivered_order_items", {
        "order_id": order_id,
        "item_ids": [exchange_item["item_id"]],
        "new_item_ids": [new_id],
        "payment_method_id": pm_id,
    }))

    reason = (f"You received order {order_id} and want to exchange {item_desc} "
              f"(change {change_desc}). If the variant you want isn't available, "
              f"you'd rather just return it instead. Use {pm_desc} for any price difference.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Branching: exchange if available, else return",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Variant is available so exchange should proceed")


def gen_hard_exchange_fallback_return(task_id, db, user_id, order_id, rng):
    """#25b: Same scenario but variant is NOT available → should return instead."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    # Find an item where specific desired variant is unavailable but return is possible
    for item in rng.sample(order["items"], len(order["items"])):
        product_id = item["product_id"]
        unav_id, unav_variant = find_unavailable_variant(db, product_id, item["item_id"])
        if unav_id:
            break
    else:
        return None

    refund_pm = pick_return_payment(user, order)
    if refund_pm is None:
        return None

    item_desc = describe_item(item)
    unav_options = unav_variant.get("options", {})
    opt_desc = ", ".join(f"{k}: {v}" for k, v in unav_options.items())
    pm_desc = describe_payment_method(user, refund_pm)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": product_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "return_delivered_order_items", {
        "order_id": order_id,
        "item_ids": [item["item_id"]],
        "payment_method_id": refund_pm,
    }))

    reason = (f"You received order {order_id} and want to exchange {item_desc} "
              f"for the variant with {opt_desc}. If that variant isn't available, "
              f"just return it instead. Refund to {pm_desc}.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Branching: exchange unavailable → fallback to return",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Desired variant is unavailable so agent should do return")


def gen_hard_cancel_or_return_fallback(task_id, db, user_id, order_id, rng):
    """#26: 'Cancel this order. If it can't be cancelled, I want to return it.'
    This is for a delivered order → can't cancel, so should return."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    refund_pm = pick_return_payment(user, order)
    if refund_pm is None:
        return None

    item_ids = [item["item_id"] for item in order["items"]]
    pm_desc = describe_payment_method(user, refund_pm)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "return_delivered_order_items", {
        "order_id": order_id,
        "item_ids": item_ids,
        "payment_method_id": refund_pm,
    }))

    reason = (f"You want to cancel order {order_id}. If the agent says it can't be cancelled, "
              f"you want to return all items instead. Refund to {pm_desc}.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Branching: cancel fails on delivered → fallback to return",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Order is delivered so cancel is impossible; should return instead")


def gen_hard_exchange_multiple_fallbacks(task_id, db, user_id, order_id, rng):
    """#28: 'Exchange for option A. If A isn't available, try B. If neither, keep original.'
    We set it up so A is unavailable, B is available."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    # Find an item with at least one unavailable and one available alternate
    for item in rng.sample(order["items"], len(order["items"])):
        product_id = item["product_id"]
        product = db["products"][product_id]

        unavailable = []
        available = []
        for vid, variant in product["variants"].items():
            if vid == item["item_id"]:
                continue
            if variant["available"]:
                available.append((vid, variant))
            else:
                unavailable.append((vid, variant))

        if unavailable and available:
            option_a_id, option_a = rng.choice(unavailable)
            option_b_id, option_b = rng.choice(available)
            break
    else:
        return None

    price_diff = option_b["price"] - item["price"]
    pm_id = pick_payment_method(user, price_difference=max(0, price_diff))
    if pm_id is None:
        return None

    item_desc = describe_item(item)
    a_opts = ", ".join(f"{k}: {v}" for k, v in option_a.get("options", {}).items())
    b_change = describe_variant_change(item.get("options", {}), option_b.get("options", {}))
    pm_desc = describe_payment_method(user, pm_id)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": product_id}))
    idx += 1
    # B is available, so exchange should proceed with B
    actions.append(make_hard_action(task_id, idx, "exchange_delivered_order_items", {
        "order_id": order_id,
        "item_ids": [item["item_id"]],
        "new_item_ids": [option_b_id],
        "payment_method_id": pm_id,
    }))

    reason = (f"You received order {order_id} and want to exchange {item_desc}. "
              f"Your first choice is the variant with {a_opts}. "
              f"If that's not available, your second choice is to change {b_change}. "
              f"If neither is available, just keep the original. "
              f"Use {pm_desc} for any price difference.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Branching: exchange A (unavail) → try B (avail) → keep",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Option A unavailable, option B available → should exchange with B")


# --- Edge cases ---

def gen_hard_single_item_cancel(task_id, db, user_id, order_id, rng):
    """#30: Order with single item — user asks to cancel just that item (equivalent to cancelling order)."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"
    assert len(order["items"]) == 1

    item = order["items"][0]
    item_desc = describe_item(item)
    reason_key = rng.choice(["no longer needed", "ordered by mistake"])

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "cancel_pending_order", {
        "order_id": order_id,
        "reason": reason_key,
    }))

    reason = (f"You want to cancel {item_desc} from order {order_id} because "
              f"you {rng.choice(CANCEL_REASONS_NATURAL[reason_key]).format(item_desc=item_desc)}. "
              f"You're asking to cancel just the item, not knowing it's the only one.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Single-item order: cancel item = cancel order",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Only 1 item in order, so cancelling the item means cancelling the order")


def gen_hard_exchange_zero_price_diff(task_id, db, user_id, order_id, rng):
    """#31: Exchange with zero or near-zero price difference."""
    order = db["orders"][order_id]
    assert order["status"] == "delivered"
    user = db["users"][user_id]

    # Find an exchange pair with very small price difference
    best_pair = None
    best_diff = float('inf')
    for item in order["items"]:
        product_id = item["product_id"]
        product = db["products"][product_id]
        for vid, variant in product["variants"].items():
            if vid == item["item_id"] or not variant["available"]:
                continue
            diff = abs(variant["price"] - item["price"])
            if diff < best_diff:
                best_diff = diff
                best_pair = (item, vid, variant, product_id)

    if best_pair is None or best_diff > 20:  # allow up to $20 diff for "near-zero"
        return None

    item, new_id, new_variant, product_id = best_pair
    price_diff = new_variant["price"] - item["price"]
    pm_id = pick_payment_method(user, price_difference=max(0, price_diff))
    if pm_id is None:
        return None

    item_desc = describe_item(item)
    change_desc = describe_variant_change(item.get("options", {}), new_variant.get("options", {}))
    pm_desc = describe_payment_method(user, pm_id)

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "get_product_details",
                                    {"product_id": product_id}))
    idx += 1
    actions.append(make_hard_action(task_id, idx, "exchange_delivered_order_items", {
        "order_id": order_id,
        "item_ids": [item["item_id"]],
        "new_item_ids": [new_id],
        "payment_method_id": pm_id,
    }))

    reason = (f"You received order {order_id} and want to exchange {item_desc} "
              f"(change {change_desc}). Use {pm_desc} for any price difference.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Exchange with minimal price difference",
                          reason, known, actions, rng,
                          reward_basis=["DB", "COMMUNICATE"],
                          notes=f"Price difference is only ${best_diff:.2f}")


def gen_hard_other_user_order(task_id, db, user_id, rng):
    """#33: User asks for info about another user's order → agent refuses."""
    user = db["users"][user_id]

    # Find an order belonging to a different user
    other_order_id = None
    for oid, order in db["orders"].items():
        if order["user_id"] != user_id:
            other_order_id = oid
            break

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": other_order_id}))

    reason = (f"You want to check on order {other_order_id} (this is actually your friend's order, "
              f"but you don't mention that). You just provide the order number.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "Access another user's order - should be denied",
                          reason, known, actions, rng,
                          nl_assertions=["The agent denied the user's request to view order details because the order belongs to a different user"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Policy line 14: can only help one user, must deny other user's data")


def gen_hard_user_backs_out(task_id, db, user_id, order_id, rng):
    """#34: User asks for action then backs out at confirmation."""
    order = db["orders"][order_id]
    assert order["status"] == "pending"

    actions, idx = auth_actions(task_id, db, user_id)
    actions.append(make_hard_action(task_id, idx, "get_order_details",
                                    {"order_id": order_id}))

    reason = (f"You initially want to cancel order {order_id} because you don't need it. "
              f"When the agent asks you to confirm the cancellation, "
              f"you change your mind and say you want to keep the order after all.")
    known = known_info_str(db, user_id)

    return make_hard_task(task_id, "User backs out at confirmation",
                          reason, known, actions, rng,
                          nl_assertions=["The agent did not cancel the order after the user changed their mind"],
                          reward_basis=["DB", "COMMUNICATE"],
                          notes="Agent must respect user's decision to not proceed; DB should be unchanged")


# ─── Hard task main generator ────────────────────────────────────────────────

def generate_hard_tasks(seed=456, output_path=None):
    rng = random.Random(seed)
    random.seed(seed)

    db = load_db()
    existing_tasks = load_existing_tasks()

    # Also load generated tasks to avoid reusing
    gen_tasks_path = BASE_DIR / "generated_tasks.json"
    if gen_tasks_path.exists():
        with open(gen_tasks_path) as f:
            generated_easy = json.load(f)
    else:
        generated_easy = []

    all_existing = existing_tasks + generated_easy
    used_order_ids = get_used_order_ids(all_existing)
    used_user_ids = get_used_user_ids(all_existing, db)

    print(f"DB: {len(db['users'])} users, {len(db['orders'])} orders, {len(db['products'])} products")
    print(f"All existing tasks: {len(all_existing)}")
    print(f"Used users: {len(used_user_ids)}, Used orders: {len(used_order_ids)}")

    # Build pools by status
    pending_orders = []
    delivered_orders = []
    processed_orders = []
    single_item_pending = []

    for oid, order in db["orders"].items():
        if oid in used_order_ids:
            continue
        uid = order["user_id"]
        if uid in used_user_ids:
            continue
        if order["status"] == "pending":
            pending_orders.append((uid, oid))
            if len(order["items"]) == 1:
                single_item_pending.append((uid, oid))
        elif order["status"] == "delivered":
            delivered_orders.append((uid, oid))
        elif order["status"] == "processed":
            processed_orders.append((uid, oid))

    rng.shuffle(pending_orders)
    rng.shuffle(delivered_orders)
    rng.shuffle(processed_orders)
    rng.shuffle(single_item_pending)

    # Multi-item pending orders (for partial cancel)
    multi_item_pending = [(uid, oid) for uid, oid in pending_orders
                          if len(db["orders"][oid]["items"]) >= 2]

    # Users with both pending and delivered orders (for multi-flow)
    user_orders = {}
    for uid, oid in pending_orders:
        user_orders.setdefault(uid, {"pending": [], "delivered": []})
        user_orders[uid]["pending"].append(oid)
    for uid, oid in delivered_orders:
        user_orders.setdefault(uid, {"pending": [], "delivered": []})
        user_orders[uid]["delivered"].append(oid)
    multi_flow_users = [(uid, data) for uid, data in user_orders.items()
                        if data["pending"] and data["delivered"]]
    rng.shuffle(multi_flow_users)

    # Users with specific payment setups for edge cases
    # Credit card orders with no gift card (for return-to-paypal refusal)
    cc_no_gc_delivered = []
    for uid, oid in delivered_orders:
        user = db["users"][uid]
        order = db["orders"][oid]
        orig_pm_id = order["payment_history"][0]["payment_method_id"]
        orig_pm = user["payment_methods"].get(orig_pm_id, {})
        has_gc = any(pm["source"] == "gift_card" for pm in user["payment_methods"].values())
        has_other = any(pm_id != orig_pm_id and pm["source"] != "gift_card"
                        for pm_id, pm in user["payment_methods"].items())
        if orig_pm.get("source") == "credit_card" and has_other:
            cc_no_gc_delivered.append((uid, oid, has_gc))

    # Low-balance gift card users
    low_gc_users = []
    for uid, user in db["users"].items():
        if uid in used_user_ids:
            continue
        for pm_id, pm in user["payment_methods"].items():
            if pm["source"] == "gift_card" and pm.get("balance", 0) < 20:
                low_gc_users.append(uid)
                break
    rng.shuffle(low_gc_users)

    available_users = [uid for uid in db["users"] if uid not in used_user_ids]
    rng.shuffle(available_users)

    print(f"\nAvailable pools:")
    print(f"  Pending orders: {len(pending_orders)}")
    print(f"  Delivered orders: {len(delivered_orders)}")
    print(f"  Processed orders: {len(processed_orders)}")
    print(f"  Multi-item pending: {len(multi_item_pending)}")
    print(f"  Single-item pending: {len(single_item_pending)}")
    print(f"  Multi-flow users: {len(multi_flow_users)}")
    print(f"  CC-no-GC delivered: {len(cc_no_gc_delivered)}")
    print(f"  Low GC users: {len(low_gc_users)}")

    generated = []
    task_id = 500  # start at 500 to separate from easy tasks
    used_orders_this_run = set()
    used_users_this_run = set()

    def use_order(oid):
        used_orders_this_run.add(oid)

    def use_user(uid):
        used_users_this_run.add(uid)

    def try_gen(gen_fn, *args, count=1, **kwargs):
        nonlocal task_id
        results = []
        for _ in range(count):
            task = gen_fn(task_id, *args, rng=rng, **kwargs)
            if task:
                generated.append(task)
                results.append(task)
                task_id += 1
        return results

    # ─── Generate each category ──────────────────────────────────────────

    print("\nGenerating hard tasks...")

    # #1: Auth - user provides user_id directly (2 tasks)
    for i in range(2):
        if i < len(pending_orders):
            uid, oid = pending_orders[i]
            if oid not in used_orders_this_run:
                use_order(oid)
                try_gen(gen_hard_auth_user_id_directly, db, uid, oid)

    # #2: Auth - wrong name/zip (2 tasks)
    p_idx = 2
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_auth_email_path, db, uid, oid)

    # #3: Cancel delivered order (2 tasks)
    d_idx = 0
    for i in range(2):
        while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
            d_idx += 1
        if d_idx < len(delivered_orders):
            uid, oid = delivered_orders[d_idx]
            use_order(oid)
            d_idx += 1
            try_gen(gen_hard_cancel_delivered, db, uid, oid)

    # #4: Return pending order (2 tasks)
    while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
        p_idx += 1
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_return_pending, db, uid, oid)

    # #5: Modify processed order (2 tasks)
    pr_idx = 0
    for i in range(2):
        while pr_idx < len(processed_orders) and processed_orders[pr_idx][1] in used_orders_this_run:
            pr_idx += 1
        if pr_idx < len(processed_orders):
            uid, oid = processed_orders[pr_idx]
            use_order(oid)
            pr_idx += 1
            try_gen(gen_hard_modify_processed, db, uid, oid)

    # #6: Exchange pending order (2 tasks)
    while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
        p_idx += 1
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_exchange_pending, db, uid, oid)

    # #7: Invalid cancel reason (2 tasks)
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_invalid_cancel_reason, db, uid, oid)

    # #8: Partial cancel (2 tasks)
    mi_idx = 0
    for i in range(2):
        while mi_idx < len(multi_item_pending) and multi_item_pending[mi_idx][1] in used_orders_this_run:
            mi_idx += 1
        if mi_idx < len(multi_item_pending):
            uid, oid = multi_item_pending[mi_idx]
            use_order(oid)
            mi_idx += 1
            try_gen(gen_hard_partial_cancel, db, uid, oid)

    # #9: Modify cross product type (2 tasks)
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_modify_cross_product_type, db, uid, oid)

    # #12: Modify same payment (2 tasks)
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_modify_same_payment, db, uid, oid)

    # #13: Modify payment gift card insufficient (1 task)
    for uid_candidate in low_gc_users:
        for oid, order in db["orders"].items():
            if order["user_id"] == uid_candidate and order["status"] == "pending" and oid not in used_orders_this_run and oid not in used_order_ids:
                use_order(oid)
                result = try_gen(gen_hard_modify_payment_gift_card_insufficient, db, uid_candidate, oid)
                if result:
                    break
        if len([t for t in generated if t["description"]["purpose"] and "gift card" in t["description"]["purpose"].lower() and "payment" in t["description"]["purpose"].lower()]) >= 1:
            break

    # #14: Return invalid payment (2 tasks)
    for i in range(2):
        while d_idx < len(cc_no_gc_delivered):
            uid, oid, has_gc = cc_no_gc_delivered[d_idx]
            d_idx_cc = d_idx
            d_idx += 1
            if oid not in used_orders_this_run and oid not in used_order_ids:
                use_order(oid)
                try_gen(gen_hard_return_invalid_payment, db, uid, oid)
                break

    # Reset d_idx for delivered orders
    d_idx = 0
    while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
        d_idx += 1

    # #15: Return transfer needed (1 task)
    for uid_c, oid_c, has_gc_c in cc_no_gc_delivered:
        if oid_c in used_orders_this_run or oid_c in used_order_ids:
            continue
        if not has_gc_c:  # must not have gift card
            user = db["users"][uid_c]
            has_paypal = any(pm["source"] == "paypal" for pm in user["payment_methods"].values())
            if has_paypal:
                use_order(oid_c)
                result = try_gen(gen_hard_return_transfer_needed, db, uid_c, oid_c)
                if result:
                    break

    # #16: Exchange cross product type (2 tasks)
    for i in range(2):
        while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
            d_idx += 1
        if d_idx < len(delivered_orders):
            uid, oid = delivered_orders[d_idx]
            use_order(oid)
            d_idx += 1
            try_gen(gen_hard_exchange_cross_product_type, db, uid, oid)

    # #17: Exchange unavailable variant (2 tasks)
    for i in range(2):
        while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
            d_idx += 1
        if d_idx < len(delivered_orders):
            uid, oid = delivered_orders[d_idx]
            use_order(oid)
            d_idx += 1
            try_gen(gen_hard_exchange_unavailable_variant, db, uid, oid)

    # #18: Exchange gift card insufficient (1 task)
    for uid_candidate in low_gc_users:
        for oid, order in db["orders"].items():
            if order["user_id"] == uid_candidate and order["status"] == "delivered" and oid not in used_orders_this_run and oid not in used_order_ids:
                use_order(oid)
                result = try_gen(gen_hard_exchange_gift_card_insufficient, db, uid_candidate, oid)
                if result:
                    break
        if result:
            break

    # #19: Out of scope request (2 tasks)
    u_idx = 0
    for i in range(2):
        while u_idx < len(available_users) and available_users[u_idx] in used_users_this_run:
            u_idx += 1
        if u_idx < len(available_users):
            uid = available_users[u_idx]
            use_user(uid)
            u_idx += 1
            try_gen(gen_hard_out_of_scope_request, db, uid)

    # #20: Impossible demand transfer (2 tasks)
    for i in range(2):
        while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
            d_idx += 1
        if d_idx < len(delivered_orders):
            uid, oid = delivered_orders[d_idx]
            use_order(oid)
            d_idx += 1
            try_gen(gen_hard_impossible_demand_transfer, db, uid, oid)

    # #21: Cancel pending + return delivered (2 tasks)
    mf_idx = 0
    for i in range(2):
        while mf_idx < len(multi_flow_users):
            uid, data = multi_flow_users[mf_idx]
            mf_idx += 1
            p_oid = None
            d_oid = None
            for oid in data["pending"]:
                if oid not in used_orders_this_run and oid not in used_order_ids:
                    p_oid = oid
                    break
            for oid in data["delivered"]:
                if oid not in used_orders_this_run and oid not in used_order_ids:
                    d_oid = oid
                    break
            if p_oid and d_oid:
                use_order(p_oid)
                use_order(d_oid)
                try_gen(gen_hard_cancel_pending_return_delivered, db, uid, p_oid, d_oid)
                break

    # #22: Exchange + modify (2 tasks)
    for i in range(2):
        while mf_idx < len(multi_flow_users):
            uid, data = multi_flow_users[mf_idx]
            mf_idx += 1
            p_oid = None
            d_oid = None
            for oid in data["pending"]:
                if oid not in used_orders_this_run and oid not in used_order_ids:
                    p_oid = oid
                    break
            for oid in data["delivered"]:
                if oid not in used_orders_this_run and oid not in used_order_ids:
                    d_oid = oid
                    break
            if p_oid and d_oid:
                use_order(p_oid)
                use_order(d_oid)
                try_gen(gen_hard_exchange_and_modify, db, uid, p_oid, d_oid)
                break

    # #24: Modify address + modify items (2 tasks)
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_modify_address_and_items, db, uid, oid)

    # #25: Exchange or return fallback (variant available → exchange) (1 task)
    while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
        d_idx += 1
    if d_idx < len(delivered_orders):
        uid, oid = delivered_orders[d_idx]
        use_order(oid)
        d_idx += 1
        try_gen(gen_hard_exchange_or_return_fallback, db, uid, oid)

    # #25b: Exchange fallback to return (variant unavailable → return) (1 task)
    while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
        d_idx += 1
    if d_idx < len(delivered_orders):
        uid, oid = delivered_orders[d_idx]
        use_order(oid)
        d_idx += 1
        try_gen(gen_hard_exchange_fallback_return, db, uid, oid)

    # #26: Cancel or return fallback (delivered → can't cancel → return) (1 task)
    while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
        d_idx += 1
    if d_idx < len(delivered_orders):
        uid, oid = delivered_orders[d_idx]
        use_order(oid)
        d_idx += 1
        try_gen(gen_hard_cancel_or_return_fallback, db, uid, oid)

    # #28: Exchange multiple fallbacks (A unavail → B avail) (1 task)
    while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
        d_idx += 1
    if d_idx < len(delivered_orders):
        uid, oid = delivered_orders[d_idx]
        use_order(oid)
        d_idx += 1
        try_gen(gen_hard_exchange_multiple_fallbacks, db, uid, oid)

    # #30: Single item cancel (2 tasks)
    si_idx = 0
    for i in range(2):
        while si_idx < len(single_item_pending) and single_item_pending[si_idx][1] in used_orders_this_run:
            si_idx += 1
        if si_idx < len(single_item_pending):
            uid, oid = single_item_pending[si_idx]
            use_order(oid)
            si_idx += 1
            try_gen(gen_hard_single_item_cancel, db, uid, oid)

    # #31: Exchange zero price diff (1 task)
    while d_idx < len(delivered_orders) and delivered_orders[d_idx][1] in used_orders_this_run:
        d_idx += 1
    if d_idx < len(delivered_orders):
        uid, oid = delivered_orders[d_idx]
        use_order(oid)
        d_idx += 1
        try_gen(gen_hard_exchange_zero_price_diff, db, uid, oid)

    # #33: Other user's order (2 tasks)
    for i in range(2):
        while u_idx < len(available_users) and available_users[u_idx] in used_users_this_run:
            u_idx += 1
        if u_idx < len(available_users):
            uid = available_users[u_idx]
            use_user(uid)
            u_idx += 1
            try_gen(gen_hard_other_user_order, db, uid)

    # #34: User backs out (2 tasks)
    for i in range(2):
        while p_idx < len(pending_orders) and pending_orders[p_idx][1] in used_orders_this_run:
            p_idx += 1
        if p_idx < len(pending_orders):
            uid, oid = pending_orders[p_idx]
            use_order(oid)
            p_idx += 1
            try_gen(gen_hard_user_backs_out, db, uid, oid)

    print(f"\nGenerated {len(generated)} hard tasks (IDs {500}-{task_id - 1})")

    # ─── Validation ──────────────────────────────────────────────────────
    print("\n--- Validation ---")
    errors = 0

    from collections import Counter
    purpose_counts = Counter()

    for task in generated:
        purpose = task["description"]["purpose"] or "unknown"
        purpose_counts[purpose] += 1

        actions = task["evaluation_criteria"]["actions"]
        action_names = [a["name"] for a in actions]

        for a in actions:
            name = a["name"]
            args = a["arguments"]

            # Skip validation for transfer_to_human_agents with empty compare_args
            if name == "transfer_to_human_agents":
                continue

            if name == "find_user_id_by_name_zip":
                fn, ln, z = args["first_name"], args["last_name"], args["zip"]
                found = any(
                    u["name"]["first_name"] == fn and u["name"]["last_name"] == ln and u["address"]["zip"] == z
                    for u in db["users"].values()
                )
                if not found:
                    print(f"  [ERROR] Task {task['id']}: User {fn} {ln} zip {z} not found")
                    errors += 1

            if name == "find_user_id_by_email":
                email = args["email"]
                if not any(u["email"] == email for u in db["users"].values()):
                    print(f"  [ERROR] Task {task['id']}: Email {email} not found")
                    errors += 1

            if "order_id" in args:
                oid = args["order_id"]
                if oid not in db["orders"]:
                    print(f"  [ERROR] Task {task['id']}: Order {oid} not found")
                    errors += 1

            if "user_id" in args:
                uid_check = args["user_id"]
                if uid_check not in db["users"]:
                    print(f"  [ERROR] Task {task['id']}: User {uid_check} not found")
                    errors += 1

            if "payment_method_id" in args:
                pm_id = args["payment_method_id"]
                for a2 in actions:
                    if "user_id" in a2["arguments"]:
                        uid_check = a2["arguments"]["user_id"]
                        if pm_id not in db["users"].get(uid_check, {}).get("payment_methods", {}):
                            print(f"  [ERROR] Task {task['id']}: Payment {pm_id} not found for user {uid_check}")
                            errors += 1
                        break

            if "item_ids" in args and "order_id" in args:
                oid = args["order_id"]
                order = db["orders"].get(oid, {})
                order_item_ids = [it["item_id"] for it in order.get("items", [])]
                for item_id in args["item_ids"]:
                    if item_id not in order_item_ids:
                        print(f"  [ERROR] Task {task['id']}: Item {item_id} not in order {oid}")
                        errors += 1

            if "new_item_ids" in args:
                for new_id in args["new_item_ids"]:
                    found = False
                    for product in db["products"].values():
                        if new_id in product["variants"]:
                            if not product["variants"][new_id]["available"]:
                                print(f"  [ERROR] Task {task['id']}: Variant {new_id} not available")
                                errors += 1
                            found = True
                            break
                    if not found:
                        print(f"  [ERROR] Task {task['id']}: Variant {new_id} not found")
                        errors += 1

    print(f"\nTask purposes:")
    for purpose, count in purpose_counts.most_common():
        print(f"  {purpose}: {count}")
    print(f"\nValidation: {errors} errors found")

    # Save
    if output_path is None:
        output_path = str(BASE_DIR / "generated_tasks_hard.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(generated, f, indent=2)

    print(f"\nSaved {len(generated)} hard tasks to {output_path}")
    return generated


# ─── Main generator ──────────────────────────────────────────────────────────

def generate_tasks(n_tasks=85, seed=123, output_path=None):
    rng = random.Random(seed)
    # Also set module-level random for helper functions
    random.seed(seed)

    db = load_db()
    existing_tasks = load_existing_tasks()

    used_order_ids = get_used_order_ids(existing_tasks)
    used_user_ids = get_used_user_ids(existing_tasks, db)

    print(f"DB: {len(db['users'])} users, {len(db['orders'])} orders, {len(db['products'])} products")
    print(f"Existing tasks: {len(existing_tasks)}")
    print(f"Used users: {len(used_user_ids)}, Used orders: {len(used_order_ids)}")

    # Build pools of available orders by status, excluding used ones
    pending_orders = []
    delivered_orders = []

    for oid, order in db["orders"].items():
        if oid in used_order_ids:
            continue
        uid = order["user_id"]
        if uid in used_user_ids:
            continue
        if order["status"] == "pending":
            pending_orders.append((uid, oid))
        elif order["status"] == "delivered":
            delivered_orders.append((uid, oid))

    rng.shuffle(pending_orders)
    rng.shuffle(delivered_orders)

    # Users without orders (for modify_user_address)
    available_users = [uid for uid in db["users"] if uid not in used_user_ids]
    rng.shuffle(available_users)

    print(f"Available: {len(pending_orders)} pending orders, {len(delivered_orders)} delivered orders, {len(available_users)} users")

    # Distribution targets
    targets = {
        "cancel": 15,
        "return": 20,
        "exchange": 20,
        "modify_items": 15,
        "modify_address": 10,
        "modify_user_address": 5,
    }

    # Adjust to fit n_tasks
    total_target = sum(targets.values())
    if n_tasks != total_target:
        scale = n_tasks / total_target
        for k in targets:
            targets[k] = max(1, int(targets[k] * scale))
        # Fix rounding
        diff = n_tasks - sum(targets.values())
        keys = list(targets.keys())
        for i in range(abs(diff)):
            if diff > 0:
                targets[keys[i % len(keys)]] += 1
            else:
                targets[keys[i % len(keys)]] -= 1

    print(f"\nGeneration targets: {targets}")

    generated = []
    task_id = 200  # start at 200 to separate from existing tasks
    used_orders_this_run = set()

    pending_idx = 0
    delivered_idx = 0
    user_idx = 0

    def next_pending():
        nonlocal pending_idx
        while pending_idx < len(pending_orders):
            uid, oid = pending_orders[pending_idx]
            pending_idx += 1
            if oid not in used_orders_this_run:
                used_orders_this_run.add(oid)
                return uid, oid
        return None, None

    def next_delivered():
        nonlocal delivered_idx
        while delivered_idx < len(delivered_orders):
            uid, oid = delivered_orders[delivered_idx]
            delivered_idx += 1
            if oid not in used_orders_this_run:
                used_orders_this_run.add(oid)
                return uid, oid
        return None, None

    def next_user():
        nonlocal user_idx
        while user_idx < len(available_users):
            uid = available_users[user_idx]
            user_idx += 1
            return uid
        return None

    # Generate cancel tasks
    for _ in range(targets["cancel"]):
        uid, oid = next_pending()
        if uid is None:
            print("[warn] Ran out of pending orders for cancel tasks")
            break
        task = generate_cancel_task(task_id, db, uid, oid, rng)
        if task:
            generated.append(task)
            task_id += 1

    # Generate return tasks
    for _ in range(targets["return"]):
        uid, oid = next_delivered()
        if uid is None:
            print("[warn] Ran out of delivered orders for return tasks")
            break
        task = generate_return_task(task_id, db, uid, oid, rng)
        if task:
            generated.append(task)
            task_id += 1

    # Generate exchange tasks
    for _ in range(targets["exchange"]):
        uid, oid = next_delivered()
        if uid is None:
            print("[warn] Ran out of delivered orders for exchange tasks")
            break
        task = generate_exchange_task(task_id, db, uid, oid, rng)
        if task:
            generated.append(task)
            task_id += 1
        else:
            # Try another order
            uid, oid = next_delivered()
            if uid:
                task = generate_exchange_task(task_id, db, uid, oid, rng)
                if task:
                    generated.append(task)
                    task_id += 1

    # Generate modify items tasks
    for _ in range(targets["modify_items"]):
        uid, oid = next_pending()
        if uid is None:
            print("[warn] Ran out of pending orders for modify-items tasks")
            break
        task = generate_modify_items_task(task_id, db, uid, oid, rng)
        if task:
            generated.append(task)
            task_id += 1
        else:
            uid, oid = next_pending()
            if uid:
                task = generate_modify_items_task(task_id, db, uid, oid, rng)
                if task:
                    generated.append(task)
                    task_id += 1

    # Generate modify address tasks
    for _ in range(targets["modify_address"]):
        uid, oid = next_pending()
        if uid is None:
            print("[warn] Ran out of pending orders for modify-address tasks")
            break
        task = generate_modify_address_task(task_id, db, uid, oid, rng)
        if task:
            generated.append(task)
            task_id += 1

    # Generate modify user address tasks
    for _ in range(targets["modify_user_address"]):
        uid = next_user()
        if uid is None:
            print("[warn] Ran out of users for modify-user-address tasks")
            break
        task = generate_modify_user_address_task(task_id, db, uid, rng)
        if task:
            generated.append(task)
            task_id += 1

    print(f"\nGenerated {len(generated)} tasks (IDs {200}-{task_id - 1})")

    # ─── Validation ───────────────────────────────────────────────────────
    print("\n--- Validation ---")
    errors = 0

    from collections import Counter
    flow_counts = Counter()

    for task in generated:
        actions = task["evaluation_criteria"]["actions"]
        action_names = [a["name"] for a in actions]

        # Determine flow type
        if "cancel_pending_order" in action_names:
            flow_counts["cancel"] += 1
        elif "return_delivered_order_items" in action_names:
            flow_counts["return"] += 1
        elif "exchange_delivered_order_items" in action_names:
            flow_counts["exchange"] += 1
        elif "modify_pending_order_items" in action_names:
            flow_counts["modify_items"] += 1
        elif "modify_pending_order_address" in action_names:
            flow_counts["modify_address"] += 1
        elif "modify_user_address" in action_names:
            flow_counts["modify_user_address"] += 1

        for a in actions:
            name = a["name"]
            args = a["arguments"]

            # Check referenced IDs exist
            if name in ("find_user_id_by_name_zip",):
                fn = args["first_name"]
                ln = args["last_name"]
                z = args["zip"]
                found = False
                for u in db["users"].values():
                    if u["name"]["first_name"] == fn and u["name"]["last_name"] == ln and u["address"]["zip"] == z:
                        found = True
                        break
                if not found:
                    print(f"  [ERROR] Task {task['id']}: User {fn} {ln} zip {z} not found in DB")
                    errors += 1

            if "order_id" in args:
                oid = args["order_id"]
                if oid not in db["orders"]:
                    print(f"  [ERROR] Task {task['id']}: Order {oid} not found in DB")
                    errors += 1
                elif name == "cancel_pending_order":
                    if db["orders"][oid]["status"] != "pending":
                        print(f"  [ERROR] Task {task['id']}: Order {oid} is {db['orders'][oid]['status']}, expected pending")
                        errors += 1
                elif name in ("return_delivered_order_items", "exchange_delivered_order_items"):
                    if db["orders"][oid]["status"] != "delivered":
                        print(f"  [ERROR] Task {task['id']}: Order {oid} is {db['orders'][oid]['status']}, expected delivered")
                        errors += 1

            if "user_id" in args:
                uid = args["user_id"]
                if uid not in db["users"]:
                    print(f"  [ERROR] Task {task['id']}: User {uid} not found in DB")
                    errors += 1

            if "payment_method_id" in args:
                pm_id = args["payment_method_id"]
                # Find which user this is for
                for a2 in actions:
                    if "user_id" in a2["arguments"]:
                        uid = a2["arguments"]["user_id"]
                        if pm_id not in db["users"].get(uid, {}).get("payment_methods", {}):
                            print(f"  [ERROR] Task {task['id']}: Payment {pm_id} not found for user {uid}")
                            errors += 1
                        break

            if "item_ids" in args:
                for item_id in args["item_ids"]:
                    # Check item exists in the order
                    if "order_id" in args:
                        oid = args["order_id"]
                        order = db["orders"].get(oid, {})
                        order_item_ids = [it["item_id"] for it in order.get("items", [])]
                        if item_id not in order_item_ids:
                            print(f"  [ERROR] Task {task['id']}: Item {item_id} not in order {oid}")
                            errors += 1

            if "new_item_ids" in args:
                for new_id in args["new_item_ids"]:
                    # Check variant exists and is available
                    found = False
                    for product in db["products"].values():
                        if new_id in product["variants"]:
                            if not product["variants"][new_id]["available"]:
                                print(f"  [ERROR] Task {task['id']}: Variant {new_id} not available")
                                errors += 1
                            found = True
                            break
                    if not found:
                        print(f"  [ERROR] Task {task['id']}: Variant {new_id} not found in any product")
                        errors += 1

            if name == "find_user_id_by_email":
                email = args["email"]
                found = any(u["email"] == email for u in db["users"].values())
                if not found:
                    print(f"  [ERROR] Task {task['id']}: Email {email} not found in DB")
                    errors += 1

    print(f"\nFlow distribution: {dict(flow_counts)}")
    print(f"Validation: {errors} errors found")

    # Save
    if output_path is None:
        output_path = str(BASE_DIR / "generated_tasks.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(generated, f, indent=2)

    print(f"Saved {len(generated)} tasks to {output_path}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate new retail tasks for tau2-bench")
    parser.add_argument("--mode", choices=["easy", "hard"], default="easy",
                        help="Generate easy (single-flow) or hard (policy-testing) tasks")
    parser.add_argument("--n_tasks", type=int, default=85, help="Number of tasks (easy mode only)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output_path", type=str, default=None, help="Output path")
    args = parser.parse_args()

    if args.mode == "hard":
        generate_hard_tasks(seed=args.seed, output_path=args.output_path)
    else:
        generate_tasks(n_tasks=args.n_tasks, seed=args.seed, output_path=args.output_path)


if __name__ == "__main__":
    main()
