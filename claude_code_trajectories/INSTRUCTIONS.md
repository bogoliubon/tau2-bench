# Claude Code Agent: Retail Task Runner

## What This Is
You (Claude Code) are playing the customer service agent role for tau2-bench retail tasks.
You interact with a simulated user via file-based communication.

## Task IDs (74 training tasks)
0,1,2,3,4,6,7,8,10,11,13,14,15,16,19,20,21,22,23,24,25,28,29,30,31,34,35,37,41,43,44,46,47,48,50,52,54,57,58,59,63,66,67,69,72,73,75,76,78,80,81,82,83,84,85,87,88,89,91,92,93,95,96,98,99,103,104,105,106,107,109,110,112,113

## How To Run Each Task

### 1. Start the agent server (background)
```
source /tmp/tau2_env.sh && cd "/Users/tudor/work/ML Research/tau2-bench" && python scripts/claude_code_agent.py --task_id TASK_ID 2>&1
```
Run this in background. Wait ~15s for ready file.

### 2. Check ready
```
cat /tmp/tau2_ready 2>/dev/null && echo "READY" || echo "NOT READY"
```

### 3. Read observation
```
cat /tmp/tau2_obs.txt
```
The first observation includes the POLICY, TOOLS, and the user's initial message.

### 4. Send response
```
echo "Your text response here" > /tmp/tau2_cmd.txt
```
Or for tool calls:
```
echo "find_user_id_by_email(email='user@example.com')" > /tmp/tau2_cmd.txt
```

### 5. Wait and read next observation
```
sleep 10 && cat /tmp/tau2_obs.txt
```

### 6. Repeat until terminated

## Agent Policy Summary (key rules)
1. AUTHENTICATE first: find user by email OR name+zip. Always do this even if user provides user_id.
2. ONE tool call at a time, no text+tool in same turn.
3. Get EXPLICIT confirmation (yes) before any DB-modifying action.
4. Remind user to confirm ALL items before modify/exchange.
5. Cancel: only pending orders, reason must be "no longer needed" or "ordered by mistake".
6. Modify items: can only be called ONCE per order, same product type only.
7. Modify payment: must be DIFFERENT from original.
8. Return: refund to original payment method or existing gift card only.
9. Exchange: same product type only, check availability.
10. Transfer to human if request is out of scope.
11. Call done() when conversation is complete.

## How To Check Progress
Read: /Users/tudor/work/ML Research/tau2-bench/claude_code_trajectories/progress.json

## Resume After Compaction
1. Read progress.json to see completed/failed tasks
2. Pick next uncompleted task ID from the list above
3. Start the agent server for that task
4. Play through it following the policy
