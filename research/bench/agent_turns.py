#!/usr/bin/env python3
"""Multi-turn measurement modelled on a coding agent

As the conversation grows, llama-server's prompt cache (context checkpoints)
starts to hit and only the delta is prefilled from the second turn on. How much
MoEStream's prefill penalty matters in real use depends on the size of that
delta.

Recorded each turn:
  prompt_n   : tokens actually evaluated (smaller when the cache hits)
  prompt_ms  : time that prefill took = the dominant component of TTFT
  predicted_ms / tok/s : the generation side
"""
import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8091"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "?"
N_TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
URL = f"http://localhost:{PORT}/completion"

# The code an agent carries in its context (about 2000 tokens)
with open("/bench/corpus/code.txt", encoding="utf-8", errors="ignore") as f:
    BASE = f.read(7000)

QUESTIONS = [
    "Summarize the purpose of this code in one sentence.",
    "What are the main data structures used here?",
    "Identify one potential performance issue.",
    "How would you add a unit test for this?",
    "What refactoring would you suggest first?",
    "Are there any thread-safety concerns?",
]

def post(prompt, n_predict=48):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.7, "cache_prompt": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d, time.time() - t0

print(f"=== {LABEL}  (port {PORT}) ===")
print(f"{'turn':>4} {'eval tok':>8} {'prefill(s)':>11} {'~TTFT':>10} "
      f"{'decode tok/s':>13} {'wall(s)':>10}")

convo = "### Code under review\n" + BASE + "\n"
total = 0.0
for i in range(N_TURNS):
    convo += f"\n### User\n{QUESTIONS[i % len(QUESTIONS)]}\n### Assistant\n"
    d, wall = post(convo)
    t = d.get("timings", {})
    pn   = t.get("prompt_n", 0)
    pms  = t.get("prompt_ms", 0.0) / 1000.0
    dtps = t.get("predicted_per_second", 0.0)
    total += wall
    print(f"{i+1:>4} {pn:>8} {pms:>11.2f} {pms:>10.2f} {dtps:>13.2f} {wall:>10.2f}")
    convo += d.get("content", "")

print(f"{'sum':>4} {'':>8} {'':>11} {'':>10} {'':>13} {total:>10.2f}")
