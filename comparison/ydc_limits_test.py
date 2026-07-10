"""
ydc_limits_test.py — probe include_domains limits on the YDC Search API.

Three sweeps:
  1. COUNT sweep:    fixed query + 5 domains, count=1..10 → find where it breaks
  2. DOMAIN sweep:   fixed query + max working count, domains 1..N → find where it breaks
  3. PARALLEL sweep: simulate GPT-5.5 parallel tool calls — fire N concurrent
                     requests simultaneously, ramp from 2..8, find the TPS threshold

Usage:
    python ydc_limits_test.py
    YDC_API_KEY=xxx python ydc_limits_test.py
"""

import os
import sys
import time
import threading
import requests

# ── Config ──────────────────────────────────────────────────────────────────

ENDPOINT = "https://ydc-index.io/v1/search"

QUERY = "2026 Winter Olympics men's ice hockey gold medal winner"
FRESHNESS = "year"

# Five domains confirmed working in production (count=5 path)
BASE_DOMAINS = [
    "olympics.com",
    "iihf.com",
    "reuters.com",
    "apnews.com",
    "wikipedia.org",
]

# Extra domains for the domain-expansion sweep
EXTRA_DOMAINS = [
    "bbc.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "espn.com",
    "sportsnet.ca",
    "tsn.ca",
    "nbcsports.com",
    "cbc.ca",
    "yahoo.com",
    "usatoday.com",
    "si.com",
    "theatlantic.com",
    "axios.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "wsj.com",
    "npr.org",
    "politico.com",
]

# ── API key ──────────────────────────────────────────────────────────────────

def get_api_key():
    key = os.environ.get("YDC_API_KEY", "")
    if key:
        return key
    # Try env.txt relative to this script's parent
    env_path = os.path.join(os.path.dirname(__file__), "..", "grounding", "env.txt")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("YDC_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""

# ── Single request ───────────────────────────────────────────────────────────

def probe(query, count, domains, freshness=None):
    """Returns (status_code, result_count, latency_ms, error_body)."""
    params = {
        "query": query,
        "count": count,
        "include_domains": ",".join(domains),
    }
    if freshness:
        params["freshness"] = freshness

    key = get_api_key()
    t0 = time.perf_counter()
    try:
        resp = requests.get(
            ENDPOINT,
            headers={"X-API-Key": key},
            params=params,
            timeout=15,
        )
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("results", {})
            results = raw.get("web", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
            return resp.status_code, len(results), latency, None
        else:
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:200]
            return resp.status_code, 0, latency, body
    except requests.Timeout:
        return None, 0, (time.perf_counter() - t0) * 1000, "timeout"
    except Exception as e:
        return None, 0, (time.perf_counter() - t0) * 1000, str(e)

# ── Display helpers ──────────────────────────────────────────────────────────

GREEN = "\033[32m"
RED   = "\033[31m"
DIM   = "\033[2m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def ok(s):  return f"{GREEN}{s}{RESET}"
def err(s): return f"{RED}{s}{RESET}"
def dim(s): return f"{DIM}{s}{RESET}"

def print_row(label, status, result_count, latency, error_body):
    if status == 200:
        status_str = ok(f"200 OK  ({result_count} results)")
    elif status is None:
        status_str = err(f"ERROR   {error_body}")
    else:
        status_str = err(f"{status} FAIL  {error_body}")
    print(f"  {label:<30}  {status_str}  {dim(f'{latency:.0f}ms')}")

# ── Sweep 1: count=1..10 ─────────────────────────────────────────────────────

def sweep_count():
    print(f"\n{BOLD}── Sweep 1: count=1..10 (fixed domains={len(BASE_DOMAINS)}) ──{RESET}")
    print(f"   Query:   {dim(QUERY)}")
    print(f"   Domains: {dim(', '.join(BASE_DOMAINS))}")
    print(f"   Freshness: {dim(FRESHNESS)}\n")

    max_passing = None
    for count in range(1, 11):
        status, result_count, latency, error_body = probe(QUERY, count, BASE_DOMAINS, FRESHNESS)
        label = f"count={count}"
        print_row(label, status, result_count, latency, error_body)
        if status == 200:
            max_passing = count
        time.sleep(0.3)

    if max_passing is not None:
        print(f"\n  {BOLD}Max passing count: {max_passing}{RESET}")
    else:
        print(f"\n  {err('All counts failed.')}")
    return max_passing

# ── Sweep 2: domain expansion ────────────────────────────────────────────────

def sweep_domains(fixed_count):
    all_domains = BASE_DOMAINS + EXTRA_DOMAINS
    print(f"\n{BOLD}── Sweep 2: domain expansion (fixed count={fixed_count}) ──{RESET}")
    print(f"   Query:     {dim(QUERY)}")
    print(f"   Freshness: {dim(FRESHNESS)}\n")

    max_passing = None
    for n in range(1, len(all_domains) + 1):
        domains = all_domains[:n]
        status, result_count, latency, error_body = probe(QUERY, fixed_count, domains, FRESHNESS)
        label = f"{n} domain{'s' if n != 1 else '':1}  ({domains[-1]})"
        print_row(label, status, result_count, latency, error_body)
        if status == 200:
            max_passing = n
        else:
            # Stop after first failure — no point continuing if it's a hard limit
            print(f"\n  {BOLD}Broke at {n} domains. Max passing: {max_passing}{RESET}")
            return max_passing
        time.sleep(0.3)

    print(f"\n  {BOLD}All {len(all_domains)} domains passed at count={fixed_count}{RESET}")
    return max_passing

# ── Sweep 3: parallel load ───────────────────────────────────────────────────

# Distinct queries so requests don't collapse into a single cache hit
PARALLEL_QUERIES = [
    "2026 Winter Olympics men's ice hockey gold medal winner",
    "2026 Winter Olympics women's ice hockey gold medal",
    "2026 Milan Cortina Winter Olympics opening ceremony",
    "2026 Winter Olympics figure skating gold medal results",
    "2026 Winter Olympics alpine skiing downhill gold medal",
    "2026 Winter Olympics speed skating 1000m results",
    "2026 Winter Olympics biathlon pursuit gold medal winner",
    "2026 Winter Olympics ski jumping individual normal hill",
]

def _fire(idx, query, count, domains, freshness, results):
    """Thread worker — stores (idx, status, result_count, latency, error_body)."""
    status, result_count, latency, error_body = probe(query, count, domains, freshness)
    results[idx] = (status, result_count, latency, error_body)

def sweep_parallel(fixed_count):
    print(f"\n{BOLD}── Sweep 3: parallel load (fixed count={fixed_count}, domains={len(BASE_DOMAINS)}) ──{RESET}")
    print(f"   Simulates GPT-5.5 firing N tool calls simultaneously.")
    print(f"   Each request uses a distinct query to avoid cache effects.\n")

    for n_parallel in range(2, 9):
        results = [None] * n_parallel
        threads = []
        for i in range(n_parallel):
            q = PARALLEL_QUERIES[i % len(PARALLEL_QUERIES)]
            t = threading.Thread(target=_fire, args=(i, q, fixed_count, BASE_DOMAINS, FRESHNESS, results))
            threads.append(t)

        # Start all threads as close together as possible
        t_burst = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        burst_ms = (time.perf_counter() - t_burst) * 1000

        statuses = [r[0] for r in results]
        fails    = [r for r in results if r[0] != 200]
        ok_count = sum(1 for s in statuses if s == 200)
        fail_count = len(fails)

        summary = ok(f"{ok_count} OK") if fail_count == 0 else err(f"{ok_count} OK  {fail_count} FAIL")
        fail_codes = "  " + " ".join(err(str(r[0])) for r in fails) if fails else ""
        print(f"  {n_parallel} concurrent  →  {summary}{fail_codes}  {dim(f'(burst {burst_ms:.0f}ms)')}")

        if fails:
            for i, r in enumerate(results):
                status, result_count, latency, error_body = r
                marker = err("✗") if status != 200 else ok("✓")
                print(f"      [{marker}] req {i+1}: {status}  {dim(str(error_body) if error_body else f'{result_count} results')}  {dim(f'{latency:.0f}ms')}")

        time.sleep(1.0)  # cool down between bursts

# ── Sweep 4: freshness values ────────────────────────────────────────────────

FRESHNESS_VALUES = [None, "day", "week", "month", "year"]

def sweep_freshness(fixed_count):
    print(f"\n{BOLD}── Sweep 4: freshness values (fixed count={fixed_count}, domains={len(BASE_DOMAINS)}) ──{RESET}")
    print(f"   Query:   {dim(QUERY)}")
    print(f"   Domains: {dim(', '.join(BASE_DOMAINS))}\n")

    for freshness in FRESHNESS_VALUES:
        label = f"freshness={freshness if freshness else '(none)'}"
        status, result_count, latency, error_body = probe(QUERY, fixed_count, BASE_DOMAINS, freshness)
        print_row(label, status, result_count, latency, error_body)
        time.sleep(0.5)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    key = get_api_key()
    if not key:
        print(err("YDC_API_KEY not found. Set it in environment or grounding/env.txt"))
        sys.exit(1)
    print(dim(f"Using API key: {key[:12]}..."))

    max_count = sweep_count()

    if max_count is None:
        print(err("\nCount sweep found no passing values — skipping domain sweep."))
        sys.exit(1)

    sweep_domains(max_count)
    sweep_parallel(max_count)
    sweep_freshness(max_count)

if __name__ == "__main__":
    main()
