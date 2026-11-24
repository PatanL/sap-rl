#!/usr/bin/env python3
"""Symbolise and summarise `envbench --sample` output.

Usage: profile_report.py <samples.txt> [top N]
Prints self time (innermost frame, inlined functions resolved by addr2line)
and inclusive time (a function anywhere in the stack) as shares of all samples.
"""
import collections, re, subprocess, sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
lines = open(path).read().splitlines()
exe = lines[0].split()[2]
# Return addresses point after the call: step back into the call instruction for the callers.
stacks = [[a if k == 0 else format(max(int(a, 16) - 4, 0), "x") for k, a in enumerate(l.split())] for l in lines[1:] if l]
addresses = sorted({a for s in stacks for a in s})
query = list(addresses)
out = subprocess.run(["addr2line", "-f", "-C", "-i", "-a", "-e", exe] + ["0x" + a for a in query],
                     capture_output=True, text=True).stdout.splitlines()
names, current = {}, None
for line in out:
    if line.startswith("0x"):
        current = line[2:].lstrip("0") or "0"
        names[current] = []
    elif current is not None and not line.startswith("/") and not line.startswith("??:") and ":" not in line[:2]:
        names[current].append(line)
clean = lambda n: re.sub(r"::h[0-9a-f]{16}$", "", n)
resolve = {a: [clean(n) for n in names.get(a.lstrip("0") or "0", ["??"])] or ["??"] for a in addresses}
self_time, inclusive = collections.Counter(), collections.Counter()
for stack in stacks:
    inner = resolve[stack[0]]
    self_time[inner[0]] += 1
    seen = set()
    for address in stack:
        for name in resolve[address]:
            seen.add(name)
    for name in seen:
        inclusive[name] += 1
total = len(stacks)
print(f"{total} samples from {exe}")
print("\nself time (innermost, inlining resolved)")
for name, count in self_time.most_common(top):
    print(f"{100 * count / total:6.2f}%  {name}")
print("\ninclusive time")
for name, count in inclusive.most_common(top):
    print(f"{100 * count / total:6.2f}%  {name}")

# Samples whose innermost frame is outside the binary (libc: memset, memcpy, malloc ...): who called it.
callers = collections.Counter()
for stack in stacks:
    if resolve[stack[0]][0] == "??":
        chain = [resolve[a][0] for a in stack[1:4]]
        callers[" <- ".join(chain[:2])] += 1
if callers:
    print("\ncallers of code outside the binary (libc)")
    for name, count in callers.most_common(15):
        print(f"{100 * count / total:6.2f}%  {name}")
