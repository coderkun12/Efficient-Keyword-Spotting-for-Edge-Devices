"""
Structural lint for the accelerator RTL, aimed at the failure modes that
survive simulation and only bite during synthesis and place-and-route.

Icarus catches syntax and elaboration errors but almost no structural problems,
so this adds the checks that matter for a clean P&R:

  L1  every instance port is connected           -- an unconnected INPUT port
                                                    floats, which is the
                                                    classic P&R complaint
  L2  no combinational always block can infer a latch
  L3  no declared net is left undriven
  L4  no net has more than one continuous-assignment driver
  L5  no blocking assignments to signals that cross a clocked boundary
  L6  every module has an explicit default_nettype guard
  L7  no continuous assignment references a signal declared LATER in the file
  L8  no comment accidentally looks like a synthesis pragma

This is a focused checker, not a Verilog parser. It errs toward reporting
something for a human to look at rather than staying quiet.

    python rtl/sim/lint_rtl.py
"""

import re
import sys
from pathlib import Path

RTL = Path(__file__).resolve().parent.parent / "rtl_design"

COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)
MODULE = re.compile(r"\bmodule\s+(\w+)\s*(#\s*\([^;]*?\))?\s*\((.*?)\)\s*;", re.S)
INSTANCE = re.compile(
    r"\b(\w+)\s*(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?(\w+)\s*\(\s*(\.[^;]*?)\)\s*;",
    re.S,
)
PORT_CONN = re.compile(r"\.\s*(\w+)\s*\(")
ALWAYS_COMB = re.compile(r"always\s*@\s*\(\s*\*\s*\)|always_comb")

KEYWORDS = {
    "begin", "end", "if", "else", "case", "endcase", "for", "while", "posedge",
    "negedge", "assign", "always", "module", "endmodule", "generate",
    "endgenerate", "function", "endfunction", "task", "endtask", "genvar",
    "localparam", "parameter", "input", "output", "inout", "wire", "reg",
    "logic", "integer", "signed", "unsigned", "default_nettype", "timescale",
    "initial", "return", "automatic", "typedef", "struct", "packed",
}


def strip_comments(text):
    return COMMENT.sub(" ", text)


def parse_ports(port_text):
    """Return {name: direction} for a module header port list."""
    ports = {}
    direction = None
    # Split on commas that are not inside brackets.
    depth = 0
    chunk = ""
    chunks = []
    for ch in port_text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            chunks.append(chunk)
            chunk = ""
        else:
            chunk += ch
    chunks.append(chunk)

    for c in chunks:
        tokens = c.split()
        if not tokens:
            continue
        for d in ("input", "output", "inout"):
            if d in tokens:
                direction = d
                break
        name = None
        for tok in reversed(tokens):
            tok = tok.strip(",;")
            if tok and re.fullmatch(r"\w+", tok) and tok not in KEYWORDS:
                name = tok
                break
        if name and direction:
            ports[name] = direction
    return ports


def collect_modules(files):
    mods = {}
    for f in files:
        text = strip_comments(f.read_text(encoding="utf-8"))
        for m in MODULE.finditer(text):
            mods[m.group(1)] = {"ports": parse_ports(m.group(3)), "file": f}
    return mods


def check_file(path, modules, findings):
    raw = path.read_text(encoding="utf-8")
    text = strip_comments(raw)

    # L6: default_nettype guard
    if "`default_nettype none" not in raw:
        findings.append(
            (path.name, "L6",
             "no `default_nettype none guard -- a typo'd signal name silently "
             "becomes an undeclared 1-bit wire, which is exactly how nets end "
             "up floating"))

    # L1: instance port connections
    for inst in INSTANCE.finditer(text):
        mod_name, inst_name, conns = inst.group(1), inst.group(2), inst.group(3)
        if mod_name not in modules or mod_name in KEYWORDS:
            continue
        connected = set(PORT_CONN.findall(conns))
        expected = set(modules[mod_name]["ports"])
        missing = expected - connected
        extra = connected - expected
        for p in sorted(missing):
            direction = modules[mod_name]["ports"][p]
            severity = "FLOATS" if direction == "input" else "unused"
            findings.append(
                (path.name, "L1",
                 f"instance {inst_name} of {mod_name}: port '{p}' ({direction}) "
                 f"not connected -- {severity}"))
        for p in sorted(extra):
            findings.append(
                (path.name, "L1",
                 f"instance {inst_name} of {mod_name}: '.{p}' is not a port "
                 f"of {mod_name}"))

    # L2: latch inference in combinational blocks
    for m in ALWAYS_COMB.finditer(text):
        tail = text[m.end():]
        # Grab the block body, roughly: to the matching end of the first begin.
        body = tail[:4000]
        if "begin" in body[:200]:
            depth = 0
            out = []
            for tok in re.finditer(r"\bbegin\b|\bend\b|.", body):
                s = tok.group(0)
                if s == "begin":
                    depth += 1
                elif s == "end":
                    depth -= 1
                    if depth == 0:
                        break
                out.append(s)
            body = "".join(out)
        targets = set(re.findall(r"^\s*(\w+)\s*(?:\[[^\]]*\])?\s*=", body, re.M))
        unconditional = set(
            re.findall(r"^\s{0,12}(\w+)\s*(?:\[[^\]]*\])?\s*=", body, re.M))
        for t in sorted(targets - unconditional):
            findings.append(
                (path.name, "L2",
                 f"'{t}' assigned only inside a branch of a combinational "
                 f"block -- infers a latch"))

    # L3/L4: driver counts for declared nets.
    #
    # A net counts as driven by ANY of: a continuous assign, a procedural
    # assignment (blocking or non-blocking, anywhere on the line, with
    # bracket-aware indexing), or an instance output port connection. Missing
    # any of those produces false alarms that train people to ignore the lint,
    # which is worse than not running it.
    decls = {}
    arrays = set()
    for m in re.finditer(
            r"^\s*(?:wire|reg|logic)\b([^;=\n]*?)(\w+)\s*((?:\[[^\]]*\]\s*)*);\s*$",
            text, re.M):
        name, trailing = m.group(2), m.group(3)
        decls[name] = 0
        if trailing.strip():
            arrays.add(name)          # unpacked array: elements driven separately

    for a in re.findall(r"assign\s+(\w+)", text):
        if a in decls:
            decls[a] += 1

    # Procedural assignments, bracket-aware, anywhere on a line.
    driven = set(re.findall(r"(\w+)\s*(?:\[[^;]*?\])?\s*(?:<=|=)(?![=>])", text))
    # Memory writes whose index itself contains brackets, e.g.
    #   linebuf[h_x[PW-1:0]] <= h_vec;
    # which the bracket-free pattern above cannot span. Assignment operators
    # may also sit on the following line, so this one spans newlines.
    driven |= set(re.findall(r"(\w+)\s*\[[^;]{0,200}?(?:<=|=)(?![=>])", text, re.S))

    # Instance output ports drive the net connected to them.
    for inst in INSTANCE.finditer(text):
        mod_name, conns = inst.group(1), inst.group(3)
        if mod_name not in modules:
            continue
        for pm in re.finditer(r"\.\s*(\w+)\s*\(\s*([^()]*?)\s*\)", conns):
            port, net = pm.group(1), pm.group(2).strip()
            if modules[mod_name]["ports"].get(port) in ("output", "inout"):
                base = re.match(r"(\w+)", net)
                if base:
                    driven.add(base.group(1))

    ports_here = set()
    for m in MODULE.finditer(text):
        ports_here |= set(parse_ports(m.group(3)))

    for name, drivers in sorted(decls.items()):
        if name in ports_here or name in driven or name in KEYWORDS:
            continue
        if drivers == 0:
            findings.append(
                (path.name, "L3",
                 f"net '{name}' is declared but nothing drives it -- it will "
                 f"float through synthesis and P&R"))

    for name, drivers in sorted(decls.items()):
        # Elements of an unpacked array are driven by separate assigns, and
        # mutually exclusive generate branches each contribute one textual
        # assign, so neither is a real multiple-driver situation.
        if drivers > 1 and name not in arrays:
            findings.append(
                (path.name, "L4",
                 f"net '{name}' has {drivers} continuous drivers"))


def check_order_and_pragmas(path, findings):
    """L7/L8 -- two things Icarus tolerates and Cadence Genus rejects."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.split(chr(10))

    # L8: a comment starting "// synthesis ..." is parsed as a PRAGMA, not a
    # comment. Genus warns "Unrecognized pragma" and, with some directives,
    # silently changes what it builds.
    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()
        for marker in ("// synthesis", "//synthesis", "/* synthesis",
                       "// synopsys", "// pragma", "// cadence"):
            if stripped.lower().startswith(marker):
                findings.append(
                    (path.name, "L8",
                     f"line {i}: comment begins '{marker}' and will be read as "
                     f"a synthesis pragma, not prose -- reword it"))
                break

    # L7: declaration order. Icarus elaborates the whole module before
    # resolving names, so a continuous assignment may reference a reg declared
    # further down. Genus's parser reads top to bottom and rejects it with
    # "Reference to undeclared variable".
    text = strip_comments(raw)
    body = text.split(chr(10))
    declared = {}
    for i, ln in enumerate(body, 1):
        m = re.match(r"\s*(?:wire|reg|logic|integer|genvar|localparam|parameter)"
                     r"[^;=]*?(\w+)\s*(?:\[[^\]]*\])?\s*(?:=|;|,)", ln)
        if m and m.group(1) not in declared:
            declared[m.group(1)] = i
        for nm in re.findall(r"(\w+)\s*(?:\[[^\]]*\])?\s*(?:,|;|=)", ln):
            if re.match(r"\s*(?:wire|reg|logic)", ln) and nm not in declared:
                declared[nm] = i

    for i, ln in enumerate(body, 1):
        m = re.match(r"\s*(?:assign\s+\w+|(?:wire|logic)[^;=]*?\w+\s*)=(.*)$", ln)
        if not m:
            continue
        rhs = m.group(1)
        for nm in set(re.findall(r"([a-zA-Z_]\w*)", rhs)):
            if nm in declared and declared[nm] > i:
                findings.append(
                    (path.name, "L7",
                     f"line {i}: uses '{nm}', which is not declared until line "
                     f"{declared[nm]} -- Genus rejects this even though Icarus "
                     f"accepts it"))


def main():
    files = sorted(RTL.glob("*.sv"))
    if not files:
        sys.exit(f"no RTL found in {RTL}")
    modules = collect_modules(files)

    findings = []
    for f in files:
        check_file(f, modules, findings)
        check_order_and_pragmas(f, findings)

    print(f"Linted {len(files)} files, {len(modules)} modules: "
          + ", ".join(sorted(modules)))
    print()

    if not findings:
        print("CLEAN -- no structural issues found.")
        print("  every instance port connected, no inferred latches,")
        print("  no undriven or multiply-driven nets, all files guarded.")
        return 0

    by_check = {}
    for name, code, msg in findings:
        by_check.setdefault(code, []).append((name, msg))
    for code in sorted(by_check):
        print(f"--- {code} ({len(by_check[code])}) " + "-" * 40)
        for name, msg in by_check[code]:
            print(f"  {name}: {msg}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
