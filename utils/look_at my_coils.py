# function to read the current values of the bi-planar coils 
import argparse
import shlex
import sys
from typing import List

# Your existing hardware API
from utils.compcoils import CompFieldControl


HELP = """\
Commands:
  set <idx> <value>         Set a single coil's field (idx 0-7, value float)
  all <v0> ... <v7>         Set all 8 coil fields at once (8 floats)
  read                      (optional) Read/print current values if your API supports it
  help                      Show this help
  exit / quit               Close and exit
"""

def parse_all(args: List[str]) -> List[float]:
    if len(args) != 8:
        raise ValueError("Expected 8 floats for 'all'.")
    return [float(x) for x in args]

def main():
    parser = argparse.ArgumentParser(description="Coil control REPL (keeps a single persistent connection).")
    parser.add_argument("--startup", nargs="*", default=None,
                        help="Optional one-shot command to run on startup, e.g. --startup all 0 0 0 0 0 0 0 0")
    parsed = parser.parse_args()

    # Establish exactly one connection up-front
    coil = CompFieldControl()
    print("Connected. Type 'help' for commands.")

    try:
        # Optional one-shot command
        if parsed.startup:
            line = " ".join(parsed.startup)
            print(f">>> {line}")
            handle_command(coil, line)

        # Interactive loop
        while True:
            try:
                line = input("coil> ").strip()
            except EOFError:
                print()
                break

            if not line:
                continue

            if line.lower() in ("exit", "quit"):
                break

            if line.lower() in ("help", "?"):
                print(HELP)
                continue

            try:
                handle_command(coil, line)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        # Clean up the background monitor thread if it exists
        ser_monitor = getattr(coil, "ser_monitor", None)
        if ser_monitor is not None:
            try:
                ser_monitor.join(timeout=1.0)
            except Exception:
                pass
        # If your API has a close()/shutdown(), call it here:
        for method in ("close", "shutdown", "disconnect"):
            f = getattr(coil, method, None)
            if callable(f):
                try:
                    f()
                    break
                except Exception:
                    pass
        print("Disconnected. Bye!")

def handle_command(coil: CompFieldControl, line: str) -> None:
    """Parse and execute a single command line."""
    parts = shlex.split(line)
    if not parts:
        return
    cmd, *rest = parts

    if cmd == "set":
        if len(rest) != 2:
            raise ValueError("Usage: set <idx> <value>")
        idx = int(rest[0])
        if idx not in range(8):
            raise ValueError("Index must be 0..7")
        val = float(rest[1])
        coil.setOffset(idx, val)
        print(f"Set coil[{idx}] = {val}")

    elif cmd == "all":
        vals = parse_all(rest)
        coil.set_coil_values(vals)
        print("Set all coils:", vals)

    elif cmd == "read":
        # Optional: adapt this if CompFieldControl exposes getters
        for getter in ("get_coil_values", "read_values", "readAll"):
            f = getattr(coil, getter, None)
            if callable(f):
                vals = f()
                print("Current:", vals)
                break
        else:
            print("No reader method found on CompFieldControl.")

    else:
        raise ValueError(f"Unknown command '{cmd}'. Type 'help'.")

if __name__ == "__main__":
    main()
