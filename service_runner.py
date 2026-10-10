"""Hosts one service's existing RunPod handler as a long-lived child process.

The three models need different library versions, so each runs in its own
virtualenv. The dispatcher talks to this process over stdin/stdout, one JSON
object per line. The service's own handler module is imported untouched: we
only intercept its `runpod.serverless.start({...})` call to grab the handler.

stdout is the protocol channel, so the real stdout is duplicated first and
fd 1 is redirected to stderr — anything the handler prints becomes a log line
instead of corrupting the protocol.
"""

import importlib
import json
import os
import sys
import traceback


def main():
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import runpod

    captured = {}
    runpod.serverless.start = lambda config: captured.update(config)
    importlib.import_module("runpod_handler")
    handler = captured.get("handler")
    if handler is None:
        print("[RUNNER] runpod_handler did not register a handler", flush=True)
        sys.exit(2)

    proto.write(json.dumps({"ready": True}) + "\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            result = handler({"id": msg.get("id", "job"), "input": msg.get("input", {})})
        except Exception:
            print(traceback.format_exc(), flush=True)
            result = {"ok": False, "error": "Internal server error"}
        proto.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
