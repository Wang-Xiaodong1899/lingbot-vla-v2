"""Monitor an existing detached refiner run and resume on bounded failures.

Create <output-dir>/STOP_TRAINING, or SIGTERM this supervisor, to stop BOTH
the supervisor and its training process group without automatic restart.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_info(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return {"alive": text[0] != "Z", "group": int(text[2]), "start": text[19], "command": command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_training_group(group):
    # Only signal groups still containing this training entrypoint.
    def owned():
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                info = process_info(int(entry.name))
                if info and info["alive"] and info["group"] == group and "tasks/vla/train_action_refiner.py" in info["command"]:
                    return True
        return False

    if not owned():
        return
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 45
    while owned() and time.monotonic() < deadline:
        time.sleep(1)
    if owned():
        os.killpg(group, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    args = parser.parse_args()
    record = json.loads(args.run_record.read_text())
    output = args.run_record.resolve().parent
    status_file = output / "supervisor_status.json"
    stop_file = output / "STOP_TRAINING"
    lock = output / "supervisor.lock"
    # flock prevents two supervisors from restarting the same job.
    import fcntl
    lock_stream = lock.open("w")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    pid = record["pid"]
    initial = process_info(pid)
    if not initial or "tasks/vla/train_action_refiner.py" not in initial["command"] or initial["group"] != pid:
        raise RuntimeError("Run record does not identify a live detached training process group")
    identity = initial["start"]
    restarts, last_step = 0, record["resume_step"]
    last_progress = time.monotonic()
    watched_log = Path(record["log"])
    active = None

    def save(status, **details):
        state = dict(status=status, supervisor_pid=os.getpid(), training_pid=pid, restarts=restarts,
                     max_restarts=args.max_restarts, last_observed_step=last_step,
                     training_log=str(watched_log), stop_file=str(stop_file),
                     updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        os.replace(temporary, status_file)

    def stop(_sig=None, _frame=None):
        stop_training_group(pid)
        save("stopped_by_user")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        if stop_file.exists():
            stop()
        info = process_info(pid)
        alive = info and info["alive"] and info["start"] == identity
        # Read only this launch's log: older metrics may contain later interrupted steps.
        if watched_log.exists():
            for line in watched_log.read_text(errors="replace").splitlines():
                if line.startswith('{"step":'):
                    try:
                        step = json.loads(line)["step"]
                    except (ValueError, KeyError):
                        continue
                    if step > last_step:
                        last_step, last_progress = step, time.monotonic()
        stalled = time.monotonic() - last_progress > args.stall_seconds
        if alive and not stalled:
            save("running", seconds_without_logged_progress=round(time.monotonic() - last_progress))
            time.sleep(20)
            continue
        reason = "no logged optimizer progress for 30 minutes" if stalled else "training process exited"
        print(datetime.now(timezone.utc).isoformat(), reason, flush=True)
        stop_training_group(pid)
        if active is not None:
            active.wait(timeout=60)
        if restarts >= args.max_restarts:
            save("failed_restart_limit", reason=reason)
            return
        save("preparing_restart", reason=reason)
        # The full base and batch-192 KV cache need more than 70 GB per GPU.
        while True:
            if stop_file.exists():
                stop()
            query = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
            free = [int(value) for value in query.stdout.splitlines()] if query.returncode == 0 else []
            if len(free) >= 4 and min(free[:4]) >= 73000:
                break
            save("waiting_for_free_gpus", free_mib=free)
            time.sleep(20)
        import torch
        torch.set_num_threads(1)
        resume = output / "latest.pt"
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        saved_step = checkpoint["step"]
        del checkpoint
        command = list(record["command"])
        command[command.index("--resume") + 1] = str(resume)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=record["cuda_visible_devices"],
                   NPROC_PER_NODE=str(record["nproc_per_node"]), PYTHONUNBUFFERED="1")
        restarts += 1
        watched_log = output / "logs" / f"auto_resume_{saved_step}_{time.time_ns()}.log"
        with watched_log.open("w") as stream:
            active = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env,
                                      stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        pid = active.pid
        identity = process_info(pid)["start"]
        last_step, last_progress = saved_step, time.monotonic()
        record.update(pid=pid, log=str(watched_log), command=command, resume_step=saved_step)
        args.run_record.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        save("restarting", resumed_from_step=saved_step)
        time.sleep(20)


if __name__ == "__main__":
    main()
