"""Persistent clean/randomized evaluation driver, waiting for free GPUs.

Snapshots must be made before running this driver. It never changes training
processes and records full episode completion rather than trusting launcher exit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def completed_results(stage_root, instructions, episodes):
    """Reuse only complete task files backed by matching final episode counts."""
    completed = {}
    for result in sorted(stage_root.glob("*/eval_results/*/_result.txt")):
        task = result.parent.name
        log = result.parents[2] / "eval_logs" / f"{task}.log"
        if not log.exists():
            continue
        text = result.read_text()
        if f"Instruction Type: {instructions}" not in text:
            continue
        contents = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
        progress = re.findall(r"Success rate: (\d+)/(\d+) => ([\d.]+)%", contents)
        if not progress or int(progress[-1][1]) != episodes:
            continue
        rate = float(text.strip().splitlines()[-1])
        if abs(rate - int(progress[-1][0]) / episodes) > 1e-6:
            continue
        completed[task] = {"success_rate": rate, "result_file": str(result), "eval_log": str(log)}
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-model", type=Path, help="Refiner mode defaults to checkpoint metadata")
    parser.add_argument("--base-only", action="store_true", help="Evaluate original Lingbot without a refiner")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-per-gpu", type=int, default=1)
    parser.add_argument("--min-free-mib", type=int, default=35000)
    parser.add_argument("--start-port", type=int, default=9430)
    parser.add_argument("--settings", choices=["both", "clean", "randomized"], default="both")
    args = parser.parse_args()
    if args.num_gpus <= 0 or args.num_per_gpu <= 0:
        parser.error("GPU and per-GPU concurrency must be positive")
    if not args.base_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --base-only is set")
    if args.base_only and args.checkpoint is not None:
        parser.error("--checkpoint and --base-only are mutually exclusive")
    if args.base_model is None:
        if args.base_only:
            parser.error("--base-model is required for --base-only")
        import torch
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        base_model = Path(payload["metadata"]["base_model"]).resolve()
        del payload
    else:
        base_model = args.base_model.resolve()
    checkpoint = (base_model / "model.safetensors.index.json") if args.base_only else args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_stream = (output / "evaluation.lock").open("w")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = output / "status.json"
    state = json.loads(status_file.read_text()) if status_file.exists() else {"stages": {}}
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if state.get("checkpoint_sha256", digest) != digest:
        raise ValueError("Checkpoint changed; use a new output directory")
    mode = "base_only" if args.base_only else "base_plus_refiner"
    if state.get("base_model", str(base_model)) != str(base_model) or state.get("inference_mode", mode) != mode:
        raise ValueError("Base model or inference mode changed; use a new output directory")
    state.update(checkpoint=str(checkpoint), checkpoint_sha256=digest, driver_pid=os.getpid(),
                 num_gpus=args.num_gpus, num_per_gpu=args.num_per_gpu,
                 base_model=str(base_model), inference_mode=mode)
    active = None

    def save(stage, status, **details):
        state.update(stage=stage, status=status, updated_at=datetime.now(timezone.utc).isoformat(), **details)
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        os.replace(temporary, status_file)
        print(state["updated_at"], stage, status, details.get("reason", ""), flush=True)

    def stop(sig, _frame):
        if active is not None:
            active.send_signal(signal.SIGTERM)
            try:
                active.wait(timeout=45)
            except subprocess.TimeoutExpired:
                active.kill()
        save(state.get("stage", "driver"), "interrupted", reason=f"signal {sig}")
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    env.update(QWEN3VL_PATH="/root/autodl-tmp/models/Qwen3-VL-4B-Instruct",
               PYTHONNOUSERSITE="1",
               XDG_CACHE_HOME="/root/autodl-tmp/cache", XDG_RUNTIME_DIR="/root/autodl-tmp/run")
    if args.base_only:
        env.pop("REFINER_CHECKPOINT", None)
    else:
        env["REFINER_CHECKPOINT"] = str(checkpoint)
    Path(env["XDG_RUNTIME_DIR"]).mkdir(exist_ok=True, mode=0o700)
    launcher = output / f"launcher_snapshot_{time.time_ns()}.sh"
    launcher_text = (ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh").read_text()
    launcher.write_text(launcher_text)
    task_match = re.search(r"task_list_all=\((.*?)\)", launcher_text, re.S)
    if task_match is None:
        raise ValueError("Could not read the launcher's task list")
    task_names = shlex.split(task_match.group(1))
    stages = [("smoke", "demo_clean", "seen", 1, 1, 1),
              ("clean", "demo_clean", "seen", 50, 10, args.num_gpus),
              ("randomized", "demo_randomized", "unseen", 50, 10, args.num_gpus)]
    stages = [stage for stage in stages if stage[0] == "smoke" or args.settings == "both" or stage[0] == args.settings]
    try:
        for name, config, instructions, tasks, episodes, gpus in stages:
            if state["stages"].get(name, {}).get("status") == "complete":
                continue
            expected = task_names[:tasks]
            reused = completed_results(output / name, instructions, episodes)
            pending = [task for task in expected if task not in reused]
            if not pending:
                rates = {task: reused[task]["success_rate"] for task in expected}
                state["stages"][name] = dict(status="complete", tasks=tasks, expected_tasks=tasks,
                    episodes_per_task=episodes, success_rates=rates,
                    mean_success_rate=sum(rates.values()) / tasks, result_sources=reused)
                save(name, "complete", reason="All task results already exist")
                continue
            concurrency = 1 if name == "smoke" else args.num_per_gpu
            required_free = max(args.min_free_mib, 33000 * concurrency)
            last_free = None
            while True:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, check=True)
                free = [int(value) for value in result.stdout.splitlines()]
                if len(free) >= gpus and min(free[:gpus]) >= required_free:
                    break
                # State stays current while the log changes only when memory does.
                if last_free != free:
                    save(name, "waiting_for_gpus", reason=f"Need {required_free} MiB free on GPUs 0..{gpus-1}",
                         free_mib=free)
                    last_free = free
                time.sleep(30)
            log = output / f"{name}_launcher_{time.time_ns()}.log"
            inference_script = "deploy/lingbot_vla_v2_policy.py" if args.base_only else "deploy/lingbot_action_refiner_policy.py"
            command = ["bash", str(launcher), "--inference_script", inference_script,
                       "--model_path", str(base_model),
                       "--inference_workdir", str(ROOT) + "/", "--eval_workdir", "/root/autodl-tmp/projects/RoboTwin",
                       "--output_base", str(output / name), "--conda_sh", "/root/miniconda3/etc/profile.d/conda.sh",
                       "--inference_env", "/root/autodl-tmp/conda/envs/lingbotvla",
                       "--sim_env", "/root/autodl-tmp/conda/envs/RoboTwin", "--task_config", config,
                       "--instruction_type", instructions, "--num_tasks", str(len(pending)),
                       "--task_list", ",".join(pending), "--num_episodes", str(episodes),
                       "--num_gpus", str(gpus), "--num_per_gpu", str(concurrency), "--use_bf16", "False",
                       "--use_fp32", "True", "--use_compile", "True" if args.base_only else "False",
                       "--start_port", str(args.start_port), "--no_video"]
            save(name, "running", command=command, launcher_log=str(log),
                 reused_completed_tasks=len(expected) - len(pending), pending_tasks=pending,
                 reused_result_sources={task: reused[task] for task in expected if task in reused},
                 interrupted_partial_policy="Restart incomplete tasks; exclude their earlier partial logs from totals",
                 reason=f"{gpus} GPUs x {concurrency} slots; reuse {len(expected)-len(pending)} completed tasks")
            with log.open("w") as stream:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                save(name, "running", launcher_pid=active.pid)
                code = active.wait()
                active = None
            text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
            directories = re.findall(r"Run directory: (.+)", text)
            run = Path(directories[-1].strip()) if directories else None
            results = list(run.glob("eval_results/*/_result.txt")) if run else []
            failures, rates = {}, {}
            for result_file in results:
                contents = result_file.read_text()
                if f"Instruction Type: {instructions}" not in contents:
                    failures[result_file.parent.name] = "wrong instruction split"
                rates[result_file.parent.name] = float(contents.strip().splitlines()[-1])
            if run:
                for task_log in run.glob("eval_logs/*.log"):
                    contents = re.sub(r"\x1b\[[0-9;]*m", "", task_log.read_text(errors="replace"))
                    progress = re.findall(r"Success rate: (\d+)/(\d+) => ([\d.]+)%", contents)
                    if not progress or int(progress[-1][1]) != episodes:
                        failures[task_log.stem] = contents[-2000:]
            merged = completed_results(output / name, instructions, episodes)
            rates = {task: merged[task]["success_rate"] for task in expected if task in merged}
            success = code == 0 and len(rates) == tasks and len(results) == len(pending) and not failures
            detail = dict(status="complete" if success else "failed", exit_code=code, run_dir=str(run),
                          tasks=len(rates), expected_tasks=tasks, episodes_per_task=episodes,
                          success_rates=rates, failures=failures, result_sources=merged,
                          num_per_gpu=concurrency, newly_evaluated_tasks=len(results))
            if rates:
                detail["mean_success_rate"] = sum(rates.values()) / len(rates)
            state["stages"][name] = detail
            save(name, detail["status"], reason=f"{len(rates)}/{tasks} tasks completed")
            if not success:
                raise RuntimeError(f"{name} incomplete; inspect {log}")
        save("all", "complete", reason=f"Requested settings ({args.settings}): each 50 tasks x 10 episodes")
    except Exception as error:
        save(state.get("stage", "driver"), "failed", reason=repr(error))
        raise


if __name__ == "__main__":
    main()
