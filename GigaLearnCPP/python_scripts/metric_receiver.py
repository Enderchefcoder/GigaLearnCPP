"""Receives training metrics from GigaLearnCPP.

By default, metrics are logged to Weights & Biases (wandb).
If wandb is not installed (or fails to initialize), metrics fall back to a local
JSONL file inside "metrics/" so training can continue without remote logging.

You can freely modify this file to send metrics wherever you want.
"""

import site
import sys
import json
import os
import time

wandb_run = None
fallback_file = None


def _try_import_wandb(py_exec_path):
	# Fix the path of our interpreter so wandb doesn't run our executable instead of Python
	# (Embedded Python reports the host executable as sys.executable)
	sys.executable = py_exec_path

	try:
		# Windows embedded interpreters sometimes miss the site-packages dir of the
		# interpreter they were built from, so add it manually if it exists
		site_packages_dir = os.path.join(os.path.dirname(py_exec_path), "Lib", "site-packages")
		if os.path.isdir(site_packages_dir):
			sys.path.append(site_packages_dir)
			site.addsitedir(site_packages_dir)

		import wandb
		return wandb
	except Exception as e:
		print(f"[metric_receiver] Failed to import wandb: {repr(e)}")
		return None


def _init_fallback(project, group, name):
	global fallback_file

	metrics_dir = "metrics"
	os.makedirs(metrics_dir, exist_ok=True)

	run_id = f"{name}-{int(time.time())}".replace(os.sep, "_")
	path = os.path.join(metrics_dir, f"{run_id}.jsonl")
	fallback_file = open(path, "a", buffering=1)  # Line-buffered

	print(f"[metric_receiver] Logging metrics locally to \"{path}\"")
	print("[metric_receiver] To use wandb instead, install it with: pip install wandb")
	return run_id


def init(py_exec_path, project, group, name, id=None):
	"""Initializes metric logging and returns the run ID (new or resumed)."""
	global wandb_run

	wandb = _try_import_wandb(py_exec_path)
	if wandb is None:
		return _init_fallback(project, group, name)

	try:
		print("[metric_receiver] Calling wandb.init()...")
		if id is not None and len(id) > 0:
			wandb_run = wandb.init(project=project, group=group, name=name, id=id, resume="allow")
		else:
			wandb_run = wandb.init(project=project, group=group, name=name)
		return wandb_run.id
	except Exception as e:
		print(f"[metric_receiver] wandb.init() failed: {repr(e)}")
		return _init_fallback(project, group, name)


def add_metrics(metrics):
	if wandb_run is not None:
		# wandb.log() can raise on network/service errors; don't let that kill training
		try:
			wandb_run.log(metrics)
		except Exception as e:
			print(f"[metric_receiver] wandb.log() failed (training continues): {repr(e)}")
	elif fallback_file is not None:
		fallback_file.write(json.dumps(metrics) + "\n")
