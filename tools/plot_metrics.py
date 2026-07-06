"""Plots training metrics from GigaLearnCPP's local JSONL metric logs.

When wandb isn't installed, the metric receiver writes metrics to
"metrics/<run-id>.jsonl" next to your executable. This script visualizes them.

Usage:
	python tools/plot_metrics.py metrics/my-run.jsonl
	python tools/plot_metrics.py metrics/my-run.jsonl --keys "Average Step Reward" "Policy Entropy"
	python tools/plot_metrics.py metrics/*.jsonl --list     # Show available metric keys
	python tools/plot_metrics.py metrics/*.jsonl --out plots.png

Requires: matplotlib (pip install matplotlib)
"""

import argparse
import json
import sys

# The default dashboard: the metrics most worth watching (see docs/TRAINING_GUIDE.md)
DEFAULT_KEYS = [
	"Average Step Reward",
	"Policy Entropy",
	"Mean KL Divergence",
	"Critic Loss",
	"Overall Steps/Second",
	"Episode Length",
]


def load_runs(paths):
	"""Loads one or more JSONL files into (name, list-of-dicts) pairs."""
	runs = []
	for path in paths:
		rows = []
		with open(path) as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					rows.append(json.loads(line))
				except json.JSONDecodeError:
					pass  # Partial last line from a live run
		if rows:
			runs.append((path, rows))
		else:
			print(f"Warning: no metrics found in {path}")
	return runs


def main():
	parser = argparse.ArgumentParser(description="Plot GigaLearnCPP JSONL metric logs")
	parser.add_argument("files", nargs="+", help="JSONL metric file(s)")
	parser.add_argument("--keys", nargs="+", default=None, help="Metric keys to plot")
	parser.add_argument("--list", action="store_true", help="List available metric keys and exit")
	parser.add_argument("--out", default=None, help="Save the figure to a file instead of showing it")
	parser.add_argument("--x", default="Total Timesteps", help="X-axis metric (default: Total Timesteps)")
	args = parser.parse_args()

	runs = load_runs(args.files)
	if not runs:
		sys.exit("No metrics to plot.")

	if args.list:
		keys = set()
		for _, rows in runs:
			for row in rows:
				keys.update(row.keys())
		print("Available metric keys:")
		for key in sorted(keys):
			print(f"  {key}")
		return

	try:
		import matplotlib
		if args.out:
			matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		sys.exit("matplotlib is required: pip install matplotlib")

	keys = args.keys if args.keys else [
		k for k in DEFAULT_KEYS if any(k in row for _, rows in runs for row in rows)
	]
	if not keys:
		sys.exit("None of the default metrics found; use --list to see available keys.")

	cols = 2
	rowsN = (len(keys) + cols - 1) // cols
	fig, axes = plt.subplots(rowsN, cols, figsize=(7 * cols, 3.2 * rowsN), squeeze=False)
	fig.suptitle("GigaLearnCPP Training Metrics")

	for i, key in enumerate(keys):
		ax = axes[i // cols][i % cols]
		for name, rows in runs:
			xs, ys = [], []
			for row in rows:
				if key in row:
					xs.append(row.get(args.x, len(xs)))
					ys.append(row[key])
			if ys:
				ax.plot(xs, ys, label=name if len(runs) > 1 else None, linewidth=0.9)
		ax.set_title(key)
		ax.set_xlabel(args.x)
		ax.grid(True, alpha=0.3)
		if len(runs) > 1:
			ax.legend(fontsize=7)

	# Hide any unused subplots
	for i in range(len(keys), rowsN * cols):
		axes[i // cols][i % cols].axis("off")

	fig.tight_layout(rect=[0, 0, 1, 0.97])

	if args.out:
		fig.savefig(args.out, dpi=120)
		print(f"Saved to {args.out}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
