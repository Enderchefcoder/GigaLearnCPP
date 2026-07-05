"""Converts checkpoints between GigaLearnCPP and Python rlgym-ppo.

Usage:
	python checkpoint_converter.py to_cpp <rlgym-ppo checkpoint folder>
	python checkpoint_converter.py to_python <GigaLearnCPP checkpoint folder>

Notes:
- GigaLearnCPP model files are POLICY.lt / CRITIC.lt (+ SHARED_HEAD.lt if shared
  layers are enabled). rlgym-ppo files are PPO_POLICY.pt / PPO_VALUE_NET.pt.
- rlgym-ppo has no shared-head equivalent, so checkpoints that use shared layers
  cannot be converted to Python (train with sharedHead.layerSizes = {} if you
  need Python interop).
- Optimizer state is not converted (optimizers reset; learning takes a few
  iterations to recover).
- Model architectures (layer sizes, obs size, action count) must match the
  target framework's configuration exactly.
"""

import sys
import os
from collections import OrderedDict

import torch

from rlgym_ppo.ppo import DiscreteFF, ValueEstimator

# GigaLearnCPP model file names (see Model::GetSavePath in the C++ source)
GGL_POLICY = "POLICY.lt"
GGL_CRITIC = "CRITIC.lt"
GGL_SHARED_HEAD = "SHARED_HEAD.lt"

# rlgym-ppo model file names
PY_POLICY = "PPO_POLICY.pt"
PY_CRITIC = "PPO_VALUE_NET.pt"


def model_info_from_dict(loaded_dict):
	state_dict = OrderedDict(loaded_dict)

	bias_counts = []
	weight_counts = []
	for key, value in state_dict.items():
		if ".weight" in key:
			weight_counts.append(value.numel())
		if ".bias" in key:
			bias_counts.append(value.size(0))

	inputs = int(weight_counts[0] / bias_counts[0])
	outputs = bias_counts[-1]
	layer_sizes = bias_counts[:-1]

	return inputs, outputs, layer_sizes


def rename_model_state_dict(state_dict):
	keys = list(state_dict.keys())

	for key in keys:
		if "model." not in key:
			state_dict["model." + key] = state_dict[key]
			del state_dict[key]

	return state_dict


def make_models_from_dicts(policy_state_dict, critic_state_dict):
	policy_inputs, policy_outputs, policy_sizes = model_info_from_dict(policy_state_dict)
	critic_inputs, critic_outputs, critic_sizes = model_info_from_dict(critic_state_dict)

	device = torch.device("cpu")
	policy = DiscreteFF(policy_inputs, policy_outputs, policy_sizes, device)
	critic = ValueEstimator(critic_inputs, critic_sizes, device)
	return policy, critic


def main():
	if len(sys.argv) != 3:
		sys.exit("Invalid argument count, arguments should be \"<to_cpp/to_python> <checkpoint path>\"")

	to_arg = sys.argv[1]
	if to_arg == 'to_cpp':
		to_cpp = True
	elif to_arg == 'to_python':
		to_cpp = False
	else:
		sys.exit("Invalid arguments, please specify \"to_cpp\" or \"to_python\" for the first argument.")

	path = sys.argv[2]

	if to_cpp:
		print("Loading state dicts...")
		policy_state_dict = torch.load(os.path.join(path, PY_POLICY))
		critic_state_dict = torch.load(os.path.join(path, PY_CRITIC))

		print("Creating models...")
		policy, critic = make_models_from_dicts(policy_state_dict, critic_state_dict)

		print("Applying state dicts...")
		policy.load_state_dict(policy_state_dict)
		critic.load_state_dict(critic_state_dict)

		print("Saving for GigaLearnCPP...")
		policy_ts = torch.jit.script(policy.model)
		critic_ts = torch.jit.script(critic.model)

		output_path = "cpp_checkpoint"
		os.makedirs(output_path, exist_ok=True)
		torch.jit.save(policy_ts, os.path.join(output_path, GGL_POLICY))
		torch.jit.save(critic_ts, os.path.join(output_path, GGL_CRITIC))

		print("NOTE: The GigaLearnCPP learner must be configured with no shared head")
		print("      (cfg.ppo.sharedHead.layerSizes = {}) to match this checkpoint.")
		print("NOTE: rlgym-ppo models have no layer norm, so the learner must also be")
		print("      configured with addLayerNorm = false on the policy and critic.")

	else:
		if os.path.exists(os.path.join(path, GGL_SHARED_HEAD)):
			sys.exit(
				"This GigaLearnCPP checkpoint uses shared layers (SHARED_HEAD.lt), which rlgym-ppo "
				"has no equivalent for, so it cannot be converted.\n"
				"To produce Python-convertible checkpoints, train with cfg.ppo.sharedHead.layerSizes = {}."
			)

		print("Loading models...")
		policy = torch.jit.load(os.path.join(path, GGL_POLICY))
		critic = torch.jit.load(os.path.join(path, GGL_CRITIC))
		policy_inputs, policy_outputs, policy_sizes = model_info_from_dict(policy.state_dict())
		print("Policy sizes:", policy_inputs, policy_sizes, policy_outputs)

		print("Creating optimizers...")
		policy_py, critic_py = make_models_from_dicts(policy.state_dict(), critic.state_dict())

		policy_optim = torch.optim.Adam(policy_py.parameters())
		critic_optim = torch.optim.Adam(critic_py.parameters())

		print("Saving for rlgym-ppo...")
		output_path = "python_checkpoint"
		os.makedirs(output_path, exist_ok=True)

		policy_state_dict = rename_model_state_dict(policy.state_dict())
		critic_state_dict = rename_model_state_dict(critic.state_dict())

		torch.save(policy_state_dict, os.path.join(output_path, PY_POLICY))
		torch.save(critic_state_dict, os.path.join(output_path, PY_CRITIC))
		torch.save(policy_optim.state_dict(), os.path.join(output_path, "PPO_POLICY_OPTIMIZER.pt"))
		torch.save(critic_optim.state_dict(), os.path.join(output_path, "PPO_VALUE_NET_OPTIMIZER.pt"))

	print(
		"Done! Partial " + ("GigaLearnCPP" if to_cpp else "rlgym-ppo") +
		" checkpoint generated at \"" + output_path + "\"."
	)
	print(
		"NOTE: Optimizer conversion is not supported, so optimizers will be reset. " +
		"It may take a few iterations for learning to recover."
	)
	print("NOTE: State JSON not included (just make a new one and copy over the vars you want).")
	print("NOTE: Make sure the obs/actions/model sizes all match.")


if __name__ == '__main__':
	main()
