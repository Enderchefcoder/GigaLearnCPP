"""Generates a synthetic arena collision mesh for testing GigaLearnCPP without game files.

Real training should use meshes dumped from Rocket League with RLArenaCollisionDumper --
this tool builds an approximate arena (flat box with rounded edges and functional goals)
so you can smoke-test the framework end to end before setting that up. Bots trained on it
will not transfer well to the real arena (no ramps or curved corners).

Usage:
	python tools/make_test_mesh.py [output_folder]

Writes <output_folder>/soccar/mesh_0.cmf (default output folder: "collision_meshes").
"""

import math
import os
import struct
import sys

# Arena dimensions (slightly inset from the real bounds so the mesh is never
# coplanar with RocketSim's built-in boundary planes)
INSET = 16
X = 4096 - INSET     # Side wall distance
Y = 5120 - INSET     # Back wall distance (the goal line)
Z = 2044 - INSET     # Ceiling height
GOAL_HALF_WIDTH = 893
GOAL_HEIGHT = 643
GOAL_BACK_Y = 5900   # Back of the goal box (real back net is at 6000)

TRI_SIZE = 256       # Tessellation resolution
ROUNDING = 300       # Edge rounding radius (sharp seams destabilize the physics)

verts = []
tris = []


def round_to_box(p):
	"""Projects a point of the sharp main box onto the rounded main box.
	Points inside the goal boxes (|y| > Y) are left alone."""
	x, y, z = p
	if abs(y) > Y - 1:
		return p

	cx = min(max(x, -X + ROUNDING), X - ROUNDING)
	cy = min(max(y, -Y + ROUNDING), Y - ROUNDING)
	cz = min(max(z, ROUNDING), Z - ROUNDING)

	dx, dy, dz = x - cx, y - cy, z - cz
	dist = math.sqrt(dx * dx + dy * dy + dz * dz)
	if dist < 1e-6:
		return p

	s = ROUNDING / dist
	return (cx + dx * s, cy + dy * s, cz + dz * s)


def add_face(origin, edge_u, edge_v, apply_rounding=True):
	"""Adds a tessellated quad defined by an origin corner and two edge vectors."""
	len_u = math.sqrt(sum(c * c for c in edge_u))
	len_v = math.sqrt(sum(c * c for c in edge_v))
	divs_u = max(1, math.ceil(len_u / TRI_SIZE))
	divs_v = max(1, math.ceil(len_v / TRI_SIZE))

	base = len(verts)
	for u in range(divs_u + 1):
		for v in range(divs_v + 1):
			fu, fv = u / divs_u, v / divs_v
			p = tuple(origin[i] + edge_u[i] * fu + edge_v[i] * fv for i in range(3))
			if apply_rounding:
				p = round_to_box(p)
			verts.append(p)

	for u in range(divs_u):
		for v in range(divs_v):
			i00 = base + u * (divs_v + 1) + v
			i01 = i00 + 1
			i10 = i00 + (divs_v + 1)
			i11 = i10 + 1
			tris.append((i00, i10, i11))
			tris.append((i00, i11, i01))


def add_wall_with_goal(side):
	"""Adds a +Y or -Y wall with a goal-sized opening, plus the goal box behind it."""
	y = Y * side
	gy = GOAL_BACK_Y * side

	# Wall sections around the goal opening: left of goal, right of goal, above goal
	add_face((-X, y, 0), (X - GOAL_HALF_WIDTH, 0, 0), (0, 0, Z))
	add_face((GOAL_HALF_WIDTH, y, 0), (X - GOAL_HALF_WIDTH, 0, 0), (0, 0, Z))
	add_face((-GOAL_HALF_WIDTH, y, GOAL_HEIGHT), (2 * GOAL_HALF_WIDTH, 0, 0), (0, 0, Z - GOAL_HEIGHT))

	# Goal box: floor, back wall, two side walls, top (no rounding inside the goal)
	depth = gy - y
	add_face((-GOAL_HALF_WIDTH, y, 0), (2 * GOAL_HALF_WIDTH, 0, 0), (0, depth, 0), False)
	add_face((-GOAL_HALF_WIDTH, gy, 0), (2 * GOAL_HALF_WIDTH, 0, 0), (0, 0, GOAL_HEIGHT), False)
	add_face((-GOAL_HALF_WIDTH, y, 0), (0, depth, 0), (0, 0, GOAL_HEIGHT), False)
	add_face((GOAL_HALF_WIDTH, y, 0), (0, depth, 0), (0, 0, GOAL_HEIGHT), False)
	add_face((-GOAL_HALF_WIDTH, y, GOAL_HEIGHT), (2 * GOAL_HALF_WIDTH, 0, 0), (0, depth, 0), False)


def main():
	out_base = sys.argv[1] if len(sys.argv) > 1 else "collision_meshes"

	# Main box
	add_face((-X, -Y, 0), (2 * X, 0, 0), (0, 2 * Y, 0))          # Floor
	add_face((-X, -Y, Z), (2 * X, 0, 0), (0, 2 * Y, 0))          # Ceiling
	add_face((-X, -Y, 0), (0, 2 * Y, 0), (0, 0, Z))              # -X wall
	add_face((X, -Y, 0), (0, 2 * Y, 0), (0, 0, Z))               # +X wall
	add_wall_with_goal(+1)                                        # +Y wall & orange goal
	add_wall_with_goal(-1)                                        # -Y wall & blue goal

	out_dir = os.path.join(out_base, "soccar")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, "mesh_0.cmf")

	with open(out_path, "wb") as f:
		f.write(struct.pack("<ii", len(tris), len(verts)))
		for t in tris:
			f.write(struct.pack("<iii", *t))
		for v in verts:
			f.write(struct.pack("<fff", *v))

	print(f"Wrote {out_path}: {len(tris)} tris, {len(verts)} verts")
	print("NOTE: This is an approximate test arena (no ramps/curves). "
		  "Use RLArenaCollisionDumper meshes for real training.")


if __name__ == "__main__":
	main()
