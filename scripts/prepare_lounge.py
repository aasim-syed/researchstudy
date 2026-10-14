#!/usr/bin/env python3
"""Room-scale benchmark scan: Open3D LoungeRGBDImages ships `lounge.ply`, the
VCG/TSDF reconstruction of a 3000-frame RGB-D sweep of a lounge (3.15M faces).
It is decimated with quadric edge collapse to ~1M faces so it falls inside the
10K-2M input range, keeping the scan's noise, holes and fragments.

usage: python scripts/prepare_lounge.py <lounge.ply> data/models/lounge_room.obj
"""
import sys

import pymeshlab

src, dst = sys.argv[1], sys.argv[2]
ms = pymeshlab.MeshSet()
ms.load_new_mesh(src)
print("input faces", ms.current_mesh().face_number())
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=1_000_000, preservetopology=False,
                                            qualitythr=0.3, planarquadric=True)
print("output faces", ms.current_mesh().face_number())
ms.save_current_mesh(dst, save_vertex_color=False, save_vertex_normal=False)
