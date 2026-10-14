"""Blender QuadriFlow baseline, run headless through the `bpy` module (Blender 5.0).

usage: python blender_quadriflow.py INPUT.obj OUTPUT.obj TARGET_FACES

Imports the mesh, merges coincident vertices (what a Blender user does before remeshing;
QuadriFlow refuses non-manifold input otherwise), runs Object > Remesh > QuadriFlow with
sharp-edge and boundary preservation, and exports quads.
"""
import sys

import bpy


def main():
    src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=src)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-7)
    bpy.ops.object.mode_set(mode="OBJECT")
    res = bpy.ops.object.quadriflow_remesh(
        target_faces=target, use_preserve_sharp=True, use_preserve_boundary=True,
        use_mesh_symmetry=False, preserve_attributes=False, smooth_normals=False,
        mode="FACES", seed=0)
    if "FINISHED" not in res:
        print("QUADRIFLOW_FAILED", res)
        sys.exit(3)
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=dst, export_selected_objects=True, export_uv=False,
                          export_normals=False, export_materials=False, forward_axis="NEGATIVE_Z",
                          up_axis="Y")
    print("faces", len(obj.data.polygons))


if __name__ == "__main__":
    main()
