"""Cycles renderer for generate_v5.py -- runs INSIDE Blender (started by generate_v5.py):

    blender -b --factory-startup -P synth/blender_render_v5.py -- --work DIR --worker k --nworkers K

Same protocol, materials and output as blender_render.py (v4), but per scene:
  * the camera (intrinsics, mounting height) comes from the scene
  * the floor is a normal, rendered surface (not a shadow catcher) carrying the scene's procedural albedo map
    (image space, mapped through the camera onto the plane z = 0) and roughness: lamps, beams and objects give
    pools of light, shadows and glossy reflections on it
  * lights: rows of x, y, z, size_x, size_y, power, target x, y, z, spread; lamps are not seen by the camera
Output: 16-bit RGBA, linear radiance * 0.5, floor included (alpha ~1 inside the frame).
"""
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_render as B  # noqa: E402


def update_camera(ob, sc, fx, fy, cx, cy, W, H, mount_h):
    cam = ob.data
    ob.location = (0.0, 0.0, mount_h)
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.lens = fx * cam.sensor_width / W
    sc.render.pixel_aspect_x, sc.render.pixel_aspect_y = (fy / fx, 1.0) if fy >= fx else (1.0, fx / fy)
    cam.shift_x = (W / 2 - (cx + 0.5)) / max(W, H)
    cam.shift_y = ((cy + 0.5) - H / 2) / max(W, H)


class Renderer5(B.Renderer):
    def __init__(self, args, first):
        self.args = args
        fx, fy, cx, cy, W, H, mh = [float(v) for v in first["cam"]]
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        B.setup_render(args.device, int(W), int(H))
        B.setup_camera(fx, fy, cx, cy, int(W), int(H), mh)
        self.cam_ob = bpy.context.scene.camera
        R = 60.0
        self.floor_xy = np.array([[-R, -R], [R, -R], [R, R], [-R, R]])
        me = bpy.data.meshes.new("floor")
        me.from_pydata([(x, y, 0.0) for x, y in self.floor_xy], [], [(0, 1, 2, 3)])
        me.uv_layers.new()
        ob = bpy.data.objects.new("floor", me)
        bpy.context.scene.collection.objects.link(ob)
        mat = B.new_material("floor")
        nt = mat.node_tree
        self.floor_bsdf = nt.nodes["Principled BSDF"]
        self.floor_tex = B.node(nt, "ShaderNodeTexImage", (-400, 0))
        self.floor_tex.extension = "EXTEND"
        nt.links.new(self.floor_tex.outputs["Color"], self.floor_bsdf.inputs["Base Color"])
        me.materials.append(mat)
        self.floor_me, self.floor_img = me, None
        self.mats = {"solid": B.solid_material(), "film": B.film_material(), "mesh": B.mesh_material()}
        world = bpy.data.worlds.new("world")
        world.use_nodes = True
        bpy.context.scene.world = world
        self.world_bg = world.node_tree.nodes["Background"]
        self.world_bg.inputs["Color"].default_value = (1, 1, 1, 1)
        self.objects, self.images, self.lights = [], [], []

    def set_floor(self, s, cam):
        fx, fy, cx, cy, W, H, mh = cam
        xy = self.floor_xy
        u = (fx * xy[:, 0] / mh + cx + 0.5) / W
        v = 1.0 - (fy * -xy[:, 1] / mh + cy + 0.5) / H  # Blender y = -camera y
        self.floor_me.uv_layers[0].data.foreach_set("uv", np.c_[u, v].astype(np.float32).ravel())
        if self.floor_img is not None:
            bpy.data.images.remove(self.floor_img)
        # grey albedo into R, G and B: a 1-channel array only fills R (G = B = 1), which turned the floor colour into
        # (a, 1, 1) -- 2/3 of its brightness independent of the painted albedo (renders before this fix)
        self.floor_img = B.image_from_array("floor", np.repeat(s["floor"].astype(np.float32)[..., None], 3, -1),
                                            is_float=True)
        self.floor_tex.image = self.floor_img
        self.floor_bsdf.inputs["Roughness"].default_value = float(s["floor_rough"])

    def add_light(self, x, y, z, sx, sy, power, tx, ty, tz, spread=np.pi):
        L = bpy.data.lights.new("lamp", "AREA")
        L.shape, L.size, L.size_y, L.energy = "RECTANGLE", float(sx), float(sy), float(power)
        L.spread = float(min(max(spread, 0.01), np.pi))
        ob = bpy.data.objects.new("lamp", L)
        ob.location = (x, y, z)
        ob.rotation_euler = Vector((tx - x, ty - y, tz - z)).to_track_quat("-Z", "Y").to_euler()
        ob.visible_camera = False
        bpy.context.scene.collection.objects.link(ob)
        self.objects.append(ob)

    def build(self, s):
        self.clear()
        sc = bpy.context.scene
        cam = [float(v) for v in s["cam"]]
        update_camera(self.cam_ob, sc, *cam)
        self.set_floor(s, cam)
        sc.cycles.samples = int(s["samples"])
        sc.cycles.seed = int(s["seed"]) % (2 ** 31)
        self.world_bg.inputs["Strength"].default_value = float(s["world"])
        for row in s["lights"]:
            self.add_light(*row)
        for g in ("solid", "film", "mesh"):
            if f"{g}_v" not in s or len(s[f"{g}_fs"]) == 0:
                continue
            mat, extra = self.mats[g]
            img = B.image_from_array(f"{g}_atlas", s[f"{g}_atlas"].astype(np.float32) / 255.0)
            self.images.append(img)
            tex = extra["tex"] if isinstance(extra, dict) else extra
            tex.image = img
            if g == "film":
                fp = s["film_params"]
                extra["noise"].inputs["Scale"].default_value = float(fp[0])
                extra["bump"].inputs["Strength"].default_value = float(fp[1])
                extra["gloss"].inputs["Roughness"].default_value = float(fp[2])
                extra["gloss_amt"].inputs[1].default_value = float(fp[3])
                extra["wave"].inputs["Scale"].default_value = float(fp[4])
                extra["fold"].inputs["Radius"].default_value = float(fp[5])
                extra["patch"].inputs["From Min"].default_value = float(fp[6])
                extra["patch"].inputs["From Max"].default_value = float(fp[6]) + 0.12
            self.add_mesh(g, s[f"{g}_v"], s[f"{g}_fs"], s[f"{g}_uv"], s[f"{g}_n"], mat)


def main():
    args = B.parse()
    todo = [int(l) for l in open(os.path.join(args.work, "todo.txt")) if l.strip()]
    mine = todo[args.worker::args.nworkers]
    parent = os.getppid()
    R = None
    t0, done = time.time(), 0
    for i in mine:
        npz = os.path.join(args.work, f"{i:08d}.npz")
        skip = os.path.join(args.work, f"{i:08d}.skip")
        out = os.path.join(args.work, f"{i:08d}.render.png")
        while not (os.path.exists(npz) or os.path.exists(skip)):
            if os.path.exists(os.path.join(args.work, "STOP")) or os.getppid() != parent:
                return
            time.sleep(0.05)
        if not os.path.exists(npz):
            continue
        s = dict(np.load(npz))
        if R is None:
            R = Renderer5(args, s)
        R.build(s)
        R.render(out)
        os.remove(npz)
        done += 1
        if done % 10 == 0:
            print(f"[blender {args.worker}] {done}/{len(mine)}  {(time.time() - t0) / done:.2f} s/img", flush=True)


if __name__ == "__main__":
    main()
