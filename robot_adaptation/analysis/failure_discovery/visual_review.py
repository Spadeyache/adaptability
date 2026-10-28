"""Evidence-preserving visual review of recorded episodes; never reruns physics.

Render a complete replay and event-focused storyboard for every indexed episode.
Annotations are supplied separately by a reviewer, never inferred from the old
failure labels or the simulated physical properties. Original data stay intact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import multiprocessing
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_adaptation.paths import ROOT


TAXONOMY = {
    "slip_or_drop": "Object escapes/downward relative motion after a visible lift or grip; speed/friction/mass cause unconfirmed.",
    "abrupt_ejection": "Object abruptly departs from the fingers or is knocked/thrown away during approach/lift; cause unconfirmed.",
    "not_secured": "Fingers miss, push, or fail to retain the object; no clear sustained object lift.",
    "held_but_task_incomplete": "Object remains with the gripper, but height, target or terminal stability is insufficient.",
    "success": "Recorded final task success; additional visual tags can still describe transient difficulties.",
    "uncertain": "Available views do not support a reliable visual type, or different behaviors overlap.",
}
TAGS = ["off_center_contact", "rotation_or_tipping", "repeated_attempts", "transient_slip", "object_pushed_on_table", "view_occluded", "mixed_behavior"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def navigation_indices(d):
    """Oracle contact/grasp flags choose frames only; they never assign a class."""
    count = len(d["meta__state_time_s"])
    obj = d["input__object_pose_world_m_wxyz"][:, :3]
    tcp = d["input__tcp_pose_world_m_wxyz"][:, :3]
    grasp = d["sim__is_grasped"].astype(bool)
    contact = d["sim__object_contact"].astype(bool)
    first_contact = int(np.flatnonzero(contact)[0]) if contact.any() else int(np.linalg.norm(obj - tcp, axis=1).argmin())
    first_grasp = int(np.flatnonzero(grasp)[0]) if grasp.any() else first_contact
    # First release after a grasp while object has risen 1 cm; no release cause is inferred.
    release = np.flatnonzero(grasp[:-1] & ~grasp[1:] & (obj[:-1, 2] > obj[0, 2] + .01)) + 1
    if len(release):
        pivot = int(release[0])
    else:
        speed = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        pivot = int(speed.argmax() + 1)
    indices = sorted({first_contact, first_grasp, max(0, pivot - 1), min(count - 1, pivot + 2), int(obj[:, 2].argmax()), count - 1})
    while len(indices) < min(6, count):
        boundaries = sorted({0, count - 1, *indices})
        left, right = max(zip(boundaries[:-1], boundaries[1:]), key=lambda ab: ab[1] - ab[0])
        midpoint = (left + right) // 2
        if midpoint in indices:
            midpoint = next(i for i in range(count) if i not in indices)
        indices = sorted({*indices, midpoint})
    return indices


def trace_panel(d, size=(420, 168)):
    """Full-duration pose trajectories accompany the sampled rendered images."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    t = d["meta__state_time_s"]
    o = d["input__object_pose_world_m_wxyz"][:, :3]
    g = d["input__tcp_pose_world_m_wxyz"][:, :3]
    fig = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax, top = fig.subplots(1, 2)
    ax.plot(t, o[:, 2], color="#d32f2f", lw=1.3, label="object")
    ax.plot(t, g[:, 2], color="#1565c0", lw=1.1, label="gripper")
    ax.axhline(o[0, 2], color="#999", lw=.5)
    ax.set(title="height (m)", xlabel="seconds")
    top.plot(o[:, 0], o[:, 1], color="#d32f2f", lw=1.1)
    top.plot(g[:, 0], g[:, 1], color="#1565c0", lw=.8, alpha=.8)
    top.scatter(o[0, 0], o[0, 1], color="#d32f2f", s=9)
    top.scatter(o[-1, 0], o[-1, 1], color="#111", marker="x", s=13)
    top.set(title="top view XY (m)", aspect="equal")
    for a in (ax, top):
        a.tick_params(labelsize=6)
        a.title.set_fontsize(8)
        a.xaxis.label.set_fontsize(7)
        a.grid(alpha=.2)
    fig.tight_layout(pad=.5)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())


def render_batch(args):
    """One worker owns its renderer; fixed poses only, no environment steps."""
    dataset, output, rows, stride = args
    import torch
    import imageio.v2 as imageio
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.structs.pose import Pose
    from robot_adaptation.simulation.environment import CollectionEnv
    torch.set_num_threads(1)
    font = ImageFont.truetype(FONT, 14)
    small = ImageFont.truetype(FONT, 12)
    env = None
    geometry = None
    completed = []
    try:
        for row in rows:
            p = Path(dataset) / row["file"]
            ident = row["review_id"]
            base = Path(output) / "episodes" / ident
            base.mkdir(parents=True, exist_ok=True)
            if (base / "evidence.json").exists():
                completed.append(ident)
                continue
            with np.load(p, allow_pickle=False) as source:
                d = {k: source[k].copy() for k in [
                    "meta__state_time_s", "input__object_pose_world_m_wxyz", "input__tcp_pose_world_m_wxyz",
                    "input__joint_position_rad_m", "input__goal_position_world_m", "sim__is_grasped", "sim__object_contact"]}
                condition = json.loads(str(source["label__condition_json"]))
            props = condition["properties"]
            key = (props["shape"], tuple(props["size_xyz_m"]))
            if key != geometry:
                if env is not None:
                    env.close()
                env = CollectionEnv(props, human_render_camera_configs={
                    "shader_pack": "minimal", "width": 640, "height": 480,
                    "pose": sapien_utils.look_at(eye=[.38, .44, .36], target=[0, 0, .12]), "fov": .85})
                env.reset(seed=int(row["seed"]), options={"reconfigure": True})
                geometry = key
            env.goal_site.set_pose(Pose.create_from_pq(d["input__goal_position_world_m"], device=env.device))
            obj = d["input__object_pose_world_m_wxyz"]
            q = d["input__joint_position_rad_m"]
            times = d["meta__state_time_s"]
            indices = navigation_indices(d)
            # A constant per-episode view frames the recorded object trajectory,
            # avoiding tracking-camera motion being confused with object motion.
            xyz = obj[:, :3]
            target = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
            target[2] = max(.10, target[2])
            radius = max(.40, float(np.ptp(xyz, axis=0).max()) * 1.65)
            eye = target + np.array([.62, .72, .55]) * radius
            camera = next(iter(env._human_render_cameras.values())).camera
            camera.set_local_pose(sapien_utils.look_at(eye=eye, target=target).sp)
            selected = set(range(0, len(times), stride)) | set(indices) | {len(times) - 1}
            snapshots = {}
            video = imageio.get_writer(str(base / "replay.mp4"), fps=round(1 / (float(times[1] - times[0]) * stride)),
                                      codec="libx264", quality=6, macro_block_size=16, ffmpeg_params=["-threads", "1"])
            try:
                for i in sorted(selected):
                    env.agent.robot.set_qpos(torch.as_tensor(q[i:i + 1], device=env.device))
                    env.cube.set_pose(Pose.create(torch.as_tensor(obj[i:i + 1], device=env.device)))
                    frame = Image.fromarray(env.render_rgb_array()[0].cpu().numpy().astype(np.uint8))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 26), fill="#111827")
                    draw.text((8, 5), f"{ident}  |  recorded poses  |  t={times[i]:.2f}s", font=font, fill="white")
                    if i in indices:
                        snapshots[i] = frame.copy()
                        frame.save(base / f"frame_{i:03d}.jpg", quality=90)
                    if i % stride == 0 or i == len(times) - 1:
                        video.append_data(np.asarray(frame))
            finally:
                video.close()
            trace = trace_panel(d)
            trace.save(base / "traces.png")
            # Full-size storyboard keeps details available; sheet is an overview.
            storyboard = Image.new("RGB", (3 * 640, 2 * 506 + 168), "white")
            thumb = Image.new("RGB", (6 * 216 + 420, 192), "white")
            td = ImageDraw.Draw(thumb)
            for slot, i in enumerate(indices):
                x, y = (slot % 3) * 640, (slot // 3) * 506
                storyboard.paste(snapshots[i], (x, y))
                ImageDraw.Draw(storyboard).text((x + 8, y + 483), f"state {i}, t={times[i]:.2f}s", font=font, fill="black")
                thumb.paste(snapshots[i].resize((216, 162)), (slot * 216, 20))
                td.text((slot * 216 + 5, 182), f"{times[i]:.2f}s", font=small, fill="#333")
            storyboard.paste(trace, (0, 1012))
            thumb.paste(trace, (1296, 20))
            td.text((5, 2), f"{ident}   {row['episode_id']}   recorded outcome: {'success' if row['success'] else 'failure'}", font=font, fill="#111")
            storyboard.save(base / "storyboard.jpg", quality=90)
            thumb.save(base / "overview.jpg", quality=90)
            dump(base / "evidence.json", {
                "review_id": ident, "source_file": str(p.relative_to(ROOT)), "source_sha256": sha256(p),
                "frame_indices": indices, "frame_times_s": [float(times[i]) for i in indices],
                "replay_stride": stride, "recorded_state_hz": float(1 / (times[1] - times[0])),
                "camera_target": target.tolist(), "camera_eye": eye.tolist(),
                "kind": "rendered original object and robot joint poses; no new simulation, interpolation or policy rollout",
                "frame_selection": "recorded contact/grasp navigation, first release or largest object displacement, peak height, final",
                "appearance_only_environment_reuse": "same-shape/size scenes reused; no physics steps are run",
                "review_status": "unreviewed",
            })
            completed.append(ident)
    finally:
        if env is not None:
            env.close()
    return completed


def assemble(output, inventory, per_sheet=8):
    sheets = output / "sheets"
    sheets.mkdir(exist_ok=True)
    for start in range(0, len(inventory), per_sheet):
        rows = inventory[start:start + per_sheet]
        paths = [output / "episodes" / r["review_id"] / "overview.jpg" for r in rows]
        if not all(p.exists() for p in paths):
            continue
        canvas = Image.new("RGB", (1716, 36 + 192 * len(paths)), "#e5e7eb")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"Sheet {start // per_sheet + 1:03d} | red=object, blue=gripper | images selected from saved poses; no cause labels", font=ImageFont.truetype(FONT, 16), fill="#111")
        for slot, path in enumerate(paths):
            with Image.open(path) as im:
                canvas.paste(im, (0, 36 + slot * 192))
        canvas.save(sheets / f"sheet_{start // per_sheet + 1:03d}.jpg", quality=93)


def closeups(dataset, output, identifiers):
    """A second fixed view resolves contact details lost in a wide trajectory view."""
    import torch
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.structs.pose import Pose
    from robot_adaptation.simulation.environment import CollectionEnv
    torch.set_num_threads(1)
    inventory = json.loads((output / "inventory.json").read_text())
    wanted = set(identifiers)
    rows = [r for r in inventory if r["review_id"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError("Unknown closeup episode")
    env, geometry = None, None
    try:
        for row in rows:
            with np.load(dataset / row["file"], allow_pickle=False) as d:
                props = json.loads(str(d["label__condition_json"]))["properties"]
                obj = d["input__object_pose_world_m_wxyz"].copy()
                q = d["input__joint_position_rad_m"].copy()
                times = d["meta__state_time_s"].copy()
                contact = np.flatnonzero(d["sim__object_contact"])
                grasp = d["sim__is_grasped"].astype(bool)
                goal = d["input__goal_position_world_m"].copy()
            c = int(contact[0]) if len(contact) else 5
            releases = np.flatnonzero(grasp[:-1] & ~grasp[1:]) + 1
            pivot = int(releases[0]) if len(releases) else int(obj[:, 2].argmax())
            indices = sorted({max(0, min(len(times) - 1, i)) for i in
                              [c - 1, c, c + 1, c + 2, c + 4, c + 8, c + 12, c + 20, pivot - 1, pivot, pivot + 2, int(obj[:, 2].argmax())]})
            key = (props["shape"], tuple(props["size_xyz_m"]))
            if key != geometry:
                if env is not None:
                    env.close()
                env = CollectionEnv(props, human_render_camera_configs={"shader_pack": "minimal", "width": 640, "height": 480})
                env.reset(seed=int(row["seed"]), options={"reconfigure": True})
                geometry = key
            env.goal_site.set_pose(Pose.create_from_pq(goal, device=env.device))
            target = obj[0, :3] + [0, 0, .075]
            eye = obj[0, :3] + [.30, .32, .30]
            next(iter(env._human_render_cameras.values())).camera.set_local_pose(sapien_utils.look_at(eye=eye, target=target).sp)
            sheet = Image.new("RGB", (1280, 30 + 260 * ((len(indices) + 3) // 4)), "white")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.truetype(FONT, 16)
            draw.text((8, 6), row["review_id"] + " | fixed camera at initial contact region | saved poses only", font=font, fill="black")
            for slot, i in enumerate(indices):
                env.agent.robot.set_qpos(torch.as_tensor(q[i:i + 1], device=env.device))
                env.cube.set_pose(Pose.create(torch.as_tensor(obj[i:i + 1], device=env.device)))
                frame = Image.fromarray(env.render_rgb_array()[0].cpu().numpy().astype(np.uint8))
                x, y = slot % 4 * 320, 30 + slot // 4 * 260
                sheet.paste(frame.resize((320, 240)), (x, y))
                draw.text((x + 4, y + 240), f"t={times[i]:.2f}s", font=font, fill="black")
            base = output / "episodes" / row["review_id"]
            sheet.save(base / "contact_closeup.jpg", quality=94)
            dump(base / "contact_closeup.json", {"frame_indices": indices, "camera_eye": eye.tolist(), "camera_target": target.tolist(), "source_sha256": row["source_sha256"]})
            print("Contact closeup:", row["review_id"], flush=True)
    finally:
        if env is not None:
            env.close()


def report(output):
    inventory = json.loads((output / "inventory.json").read_text())
    ap = output / "annotations.json"
    annotations = json.loads(ap.read_text()) if ap.exists() else []
    by_id = {}
    allowed = {r["review_id"] for r in inventory}
    for a in annotations:
        ident = a["review_id"]
        if ident not in allowed or ident in by_id:
            raise ValueError(f"Unknown/duplicate annotation: {ident}")
        if a["primary_visual_type"] not in TAXONOMY:
            raise ValueError(f"Unknown type: {a['primary_visual_type']}")
        if not a.get("evidence") or not a.get("review_basis"):
            raise ValueError(f"Missing visual evidence/review basis: {ident}")
        if a.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid confidence: {ident}")
        if set(a.get("visual_tags", [])) - set(TAGS):
            raise ValueError(f"Unknown visual tag: {ident}")
        by_id[ident] = a
    rows = []
    for r in inventory:
        a = by_id.get(r["review_id"], {})
        rows.append({"review_id": r["review_id"], "episode_id": r["episode_id"], "source_file": r["source_file"],
                     "recorded_success": r["success"], "original_failure_type": r.get("failure_type", ""),
                     "review_status": "reviewed" if a else "unreviewed",
                     "primary_visual_type": a.get("primary_visual_type", ""),
                     "visual_tags": ";".join(a.get("visual_tags", [])), "confidence": a.get("confidence", ""),
                     "event_interval_s": json.dumps(a.get("event_interval_s")),
                     "evidence": a.get("evidence", ""), "review_basis": a.get("review_basis", ""),
                     "reviewed_sheet": a.get("reviewed_sheet", ""),
                     "contact_closeup": a.get("additional_visual_evidence", {}).get("file", ""),
                     "cause_status": "not established by visual review", "source_sha256": r["source_sha256"]})
    write_csv(output / "visual_labels.csv", rows)
    groups = output / "groups"
    groups.mkdir(exist_ok=True)
    for kind in TAXONOMY:
        members = [r for r in rows if r["primary_visual_type"] == kind]
        # Always write a header, including categories with zero confirmed members.
        with (groups / f"{kind}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["review_id"])
            writer.writeheader()
            writer.writerows(members)
    counts = {k: sum(a["primary_visual_type"] == k for a in annotations) for k in TAXONOMY}
    summary = {"total_episodes": len(inventory), "visually_reviewed": len(annotations), "unreviewed": len(inventory) - len(annotations),
               "recorded_successes": sum(r["success"] for r in inventory), "visual_type_counts": counts,
               "off_center_tag_count": sum("off_center_contact" in a.get("visual_tags", []) for a in annotations),
               "additional_contact_closeups_inspected": sum(bool(a.get("additional_visual_evidence")) for a in annotations),
               "recorded_failure_visual_type_counts": {k: sum(not r["recorded_success"] and r["primary_visual_type"] == k for r in rows) for k in TAXONOMY},
               "review_method": "First-pass AI visual inspection of six event-focused saved-pose keyframes per episode and full-duration height/XY traces; selected contact closeups. Full videos were generated, not watched end-to-end.",
               "scope": "One historical collection; types describe observed behavior, not verified causes",
               "reviewer": "Codex visual inspection; not independent human ground truth"}
    dump(output / "summary.json", summary)
    dump(output / "review_data.json", {"summary": summary, "taxonomy": TAXONOMY, "rows": rows})
    data = json.dumps({"summary": summary, "taxonomy": TAXONOMY, "rows": rows}).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><title>Visual episode review</title>
<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;background:#f5f5f2;color:#192b33}h1{font-size:28px}button,select,input{padding:9px;margin:4px}#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:18px}article{background:white;padding:16px;border:1px solid #ddd;border-radius:8px}video{width:100%}a{color:#086e78}.note{color:#59656a}details{margin-top:12px}img{width:100%}</style>
<h1>Visual review of 600 recorded attempts</h1><p id="coverage"></p><p id="method" class="note"></p><p><a href="REVIEW.md">Read findings and review limitations</a></p>
<p>These are reconstructions of saved poses. Source recordings are unchanged. Visual categories describe behavior; mass, friction, speed and grasp placement remain possible explanations rather than proven causes. Off-center contact can overlap with slip or ejection. No confirmed off-center-contact or abrupt-ejection cases were assigned in this first pass; this does not establish their absence.</p>
<p><a href="visual_labels.csv">All episode labels (CSV)</a> · <a href="summary.json">Counts and coverage</a> · <a href="manifest.json">Source provenance</a></p>
<label>Type <select id="filter"><option value="all">All types</option><option value="unreviewed">Unreviewed</option></select></label><label>Search <input id="search" placeholder="episode, tag, notes"></label><label><input type="checkbox" id="failures" checked>Recorded failures only</label><button id="more">Show next 24</button><div id="cards"></div>
<script>const D=__DATA__;let limit=24;const $=x=>document.getElementById(x);$('coverage').textContent=`Visually reviewed: ${D.summary.visually_reviewed} / ${D.summary.total_episodes}. Unreviewed: ${D.summary.unreviewed}.`;
$('method').textContent=D.summary.review_method;for(const [k,v] of Object.entries(D.taxonomy)){const o=document.createElement('option');o.value=k;o.textContent=k+' ('+D.summary.visual_type_counts[k]+')';$('filter').append(o)}
function el(tag,text,parent){const e=document.createElement(tag);if(text)e.textContent=text;if(parent)parent.append(e);return e}
function render(){const q=$('search').value.toLowerCase(),f=$('filter').value;const rows=D.rows.filter(r=>(f==='all'||r.primary_visual_type===f||(f==='unreviewed'&&r.review_status==='unreviewed'))&&(!$('failures').checked||!r.recorded_success)&&JSON.stringify(r).toLowerCase().includes(q));$('cards').replaceChildren();for(const r of rows.slice(0,limit)){const a=el('article','',$('cards'));el('h3',r.review_id+' — '+(r.primary_visual_type||'UNREVIEWED'),a);el('p','Recorded outcome: '+(r.recorded_success?'success':'failure')+' · confidence: '+(r.confidence||'—'),a);const v=el('video','',a);v.controls=true;v.preload='none';v.src='episodes/'+r.review_id+'/replay.mp4';el('p',r.evidence||'No visual annotation yet.',a);el('p',r.visual_tags?'Tags: '+r.visual_tags:'',a);el('p',r.review_basis||'',a).className='note';const d=el('details','',a);el('summary','Storyboard, traces and source',d);const im=el('img','',d);im.loading='lazy';im.src='episodes/'+r.review_id+'/storyboard.jpg';const link=el('a','Open full storyboard',d);link.href=im.src;if(r.contact_closeup){const c=el('img','',d);c.loading='lazy';c.src=r.contact_closeup;const cl=el('a','Open inspected contact closeup',d);cl.href=r.contact_closeup}el('p',r.episode_id,d);el('p','Cause: unconfirmed',d)}$('more').textContent=`Show next 24 (${Math.min(limit,rows.length)} / ${rows.length})`}
for(const id of ['filter','search','failures'])$(id).addEventListener('input',()=>{limit=24;render()});$('more').onclick=()=>{limit+=24;render()};render();</script>'''
    (output / "index.html").write_text(page.replace("__DATA__", data).replace("Visual review of 600 recorded attempts", f"Visual review of {len(inventory)} recorded attempts"))
    return summary


def prepare(dataset, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    source_rows = json.loads((dataset / "episodes.json").read_text())
    if limit:
        source_rows = source_rows[:limit]
    inventory = []
    for i, row in enumerate(source_rows, 1):
        p = dataset / row["file"]
        inventory.append({**row, "review_id": f"E{i:03d}", "source_file": str(p.relative_to(ROOT)), "source_sha256": sha256(p)})
    dump(output / "inventory.json", inventory)
    dump(output / "manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
         "dataset": str(dataset.relative_to(ROOT)), "episode_count": len(inventory), "index_sha256": sha256(dataset / "episodes.json"),
         "render_code_sha256": sha256(Path(__file__)), "source_files": [{"review_id": r["review_id"], "file": r["source_file"], "sha256": r["source_sha256"]} for r in inventory],
         "taxonomy": TAXONOMY, "available_tags": TAGS,
         "annotation_rule": "No automatic assignments. Reviewer supplies annotations after inspecting saved-pose visual evidence. Physical properties and original heuristic failure names are not shown on review sheets.",
         "scope": "Not causal diagnosis. Image sequences and full motion traces are visual evidence; video-watched vs storyboard-reviewed must be recorded separately."})
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "runs/failure_collection/varied_fixed40g")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--closeups", nargs="+")
    a = p.parse_args()
    if a.closeups:
        closeups(a.dataset, a.output, a.closeups)
        return
    if a.report_only:
        print(json.dumps(report(a.output)), flush=True)
        return
    inventory = prepare(a.dataset, a.output, a.limit)
    report(a.output)
    # Keep ten starts of a condition together to reuse visual geometry.
    batches = [inventory[start:start + 10] for start in range(0, len(inventory), 10)]
    complete = 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(render_batch, (str(a.dataset), str(a.output), batch, a.stride)) for batch in batches]
        for future in as_completed(futures):
            complete += len(future.result())
            assemble(a.output, inventory)
            print(f"Rendered {complete}/{len(inventory)} episodes", flush=True)
    print(json.dumps(report(a.output)), flush=True)


if __name__ == "__main__":
    main()
