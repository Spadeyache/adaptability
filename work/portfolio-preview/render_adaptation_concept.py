"""Scripted visual concept. No policy learning or online physical inference.

Recorded robot joint poses seed a staged animation; object slip, retries,
finger spacing and captions are authored. Never use as an experiment result.
"""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils
from robot_adaptation.simulation.environment import CollectionEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'adaptation-concept'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / 'runs/deployment/20261022_014201_067220/episode_001/trajectory.npz'
data = np.load(SOURCE)
q = data['joint_position_rad_m']
p = data['actual_object_pose_world_m_wxyz']
font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fonts = {size: ImageFont.truetype(font, size) for size in (15, 18, 22, 30)}
heading = ImageFont.truetype(bold, 30)
torch.set_num_threads(2)
env = CollectionEnv(
    {'size_xyz_m': [.05]*3, 'mass_kg': .125,
     'initial_pose_world': [0, 0, .025, 1, 0, 0, 0]},
    human_render_camera_configs={
        'shader_pack': 'minimal', 'width': 640, 'height': 480,
        'pose': sapien_utils.look_at(eye=[.45,.50,.38], target=[0,0,.12]),
        'fov': .85,
    },
)
env.reset(seed=2001, options={'reconfigure': True})
frames, durations, records = [], [], []
cache = {}

def scene(joints, cube):
    key = joints.tobytes() + cube.tobytes()
    if key not in cache:
        env.agent.robot.set_qpos(torch.tensor(joints[None], device=env.device))
        env.cube.set_pose(Pose.create(torch.tensor(cube[None], device=env.device)))
        cache[key] = Image.fromarray(env.render_rgb_array()[0].cpu().numpy().astype('uint8'))
    return cache[key]

def add(joints, cube, stage, line, memory, ms=100):
    canvas = Image.new('RGB', (760, 672), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((36,24), 'Learning as it goes', font=heading, fill='#263929')
    draw.text((36,70), stage, font=fonts[18], fill='#77776e')
    canvas.paste(scene(joints, cube), (60,108))
    draw.text((36,598), line, font=fonts[22], fill='#263929')
    draw.text((36,638), memory, font=fonts[15], fill='#66685f')
    draw.text((724,28), 'CONCEPT', anchor='ra', font=fonts[15], fill='#88887f')
    frames.append(canvas)
    durations.append(ms)
    records.append({'stage':stage, 'caption':line, 'memory':memory, 'duration_ms':ms})

def segment(start, end, stage, line, memory, cube_fn=None, tighter=False):
    # Interpolate recorded poses for a slower, legible visual sample.
    ids = np.linspace(start, end, max(2, abs(end-start)*3+1))
    for t in ids:
        lo, hi = int(np.floor(t)), int(np.ceil(t))
        a = t-lo
        joints = (1-a)*q[lo] + a*q[hi]
        cube = (1-a)*p[lo] + a*p[hi]
        if tighter and t >= 5:
            joints[-2:] -= .001
        if cube_fn is not None:
            cube = cube_fn(t, cube)
        add(joints, cube, stage, line, memory)

try:
    base = p[0].copy()
    add(q[0], base, 'FIRST ENCOUNTER', 'This grip should hold.', 'Concept sample · scripted motion, not a learning result', 1100)
    segment(0,5,'FIRST ENCOUNTER','This grip should hold.','Same robot. Same object.')
    # Author a slip on the first attempt. Physical inference is not being claimed.
    def slip(t, cube):
        cube[2] = max(.025, cube[2] - .016*(t-5)**2)
        return cube
    segment(5,8,'NOTICE','The cube is slipping.','What happened differs from what it expected.', slip)
    dropped = slip(8, p[8].copy())
    add(q[8],dropped,'NOTICE','The cube is slipping.','What happened differs from what it expected.',1100)
    segment(8,4,'ADJUST','Try a firmer grip.','Use the slip as feedback.', lambda t,c: dropped.copy())
    segment(4,10,'ADJUST','Try a firmer grip.','Use the slip as feedback.', tighter=True)
    add(q[10],p[10],'ADJUST','This time, it holds.','Keep the adjustment for the next attempt.',1100)
    segment(10,5,'KEEP WHAT WORKED','Put it down. Keep what worked.','The next encounter starts with the revised grip.', tighter=True)
    segment(5,0,'KEEP WHAT WORKED','Put it down. Keep what worked.','The next encounter starts with the revised grip.',lambda t,c:base.copy())
    add(q[0],base,'NEXT ENCOUNTER','Start with the firmer grip.','It carries the adjustment into the next lift.',1100)
    segment(0,10,'NEXT ENCOUNTER','Start with the firmer grip.','It carries the adjustment into the next lift.',tighter=True)
    add(q[10],p[10],'NEXT ENCOUNTER','Learn once. Use it next time.','Concept sample · scripted motion, not a learning result',2000)
    gif = OUT/'learning-as-it-goes.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    for i in [0,25,55,len(frames)-1]:
        frames[i].save(OUT/f'frame-{i:03}.png')
    (OUT/'provenance.json').write_text(json.dumps({
        'kind':'scripted research-vision concept, not an experimental result',
        'source':str(SOURCE.relative_to(ROOT)),
        'authored':['object slip','trajectory replay and interpolation','tighter finger spacing','retry sequence','memory and feedback captions'],
        'online_inference':False,'policy_learning':False,
        'frames':len(frames),'duration_ms':sum(durations),'sequence':records,
    },indent=2))
    print(json.dumps({'gif':str(gif),'frames':len(frames),'duration_s':sum(durations)/1000}),flush=True)
finally:
    env.close()
