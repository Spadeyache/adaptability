"""Authored, three-part adaptation concept; no learning results are claimed."""
from pathlib import Path
import json
import time
import numpy as np
import torch
import sapien
from PIL import Image, ImageDraw, ImageFont
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from robot_adaptation.simulation.environment import CollectionEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent/'adaptation-studio'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT/'runs/deployment/20261022_014201_067220/episode_001/trajectory.npz'
d = np.load(SOURCE)
q, p = d['joint_position_rad_m'], d['actual_object_pose_world_m_wxyz']
torch.set_num_threads(2)

class Studio(CollectionEnv):
    def object_color_rgba(self):
        return [.76,.10,.075,1]

    def _load_scene(self, options):
        super()._load_scene(options)
        for actor in (self.table_scene.table,self.table_scene.ground):
            for entity in actor._objs:
                body=entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if body is not None: body.visibility=0
        self.studio = actors.build_box(self.scene, [4,4,.04], [.83,.85,.85,1],
            'matte-workbench', body_type='kinematic', add_collision=False,
            initial_pose=sapien.Pose([0,0,-.04]))
        self.ball = actors.build_sphere(self.scene, .029, [.045,.32,.48,1],
            'blue-sphere', body_type='kinematic', add_collision=False,
            initial_pose=sapien.Pose([0,-.14,.029]))

    def _load_lighting(self, options):
        self.scene.set_ambient_light([.45,.45,.45])
        self.scene.add_directional_light([.4,.5,-1], [.9,.86,.80],
            shadow=True, shadow_scale=1.8, shadow_map_size=2048)
        self.scene.add_directional_light([-.5,-.3,-1], [.3,.35,.4])

env = Studio({'size_xyz_m':[.05]*3, 'mass_kg':.125,
              'initial_pose_world':[0,0,.025,1,0,0,0]},
    sensor_configs={'shader_pack':'default'},
    human_render_camera_configs={'shader_pack':'default','width':1152,'height':720,
        'pose':sapien_utils.look_at(eye=[.71,.72,.59],target=[-.18,-.015,.22]),'fov':.78})
env.reset(seed=2001, options={'reconfigure':True})
# Human-view rendering otherwise reveals the simulator's green goal marker.
for hidden in env._hidden_objects:
    hidden.hide_visual()
env._hidden_objects=[]
base = p[0].copy()
ball_base = np.array([0,-.14,.029,1,0,0,0], dtype=np.float32)
frames, meta = [], []
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
title = ImageFont.truetype(BOLD,26)
label = ImageFont.truetype(FONT,19)
small = ImageFont.truetype(FONT,13)
headings = ['Notices the slip','Adjusts the grip','Uses what it learned']
captions = ['The cube slips.','Enough grip to hold it.','A different object. A different grasp.']

def render(joints,cube,ball,stage):
    env.agent.robot.set_qpos(torch.tensor(joints[None], device=env.device))
    env.cube.set_pose(Pose.create(torch.tensor(cube[None], device=env.device)))
    env.ball.set_pose(Pose.create(torch.tensor(ball[None], device=env.device)))
    pic = Image.fromarray(env.render_rgb_array()[0].cpu().numpy().astype('uint8'))
    pic = pic.resize((864,540),Image.Resampling.LANCZOS)
    canvas = Image.new('RGB',(864,650),'white')
    canvas.paste(pic,(0,58))
    draw = ImageDraw.Draw(canvas)
    draw.text((30,17),headings[stage],font=title,fill='#263929')
    draw.text((834,22),f'{stage+1} / 3',font=label,fill='#73796f',anchor='ra')
    draw.text((30,610),captions[stage],font=label,fill='#263929')
    draw.text((834,615),'SCRIPTED CONCEPT',font=small,fill='#7d827b',anchor='ra')
    frames.append(canvas)
    meta.append({'stage':stage,'caption':captions[stage]})

def interp(arr,t):
    lo,hi=int(np.floor(t)),int(np.ceil(t))
    return (1-(t-lo))*arr[lo]+(t-lo)*arr[hi]

def move(a,b,n,stage,cube_fn=None,tight=False,ball_q=None):
    for t in np.linspace(a,b,n):
        joints=interp(q,t) if ball_q is None else interp(ball_q,t)
        cube=interp(p,t)
        ball=ball_base.copy()
        if tight and t>=5:
            joints[-2:]=.0245
        if cube_fn: cube=cube_fn(t,cube)
        if ball_q is not None:
            ball[:3]=interp(p,t)[:3]+np.array([0,-.14,.004])
            cube=base.copy()
        render(joints,cube,ball,stage)

try:
    # Preview the studio before rendering the complete sample.
    render(q[0],base,ball_base,0)
    frames[-1].save(OUT/'studio-preview.png')
    print('Studio preview rendered.',flush=True)
    move(0,5,12,0)
    def slip(t,c):
        c[2]=max(.025,c[2]-.016*(t-5)**2)
        return c
    move(5,8,10,0,slip)
    dropped=slip(8,p[8].copy())
    for _ in range(5): render(q[8],dropped,ball_base,0)
    move(8,4,12,1,lambda t,c:dropped.copy())
    move(4,10,16,1,tight=True)
    for _ in range(5): render(q[10],p[10],ball_base,1)
    move(10,5,10,1,tight=True)
    move(5,0,8,1,lambda t,c:base.copy())

    # Solve the same tool poses above the second object. This is authored IK,
    # not transfer learned by a policy. Keep the original arm base fixed.
    arm=env.agent.controller.controllers['arm']
    q_ball=[]
    for i in range(11):
        env.agent.robot.set_qpos(torch.tensor(q[i:i+1],device=env.device))
        target=arm.ee_pose_at_base
        target=Pose.create(target.raw_pose.clone())
        target.raw_pose[:,1]-=.14
        target.raw_pose[:,2]+=.004
        sol=arm.kinematics.compute_ik(target,torch.tensor(q[i:i+1],device=env.device))
        if sol is None: raise RuntimeError(f'IK failed at pose {i}')
        value=q[i].copy()
        value[:7]=sol[0].cpu().numpy()
        value[-2:]=.04 if i<4 else .028
        q_ball.append(value)
    q_ball=np.array(q_ball)
    for a in np.linspace(0,1,8):
        render((1-a)*q[0]+a*q_ball[0],base,ball_base,2)
    move(0,10,24,2,ball_q=q_ball)
    for _ in range(10): render(q_ball[10],base, p[10]+np.array([0,-.14,.004,0,0,0,0]),2)

    # A shared palette reduces flicker and GIF size on the neutral workbench.
    samples=Image.new('RGB',(864*4,650))
    for x,i in enumerate([0,25,60,len(frames)-1]):samples.paste(frames[i],(864*x,0))
    palette=samples.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
    quantized=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(OUT/'learning-as-it-goes-v2.gif',save_all=True,
        append_images=quantized[1:],duration=60,loop=0,optimize=True)
    for i in [0,25,60,len(frames)-1]:frames[i].save(OUT/f'frame-{i:03}.png')
    (OUT/'provenance.json').write_text(json.dumps({
        'kind':'scripted research-vision concept', 'source':str(SOURCE.relative_to(ROOT)),
        'scripted':['slip','retry','grip spacing','IK transfer to sphere','captions'],
        'learned_adaptation':False,'frames':len(frames),'duration_s':len(frames)*.06,
        'stages':headings,'captions':captions},indent=2))
    # Save numbered frames for an MP4 export without GIF palette limitations.
    movie=OUT/'movie-frames';movie.mkdir(exist_ok=True)
    for i,f in enumerate(frames):f.save(movie/f'{i:04}.png')
    print(json.dumps({'frames':len(frames),'seconds':len(frames)*.06,'output':str(OUT)}),flush=True)
finally:
    env.close()
