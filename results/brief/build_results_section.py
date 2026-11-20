"""Typeset a one-page summary of saved results; no model execution."""
from pathlib import Path
import os
import textwrap
os.environ.setdefault('MPLCONFIGDIR', '/tmp/brief-mpl-page')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
blocks = [
('Setup',
 'We tested a shared next-state predictor with one learned belief row per cube, '
 'using simulated shoves. The proposed separation is common dynamics in the network '
 'and persistent object information in the row; interpretable physical constants are '
 'not established. Step 2 used 180 weight-training cubes, 20 cubes for adaptation '
 'settings, 30 unseen test cubes and five independent training runs. Each test cube supplied '
 'five fitting shoves and five separate scoring shoves. [S1]'),
('Leak check',
 'Applied forces were generated independently of mass and friction. Two actions-only '
 'prediction checks returned negative R²: mass −0.113/−0.117 and combined friction '
 '−0.022/−0.025, meaning worse than predicting the test-set average. These checks '
 'found no useful property signal in commands; they do not prove every possible '
 'leak absent. [S1, S2]'),
('Known-physics check',
 'Compared with Plain (state and force only), giving the predictor true mass AND combined cube–floor friction reduced push '
 'velocity-change error from 0.209563 ± 0.00515 to 0.080712 ± 0.00937 (61.49%). '
 'The metric is squared prediction error divided by a fixed training-data scale '
 '(NMSE; lower is better); ± denotes standard deviation across training seeds. '
 'This comparison does not isolate mass from friction. [S1]'),
('Belief test',
 'With the network frozen, fitting only the new row reduced query push NMSE to '
 '0.161539 ± 0.0307 (22.92% below Plain), beating both the average row and another '
 'cube’s fitted row. Mean error fell from 0.207674 to 0.161539 with one to five '
 'fitting shoves. A separate linear mass decoder achieved test R² 0.876836 ± 0.0553; '
 'the beliefs used no mass labels, but this decoder used training-cube mass labels. '
 'Belief fitting recovered 37.27% of the Plain-to-Oracle gap, below the predeclared '
 '50% target, and did not beat the recent-history model beyond seed spread. [S1]'),
('Drift comparison',
 'Step 3 reused frozen checkpoints and visited 30 new cubes in one fixed order. '
 'At learning rate 1e−4, independently resetting and adapting all weights per cube '
 'gave mean push NMSE 0.107807 versus Plain’s 0.127834. In the sequential run, '
 'the mean over cubes immediately after their own adaptation was 0.170115; '
 'training-cube validation composite error rose from 0.366298 to 0.637299. '
 'At the validation-selected rate 1e−5, the corresponding sequential means were '
 '0.134966 and 0.378233. Composite means the average of push-velocity, moving-coast '
 'velocity and full-state normalized errors. These are not final-model scores on '
 'all visited cubes; the predeclared forgetting and speed criteria both failed. [S3, S4]'),
('Scope',
 'Oracle did not improve prediction during coast (sliding after force release), although the explicit friction '
 'formula predicted coast changes accurately. Push-focused belief fitting worsened '
 'coast and full-state prediction, and longer forecasts were unreliable. This '
 'is idealized direct-force simulation, not demonstrated robot contact or learning '
 'that accumulates across objects. [S1, S3, S5]'),
]
sources = [
 ('S1', 'clean/shove_gate/push_latent/RESULTS.md', '7–84; 140–177; 283–290', 7),
 ('S2', 'clean/shove_gate/push_latent/collection.py', '37–45; 84–90', 37),
 ('S3', 'clean/shove_gate/adaptation_compare/RESULTS.md', '3–18; 58–89; 150–208; 224', 3),
 ('S4', 'clean/shove_gate/adaptation_compare/results/gain_damage_partial.csv', 'rows 2–3', 2),
 ('S5', 'clean/shove_gate/revised_velocity/RESULTS.md', '3–29; 44', 3),
]
md = '# Object-specific belief adaptation in simulated shoves\n\n'
md += 'Saved-results brief · Kent · 20 November 2026\n\n'
for heading, body in blocks:
    md += f'**{heading}.** {body}\n\n'
md += '## Sources\n\n'
for key,path,lines,line in sources:
    locator = lines if lines.startswith('rows ') else 'lines ' + lines
    md += f'- **{key}:** [{path}]({ROOT/path}:{line}), {locator}.\n'
(OUT/'results_section.md').write_text(md)

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig = plt.figure(figsize=(8.27,11.69),facecolor='white')
fig.text(.075,.955,'Object-specific belief adaptation',size=19,weight='bold',color='#172b40')
fig.text(.075,.928,'Simulated shoves | Saved-results brief | Kent | 20 November 2026',
         size=9.5,color='#64748b')
fig.add_artist(plt.Line2D([.075,.925],[.912,.912],transform=fig.transFigure,
                         color='#00897b',lw=2))
y=.891
renderer = fig.canvas.get_renderer()
font = FontProperties(family='DejaVu Sans',size=10)
def wrap_measured(body):
    lines=[]; current=''
    for word in body.split():
        candidate=(current+' '+word).strip()
        width=renderer.get_text_width_height_descent(candidate,font,False)[0]
        if current and width > fig.bbox.width*.845:
            lines.append(current); current=word
        else:
            current=candidate
    if current: lines.append(current)
    return '\n'.join(lines)
for heading,body in blocks:
    fig.text(.075,y,heading.upper(),size=10.3,weight='bold',color='#00897b',va='top')
    y-=.020
    wrapped=wrap_measured(body)
    fig.text(.075,y,wrapped,size=10,linespacing=1.28,color='#172b40',va='top')
    y -= (wrapped.count('\n')+1)*.0152 + .017
fig.text(.075,y,'SOURCES  |  Full figure data and line references: numbers.md',
         size=8.1,weight='bold',color='#64748b',va='top')
y-=.017
for key,path,lines,_ in sources:
    short=path.replace('clean/shove_gate/','')
    fig.text(.075,y,f'{key}  clean/shove_gate/{short}  ·  {lines}',
             size=6.9,color='#64748b',va='top')
    y-=.013
if y<.035:
    raise RuntimeError(f'Content exceeds single-page layout: bottom={y}')
fig.savefig(OUT/'results_section.pdf')
fig.savefig(OUT/'results_section.png',dpi=170)
plt.close(fig)
print(f'Single-page summary saved; final baseline {y:.3f}')
