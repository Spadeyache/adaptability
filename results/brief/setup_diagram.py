"""Render the experimental design; does not load or evaluate any model."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/brief-mpl-setup')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
INK, GREEN, GRAY = '#172b40', '#00897b', '#64748b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(15, 7.1))
fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis('off')

def box(x, y, w, h, label, color=INK, fill='white', size=15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=1.2',
        edgecolor=color,facecolor=fill,linewidth=1.6))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=color,linespacing=1.4)

def arrow(a,b,color=GRAY,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,
                 linewidth=1.7,color=color,connectionstyle=f'arc3,rad={rad}'))

for x in [1,52]:
    ax.add_patch(FancyBboxPatch((x,3),47,88,boxstyle='round,pad=.5,rounding_size=1.5',
        linewidth=0,facecolor='#f5f8fa'))
ax.text(2,97,'TRAINING',weight='bold',fontsize=20,color=INK,va='center')
ax.text(53,97,'DEPLOYMENT',weight='bold',fontsize=20,color=INK,va='center')
ax.text(24.5,86,'Many cubes • one shared network',ha='center',color=GRAY,fontsize=14)
ax.text(75.5,86,'New cube • shared network stays frozen',ha='center',color=GRAY,fontsize=14)

box(4,62,19,16,'Cube state\n+ applied force')
box(29,61,15,18,'Shared\nnetwork')
arrow((23.5,70),(28.3,70))
ax.text(13.5,54,'Belief table',ha='center',weight='bold',color=GREEN,fontsize=15)
for i,label in enumerate(['Cube A','Cube B','Cube C']):
    yy=47-i*6
    ax.text(4,yy+2.5,label,color=INK,fontsize=12,va='center')
    for j in range(5):
        ax.add_patch(Rectangle((11+j*2.4,yy),2.2,5,
            facecolor=GREEN if i==1 else '#cbd5e1',edgecolor='white',linewidth=.6))
ax.text(13.5,31,'One learned row per cube',ha='center',fontsize=11.5,color=GRAY)
arrow((23.5,43.5),(33,60),GREEN)
ax.text(29,47,'Use its row',color=GREEN,fontsize=12,rotation=39,ha='center')
arrow((36.5,60),(36.5,48))
box(29,35,15,12,'Predicted\nnext state',size=13.5)
arrow((36.5,34.5),(36.5,25))
box(8,10,36,14,'Compare with observed next state\nPrediction error updates network + row',
    color=GREEN,fill='#e7f4f1',size=13)
arrow((8,19),(6,36),GREEN,rad=-.25)

box(55,65,19,13,'New belief row\nStart at training average',color=GREEN,fill='#e7f4f1',size=13.5)
box(80,65,15,13,'Shared network\nFROZEN',size=13)
box(55,43,40,14,'First 5 shoves: compare predictions with motion\nFit only the new cube’s belief row',
    color=GREEN,fill='#e7f4f1',size=13.5)
arrow((64.5,64),(67,58),GREEN)
arrow((87.5,64),(84,58))
arrow((75,42),(75,34),GREEN)
box(55,21,40,12,'Other 5 shoves: test prediction\nKeep network and fitted row fixed',size=13.5)
arrow((75,20),(75,14),GREEN)
ax.text(75,9,'Save this cube’s row for reuse',ha='center',va='center',
        weight='bold',color=GREEN,fontsize=15)

fig.savefig(OUT/'setup.svg',facecolor='white')
fig.savefig(OUT/'setup.png',dpi=220,facecolor='white')
plt.close(fig)
