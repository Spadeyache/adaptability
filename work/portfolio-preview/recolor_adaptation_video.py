"""Match authored video typography to the existing homepage palette."""
from pathlib import Path
import json
import re
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).parent
SOURCE=ROOT/'adaptation-studio'
OUT=ROOT/'adaptation-bottom-captions'
MOVIE=OUT/'movie-frames'
MOVIE.mkdir(parents=True,exist_ok=True)
css=(ROOT/'served-simple.css').read_text()
def color(name):
    return re.search(r'--'+re.escape(name)+r':\s*(#[0-9a-fA-F]{6})',css).group(1)
bg,ink,secondary=color('bg'),color('text'),color('secondary')
title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',19)
label=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
headings=['Notices the slip','Adjusts the grip','Uses what it learned']
captions=['The cube slips.','Enough grip to hold it.','A different object. A different grasp.']
frames=[]
for i,path in enumerate(sorted((SOURCE/'movie-frames').glob('*.png'))):
    stage=0 if i<28 else 1 if i<79 else 2
    source=Image.open(path).convert('RGB')
    image=Image.new('RGB',(864,598),bg)
    image.paste(source.crop((0,58,864,598)),(0,0))
    draw=ImageDraw.Draw(image)
    x,y=30,559
    draw.text((x,y),headings[stage],font=title,fill=ink)
    x+=draw.textlength(headings[stage],font=title)
    draw.text((x,y),'  |  ',font=label,fill=secondary)
    x+=draw.textlength('  |  ',font=label)
    draw.text((x,y),captions[stage],font=label,fill=ink)
    assert x+draw.textlength(captions[stage],font=label)<=834
    image.save(MOVIE/f'{i:04}.png')
    frames.append(image)
samples=Image.new('RGB',(864*4,598),bg)
for x,i in enumerate([0,25,60,len(frames)-1]):samples.paste(frames[i],(864*x,0))
palette=samples.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
quantized=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
quantized[0].save(OUT/'learning-as-it-goes.gif',save_all=True,
    append_images=quantized[1:],duration=60,loop=0,optimize=True)
frames[0].save(OUT/'poster.webp',quality=82)
frames[60].save(OUT/'preview.png')
provenance=json.loads((SOURCE/'provenance.json').read_text())
provenance.update({'palette_source':'http://localhost:8000/simple.css?v=20261019-headline',
    'palette':{'background':bg,'text':ink,'secondary':secondary},
    'scope':'Single bottom caption: title | description. Top title and counter removed; simulation frames unchanged'})
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps({'frames':len(frames),'palette':provenance['palette']}))
