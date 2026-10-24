const {execFileSync}=require('child_process');
const probe=`import os,json
from pathlib import Path
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]))
  decoded=[a.decode(errors='replace') for a in args if a]
  if any('http.server' in a for a in decoded) and any(a=='8000' for a in decoded):
   print(json.dumps({'pid':p.name,'args':decoded,'cwd':os.readlink(p/'cwd')}))
 except (OSError,PermissionError): pass
`;
console.log(execFileSync('wsl.exe',['-d','Ubuntu','--exec','python3','-c',probe],{encoding:'utf8'}));
