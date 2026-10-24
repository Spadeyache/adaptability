const fs=require('fs');
const {execFileSync}=require('child_process');
(async()=>{
  const response=await fetch('http://localhost:8000/index.html');
  const html=await response.text();
  fs.writeFileSync('\\\\wsl.localhost\\Ubuntu\\home\\ubuntu\\faliure\\work\\portfolio-preview\\served-index.html',html);
  console.log(JSON.stringify({status:response.status,server:response.headers.get('server'),html}));
  const command="$connections=Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; $connections | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Get-CimInstance Win32_Process -Filter ('ProcessId = '+$_) | Select-Object Name,CommandLine,ParentProcessId | ConvertTo-Json -Compress }";
  console.log(execFileSync('powershell.exe',['-NoProfile','-Command',command],{encoding:'utf8'}));
})().catch(e=>{console.error(e.message);process.exit(1)});
