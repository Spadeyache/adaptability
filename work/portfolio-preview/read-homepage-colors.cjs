const fs=require('fs');
(async()=>{
  const css=await(await fetch('http://localhost:8000/simple.css?v=20261019-headline')).text();
  fs.writeFileSync('\\\\wsl.localhost\\Ubuntu\\home\\ubuntu\\faliure\\work\\portfolio-preview\\served-simple.css',css);
  console.log(css.slice(0,7500));
})().catch(e=>{console.error(e.message);process.exit(1)});
