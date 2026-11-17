const API_BASE = "http://localhost:5000";
const $ = (id) => document.getElementById(id);
const viewer = $("viewer"), viewerEmpty = $("viewerEmpty"), viewerImage = $("viewerImage"), scanline = $("scanline");
const fileInput = $("fileInput"), fileMeta = $("fileMeta"), clearBtn = $("clearBtn"), analyzeBtn = $("analyzeBtn");
const apiStatus = $("apiStatus"), navStatus = $("navStatus"), connectionBadge = $("connectionBadge");
const resultsSection = $("results"), resultLabel = $("resultLabel"), resultDescription = $("resultDescription");
const barsContainer = $("bars"), ringProgress = $("ringProgress"), confidenceValue = $("confidenceValue"), reportBtn = $("reportBtn");
let currentFile = null, lastPrediction = null;
const circumference = 2 * Math.PI * 57;
ringProgress.style.strokeDasharray = circumference;
ringProgress.style.strokeDashoffset = circumference;

async function checkHealth(){
  try{
    const res=await fetch(`${API_BASE}/api/health`,{cache:"no-store"});
    const data=await res.json();
    if(data.model_loaded){
      apiStatus.textContent="Backend connected · model loaded"; apiStatus.className="api-status ok";
      navStatus.innerHTML='<b></b> Model online'; connectionBadge.innerHTML='<span></span><b>Backend connected · model loaded</b>';
    }else{throw new Error(data.load_error||"Model not loaded");}
  }catch(e){
    apiStatus.textContent="Backend offline — start app.py on port 5000"; apiStatus.className="api-status err";
    navStatus.innerHTML='<b style="background:#e77f6d"></b> Backend offline'; connectionBadge.innerHTML='<span style="background:#e77f6d"></span><b>Backend offline</b>';
  }
}
checkHealth();

function setProcess(active){document.querySelectorAll('.process-step').forEach((el,i)=>el.classList.toggle('active',i<=active));}
function loadFile(file){
  if(!file || !file.type.startsWith('image/')) return;
  currentFile=file; viewerImage.src=URL.createObjectURL(file); viewerImage.hidden=false; viewerEmpty.style.display='none'; clearBtn.hidden=false;
  fileMeta.textContent=`${file.name.toUpperCase()} · ${(file.size/1024).toFixed(0)} KB`; analyzeBtn.disabled=false; resultsSection.hidden=true; setProcess(0);
}
function clearFile(){currentFile=null;viewerImage.hidden=true;viewerImage.src='';viewerEmpty.style.display='flex';clearBtn.hidden=true;fileMeta.textContent='NO IMAGE LOADED';analyzeBtn.disabled=true;resultsSection.hidden=true;setProcess(0)}
fileInput.addEventListener('change',e=>loadFile(e.target.files[0])); clearBtn.addEventListener('click',clearFile);
['dragover','dragenter'].forEach(ev=>viewer.addEventListener(ev,e=>{e.preventDefault();viewer.classList.add('dragover')}));
['dragleave','dragend'].forEach(ev=>viewer.addEventListener(ev,()=>viewer.classList.remove('dragover')));
viewer.addEventListener('drop',e=>{e.preventDefault();viewer.classList.remove('dragover');loadFile(e.dataTransfer.files[0])});

analyzeBtn.addEventListener('click',async()=>{
  if(!currentFile)return; analyzeBtn.disabled=true; analyzeBtn.innerHTML='Analyzing <span>…</span>'; scanline.hidden=false; setProcess(1);
  const formData=new FormData();formData.append('image',currentFile);
  try{
    const res=await fetch(`${API_BASE}/api/predict`,{method:'POST',body:formData}); const data=await res.json(); if(!res.ok)throw new Error(data.error||'Prediction failed');
    setProcess(3); renderResults(data);
  }catch(err){resultsSection.hidden=false;resultLabel.textContent='Prediction unavailable';resultDescription.textContent=err.message;barsContainer.innerHTML='';}
  finally{analyzeBtn.disabled=false;analyzeBtn.innerHTML='Analyze scan <span>→</span>';scanline.hidden=true;}
});

function renderResults(data){
  resultsSection.hidden=false; const prediction=data.prediction, probabilities=data.probabilities;
  if(!prediction||!probabilities)throw new Error('Invalid prediction response from backend'); lastPrediction=data;
  resultLabel.textContent=prediction.label;
  const descriptions={glioma:'The model classified this MRI slice as showing features associated with glioma.',healthy:'The model classified this MRI slice as healthy brain tissue.',meningioma:'The model classified this MRI slice as showing features associated with meningioma.',pituitary:'The model classified this MRI slice as showing features associated with pituitary tumor.'};
  resultDescription.textContent=descriptions[prediction.class]||'The model completed the classification successfully.';
  const confidence=Number(prediction.confidence);confidenceValue.textContent=`${confidence.toFixed(1)}%`;
  requestAnimationFrame(()=>ringProgress.style.strokeDashoffset=circumference*(1-Math.max(0,Math.min(100,confidence))/100));
  const labels={glioma:'Glioma',healthy:'Healthy',meningioma:'Meningioma',pituitary:'Pituitary Tumor'};barsContainer.innerHTML='';
  Object.entries(probabilities).forEach(([cls,val])=>{const pct=Number(val);const row=document.createElement('div');row.className='bar-row';row.innerHTML=`<span class="bar-name">${labels[cls]||cls}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value">${pct.toFixed(1)}%</span>`;barsContainer.appendChild(row);requestAnimationFrame(()=>row.querySelector('.bar-fill').style.width=`${Math.max(0,Math.min(100,pct))}%`)});
  resultsSection.scrollIntoView({behavior:'smooth',block:'start'});
}

reportBtn.addEventListener('click',()=>{
  if(!lastPrediction)return;
  const p=lastPrediction.prediction, probs=lastPrediction.probabilities;
  const lines=['CORTICODE — RESEARCH MODEL OUTPUT','',`Prediction: ${p.label}`,`Confidence: ${Number(p.confidence).toFixed(2)}%`,'','Class probabilities:',...Object.entries(probs).map(([k,v])=>`- ${k}: ${Number(v).toFixed(2)}%`),'',`Model: Hybrid ConvNeXtTiny + DenseNet121 + CBAM`,'Input: 224 × 224','Purpose: Academic research prototype only','',`Generated: ${new Date().toLocaleString()}`];
  const blob=new Blob([lines.join('\n')],{type:'text/plain'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='corticode_model_report.txt';a.click();URL.revokeObjectURL(a.href);
});
