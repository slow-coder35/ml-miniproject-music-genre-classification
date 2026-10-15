import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES ??= '/home/bae/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
try {
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const SKILL_DIR='/home/bae/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PYTHON='/home/bae/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const read=async p=>JSON.parse(await fs.readFile(path.join(ROOT,p),'utf8'));
const evaluation=await read('results/evaluation.json');
const inventory=await read('data/cache/inventory.json');
const splits=await read('data/cache/splits.json');
const features=await read('data/cache/feature_meta.json');
const team=await read('team.json');
const base=evaluation.rows.filter(r=>!r.tuned);
const chosen=evaluation.selected_demo.experiment;
const best=base.reduce((a,b)=>a.test_macro_f1>b.test_macro_f1?a:b);
const multi=evaluation.multiwindow.test;
const pct=x=>(100*x).toFixed(1)+'%';
const usable=inventory.tracks.filter(t=>t.usable).length;
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='DejaVu Sans',ink='#17343E',teal='#007F8B',muted='#526975';
p.theme.defaultFont=font;
function text(slide,content,x,y,w,h,size=28,bold=false,color=ink){
 const t=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=content;t.text.style={typeface:font,fontSize:size,bold,color};return t;
}
function slide(title,notes=''){
 const s=p.slides.add();s.background.fill='#F8FAF9';
 text(s,title,64,46,1152,80,44,true);
 text(s,`${p.slides.items.length} / 8`,1130,668,90,26,16,false,muted);
 s.speakerNotes.text=notes;return s;
}
function table(s,values,x,y,w,h,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:'#D5E3E5',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?ink:(r%2?'#FFFFFF':'#ECF3F3');
  cell.text.style={typeface:font,fontSize:24,bold:r===0,color:r===0?'#FFFFFF':ink};
 }
 return t;
}
async function image(s,relative,x,y,w,h){
 const b=await fs.readFile(path.join(ROOT,relative));
 s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:path.basename(relative),fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function chart(s,metric,title,x,y,w,h){
 const chart=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},title,titleTextStyle:{fontSize:25,bold:true,fill:ink},categories:['SVM','k-NN','FFNN','CNN'],series:['raw','mel'].map((kind,i)=>({name:kind==='raw'?'Raw waveform':'Log-mel',values:['svm','knn','ffnn','cnn'].map(m=>Number(base.find(r=>r.model===m&&r.input===kind)[metric].toFixed(6))),fill:i?'#007F8B':'#ED9459'})),hasLegend:true,barOptions:{direction:'column',grouping:'clustered'},legend:{textStyle:{fontSize:19}},xAxis:{textStyle:{fontSize:21}},yAxis:{numberFormatCode:'0%',min:0,max:1,textStyle:{fontSize:18}},dataLabels:{showValue:true,numberFormatCode:'0%',position:'outEnd',textStyle:{fontSize:17,fill:ink}},chartFill:'#F8FAF9'});
 applyPresentationChartFont(chart,{fontFamily:font});
 return chart;
}
const members=team.members.filter(m=>m.name).map(m=>`${m.name} (${m.student_id})`).join(' / ');
let s=slide('Music Genre Classification','Project objective: predict one of ten GTZAN genre labels. Sources: supplied Stanford CS229 report/poster; local results/evaluation.json.');
text(s,'Raw audio and log-mel features',64,176,1130,72,50,true,teal);
text(s,'Four model families. Eight baseline experiments.',64,281,1100,52,32);
text(s,'Local NVIDIA RTX 4060 training',64,361,1100,48,30,false,muted);
text(s,team.course,64,507,1100,45,27);
text(s,members||'Team names, student IDs, and section to be completed',64,567,1100,60,23,false,muted);
s=slide('Dataset and genre labels','Source: GTZAN, Tzanetakis and Cook (2002). Mirror: https://huggingface.co/datasets/marsyas/gtzan. Local data/download.json and data/cache/inventory.json document exclusions.');
text(s,String(usable),64,155,450,130,94,true,teal);
text(s,'usable recordings',64,294,480,52,34);
text(s,'Approximately 30 seconds each\nMono audio at 22,050 Hz',64,377,530,111,29);
text(s,'One unreadable file and two conflicting-label recordings excluded',64,538,530,90,24,false,muted);
table(s,[['Genre','Genre'],['Blues','Jazz'],['Classical','Metal'],['Country','Pop'],['Disco','Reggae'],['Hip-hop','Rock']],680,160,530,430,[265,265]);
s=slide('Recording splits and leakage prevention','Exact decoded-audio hashes define groups. StratifiedGroupKFold(10, shuffle=True, random_state=42): fold 0 test, fold 1 validation. Source: src/data.py and data/cache/splits.json. No artist-disjoint claim.');
table(s,[['Partition','Tracks','Windows'],['Training',String(splits.train.length),String(features.counts.train)],['Validation',String(splits.val.length),'100'],['Test',String(splits.test.length),'100']],64,169,615,330,[235,165,215]);
text(s,'Split complete recordings before sampling clips',738,171,470,112,34,true,teal);
text(s,'Exact audio duplicates stay together.\n\nFour random training windows.\nOne central evaluation window.',738,325,470,220,27);
text(s,'Artist overlap and near duplicates may remain.',64,558,1120,60,26,false,muted);
s=slide('Raw waveform and log-mel representation','Measured evidence: a training window from the local GTZAN split. Librosa mel power -> decibels; no DCT. FFT/window 512, hop 256, 64 mel bands. Source: src/features.py.');
await image(s,'results/figures/feature_example.png',64,152,1152,370);
text(s,'44,100 waveform values',64,548,540,52,29,true,teal);
text(s,'64 x 173 log-mel values',680,548,530,52,29,true,teal);
text(s,'Normalization and PCA learn from training data only.',64,615,1100,40,24,false,muted);
s=slide('Models and GPU training','Architectures extend the local starter code and follow the supplied CS229 comparison. TensorFlow GPU evidence: results/gpu_check.json. Hyperparameters: results/*_run.json.');
table(s,[['Model','Main configuration'],['SVM','Scale, PCA(15), RBF kernel'],['k-NN','Scale, PCA(15), k=10, distance weighting'],['FFNN','Five hidden dense layers, ten-class softmax'],['CNN','Three convolution blocks, dense layers, softmax']],64,155,1152,320,[230,922]);
text(s,'Adam 0.001, batch 32, up to 50 epochs',64,511,1120,50,29,true,teal);
text(s,'Early stopping restores the best validation-loss weights.\nTwo additional mel-CNN trials vary learning rate or dropout.',64,578,1120,78,25);
s=slide('Eight-model test comparison','All chart values are measured local results/results.csv, rounded to six decimal places for Excel portability. Main protocol: one central two-second window per held-out recording. Macro-F1 gives each genre equal weight. Published report values are not substituted.');
chart(s,'test_acc','Accuracy',64,154,550,370);
chart(s,'test_macro_f1','Macro-F1',660,154,550,370);
text(s,`Highest baseline test macro-F1: ${best.experiment} (${best.test_macro_f1.toFixed(3)})`,64,558,1145,58,29,true,teal);
text(s,'100 test recordings. One additional correct prediction changes accuracy by 1 percentage point.',64,625,1140,38,21,false,muted);
s=slide('Error analysis and dataset limits','Confusion matrix: results/figures/demo_multiwindow_confusion.png, selected CNN, five windows averaged per recording. GTZAN limitations: Sturm (2013), https://arxiv.org/abs/1306.1461.');
await image(s,'results/figures/demo_multiwindow_confusion.png',50,143,685,500);
text(s,'Genre boundaries overlap',780,165,430,65,31,true,teal);
text(s,'Short clips omit musical structure.\n\nGTZAN includes repetitions and label faults.\n\nA recording split cannot guarantee unseen-artist generalization.',780,264,430,300,27);
s=slide('Live demonstration and conclusions','Start scripts/run.sh app. Show two held-out examples, one error case, and silence handling. The app averages five windows from the first 30 seconds. Selection uses validation macro-F1, not test scores. See docs/DEMO_AND_VIVA.md.');
text(s,'Upload a WAV or FLAC music clip',64,165,1150,60,35,true,teal);
text(s,'Listen, inspect the spectrogram, and predict its genre.',64,254,1150,70,30);
text(s,`Selected model: ${chosen}`,64,355,1130,50,30,true);
text(s,`Five-window test accuracy ${pct(multi.accuracy)}\nMacro-F1 ${multi.macro_f1.toFixed(3)} on ${multi.count} recordings`,64,430,1120,100,31);
text(s,'Model scores can be overconfident. Interpret errors and report the evaluation protocol.',64,579,1120,72,25,false,muted);
const out=path.join(ROOT,'deliverables');const staging=path.join(ROOT,'work/deck');
await fs.mkdir(out,{recursive:true});await fs.mkdir(staging,{recursive:true});
for(const [i,sl] of p.slides.items.entries()){
 const png=await p.export({slide:sl,format:'png',scale:1});
 await fs.writeFile(path.join(staging,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidatePath=path.join(staging,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
let finalPath=path.join(out,'project_presentation.pptx');
try { await fs.access(finalPath); finalPath=path.join(out,`project_presentation_${Date.now()}.pptx`); } catch {}
await finalizePresentation({workspaceDir:ROOT,candidatePath,finalPath,pythonExecutable:PYTHON,
 integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','3','--require-native-table-slide','5'],
 explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[2,3,5],requiredNativeChartOwnerSlides:[6],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},
 verifyArtifactToolImport:true,receiptPath:path.join(staging,'validation.json')});
console.log(finalPath);

} catch(error) { console.error(error.stack ?? String(error)); process.exitCode=1; }
