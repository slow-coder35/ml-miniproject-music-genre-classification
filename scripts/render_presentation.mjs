import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
try {
 const p=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'deliverables/project_presentation.pptx')));
 for(const [i,slide] of p.slides.items.entries()){
  const png=await p.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(root,`work/deck/final-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 }
 console.log(`Rendered ${p.slides.items.length} final slides.`);
} catch(error) { console.error(error.stack??String(error));process.exitCode=1; }
