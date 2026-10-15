"""Generate the two-page report and one-page alternative from measured results."""
from pathlib import Path
import json
import sys
import html
from collections import Counter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
read=lambda path:json.loads(path.read_text())
evaluation=read(ROOT/'results/evaluation.json')
meta=read(ROOT/'data/cache/feature_meta.json')
inventory=read(ROOT/'data/cache/inventory.json')
download=read(ROOT/'data/download.json')
splits=read(ROOT/'data/cache/splits.json')
team=read(ROOT/'team.json')
rows=[r for r in evaluation['rows'] if not r['tuned']]
selected=evaluation['selected_demo']['experiment']
chosen=next(r for r in evaluation['rows'] if r['experiment']==selected)
best=max(rows,key=lambda r:r['test_macro_f1'])
multi=evaluation['multiwindow']['test']
fonts='/usr/share/fonts/truetype/dejavu/'
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('DejaVu',fonts+'DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',fonts+'DejaVuSans-Bold.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportBody',fontName='DejaVu',fontSize=9.0,leading=13,spaceAfter=6,textColor=colors.HexColor('#20303c')))
styles.add(ParagraphStyle(name='ReportTitle',fontName='DejaVu-Bold',fontSize=22,leading=26,spaceAfter=8,textColor=colors.HexColor('#123d49')))
styles.add(ParagraphStyle(name='ReportHeading',fontName='DejaVu-Bold',fontSize=11,leading=15,spaceBefore=9,spaceAfter=5,textColor=colors.HexColor('#007f8b')))
styles.add(ParagraphStyle(name='SmallReport',fontName='DejaVu',fontSize=7.6,leading=10.5,spaceAfter=5,textColor=colors.HexColor('#53616c')))
def p(text,style='ReportBody'):return Paragraph(text,styles[style])
def head(text):return p(text,'ReportHeading')
def pct(value):return f'{value*100:.1f}%'
def identity():
 members=[f'{m["name"]} ({m["student_id"]})' for m in team['members'] if m['name']]
 return ' / '.join(members) if members else 'Team names, student IDs, and section to be completed before submission'
def table():
 values=[['Model','Input','Train acc.','Val. acc.','Test acc.','Test F1']]
 for row in rows:
  values.append([row['model'].upper(),row['input'],pct(row['train_acc']),pct(row['val_acc']),pct(row['test_acc']),f'{row["test_macro_f1"]:.3f}'])
 t=Table(values,colWidths=[65,49,85,85,85,85],repeatRows=1)
 t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'),('FONTNAME',(0,1),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#123d49')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('TEXTCOLOR',(0,1),(-1,-1),colors.HexColor('#20303c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf4f4')]),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ALIGN',(2,0),(-1,-1),'RIGHT')]))
 return t
usable=sum(r['usable'] for r in inventory['tracks'])
conflicts=sum(not r['usable'] for r in inventory['tracks'])
counts=', '.join(f'{s}: {len(splits[s])}' for s in ('train','val','test'))
rawmel={m:{r['input']:r['test_acc'] for r in rows if r['model']==m} for m in ('svm','knn','ffnn','cnn')}
improved=sum(v['mel']>v['raw'] for v in rawmel.values())
records=read(ROOT/'results/predictions/demo_multiwindow_test.json')
confusions=Counter((r['true_genre'],r['genre']) for r in records if r['true_genre']!=r['genre'])
pairs='; '.join(f'{a} predicted as {b} ({n})' for (a,b),n in confusions.most_common(3)) or 'No test misclassifications in this run'
references='[1] Huang, Serafini &amp; Pugh. Music Genre Classification, Stanford CS229, 2018 (supplied report/poster). [2] Tzanetakis &amp; Cook, Musical Genre Classification of Audio Signals, 2002. [3] Sturm, The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use, arXiv:1306.1461. [4] Dataset mirror: huggingface.co/datasets/marsyas/gtzan.'

def footer(canvas,doc):
 canvas.setFont('DejaVu',7)
 canvas.setFillColor(colors.HexColor('#63747b'))
 canvas.drawString(43,27,'UE24CS352A - Machine Learning | Music Genre Classification')
 canvas.drawRightString(A4[0]-43,27,str(doc.page))

def build(two_pages=True):
 story=[p('Music Genre Classification','ReportTitle'),p(identity(),'SmallReport')]
 story += [head('Problem and objective'),p('Predict one of ten music genres from audio and compare classical classifiers with neural networks. The implementation investigates how raw waveforms and log-mel features affect performance, then provides a local upload demonstration using a validation-selected CNN.')]
 story += [head('Dataset and split'),p(f'GTZAN nominally contains 1,000 approximately 30-second recordings in ten genres. The archive was checksum-verified and stored as lossless PCM16 FLAC. One recording could not be decoded. Auditing excluded {conflicts} decoded recordings with conflicting labels, leaving <b>{usable} usable tracks</b>. A seeded stratified group split keeps exact decoded-audio duplicates together: <b>{counts}</b>. Artist overlap and near duplicates may remain.')]
 story += [head('Features and implementation'),p(f'Audio is mono at 22,050 Hz. Four seeded random two-second windows per training track give <b>{meta["counts"]["train"]} training examples</b>. Validation and the main test use one central window per recording. Raw inputs contain 44,100 values and use per-window standardization. Log-mel inputs are 64 x 173, with a 512-sample FFT, 256-sample hop, power 2 and decibel scaling. Mel statistics, scaling and PCA are fitted only on training data. No DCT is applied.')]
 if two_pages:
  story += [p('SVM and k-NN use standardized flattened features and PCA(15). SVM uses an RBF kernel, C=1 and gamma=scale. k-NN uses ten distance-weighted neighbors. The FFNN has hidden widths 1024, 512, 256, 128 and 64. CNNs use three convolution blocks with 32, 64 and 128 filters, batch normalization, ReLU, pooling, dropout and L2 regularization, then dense layers 256, 64 and ten-class softmax.')]
  story += [p('Neural models trained on the local NVIDIA RTX 4060 using TensorFlow. Adam uses learning rate 0.001 and batch size 32, with up to 50 epochs and validation-loss early stopping (patience 8). Two additional mel-CNN trials vary learning rate or dropout. Validation macro-F1 selects the demo checkpoint, with validation loss breaking ties. Frozen test evaluation runs separately from training.')]
 if two_pages:
  story += [Image(str(ROOT/'results/figures/comparison.png'),width=500,height=182),PageBreak(),p('Results, demonstration and conclusions','ReportTitle')]
 story += [head('Eight-model comparison'),table(),p('Accuracy is a fraction of correct predictions. Macro-F1 weights all ten genres equally. Test columns use one central two-second window per held-out recording. Training accuracy uses augmented training windows.','SmallReport')]
 if two_pages:
  story += [p(f'The highest baseline test macro-F1 belongs to <b>{best["experiment"]}</b>: accuracy <b>{pct(best["test_acc"])}</b> and macro-F1 <b>{best["test_macro_f1"]:.3f}</b>. Mel input improves test accuracy for {improved} of the four model families in this run. This is a measured result on the saved split, not a claim that the source paper was reproduced exactly.')]
  story += [head('Upload demonstration'),p(f'The local Streamlit application accepts WAV or FLAC music, shows playback, waveform and a spectrogram, and averages CNN scores from five evenly spaced two-second windows in up to 30 seconds. The selected model is <b>{selected}</b>. Its separate five-window test evaluation gives <b>{pct(multi["accuracy"])}</b> accuracy and <b>{multi["macro_f1"]:.3f}</b> macro-F1 on {multi["count"]} recordings. Model scores are uncalibrated and can be overconfident.')]
  story += [head('Error analysis and limitations'),p(f'Frequent five-window errors include {html.escape(pairs)}. Genre boundaries overlap, and short excerpts omit longer musical structure. GTZAN has known dataset faults, limited historical coverage and possible artist overlap across partitions. With only 100 test tracks, each additional correct prediction changes accuracy by one percentage point. Published accuracy from a different split and protocol is not a direct comparison.')]
  story += [head('Conclusion'),p(f'The project delivers eight reproducible baseline experiments, bounded CNN tuning, auditable predictions and a saved-model application. On this split, {best["experiment"]} has the strongest baseline macro-F1. Evaluation separates central-window classification from the multi-window user workflow. Further work should prioritize artist-disjoint evaluation, larger datasets and calibrated uncertainty.')]
  story += [head('Reproducibility and contributions'),p('The repository includes dependency versions, split/window manifests, preprocessing metadata, training histories, GPU execution evidence and per-recording predictions. Team members must complete the verified contribution record. Implementation and document preparation used AI assistance, extending the supplied local starter code.','SmallReport')]
 else:
  story += [head('Results and conclusion'),p(f'The strongest baseline test macro-F1 is {best["test_macro_f1"]:.3f} ({best["experiment"]}, accuracy {pct(best["test_acc"])}). Mel input improves accuracy in {improved}/4 families. The validation-selected app model ({selected}) averages five windows and reaches {pct(multi["accuracy"])} accuracy and {multi["macro_f1"]:.3f} macro-F1 in its separate track-level test. GTZAN faults, artist overlap and only 100 test recordings limit generalization. Code, saved models and prediction records support reproduction. AI assistance extended the starter implementation; students must verify contributions.')]
 story += [head('References'),p(references,'SmallReport')]
 path=OUT/('project_report_two_pages.pdf' if two_pages else 'project_summary_one_page.pdf')
 doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=36,bottomMargin=40,title='Music Genre Classification',author='Project team')
 doc.build(story,onFirstPage=footer,onLaterPages=footer)
 return path
if __name__=='__main__':
 for flag in (True,False):print(build(flag))
