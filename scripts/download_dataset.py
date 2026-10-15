"""Stream the original GTZAN archive and store lossless FLAC, without an archive copy."""
import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
import requests
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from config import DATA_DIR, GENRES, ROOT, write_json
URL='https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz'
SHA256='b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531'
class HashReader:
    def __init__(self, raw): self.raw, self.sha = raw, hashlib.sha256()
    def read(self, size=-1):
        value=self.raw.read(size)
        self.sha.update(value)
        return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=URL)
    args=parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    receipt=ROOT/'data'/'download.json'
    if receipt.exists() and receipt.read_text() and json.loads(receipt.read_text()).get('complete'):
        expected=json.loads(receipt.read_text()).get('saved',0)
        if len(list(DATA_DIR.glob('*/*.flac')))==expected:
            print('Existing download is complete.'); return
    records=[]
    with requests.get(args.url, stream=True, timeout=(30,120)) as response:
        response.raise_for_status()
        reader=HashReader(response.raw)
        with tarfile.open(fileobj=reader, mode='r|gz') as archive:
            for member in archive:
                path=Path(member.name)
                if path.name.startswith('._') or '__MACOSX' in path.parts: continue
                if not member.isfile() or path.suffix.lower() not in ('.au','.wav') or path.parent.name not in GENRES:
                    continue
                target=DATA_DIR/path.parent.name/(path.stem+'.flac')
                row=dict(source=member.name, target=str(target.relative_to(ROOT)), saved=False)
                try:
                    content=archive.extractfile(member).read()
                    audio,sr=sf.read(io.BytesIO(content),dtype='int16')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(target,audio,sr,subtype='PCM_16',format='FLAC')
                    row['saved']=True
                except Exception as error:
                    row['error']=str(error)
                records.append(row)
                if len(records)%50==0: print(f'Converted {len(records)} recordings',flush=True)
        while reader.read(1024*1024): pass
        sha=reader.sha.hexdigest()
    if args.url==URL and sha!=SHA256:
        raise RuntimeError(f'Archive checksum mismatch: {sha}')
    write_json(receipt,dict(complete=True,url=args.url,archive_sha256=sha,
        saved=sum(r['saved'] for r in records), failed=[r for r in records if not r['saved']],
        storage='PCM16 lossless FLAC converted from the original audio archive',records=records))
    print(f'Done: {sum(r["saved"] for r in records)} saved, {sum(not r["saved"] for r in records)} exclusions.',flush=True)

if __name__=='__main__': main()
