import os, re, json, uuid, shutil, sqlite3, subprocess, threading, time
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)
DB = BASE / 'jobs.db'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
MAX_CLIPS = int(os.getenv('MAX_CLIPS', '8'))

app = FastAPI(title='CLIPGEN AI')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')


def db():
    con = sqlite3.connect(DB)
    con.execute('''CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, progress INTEGER, message TEXT, result TEXT, created REAL)''')
    con.commit()
    return con


def update_job(job_id, status=None, progress=None, message=None, result=None):
    con = db()
    row = con.execute('SELECT status,progress,message,result FROM jobs WHERE id=?', (job_id,)).fetchone()
    if not row:
        con.close(); return
    vals = [status if status is not None else row[0], progress if progress is not None else row[1], message if message is not None else row[2], json.dumps(result) if result is not None else row[3], job_id]
    con.execute('UPDATE jobs SET status=?,progress=?,message=?,result=? WHERE id=?', vals)
    con.commit(); con.close()


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout


def safe_video_url(url):
    host = (urlparse(url).hostname or '').lower()
    allowed = ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be', 'youtube-nocookie.com', 'www.youtube-nocookie.com')
    if host not in allowed:
        raise ValueError('এই MVP-তে আপাতত YouTube ও youtu.be লিংক সাপোর্ট করে।')


def download_video(url, folder):
    safe_video_url(url)
    target = str(folder / 'source.%(ext)s')
    cmd = ['yt-dlp', '--no-playlist', '--no-warnings', '--restrict-filenames', '-f', 'bv*+ba/b', '--merge-output-format', 'mp4', '-o', target, url]
    run(cmd)
    files = list(folder.glob('source.*'))
    files = [f for f in files if f.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')]
    if not files: raise RuntimeError('ভিডিও ডাউনলোড হয়নি। লিংকটি পাবলিক কি না এবং অ্যাক্সেসযোগ্য কি না যাচাই করো।')
    return files[0]


def get_duration(video):
    out = run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(video)])
    return float(out.strip())


def transcribe(video, folder):
    # Extract compressed audio first to keep uploads manageable.
    audio = folder / 'audio.mp3'
    run(['ffmpeg', '-y', '-i', str(video), '-vn', '-ac', '1', '-ar', '16000', '-b:a', '48k', str(audio)])
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(audio, 'rb') as f:
        tr = client.audio.transcriptions.create(model='whisper-1', file=f, response_format='verbose_json', timestamp_granularities=['segment'])
    segments = []
    raw = getattr(tr, 'segments', None) or []
    for s in raw:
        if isinstance(s, dict): segments.append({'start': float(s.get('start', 0)), 'end': float(s.get('end', 0)), 'text': s.get('text', '').strip()})
        else: segments.append({'start': float(s.start), 'end': float(s.end), 'text': s.text.strip()})
    if not segments:
        text = getattr(tr, 'text', '')
        if text: segments = [{'start': 0, 'end': min(30, get_duration(video)), 'text': text}]
    return segments


def choose_highlights(segments, duration):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    transcript = '\n'.join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    prompt = '''Select the most engaging, funny, surprising, or emotionally compelling short-video moments from this transcript. Return ONLY valid JSON: {"clips":[{"start": number,"end": number,"title":"short title","reason":"why this moment works"}]}. Each clip must be 15-60 seconds, within the video duration, avoid overlap where possible, and select only moments supported by the transcript. Aim for up to 6 strong clips; fewer is better than weak clips. If the transcript is not funny, choose the most engaging moments.''' 
    response = client.chat.completions.create(model=os.getenv('AI_MODEL', 'gpt-4o-mini'), temperature=0.2, response_format={'type':'json_object'}, messages=[{'role':'system','content':prompt},{'role':'user','content':f'Video duration: {duration:.1f} seconds\nTranscript:\n{transcript[:90000]}'}])
    data = json.loads(response.choices[0].message.content)
    clips = []
    for c in data.get('clips', [])[:MAX_CLIPS]:
        try:
            start, end = max(0, float(c['start'])), min(duration, float(c['end']))
            if end-start < 15: end = min(duration, start+15)
            if end-start > 60: end = start+60
            if end-start >= 12:
                clips.append({'start': start, 'end': end, 'title': str(c.get('title','Best moment'))[:80], 'reason': str(c.get('reason',''))[:240]})
        except Exception: pass
    if not clips and duration >= 15:
        clips = [{'start': 0, 'end': min(45, duration), 'title':'প্রথম হাইলাইট', 'reason':'AI কোনো নির্দিষ্ট মুহূর্ত শনাক্ত করতে পারেনি; প্রথম অংশ থেকে নমুনা ক্লিপ।'}]
    return clips


def srt_time(seconds):
    ms = int(seconds * 1000); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def write_srt(segments, start, end, path):
    lines=[]; idx=1
    for s in segments:
        a=max(start, s['start']); b=min(end, s['end'])
        if b <= a or not s['text']: continue
        lines += [str(idx), f'{srt_time(a-start)} --> {srt_time(b-start)}', s['text'].replace('-->', '—'), '']; idx += 1
    path.write_text('\n'.join(lines), encoding='utf-8')


def render_clip(video, folder, idx, clip, segments):
    start, end = clip['start'], clip['end']; dur=end-start
    srt=folder/f'clip_{idx}.srt'; write_srt(segments, start, end, srt)
    output=folder/f'clip_{idx}.mp4'
    # Crop to 9:16 around the centre; scale to 1080x1920 and burn captions.
    srt_arg=str(srt).replace('\\','/').replace(':','\\:').replace("'", "\\'")
    vf=f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920, subtitles='{srt_arg}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=110'"
    run(['ffmpeg','-y','-ss',str(start),'-i',str(video),'-t',str(dur),'-vf',vf,'-c:v','libx264','-preset','veryfast','-crf','23','-c:a','aac','-b:a','128k','-movflags','+faststart',str(output)])
    thumb=folder/f'thumb_{idx}.jpg'
    run(['ffmpeg','-y','-ss',str(min(2,dur/2)),'-i',str(output),'-frames:v','1','-q:v','3',str(thumb)])
    return {'title':clip['title'],'reason':clip['reason'],'start':round(start,1),'end':round(end,1),'duration':round(dur,1),'video':f'/files/{folder.name}/{output.name}','thumbnail':f'/files/{folder.name}/{thumb.name}'}


def process_job(job_id, url):
    folder=OUT/job_id; folder.mkdir(parents=True, exist_ok=True)
    try:
        if not OPENAI_API_KEY: raise RuntimeError('AI চালাতে OPENAI_API_KEY প্রয়োজন। সার্ভারে API key সেট করে আবার চেষ্টা করো।')
        update_job(job_id,'processing',5,'ভিডিও লিংক যাচাই করা হচ্ছে…')
        video=download_video(url, folder)
        duration=get_duration(video)
        if duration < 15: raise RuntimeError('ভিডিওটি ১৫ সেকেন্ডের চেয়ে ছোট।')
        update_job(job_id,'processing',20,'ভিডিও থেকে অডিও ও ট্রান্সক্রিপ্ট তৈরি হচ্ছে…')
        segments=transcribe(video, folder)
        update_job(job_id,'processing',55,'AI সেরা মুহূর্তগুলো বেছে নিচ্ছে…')
        clips=choose_highlights(segments, duration)
        results=[]
        for i,c in enumerate(clips,1):
            update_job(job_id,'processing',55+int(40*i/max(1,len(clips))),f'{i}/{len(clips)} নম্বর ক্লিপ তৈরি হচ্ছে…')
            results.append(render_clip(video,folder,i,c,segments))
        update_job(job_id,'completed',100,f'{len(results)}টি ক্লিপ তৈরি হয়েছে।',{'clips':results})
    except Exception as e:
        update_job(job_id,'failed',0,str(e)[:1000])


class CreateJob(BaseModel):
    url: HttpUrl

@app.get('/', response_class=HTMLResponse)
def home():
    return (BASE/'templates'/'index.html').read_text(encoding='utf-8')

@app.post('/api/jobs')
def create_job(body: CreateJob, background: BackgroundTasks):
    job_id=uuid.uuid4().hex[:12]
    con=db(); con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)',(job_id,'queued',0,'কিউতে যোগ হয়েছে…',None,time.time())); con.commit(); con.close()
    background.add_task(process_job,job_id,str(body.url))
    return {'id':job_id}

@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    con=db(); row=con.execute('SELECT status,progress,message,result FROM jobs WHERE id=?',(job_id,)).fetchone(); con.close()
    if not row: raise HTTPException(404,'Job পাওয়া যায়নি')
    return {'id':job_id,'status':row[0],'progress':row[1],'message':row[2],'result':json.loads(row[3]) if row[3] else None}

@app.get('/files/{job_id}/{filename}')
def get_file(job_id: str, filename: str):
    if not re.fullmatch(r'[a-f0-9]{12}',job_id) or not re.fullmatch(r'[A-Za-z0-9_.-]+',filename): raise HTTPException(400,'Invalid path')
    path=OUT/job_id/filename
    if not path.is_file(): raise HTTPException(404,'File পাওয়া যায়নি')
    return FileResponse(path)

@app.get('/health')
def health(): return {'ok':True,'service':'CLIPGEN AI'}
