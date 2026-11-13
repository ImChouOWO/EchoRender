"""Serve SR comparisons and training performance: --ip 127.0.0.1 --port 8080."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import uuid

import torch
from PIL import Image
from train import resolve_device
from model.student import EchoRenderSR
from tool.dataloader import read_rgb
from tool.prediction_report import image_array, save_heatmap
from tool.resume import read_checkpoint
from tool.temporal import FORMAT, tiled_forward, autocast, gaussian_blur
from tool.training_config import PROJECT_ROOT, load_training_config, project_path

OUTPUT = PROJECT_ROOT / "output"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def checkpoints():
    paths = []
    for path in OUTPUT.glob("*/best.pt"):
        try:
            history = json.loads(path.with_name("history.json").read_text())
            if isinstance(history, list) and history and history[-1].get("format") == FORMAT:
                paths.append(path)
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True)


def defaults(weights=None):
    cfg = load_training_config()
    paths = checkpoints()
    chosen = project_path(weights) if weights else paths[0] if paths else None
    if chosen:
        cfg = read_checkpoint(chosen)["train_config"]
    settings = cfg["dataset"]["splits"]["val"]
    source = settings["source"]
    lr_root = Path(cfg["dataset"]["root"]) / source / f"{source}_sharp_bicubic"
    if (lr_root / f"X{cfg['dataset']['scale']}").is_dir():
        lr_root = lr_root / f"X{cfg['dataset']['scale']}"
    scenes = settings["scenes"]
    scene = (
        lr_root / scenes[0]
        if scenes
        else next(iter(sorted(lr_root.glob("*"))), lr_root)
    )
    sample = next(iter(sorted(scene.glob("*.png"))), None)
    return {
        "weights": str(chosen) if chosen else "",
        "checkpoints": [str(p) for p in paths],
        "input": str(sample) if sample else "",
        "device": cfg["runtime"]["device"],
        "tile_size": cfg["validation"]["tile_size"],
        "tile_overlap": cfg["validation"]["tile_overlap"],
        "max_frames": 1,
        "mode": "original",
        "strength": 0.25,
        "blur_sigma": cfg["augmentation"]["blur_sigma"],
        "noise_std": cfg["augmentation"]["noise_std"],
    }


def validate_request(request):
    required = {
        "weights",
        "input",
        "device",
        "tile_size",
        "tile_overlap",
        "max_frames",
        "mode",
        "strength",
        "blur_sigma",
        "noise_std",
    }
    if set(request) != required:
        raise ValueError("Incomplete inference settings")
    if request["mode"] not in ("original", "noise", "blur", "combined"):
        raise ValueError("Unknown degradation")
    for name, low, high in [
        ("strength", 0, 1),
        ("noise_std", 0, 1),
        ("blur_sigma", 0, 10),
        ("tile_size", 16, 512),
        ("tile_overlap", 0, 511),
        ("max_frames", 1, 10),
    ]:
        value = request[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not low <= value <= high
        ):
            raise ValueError(f"Invalid {name}")
    for name in ("tile_size", "tile_overlap", "max_frames"):
        if type(request[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if request["tile_overlap"] >= request["tile_size"]:
        raise ValueError("Overlap must be smaller than tile size")
    source = project_path(request["input"])
    paths = (
        [source]
        if source.is_file()
        else sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES)[
            : request["max_frames"]
        ]
    )
    if not paths or any(p.suffix.lower() not in IMAGE_TYPES for p in paths):
        raise ValueError("Select an RGB image or frame directory")
    return paths


class PreviewService:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        self.lock = threading.Lock()
        self.model = None
        self.key = None
        self.tensorboard = None

    def submit(self, request):
        validate_request(request)
        with self.lock:
            if any(
                job["status"] in ("queued", "running") for job in self.jobs.values()
            ):
                raise ValueError("Inference is already running")
            identifier = uuid.uuid4().hex
            self.jobs[identifier] = {"status": "queued"}
            self.pool.submit(self.run, identifier, deepcopy(request))
        return identifier

    @torch.inference_mode()
    def run(self, identifier, request):
        self.jobs[identifier] = {"status": "running", "message": "Loading model"}
        try:
            paths = validate_request(request)
            weights = project_path(request["weights"])
            stamp = (str(weights), weights.stat().st_mtime_ns, request["device"])
            if self.key != stamp:
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                payload = read_checkpoint(weights)
                device = resolve_device(request["device"])
                self.model = EchoRenderSR(payload["model_config"]).to(device).eval()
                self.model.load_state_dict(payload["model"])
                self.config = payload["train_config"]
                self.key = stamp
            model = self.model
            device = next(model.parameters()).device
            torch.set_num_threads(self.config["runtime"]["cpu_threads"])
            directory = OUTPUT / "preview" / identifier
            directory.mkdir(parents=True)
            previous = None
            frames = []
            durations = []
            generator = torch.Generator(device=device).manual_seed(42)
            for index, path in enumerate(paths):
                self.jobs[identifier] = {
                    "status": "running",
                    "message": f"Frame {index + 1}/{len(paths)}",
                }
                rgb = read_rgb(path)[None].to(device)
                if previous is not None and previous.shape != rgb.shape:
                    previous = None
                strength = request["strength"]
                mode = request["mode"]
                if mode in ("blur", "combined"):
                    rgb = gaussian_blur(rgb, request["blur_sigma"] * strength)
                if mode in ("noise", "combined"):
                    rgb = (
                        rgb
                        + torch.randn(rgb.shape, device=device, generator=generator)
                        * request["noise_std"]
                        * strength
                    ).clamp(0, 1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with autocast(model, self.config["training"]["precision"]):
                    predictions = tiled_forward(
                        model,
                        rgb,
                        previous,
                        tile_size=request["tile_size"],
                        overlap=request["tile_overlap"],
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - start)
                previous = rgb.detach()

                def save(name, tensor):
                    filename = f"{index:02d}_{name}.png"
                    Image.fromarray(image_array(tensor)).save(directory / filename)
                    return f"/assets/preview/{identifier}/{filename}"

                up = torch.nn.functional.interpolate(
                    rgb, scale_factor=model.scale, mode="bicubic", align_corners=False
                )
                entry = {
                    "name": path.name,
                    "input": save("input", up),
                    **{k: save(k, v) for k, v in predictions.items()},
                    "seconds": durations[-1],
                }
                # Resolve GT only for an unambiguous official REDS LR path.
                candidate = Path(
                    str(path)
                    .replace("_sharp_bicubic", "_sharp")
                    .replace(f"/X{model.scale}/", "/")
                )
                if candidate != path and candidate.is_file():
                    gt = read_rgb(candidate)[None].to(device)
                    if gt.shape == predictions["output"].shape:
                        entry["gt"] = save("gt", gt)
                        heat = directory / f"{index:02d}_heatmap.png"
                        save_heatmap(heat, predictions["output"], gt)
                        entry["heatmap"] = f"/assets/preview/{identifier}/{heat.name}"
                frames.append(entry)
            result = {
                "status": "done",
                "frames": frames,
                "scale": model.scale,
                "total_seconds": sum(durations),
                "avg_seconds": sum(durations) / len(durations),
                "degradation_space": "LR",
                "mode": request["mode"],
                "strength": strength,
            }
            (directory / "result.json").write_text(json.dumps(result, indent=2))
            self.jobs[identifier] = result
        except Exception as exc:
            self.jobs[identifier] = {"status": "error", "message": str(exc)}

    def tensorboard_port(self):
        if self.tensorboard is None or self.tensorboard.poll() is not None:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.tb_port = sock.getsockname()[1]
            self.tensorboard = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(OUTPUT),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.tb_port),
                    "--path_prefix",
                    "/tensorboard",
                    "--load_fast",
                    "false",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return self.tb_port

    def close(self):
        self.pool.shutdown(wait=True)
        if self.tensorboard:
            self.tensorboard.terminate()
            self.tensorboard.wait(timeout=10)


HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EchoRender Studio</title>
<style>
:root{color-scheme:dark;--bg:#0c111b;--panel:#141d2b;--line:#2b3a50;--text:#e7edf7;--muted:#9eafc6;--accent:#6ce4c7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}header{padding:24px 32px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:20px}h1{font-size:21px;margin:0}header span{color:var(--muted)}nav{margin-left:auto;display:flex;gap:8px}button{border:1px solid var(--line);border-radius:8px;background:#202d40;color:var(--text);min-height:44px;padding:10px 16px;cursor:pointer;font:inherit}button[aria-selected=true],button.primary{background:var(--accent);color:#07271e;font-weight:700}button:disabled{opacity:.45;cursor:wait}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:3px}main{padding:24px 32px}.workspace{display:grid;grid-template-columns:300px minmax(0,1fr);gap:24px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px}h2{font-size:15px;margin:0 0 18px}label{display:block;margin:14px 0 6px;color:var(--muted);font-size:12px}input,select{width:100%;min-height:44px;background:#0c1420;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:10px;font:inherit}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.primary{width:100%;margin-top:20px}.hint{font-size:12px;line-height:1.7;color:var(--muted)}.toolbar{display:flex;align-items:end;gap:16px;flex-wrap:wrap;margin-bottom:18px}.toolbar>div{min-width:130px}.toolbar label{margin-top:0}.status{color:var(--muted);min-height:24px;line-height:1.7;margin:12px 0}.status.error{color:#ffaba9}.compare{position:relative;overflow:hidden;border-radius:8px;background:#070b12;touch-action:pan-y;cursor:ew-resize}.compare img{width:100%;display:block}.compare .before{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}.divider{position:absolute;top:0;bottom:0;left:50%;width:2px;background:white;pointer-events:none}.handle{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:white;color:#102236;border-radius:24px;padding:12px 10px;font-weight:bold;box-shadow:0 0 0 6px #ffffff33}.badges{position:absolute;top:12px;left:12px;right:12px;display:flex;justify-content:space-between;pointer-events:none}.badges span{background:#08101bcc;padding:7px 10px;border-radius:6px;font-weight:600}.empty{min-height:360px;display:grid;place-items:center;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:8px}.hidden,[hidden]{display:none!important}.zoom{max-height:78vh;overflow:auto}.zoom.native .compare{width:max-content}.zoom.native img{width:auto;max-width:none}.heatmap{width:100%;margin-top:16px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}.card{padding:20px;background:var(--panel);border:1px solid var(--line);border-radius:10px}.card strong{display:block;font-size:25px;margin-top:10px}.card small{color:var(--muted)}.charts{display:grid;grid-template-columns:1fr 1fr;gap:20px}.chart{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px}.chart svg{width:100%;height:230px}.chart text{fill:var(--muted);font:12px system-ui}.chart line{stroke:var(--line)}iframe{width:100%;height:78vh;border:1px solid var(--line);border-radius:10px;background:white;margin-top:20px}a{color:var(--accent)}input[type=range]{padding:0;accent-color:var(--accent)}.check{display:flex;align-items:center;gap:8px}.check input{width:18px;min-height:18px}.footer{margin-top:14px;color:var(--muted);font-size:12px}@media(max-width:850px){header{padding:18px;flex-wrap:wrap}header span{display:none}main{padding:16px}.workspace{grid-template-columns:1fr}.cards,.charts{grid-template-columns:1fr 1fr}.panel{padding:16px}.cards{gap:8px}.card{padding:12px}}@media(max-width:520px){.charts{grid-template-columns:1fr}nav{margin-left:0;width:100%}}
</style><header><h1>EchoRender <span>Studio</span></h1><span>RGB super-resolution & restoration</span><nav role="tablist" aria-label="Workspace"><button role="tab" id="previewTab" aria-selected="true" aria-controls="previewPanel">Preview</button><button role="tab" id="trainingTab" aria-selected="false" aria-controls="trainingPanel">Training</button></nav></header>
<main><section id="previewPanel" role="tabpanel" aria-labelledby="previewTab"><div class="workspace"><aside class="panel"><h2>Inference settings</h2><label for="weights">Checkpoint</label><input id="weights" list="checkpointList" placeholder="Path to best.pt"><datalist id="checkpointList"></datalist><label for="input">LR image or frame directory</label><input id="input" placeholder="/path/to/image.png"><label for="upload">Or upload an RGB image</label><input id="upload" type="file" accept="image/png,image/jpeg,image/webp"><div class="pair"><div><label for="device">Device</label><input id="device" value="cuda:1"></div><div><label for="max_frames">Frames (1–10)</label><input id="max_frames" type="number" min="1" max="10" value="1"></div></div><label for="mode">Input degradation</label><select id="mode"><option value="original">Original</option><option value="noise">Noise</option><option value="blur">Blur</option><option value="combined">Blur + noise</option></select><label for="strength">Strength <output id="strengthValue">25%</output></label><input id="strength" type="range" min="0" max="1" step=".01" value=".25"><div class="pair"><div><label for="noise_std">Maximum noise σ</label><input id="noise_std" type="number" min="0" max="1" step=".01"></div><div><label for="blur_sigma">Maximum blur σ</label><input id="blur_sigma" type="number" min="0" max="10" step=".1"></div></div><div class="pair"><div><label for="tile_size">LR tile size</label><input id="tile_size" type="number" min="16" max="512"></div><div><label for="tile_overlap">LR overlap</label><input id="tile_overlap" type="number" min="0"></div></div><button id="run" class="primary">Run inference</button><p class="hint">Preview degradation is applied directly to LR RGB. Strength scales the maximum σ. Training uses matched HR intermediate targets.</p></aside>
<article class="panel"><div class="toolbar"><div><label for="frame">Frame</label><select id="frame" disabled><option>No result</option></select></div><div><label for="stage">Prediction</label><select id="stage"><option value="output">Final · denoised</option><option value="deblur">Stage 2 · deblurred</option><option value="sr">Stage 1 · upscaled</option></select></div><div><label for="reference">Compare against</label><select id="reference"><option value="input">Input · bicubic</option><option value="gt">Ground truth</option></select></div><button id="native">1:1 pixels</button></div><div id="status" class="status" role="status" aria-live="polite">Choose current SR weights and an LR image to begin.</div><div id="empty" class="empty"><div><h2>Inspect the details</h2><p>Drag the divider to compare one input and prediction.</p><p>Noise strengths are shown separately.</p></div></div><div id="result" hidden><div class="zoom" id="zoom"><div class="compare" id="compare"><img id="after" alt="Model prediction"><img class="before" id="before" alt="Reference image"><div class="divider" id="divider"><span class="handle">↔</span></div><div class="badges"><span id="beforeLabel">Input · bicubic</span><span id="afterLabel">Prediction</span></div></div></div><label for="dividerRange">Comparison position</label><input id="dividerRange" type="range" min="0" max="100" value="50"><label class="check"><input type="checkbox" id="showHeat">Show final RGB and gradient error maps (requires GT)</label><img id="heatmap" class="heatmap" hidden alt="Final prediction RGB and gradient errors against ground truth"><p class="footer" id="timing"></p><a id="download" download>Open full-resolution prediction</a></div></article></div></section>
<section id="trainingPanel" role="tabpanel" aria-labelledby="trainingTab" hidden><div class="toolbar"><div><label for="runSelect">Training run</label><select id="runSelect"></select></div><button id="refresh">Refresh</button><button id="openTB">Open TensorBoard</button><span class="hint">PSNR / SSIM ↑ · LPIPS / loss ↓</span></div><div id="trainingStatus" class="status" role="status">No SR runs yet.</div><div id="cards" class="cards"></div><div id="charts" class="charts"></div><iframe id="tensorboard" title="TensorBoard training dashboard" hidden></iframe></section></main>
<script>
const $=id=>document.getElementById(id);let results=null;let runData=[];
async function api(path,options){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data}
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error)}
for(const tab of ['preview','training'])$(tab+'Tab').onclick=()=>{for(const name of ['preview','training']){$(name+'Tab').setAttribute('aria-selected',name===tab);$(name+'Panel').hidden=name!==tab}if(tab==='training')loadRuns()};
$('strength').oninput=()=>$('strengthValue').textContent=Math.round($('strength').value*100)+'%';
function divider(value){$('before').style.clipPath=`inset(0 ${100-value}% 0 0)`;$('divider').style.left=value+'%';$('dividerRange').value=value}
$('dividerRange').oninput=event=>divider(event.target.value);
$('compare').onpointerdown=event=>{if(event.button!==0)return;$('compare').setPointerCapture(event.pointerId);move(event)};
function move(event){if(!event.buttons)return;const rect=$('compare').getBoundingClientRect();divider(Math.max(0,Math.min(100,(event.clientX-rect.left)/rect.width*100)))}$('compare').onpointermove=move;
$('native').onclick=()=>{$('zoom').classList.toggle('native');$('native').textContent=$('zoom').classList.contains('native')?'Fit image':'1:1 pixels'};
function render(){if(!results)return;const frame=results.frames[+$('frame').value];const stage=$('stage').value;let ref=$('reference').value;if(ref==='gt'&&!frame.gt){$('reference').value='input';ref='input';status('No paired GT for this image. Comparing with the upscaled input.')} $('after').src=frame[stage];$('before').src=frame[ref];$('beforeLabel').textContent=ref==='gt'?'Ground truth':'Input · bicubic';$('afterLabel').textContent=$('stage').selectedOptions[0].textContent;$('heatmap').hidden=!$('showHeat').checked||!frame.heatmap;if(frame.heatmap)$('heatmap').src=frame.heatmap;$('showHeat').disabled=!frame.heatmap;$('download').href=frame[stage];$('timing').textContent=`${results.scale}× SR · ${results.mode} ${Math.round(results.strength*100)}% | total infer ${results.total_seconds.toFixed(3)} s | avg infer ${results.avg_seconds.toFixed(3)} s |`;divider(+$('dividerRange').value)}
for(const id of ['frame','stage','reference','showHeat'])$(id).onchange=render;
$('run').onclick=async()=>{try{$('run').disabled=true;status('Preparing inference…');const data={};for(const id of ['weights','input','device','tile_size','tile_overlap','max_frames','mode','strength','blur_sigma','noise_std'])data[id]=['weights','input','device','mode'].includes(id)?$(id).value:+$(id).value;const job=await api('/api/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});while(true){await new Promise(r=>setTimeout(r,700));const state=await api('/api/job/'+job.id);status(state.message||state.status);if(state.status==='error')throw Error(state.message);if(state.status==='done'){results=state;$('frame').replaceChildren(...state.frames.map((f,i)=>new Option(`${i+1} · ${f.name}`,i)));$('frame').disabled=false;$('empty').hidden=true;$('result').hidden=false;status('Inference complete. Drag the divider or use the slider.');render();break}}}catch(error){status(error.message,true)}finally{$('run').disabled=false}};
$('upload').onchange=async()=>{const file=$('upload').files[0];if(!file)return;try{if(file.size>20*1024*1024)throw Error('Upload limit is 20 MB');const encoded=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file)});const value=await api('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data:encoded})});$('input').value=value.path;status('Image uploaded. Ready for inference.')}catch(error){status(error.message,true)}};
async function loadRuns(){try{const runs=await api('/api/runs');const prior=$('runSelect').value;$('runSelect').replaceChildren(...runs.map(r=>new Option(r,r)));if(runs.includes(prior))$('runSelect').value=prior;if(runs.length)await loadHistory();else $('trainingStatus').textContent='No current-format SR history yet. Start training to populate this page.'}catch(error){$('trainingStatus').textContent=error.message}}
function chart(key,title,series){const card=document.createElement('div');card.className='chart';const heading=document.createElement('h2');heading.textContent=title;card.append(heading);const values=series.flatMap(s=>s.values);let min=Math.min(...values),max=Math.max(...values);if(min===max){min-=.01;max+=.01}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 540 250');svg.setAttribute('role','img');svg.setAttribute('aria-label',title+' by epoch');const make=(tag,attrs)=>{const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);svg.append(e);return e};for(let i=0;i<4;i++){const y=20+i*60;make('line',{x1:65,x2:520,y1:y,y2:y});make('text',{x:0,y:y+4}).textContent=(max-i*(max-min)/3).toFixed(4)}for(const s of series){const points=s.values.map((v,i)=>`${65+i*455/Math.max(1,s.values.length-1)},${20+(max-v)/(max-min)*180}`).join(' ');make('polyline',{points,fill:'none',stroke:s.color,'stroke-width':2});for(let i=0;i<s.values.length;i++){const dot=make('circle',{cx:65+i*455/Math.max(1,s.values.length-1),cy:20+(max-s.values[i])/(max-min)*180,r:3,fill:s.color});const tip=document.createElementNS(ns,'title');tip.textContent=`Epoch ${runData[i].epoch}: ${s.values[i].toFixed(5)}`;dot.append(tip)}make('text',{x:70+series.indexOf(s)*160,y:245,fill:s.color}).textContent=s.name}make('text',{x:65,y:223}).textContent='Epoch '+runData[0].epoch;make('text',{x:440,y:223}).textContent=runData.at(-1).epoch;card.append(svg);return card}
async function loadHistory(){try{runData=await api('/api/history?run='+encodeURIComponent($('runSelect').value));if(!runData.length)return;const latest=runData.at(-1);$('trainingStatus').textContent=`Epoch ${latest.epoch} · ${latest.step} optimizer updates · LR ${latest.lr.toExponential(2)} · latest completed epoch`;$('cards').replaceChildren();for(const key of ['loss','psnr','ssim','lpips']){const card=document.createElement('div');card.className='card';const label=document.createElement('small');label.textContent=key.toUpperCase()+(key==='psnr'?' (dB)':'');const value=document.createElement('strong');value.textContent=latest.validation[key].toFixed(key==='psnr'?2:4);card.append(label,value);$('cards').append(card)}$('charts').replaceChildren();for(const key of ['loss','psnr','ssim','lpips']){const series=[{name:'Validation',color:'#6ce4c7',values:runData.map(r=>r.validation[key])}];if(key==='loss')series.push({name:'Training',color:'#91b4ff',values:runData.map(r=>r.train.loss)});$('charts').append(chart(key,key.toUpperCase(),series))}for(const key of ['sr','deblur','denoise','edge'])$('charts').append(chart(key,'Train · '+key,[{name:'Training',color:'#91b4ff',values:runData.map(r=>r.train[key])}]))}catch(error){$('trainingStatus').textContent=error.message}}
$('tensorboard').onload=()=>{$('trainingStatus').textContent='TensorBoard ready · standard event data'};setInterval(()=>{if(!$('trainingPanel').hidden&&$('runSelect').value)loadHistory()},15000);$('refresh').onclick=loadRuns;$('runSelect').onchange=loadHistory;$('openTB').onclick=async()=>{try{await api('/api/tensorboard');$('tensorboard').src='/tensorboard/';$('tensorboard').hidden=false;$('trainingStatus').textContent='TensorBoard is starting; the embedded view will load shortly.'}catch(error){$('trainingStatus').textContent=error.message}};
api('/api/defaults').then(data=>{for(const [key,value]of Object.entries(data)){if($(key))$(key).value=value}for(const path of data.checkpoints){const option=document.createElement('option');option.value=path;$('checkpointList').append(option)}$('strength').oninput();if(!data.weights)status('No compatible SR checkpoint yet. Train the new SR head to create one.')}).catch(error=>status(error.message,true));
</script></html>"""


def make_handler(service, weights=None):
    class Handler(BaseHTTPRequestHandler):
        def send(self, data, status=200, content_type="application/json"):
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/":
                    return self.send(
                        HTML.encode(), content_type="text/html; charset=utf-8"
                    )
                if parsed.path == "/api/defaults":
                    return self.send(defaults(weights))
                if parsed.path == "/api/runs":
                    return self.send(
                        [
                            p.parent.name
                            for p in sorted(
                                OUTPUT.glob("*/history.json"),
                                key=lambda p: p.stat().st_mtime_ns,
                                reverse=True,
                            )
                        ]
                    )
                if parsed.path == "/api/history":
                    from urllib.parse import parse_qs

                    name = parse_qs(parsed.query).get("run", [""])[0]
                    if not name or Path(name).name != name:
                        raise ValueError("Invalid run")
                    return self.send(
                        json.loads((OUTPUT / name / "history.json").read_text())
                    )
                if parsed.path.startswith("/api/job/"):
                    identifier = parsed.path.rsplit("/", 1)[-1]
                    if identifier not in service.jobs:
                        raise ValueError("Unknown inference job")
                    return self.send(service.jobs[identifier])
                if parsed.path == "/api/tensorboard":
                    return self.send({"port": service.tensorboard_port()})
                if parsed.path.startswith("/tensorboard"):
                    return self.proxy_tensorboard()
                if parsed.path.startswith("/assets/"):
                    path = (OUTPUT / unquote(parsed.path[len("/assets/") :])).resolve()
                    if (
                        not path.is_relative_to(OUTPUT.resolve())
                        or path.suffix.lower() not in IMAGE_TYPES
                    ):
                        raise ValueError("Invalid asset")
                    return self.send(
                        path.read_bytes(), content_type=mimetypes.guess_type(path)[0]
                    )
                self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, RuntimeError) as exc:
                self.send({"error": str(exc)}, 400)

        def proxy_tensorboard(self, body=None):
            port = service.tensorboard_port()
            for attempt in range(30):
                try:
                    request = Request(
                        f"http://127.0.0.1:{port}{self.path}",
                        data=body,
                        method=self.command,
                        headers={
                            "Accept-Encoding": "identity",
                            "Content-Type": self.headers.get(
                                "Content-Type", "application/json"
                            ),
                        },
                    )
                    with urlopen(request, timeout=5) as response:
                        return self.send(
                            response.read(),
                            status=response.status,
                            content_type=response.headers.get(
                                "Content-Type", "application/octet-stream"
                            ),
                        )
                except HTTPError as exc:
                    return self.send(
                        exc.read(),
                        status=exc.code,
                        content_type=exc.headers.get(
                            "Content-Type", "application/octet-stream"
                        ),
                    )
                except OSError:
                    if service.tensorboard.poll() is not None:
                        raise RuntimeError("TensorBoard failed to start")
                    time.sleep(0.3)
            raise RuntimeError("TensorBoard is still starting; refresh the view")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length < 0 or length > 30 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                body = self.rfile.read(length)
                if self.path.startswith("/tensorboard"):
                    return self.proxy_tensorboard(body)
                if not body:
                    raise ValueError("Empty request")
                data = json.loads(body)
                if self.path == "/api/infer":
                    return self.send({"id": service.submit(data)}, 202)
                if self.path == "/api/upload":
                    from io import BytesIO

                    raw = base64.b64decode(data["data"], validate=True)
                    with Image.open(BytesIO(raw)) as image:
                        image.load()
                        directory = OUTPUT / "preview/uploads"
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / f"{uuid.uuid4().hex}.png"
                        image.convert("RGB").save(path)
                    return self.send({"path": str(path)})
                self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                self.send({"error": str(exc)}, 400)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--weights")
    args = parser.parse_args(argv)
    service = PreviewService()
    server = ThreadingHTTPServer(
        (args.ip, args.port), make_handler(service, args.weights)
    )
    print(f"Preview: http://{args.ip}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
