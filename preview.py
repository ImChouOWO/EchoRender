"""Serve an interactive Student/Teacher preview: python preview.py --ip 0.0.0.0 --port 8080."""
import argparse
import base64
from copy import deepcopy
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import pickle
import threading
from time import perf_counter
from urllib.parse import unquote, urlsplit
import uuid

import numpy as np
import torch

from model.student import EchoRenderStudent
from model.teacher import EchoRenderTeacher, resolve_device
from teacher_infer import compose_output, dataset_motion, resample
from tool.teacher_pipeline import load_rgb, save_rgb
from tool.temporal import (FORMAT, PROFILES, compose_tensor, prepare_frame,
                           prepare_student_frame, reset_history, student_inputs, student_previous)
from tool.training_config import PROJECT_ROOT, load_training_config

OUTPUT = PROJECT_ROOT / 'output/preview'
IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
CONTROL_RANGES = {'style_index': (0, 2), 'tone': (0, 2), 'structure': (0, 2),
                  'skin_structure': (-1, 2), 'auto_structure': (-1, 2),
                  'intensity': (0, 2), 'detail_strength': (0, 2),
                  'colour_strength': (0, 2), 'detail_radius': (.1, 64)}


def project_path(value):
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve()


def checkpoint(path):
    try:
        payload = torch.load(project_path(path), map_location='cpu', weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f'Cannot load Student checkpoint: {exc}') from exc
    required = {'student', 'student_config', 'train_config', 'epoch', 'teacher_blend_scale'}
    if not isinstance(payload, dict) or payload.get('format') != FORMAT or not required <= payload.keys():
        raise ValueError(f'Expected a complete {FORMAT} checkpoint')
    return payload


def defaults(weights=None):
    paths = sorted((PROJECT_ROOT / 'output').glob('*/best.pt'),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    selected = project_path(weights) if weights else (paths[0] if paths else None)
    payload = checkpoint(selected) if selected else None
    config = payload['train_config'] if payload else load_training_config()
    profiles = config['training']['profiles']
    settings = deepcopy(config['rendering']['profiles'])
    for name in PROFILES:
        if name not in settings:
            settings[name] = {**settings[profiles[0]], 'style_index': None, 'tone': None}
    root = Path(config['dataset']['root'])
    if config['dataset'].get('type') == 'reds':
        base = root / 'val/val_sharp'
        scenes = config['dataset']['splits']['val']['scenes']
        source = base / scenes[0] if scenes else next(iter(sorted(p for p in base.glob('*') if p.is_dir())), base)
    else:
        source = root / 'training/clean' / config['dataset']['splits']['val']['scenes'][0]
    return {'weights': str(selected) if selected else '', 'checkpoints': [str(p) for p in paths],
            'epoch': payload['epoch'] if payload else None, 'trained_profiles': profiles,
            'profile': profiles[0], 'profiles': settings,
            'teacher_weights': config['model']['teacher_weights'],
            'input': str(source),
            'device': config['runtime']['device'], 'adapter_size': config['training']['adapter_size'],
            'rendering': {k: v for k, v in config['rendering'].items() if k != 'profiles'}}


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and numeric')
    if not low <= value <= high or integer and type(value) is not int:
        raise ValueError(f'{name} must be within [{low}, {high}]' + (' and an integer' if integer else ''))
    return value


def rendering_config(payload, request):
    config = deepcopy(payload['train_config'])
    profile = request['profile']
    if profile not in PROFILES:
        raise ValueError('Unknown profile')
    if type(request['temporal']) is not bool:
        raise ValueError('temporal must be boolean')
    size = number(request['adapter_size'], 'adapter_size', 320, 4096, integer=True)
    if size % 64:
        raise ValueError('adapter_size must be a multiple of 64')
    config['training']['adapter_size'] = size
    settings = request['controls']
    if set(settings) != set(CONTROL_RANGES):
        raise ValueError('Incomplete profile controls')
    for key, (low, high) in CONTROL_RANGES.items():
        if settings[key] is None and key in ('style_index', 'tone', 'structure', 'skin_structure', 'auto_structure'):
            continue
        number(settings[key], key, low, high, integer=key == 'style_index')
    config['rendering']['profiles'][profile] = dict(settings)
    config['rendering']['temporal'] = request['temporal']
    config['rendering']['max_frames'] = number(request['max_frames'], 'max_frames', 1, 10, integer=True)
    for key, low, high, integer in (('noise_index', 0, 2**31 - 1, True),
                                    ('history_strength', 0, 1, False),
                                    ('scene_cut_threshold', 0, 1, False)):
        config['rendering'][key] = number(request[key], key, low, high, integer)
    return config


def input_paths(request, directory):
    maximum = number(request['max_frames'], 'max_frames', 1, 10, integer=True)
    uploads = request.get('uploads', [])
    if uploads:
        if not isinstance(uploads, list) or len(uploads) > 10:
            raise ValueError('Upload between one and ten ordered frames')
        paths = []
        for index, item in enumerate(uploads[:maximum]):
            suffix = Path(item['name']).suffix.lower()
            if suffix not in IMAGE_TYPES:
                raise ValueError('Unsupported image type')
            path = directory / f'upload_{index + 1:02d}{suffix}'
            path.write_bytes(base64.b64decode(item['data'], validate=True))
            paths.append(path)
        return paths
    source = project_path(request['input'])
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES) if source.is_dir() else [source]
    if not paths or any(not p.is_file() or p.suffix.lower() not in IMAGE_TYPES for p in paths[:maximum]):
        raise ValueError('Input must be an image or a folder containing ordered image frames')
    return paths[:maximum]


def release(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_preview(request, update):
    payload = checkpoint(request['weights'])
    config = rendering_config(payload, request)
    device = resolve_device(request['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(config['runtime']['cpu_threads'])
    teacher_weights = project_path(request['teacher_weights'])
    if not teacher_weights.is_relative_to((PROJECT_ROOT / 'weights').resolve()) or not teacher_weights.is_file():
        raise ValueError('Teacher weights must exist in the project weights folder')
    directory = OUTPUT / uuid.uuid4().hex
    directory.mkdir(parents=True)
    paths = input_paths(request, directory)
    images = [load_rgb(path) for path in paths]
    profile = request['profile']
    rows = []
    for index, (path, rgb) in enumerate(zip(paths, images)):
        filename = f'{index + 1:02d}_input.png'
        save_rgb(rgb, directory / filename)
        name = request['uploads'][index]['name'] if request.get('uploads') else path.name
        rows.append({'name': name, 'original': f'/assets/{directory.name}/{filename}'})
    # Each model owns its recurrent neural history; unload one before loading the other.
    for name in ('student', 'teacher'):
        model = None
        history = previous = previous_student = frame = head = inputs = value = None
        try:
            update(message=f'Loading {name.title()}')
            if name == 'student':
                model = EchoRenderStudent(payload['student_config'], adapter_size=config['training']['adapter_size'])
                model.load_state_dict(payload['student'], strict=True)
            else:
                model = EchoRenderTeacher(teacher_weights, device='cpu')
            model.eval().requires_grad_(False).to(device)
            for index, (path, rgb) in enumerate(zip(paths, images)):
                if reset_history(previous, rgb, config):
                    history = previous_student = None
                synchronize(device)
                started = perf_counter()
                with torch.inference_mode():
                    if name == 'student':
                        frame = prepare_student_frame(rgb, config, profile, model,
                                                      previous=previous_student, history=history)
                        inputs = student_inputs(frame, device, model.layout)
                        head = model(inputs)
                        if model.layout == 'BCHW':
                            head = head.permute(0, 2, 3, 1)
                        value, next_history = compose_tensor(head, frame, payload['teacher_blend_scale'])
                        prediction = value[0].permute(1, 2, 0).cpu().numpy()
                        previous_student = student_previous(frame, model)
                    else:
                        motion = confidence = None
                        motion_source = 'none'
                        if history is not None and config['rendering']['temporal']:
                            root = Path(config['dataset']['root'])
                            relative = path.relative_to(root) if path.is_relative_to(root) else None
                            if (relative is not None and len(relative.parts) == 4
                                    and relative.parts[1] == 'clean' and path.parent == paths[index - 1].parent):
                                motion, confidence, _ = dataset_motion(root, relative.parts[0], relative.parts[2],
                                                                       paths[index - 1], path, rgb.shape[:2])
                                motion_source = 'sintel'
                            else:
                                from model.motion import raw_motion
                                motion, confidence = raw_motion(previous, rgb)
                                motion_source = 'rgb-estimated'
                        frame = prepare_frame(rgb, config, profile, history=history, motion=motion, confidence=confidence)
                        head = model(frame.inputs.to(device))
                        h, w = frame.small.shape[:2]
                        raw = resample(head[0, :h, :w].cpu().numpy(), rgb.shape[1], rgb.shape[0])
                        prediction, next_history = compose_output(raw, rgb, frame.controls,
                            blend_scale=float(model.blend_scale), history=frame.history, confidence=frame.confidence,
                            history_strength=frame.rendering['history_strength'], **{key: frame.settings[key] for key in
                            ('intensity', 'detail_strength', 'colour_strength', 'detail_radius')})
                        rows[index]['teacher_motion'] = motion_source
                    synchronize(device)
                    seconds = perf_counter() - started
                    if prediction.shape != rgb.shape or not np.isfinite(prediction).all():
                        raise RuntimeError('Non-finite prediction or unexpected output dimensions')
                    filename = f'{index + 1:02d}_{name}.png'
                    save_rgb(prediction, directory / filename)
                    rows[index][name] = f'/assets/{directory.name}/{filename}'
                    rows[index][name + '_seconds'] = seconds
                    rows[index][name + '_history'] = frame.history is not None
                    history = next_history if config['rendering']['temporal'] else None
                    previous = rgb
                    frame = head = inputs = value = next_history = None
                update(message=f'{name.title()} {index + 1}/{len(paths)}',
                       completed=(len(paths) if name == 'teacher' else 0) + index + 1, total=2 * len(paths))
        finally:
            model = history = previous_student = frame = head = inputs = value = next_history = None
            release(device)
    result = {'frames': rows, 'epoch': payload['epoch'], 'weights': str(project_path(request['weights'])),
              'profile': profile, 'adapter_size': config['training']['adapter_size'], 'device': str(device),
              'trained_profiles': payload['train_config']['training']['profiles'],
              'output': str(directory), 'settings': config['rendering'],
              'timing': 'Inference includes preparation, motion, forward and composition; excludes model loading and PNG writing.'}
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


class PreviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, Handler)
        self.lock = threading.Lock()
        self.state = {'status': 'idle', 'message': 'Ready', 'completed': 0, 'total': 1}

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start_job(self, request):
        with self.lock:
            if self.state['status'] == 'running':
                raise ValueError('A preview is already running')
            self.state = {'status': 'running', 'message': 'Preparing inputs', 'completed': 0, 'total': 1}
        def work():
            try:
                self.update(result=run_preview(request, self.update), status='done', message='Preview ready')
            except Exception as exc:
                self.update(status='error', message=f'{type(exc).__name__}: {exc}')
                print(f'[Preview]: {type(exc).__name__}: {exc}', flush=True)
        threading.Thread(target=work, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def send(self, data, content_type='application/json', status=200):
        body = json.dumps(data).encode() if content_type == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/':
                self.send(HTML.encode(), 'text/html; charset=utf-8')
            elif path == '/api/defaults':
                self.send(defaults())
            elif path == '/api/status':
                with self.server.lock:
                    state = dict(self.server.state)
                self.send(state)
            elif path.startswith('/assets/'):
                file = (OUTPUT / path.removeprefix('/assets/')).resolve()
                if not file.is_relative_to(OUTPUT.resolve()) or not file.is_file() or file.suffix != '.png':
                    raise FileNotFoundError('Image not found')
                self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0])
            else:
                raise FileNotFoundError('Not found')
        except (OSError, ValueError, RuntimeError) as exc:
            self.send({'error': str(exc)}, status=404 if isinstance(exc, FileNotFoundError) else 400)

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                raise ValueError('Cross-origin requests are not allowed')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('Expected application/json')
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 120 * 1024 * 1024:
                raise ValueError('Request must be under 120 MiB')
            request = json.loads(self.rfile.read(length))
            if self.path == '/api/settings':
                self.send(defaults(request['weights']))
            elif self.path == '/api/run':
                self.server.start_job(request)
                self.send({'status': 'running'}, status=202)
            else:
                self.send({'error': 'Not found'}, status=404)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            self.send({'error': str(exc)}, status=400)

    def log_message(self, format, *args):
        if self.path != '/api/status':
            super().log_message(format, *args)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EchoRender | Preview</title><style>
:root{color-scheme:dark;--bg:#101217;--surface:#191d25;--text:#eef0f5;--muted:#9ba4b4;--border:#2e3440;--accent:#a7edce;--accent-ink:#132b21;--focus:#8fb7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{width:min(1200px,100%);margin:0 auto;padding:28px 30px 28px}.brand{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:25px}.brand-name{font-weight:600;letter-spacing:-.3px}.brand-name span{font-weight:400;color:var(--muted);margin-left:6px}.badge{color:var(--muted);border:1px solid var(--border);border-radius:5px;padding:3px 8px;font-size:11px;letter-spacing:.3px}.title-row{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px;flex-wrap:wrap}h1{font-size:30px;line-height:1.2;letter-spacing:-.8px;margin:0 0 9px;font-weight:600}h2{font-size:17px;font-weight:600;margin:0;letter-spacing:-.2px}.subtitle{margin:0;color:var(--muted)}.controls{display:flex;align-items:center;gap:18px;flex-wrap:wrap}button{font:inherit;cursor:pointer;color:var(--muted);background:transparent;border:1px solid var(--border);border-radius:6px;padding:8px 13px;min-height:38px}button:hover{color:var(--text);background:#242a35}button:focus-visible{outline:2px solid var(--focus);outline-offset:4px}.sync-control{display:flex;align-items:center;gap:9px;cursor:pointer;min-height:44px}.sync-control input{appearance:none;-webkit-appearance:none;flex-shrink:0;width:35px;height:21px;border:1px solid var(--border);border-radius:999px;background:var(--surface);position:relative;margin:0;cursor:pointer}.sync-control input::before{content:'';position:absolute;left:3px;top:3px;width:13px;height:13px;border-radius:50%;background:var(--muted);transition:transform .15s,background .15s}.sync-control input:checked{background:var(--accent);border-color:var(--accent)}.sync-control input:checked::before{background:var(--accent-ink);transform:translateX(14px)}.sync-control input:focus-visible{outline:2px solid var(--focus);outline-offset:4px}.context{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding-bottom:18px;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted)}.filename{color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.sync-status{color:var(--muted);font-size:12px}.comparison{margin-top:24px}.comparison+.comparison{margin-top:35px}.section-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:11px}.section-meta{font-size:12px;color:var(--muted)}
.stage{--position:50%;position:relative;aspect-ratio:1024/436;width:100%;overflow:hidden;border-radius:10px;isolation:isolate;background:var(--surface);cursor:ew-resize;touch-action:pan-y;user-select:none;outline:1px solid var(--border);outline-offset:0}.stage.dragging{touch-action:none}.stage:focus-within{outline:3px solid var(--focus);outline-offset:4px}.stage img{position:absolute;inset:0;width:100%;height:100%;display:block;object-fit:contain;pointer-events:none}.prediction{clip-path:inset(0 0 0 var(--position))}.label{position:absolute;z-index:2;top:17px;padding:5px 10px;border:1px solid #ffffff35;border-radius:5px;background:#14171ed9;color:#fff;font-size:12px;pointer-events:none;box-shadow:0 2px 6px #0002}.label.before{left:17px}.label.after{right:17px}.divider{position:absolute;top:0;bottom:0;left:var(--position);width:2px;transform:translateX(-1px);background:var(--accent);box-shadow:0 0 0 1px #132b2130;z-index:3;pointer-events:none}.handle{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:48px;height:48px;display:flex;align-items:center;justify-content:center;gap:10px;border-radius:50%;background:var(--accent);color:var(--accent-ink);box-shadow:0 4px 18px #0005;font-size:25px;line-height:1;transition:box-shadow .15s}.stage.dragging .handle{box-shadow:0 0 0 7px #a7edce20,0 4px 18px #0005}.keyboard-range{position:absolute;inset:0;opacity:0;pointer-events:none;width:100%;height:100%;margin:0}.hint-row{margin-top:13px;display:flex;align-items:center;justify-content:space-between;gap:14px;color:var(--muted);font-size:12px;flex-wrap:wrap}.hint{display:flex;align-items:center;gap:7px}.position{display:flex;gap:9px;align-items:center;font-variant-numeric:tabular-nums}.position strong{color:var(--text);font-weight:500;min-width:37px;text-align:right}.keys{font-size:11px;padding:2px 5px;border:1px solid var(--border);border-radius:4px;color:var(--muted)}.info{margin-top:28px;padding-top:18px;border-top:1px solid var(--border);display:flex;align-items:start;justify-content:space-between;gap:22px;font-size:12px;color:var(--muted);flex-wrap:wrap}.note{max-width:490px;margin:0}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:650px){main{padding:20px 16px}.brand{margin-bottom:24px}h1{font-size:26px}.title-row{gap:14px;margin-bottom:15px}.controls{width:100%;justify-content:space-between}.stage{border-radius:7px}.label{top:10px;font-size:11px;padding:3px 6px}.label.before{left:10px}.label.after{right:10px}.handle{width:42px;height:42px;gap:7px;font-size:22px}.section-head{align-items:start}.section-meta{font-size:11px;text-align:right}.hint-row{font-size:11px}.comparison+.comparison{margin-top:26px}.note{max-width:none}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
.frame-control{display:inline-flex;align-items:center;gap:8px}.frame-control select{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 9px}.frame-control select:focus-visible{outline:2px solid var(--focus);outline-offset:3px}.target-settings{margin-top:18px;padding-top:14px;border-top:1px solid var(--border);color:var(--muted);font-size:12px}.target-settings summary{cursor:pointer}.target-settings p{line-height:1.8;overflow-wrap:anywhere}
.panel{padding:20px;border:1px solid var(--border);border-radius:10px;margin:22px 0;background:var(--surface)}
.fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:16px 0}.fields label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:12px}.wide{grid-column:1/-1}.fields input,.fields select{font:inherit;min-height:38px;width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text)}
.fields input:focus-visible,.fields select:focus-visible{outline:2px solid var(--focus)}.actions{display:flex;gap:14px;align-items:center;flex-wrap:wrap}#run{background:var(--accent);color:var(--accent-ink);font-weight:600}button:disabled{opacity:.5;cursor:wait}progress{accent-color:var(--accent);width:180px}#status{overflow-wrap:anywhere}.empty{padding:60px 20px;text-align:center;color:var(--muted);border:1px dashed var(--border);border-radius:10px}#results[hidden],#empty[hidden]{display:none}.error{color:#ff9999}details summary{cursor:pointer}#settings-note{color:var(--muted);font-size:12px;margin:12px 0 0;overflow-wrap:anywhere}@media(max-width:650px){.fields{grid-template-columns:1fr 1fr}.panel{padding:14px}.section-head{flex-wrap:wrap}}
</style></head><body><main>
<header class="brand"><div class="brand-name">EchoRender <span>/ Preview</span></div><span class="badge">RGB COMPARISON</span></header>
<div class="title-row"><div><h1>Prediction comparison</h1><p class="subtitle">Choose weights and images, then compare Student and Teacher.</p></div><div class="controls"><label class="sync-control"><input id="sync-sliders" type="checkbox" checked><span>Sync sliders</span></label><button id="reset-button" aria-label="Reset both dividers to the middle">Reset</button></div></div>
<form class="panel" id="preview-form"><h2>Controls</h2><div class="fields">
<label class="wide">Student weights<input id="weights" list="checkpoints" required placeholder="output/neutral_mode/best.pt"><datalist id="checkpoints"></datalist></label>
<label class="wide">Input image or frame folder<input id="input" placeholder="Server path, relative to the project root"></label>
<label class="wide">Upload images<input id="uploads" type="file" accept="image/*" multiple><span>Optional. Uploads replace the input path. Files are sorted by name; maximum 10 frames.</span></label>
<label>Profile<select id="profile"><option>standard</option><option>natural</option><option>cinematic</option><option>neutral</option></select></label>
<label>Device<input id="device" value="cuda:1" required></label><label>Frames<input id="max_frames" type="number" min="1" max="10" step="1" value="10" required></label>
<label>Temporal<select id="temporal"><option value="true">On</option><option value="false">Off</option></select></label><label>Adapter size<input id="adapter_size" type="number" min="320" max="4096" step="64" value="320" required></label>
<label>Noise index<input id="noise_index" type="number" min="0" step="1" value="1" required></label>
</div><details><summary>Effect settings</summary><div class="fields" id="effect-fields"></div>
<div class="fields"><label class="wide">Teacher weights<input id="teacher_weights" required></label><label>History strength<input id="history_strength" type="number" min="0" max="1" step="0.05" required></label><label>Scene cut threshold<input id="scene_cut_threshold" type="number" min="0" max="1" step="0.01" required></label></div></details>
<p id="settings-note"></p><p class="subtitle">Blank effect values follow the profile. Structure controls and internal noise affect Teacher only; Student uses RGB, history, style and tone. Untrained profiles may differ from Teacher. Sintel sequences use dataset flow; other sequences use RGB-estimated Teacher motion.</p>
<div class="actions" style="margin-top:16px"><button id="run" type="submit" disabled>Run preview</button><button id="load-settings" type="button">Load checkpoint settings</button><progress id="progress" value="0" max="1"></progress><span id="status" role="status" aria-live="polite">Loading settings...</span></div></form>
<div id="empty" class="empty">Your comparison will appear here.</div><div id="results" hidden>
<div class="context"><span class="filename" id="frame-name"></span><label class="frame-control">Frame <select id="frame-select"></select></label><span id="history-status"></span><span id="sync-status">Move either divider to adjust both comparisons.</span></div>
<section class="comparison"><div class="section-head"><h2>Student</h2><span class="section-meta" id="student-meta"></span></div>
<div class="stage" id="student-stage" aria-label="Original input and Student comparison"><img data-source="original" alt="Original input" draggable="false"><img class="prediction" data-source="student" alt="Student prediction" draggable="false"><span class="label before">Original input</span><span class="label after">Student prediction</span><div class="divider" aria-hidden="true"><div class="handle"><span>‹</span><span>›</span></div></div><input class="keyboard-range" id="student-divider" type="range" min="0" max="100" step="1" value="50" aria-label="Original image visible in Student comparison"></div>
<div class="hint-row"><span>Drag the divider · ← → to fine-tune</span><div class="position"><span>Original</span><strong id="student-original-percent">50%</strong><span>/</span><strong id="student-prediction-percent">50%</strong><span>Student</span></div></div></section><section class="comparison"><div class="section-head"><h2>Teacher</h2><span class="section-meta" id="teacher-meta"></span></div>
<div class="stage" id="teacher-stage" aria-label="Original input and Teacher comparison"><img data-source="original" alt="Original input" draggable="false"><img class="prediction" data-source="teacher" alt="Teacher prediction" draggable="false"><span class="label before">Original input</span><span class="label after">Teacher prediction</span><div class="divider" aria-hidden="true"><div class="handle"><span>‹</span><span>›</span></div></div><input class="keyboard-range" id="teacher-divider" type="range" min="0" max="100" step="1" value="50" aria-label="Original image visible in Teacher comparison"></div>
<div class="hint-row"><span>Drag the divider · ← → to fine-tune</span><div class="position"><span>Original</span><strong id="teacher-original-percent">50%</strong><span>/</span><strong id="teacher-prediction-percent">50%</strong><span>Teacher</span></div></div></section>
<footer class="info"><span id="output-note"></span><p class="note">Move left for more prediction, right for more input. Each model uses its own prediction history. Times include preprocessing, motion, forward and composition; exclude loading and PNG writing.</p></footer></div>
</main><script>
'use strict';
const $=id=>document.getElementById(id);
const controlSpecs={style_index:['Style index',0,2,1],tone:['Tone',0,2,.1],structure:['Structure',0,2,.1],skin_structure:['Skin structure',-1,2,.1],auto_structure:['Auto structure',-1,2,.1],intensity:['Intensity',0,2,.1],detail_strength:['Detail strength',0,2,.1],colour_strength:['Colour strength',0,2,.1],detail_radius:['Detail radius',.1,64,.1]};
for(const [key,[label,min,max,step]] of Object.entries(controlSpecs)){
 const field=document.createElement('label');field.textContent=label;
 const input=document.createElement('input');Object.assign(input,{id:key,type:'number',min,max,step,placeholder:'Profile default'});
 if(['intensity','detail_strength','colour_strength','detail_radius'].includes(key))input.required=true;
 field.append(input);$('effect-fields').append(field);
}
let profiles={},assets=[],result=null;
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data;}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function profileSettings(){const settings=profiles[$('profile').value];for(const key of Object.keys(controlSpecs))$(key).value=settings[key]??'';}
function applyDefaults(data){
 profiles=data.profiles;
 for(const key of ['weights','input','teacher_weights','device','adapter_size','profile'])$(key).value=data[key];
 for(const key of ['max_frames','noise_index','history_strength','scene_cut_threshold','temporal'])$(key).value=String(data.rendering[key]);
 $('checkpoints').replaceChildren(...data.checkpoints.map(path=>{const option=document.createElement('option');option.value=path;return option;}));
 profileSettings();$('settings-note').textContent=`Checkpoint epoch ${data.epoch??'—'} · trained profiles: ${data.trained_profiles.join(', ')} · changing weights requires loading their settings.`;
 $('run').disabled=false;status('Ready');
}
$('profile').addEventListener('change',profileSettings);
$('load-settings').addEventListener('click',async()=>{try{status('Loading checkpoint settings...');applyDefaults(await api('/api/settings',{weights:$('weights').value}));}catch(error){status(error.message,true);}});
function showFrame(){
 const frame=assets[Number($('frame-select').value)];if(!frame)return;
 for(const image of document.querySelectorAll('img[data-source]'))image.src=frame[image.dataset.source];
 for(const stage of document.querySelectorAll('.stage')){const image=stage.querySelector('img');image.onload=()=>stage.style.aspectRatio=`${image.naturalWidth}/${image.naturalHeight}`;}
 $('frame-name').textContent=frame.name;
 $('history-status').textContent=`History: Student ${frame.student_history?'on':'off'} · Teacher ${frame.teacher_history?'on':'off'}`;
 for(const name of ['student','teacher']){const total=assets.reduce((sum,f)=>sum+f[name+'_seconds'],0);$(name+'-meta').textContent=`${name==='student'?'Epoch '+result.epoch+' · ':''}${result.profile} · total infer ${total.toFixed(3)} s · avg infer ${(total/assets.length).toFixed(3)} s`;}
}
$('frame-select').addEventListener('change',showFrame);
function showResult(data){
 result=data;assets=data.frames;
 $('frame-select').replaceChildren(...assets.map((frame,index)=>{const option=document.createElement('option');option.value=index;option.textContent=`Frame ${index+1}`;return option;}));
 $('frame-select').value=assets.length-1;$('results').hidden=false;$('empty').hidden=true;
 $('output-note').textContent=`${data.adapter_size} × ${data.adapter_size} inference → native-size output · ${data.device} · ${data.output}`;
 showFrame();
}
async function poll(){
 try{const state=await api('/api/status');$('progress').max=state.total;$('progress').value=state.completed;status(state.message,state.status==='error');
 if(state.status==='running'){setTimeout(poll,700);return;}
 $('run').disabled=false;$('load-settings').disabled=false;
 if(state.status==='done')showResult(state.result);
 }catch(error){status(error.message,true);$('run').disabled=false;$('load-settings').disabled=false;}
}
async function upload(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve({name:file.name,data:reader.result.split(',')[1]});reader.onerror=()=>reject(Error('Cannot read upload'));reader.readAsDataURL(file);});}
$('preview-form').addEventListener('submit',async event=>{
 event.preventDefault();$('run').disabled=true;$('load-settings').disabled=true;
 try{
 const files=[...$('uploads').files].sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));
 if(files.length>10)throw Error('Choose at most 10 frames');
 if(files.reduce((sum,file)=>sum+file.size,0)>80*1024*1024)throw Error('Uploads must total under 80 MiB');
 const body={controls:{},uploads:await Promise.all(files.map(upload))};
 for(const key of ['weights','input','teacher_weights','device','profile'])body[key]=$(key).value.trim();
 for(const key of ['adapter_size','max_frames','noise_index','history_strength','scene_cut_threshold'])body[key]=Number($(key).value);
 body.temporal=$('temporal').value==='true';
 for(const key of Object.keys(controlSpecs))body.controls[key]=$(key).value===''?null:Number($(key).value);
 await api('/api/run',body);status('Preparing inputs...');poll();
 }catch(error){status(error.message,true);$('run').disabled=false;$('load-settings').disabled=false;}
});
const sync=document.getElementById('sync-sliders');
const pairs=['student','teacher'].map(name=>({
 name,label:name==='student'?'Student':'Teacher',
 stage:document.getElementById(name+'-stage'),
 range:document.getElementById(name+'-divider')
}));
let lastActive=pairs[0];
function renderPosition(pair,value){
 const position=Math.min(100,Math.max(0,Math.round(value)));
 pair.range.value=String(position);
 pair.stage.style.setProperty('--position',position+'%');
 document.getElementById(pair.name+'-original-percent').textContent=position+'%';
 document.getElementById(pair.name+'-prediction-percent').textContent=(100-position)+'%';
 pair.range.setAttribute('aria-valuetext',position+' percent original input, '+(100-position)+' percent '+pair.label+' prediction');
 pair.stage.querySelector('.before').hidden=position<13;
 pair.stage.querySelector('.after').hidden=position>87;
}
function setPosition(pair,value){
 lastActive=pair;
 for(const target of (sync.checked?pairs:[pair]))renderPosition(target,value);
}
function positionFromPointer(pair,event){
 const bounds=pair.stage.getBoundingClientRect();
 setPosition(pair,(event.clientX-bounds.left)/bounds.width*100);
}
for(const pair of pairs){
 pair.stage.addEventListener('pointerdown',event=>{
  if(event.button!==0)return;
  event.preventDefault();
  pair.range.focus({preventScroll:true});
  pair.stage.setPointerCapture(event.pointerId);
  pair.stage.classList.add('dragging');
  positionFromPointer(pair,event);
 });
 pair.stage.addEventListener('pointermove',event=>{
  if(pair.stage.hasPointerCapture(event.pointerId))positionFromPointer(pair,event);
 });
 pair.stage.addEventListener('pointerup',event=>{
  if(pair.stage.hasPointerCapture(event.pointerId))positionFromPointer(pair,event);
  pair.stage.classList.remove('dragging');
  if(pair.stage.hasPointerCapture(event.pointerId))pair.stage.releasePointerCapture(event.pointerId);
 });
 for(const type of ['pointercancel','lostpointercapture'])pair.stage.addEventListener(type,event=>{
  pair.stage.classList.remove('dragging');
  if(pair.stage.hasPointerCapture(event.pointerId))pair.stage.releasePointerCapture(event.pointerId);
 });
 pair.range.addEventListener('input',()=>setPosition(pair,Number(pair.range.value)));
 renderPosition(pair,50);
}
sync.addEventListener('change',()=>{
 if(sync.checked)setPosition(lastActive,Number(lastActive.range.value));
 document.getElementById('sync-status').textContent=sync.checked?
  'Move either divider to adjust both comparisons.':'Each divider moves independently.';
});
document.getElementById('reset-button').addEventListener('click',()=>{
 for(const pair of pairs)renderPosition(pair,50);
 lastActive=pairs[0];
});

api('/api/defaults').then(data=>{applyDefaults(data);poll();}).catch(error=>status(error.message,true));
</script></body></html>
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default='127.0.0.1', help='Listening IP address')
    parser.add_argument('--port', type=int, default=8080, help='Listening port (1..65535)')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be within [1,65535]')
    server = PreviewServer((args.ip, args.port))
    host = f'[{args.ip}]' if ':' in args.ip else args.ip
    print(f'[Preview]: http://{host}:{server.server_port} | output: {OUTPUT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
