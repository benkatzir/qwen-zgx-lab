"""Functional text/image/video probes; deliberately NOT an accuracy certification.

Requires aiohttp and Pillow. Video creation uses existing ffmpeg or PyAV; this
script never installs dependencies. --assets-only creates assets without HTTP.
Use --video FILE.mp4 to reuse an asset created in a runtime with a video encoder.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw, ImageFont

NOTICE = ('Functional probes only. HTTP 200 establishes request handling, not answer '
          'correctness. Inspect answers against expected observations. These probes '
          'do not certify native accuracy retention, 128K recall, or production quality.')


def load_font(size):
    for candidate in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/dejavu/DejaVuSans.ttf',
                      '/System/Library/Fonts/Supplemental/Arial.ttf'):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def asset_info(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def make_assets(folder):
    image_path = folder / 'image_probe.png'
    img = Image.new('RGB', (800, 500), 'white')
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), 'ACCESS CODE: 7319', fill='black', font=load_font(48))
    draw.rectangle((70, 180, 240, 350), fill='red')
    draw.ellipse((450, 180, 620, 350), fill='blue')
    img.save(image_path)
    frames = []
    for index, (label, color, shape) in enumerate((('FIRST', 'red', 'square'),
            ('MIDDLE', 'green', 'circle'), ('LAST', 'blue', 'triangle'))):
        img = Image.new('RGB', (800, 500), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((45, 35), label, fill='black', font=load_font(64))
        if shape == 'square':
            draw.rectangle((290, 190, 510, 410), fill=color)
        elif shape == 'circle':
            draw.ellipse((290, 190, 510, 410), fill=color)
        else:
            draw.polygon(((400, 180), (265, 415), (535, 415)), fill=color)
        img.save(folder / f'video_card_{index + 1}_{label.lower()}.png')
        frames.extend([img] * 8)  # Two seconds per scene, four frames per second.
    return image_path, frames


def encode_video(folder, frames):
    destination = folder / 'temporal_order.mp4'
    failures = []
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        frame_folder = folder / 'video_frames'
        frame_folder.mkdir(exist_ok=True)
        for index, image in enumerate(frames):
            image.save(frame_folder / f'frame_{index:03d}.png')
        for codec in ('libx264', 'mpeg4'):
            command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '4',
                       '-i', str(frame_folder / 'frame_%03d.png'), '-c:v', codec,
                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=90)
                if process.returncode == 0 and destination.is_file() and destination.stat().st_size:
                    return destination, {'backend': 'ffmpeg', 'codec': codec}
                failures.append(f'ffmpeg/{codec}: {process.stderr[-1500:]}')
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f'ffmpeg/{codec}: {exc}')
    else:
        failures.append('ffmpeg executable unavailable')
    try:
        import av
        for codec in ('libx264', 'mpeg4'):
            try:
                with av.open(str(destination), mode='w') as container:
                    stream = container.add_stream(codec, rate=4)
                    stream.width, stream.height, stream.pix_fmt = 800, 500, 'yuv420p'
                    for image in frames:
                        for packet in stream.encode(av.VideoFrame.from_image(image)):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                if destination.is_file() and destination.stat().st_size:
                    return destination, {'backend': 'PyAV', 'codec': codec}
            except Exception as exc:
                failures.append(f'PyAV/{codec}: {type(exc).__name__}: {exc}')
    except ImportError:
        failures.append('PyAV unavailable')
    return None, {'backend': None, 'errors': failures}


def data_uri(path, mime):
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='qwen-lab')
    parser.add_argument('--out', required=True)
    parser.add_argument('--assets-dir')
    parser.add_argument('--assets-only', action='store_true')
    parser.add_argument('--video', help='Existing MP4; ideally the generated temporal-order asset')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--api-key-env', default='VLLM_API_KEY')
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_tokens <= 0:
        parser.error('timeout and max-tokens must be positive')
    return args


async def main():
    args = arguments()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(args.assets_dir) if args.assets_dir else out.with_name(out.stem + '_assets')
    folder.mkdir(parents=True, exist_ok=True)
    image_path, frames = make_assets(folder)
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise FileNotFoundError(f'Supplied video does not exist: {video_path}')
        encoder = {'backend': 'supplied MP4', 'note': 'Confirm supplied content matches expected observations.'}
    else:
        video_path, encoder = encode_video(folder, frames)
    expected_video = ['FIRST: red square', 'MIDDLE: green circle', 'LAST: blue triangle']
    report = {'not_accuracy_certification': True, 'notice': NOTICE, 'model': args.model,
              'base_url': args.base_url, 'assets_only': args.assets_only,
              'assets': {'image': asset_info(image_path),
                         'video': asset_info(video_path) if video_path else None,
                         'video_encoder': encoder, 'video_spec': {
                             'duration_seconds': 6, 'fps': 4, 'resolution': [800, 500],
                             'frame_count': 24, 'expected_order': expected_video}}, 'results': []}
    if args.assets_only:
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0 if video_path else 2
    cases = [
        ('text', [{'type': 'text', 'text': 'What is 17 multiplied by 23? Give the answer.'}],
         {'answer': 391}),
        ('image', [{'type': 'image_url', 'image_url': {'url': data_uri(image_path, 'image/png')}},
                   {'type': 'text', 'text': 'Read the access code in the image, then describe the shapes and their colors from left to right.'}],
         {'access_code': '7319', 'left': 'red square', 'right': 'blue circle'}),
    ]
    if video_path:
        cases.append(('video', [
            {'type': 'video_url', 'video_url': {'url': data_uri(video_path, 'video/mp4')}},
            {'type': 'text', 'text': 'Describe the changes in this video in chronological order. For each distinct scene, read its visible word and identify the shape and its color. Do not omit the middle scene.'},
        ], {'chronological_order': expected_video}))
    else:
        report['results'].append({'probe': 'video', 'status': 'untested', 'reason': encoder,
                                  'expected': {'chronological_order': expected_video}})
    headers = {}
    if os.environ.get(args.api_key_env):
        headers['Authorization'] = 'Bearer ' + os.environ[args.api_key_env]
    endpoint = args.base_url.rstrip('/')
    endpoint += '/chat/completions' if endpoint.endswith('/v1') else '/v1/chat/completions'
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout), headers=headers) as session:
        for label, content, expected in cases:
            started = time.monotonic()
            result = {'probe': label, 'expected': expected, 'correctness': 'requires_answer_review',
                      'status': 'not_started', 'http_status': None, 'raw_response_file': None}
            try:
                payload = {'model': args.model, 'messages': [{'role': 'user', 'content': content}],
                           'max_tokens': args.max_tokens, 'temperature': 0}
                async with session.post(endpoint, json=payload) as response:
                    body = await response.text()
                    raw_path = folder / f'{label}_response.txt'
                    raw_path.write_text(body)
                    result.update(http_status=response.status, raw_response_file=str(raw_path.resolve()),
                                  status='api_accepted' if response.status == 200 else 'api_error')
                    try:
                        result['response'] = json.loads(body)
                    except json.JSONDecodeError:
                        result['response_text'] = body
                        result['status'] = 'invalid_json_response'
            except Exception as exc:
                result.update(status='request_error', error=f'{type(exc).__name__}: {exc}')
            result['seconds'] = time.monotonic() - started
            report['results'].append(result)
            out.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(result, indent=2), flush=True)
    report['all_three_modalities_api_accepted'] = (len(report['results']) == 3 and
        all(result['status'] == 'api_accepted' for result in report['results']))
    out.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['all_three_modalities_api_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
