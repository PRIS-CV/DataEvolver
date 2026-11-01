"""Durable OpenAI transport bound to an exact host-owned Linux process.

The existing Gateway owns requests and budgets. This provider adds native
readback and checks the listener/process/source identity before a cache miss.
It never starts a service or resends an uncertain HTTP request.
"""
from pathlib import Path
import base64
import hashlib
import json
import os
from urllib.parse import urlsplit

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from .model_gateway import LocalEndpointProvider, ModelError


class LocalProcessEndpointProvider(LocalEndpointProvider):
    @staticmethod
    def profile_revision(deployment, video_sampling=None):
        if video_sampling is None:
            return 'sha256:' + payload_sha256(deployment)
        if (set(video_sampling) != {'max_frames', 'image_side'}
                or type(video_sampling['max_frames']) is not int or not 3 <= video_sampling['max_frames'] <= 256
                or type(video_sampling['image_side']) is not int or not 32 <= video_sampling['image_side'] <= 2048):
            raise ValueError('video sampling requires explicit frame and image bounds')
        return 'sha256:' + payload_sha256({'deployment': deployment, 'video_sampling': video_sampling,
            'transport_sha256': file_sha256(Path(__file__)),
            'decoder_sha256': file_sha256(Path(__file__).with_name('local_visual_worker.py'))})

    def __init__(self, profile, deployment, store, native_root, *, video_sampling=None):
        url = urlsplit(deployment['origin'])
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port
                or url.path or url.query or url.fragment
                or profile.model != deployment['model']
                or profile.revision != self.profile_revision(deployment, video_sampling)):
            raise ValueError('local endpoint profile must bind an exact loopback deployment')
        self.deployment, self.root = dict(deployment), Path(native_root)
        self.video_sampling = dict(video_sampling) if video_sampling is not None else None
        self.deployment_hash = payload_sha256(deployment)
        prefix = deployment.get('api_prefix', '/v1')
        if prefix not in {'', '/v1'}:
            raise ValueError('registered OpenAI endpoint requires an explicit supported API prefix')
        super().__init__(profile, deployment['origin'] + prefix, deployment['served_model'], store)

    def _check_binding(self):
        try:
            valid = self.profile.revision == self.profile_revision(self.deployment, self.video_sampling)
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ModelError('DEPLOYMENT_MISMATCH', 'frozen process or sampled-video binding changed')

    def verify_deployment(self):
        d = self.deployment
        process = Path('/proc') / str(d['pid'])
        try:
            if (process.joinpath('stat').read_text().split(') ', 1)[1].split()[19] != d['process_start_ticks']
                    or hashlib.sha256(process.joinpath('cmdline').read_bytes()).hexdigest() != d['cmdline_sha256']):
                raise ValueError('service process or command changed')
            environment = dict(v.split(b'=', 1) for v in process.joinpath('environ').read_bytes().split(b'\0') if b'=' in v)
            if environment.get(b'CUDA_VISIBLE_DEVICES', b'').decode().split(',') != d['gpu_uuids']:
                raise ValueError('service physical GPU mapping changed')
            inodes = {os.readlink(p)[8:-1] for p in process.joinpath('fd').iterdir()
                      if p.is_symlink() and os.readlink(p).startswith('socket:[')}
            port = urlsplit(d['origin']).port
            listeners = [row.split()[9] for row in Path('/proc/net/tcp').read_text().splitlines()[1:]
                         if row.split()[1].upper() == '0100007F:' + f'{port:04X}' and row.split()[3] == '0A']
            if not set(listeners) & inodes:
                raise ValueError('the frozen process does not own the frozen listening port')
            for path, digest in d['source_hashes'].items():
                if file_sha256(Path(path)) != digest:
                    raise ValueError('registered endpoint implementation changed')
        except (OSError, KeyError, ValueError, IndexError) as exc:
            raise ModelError('DEPLOYMENT_MISMATCH', str(exc)) from exc

    def health(self):
        self.verify_deployment()
        health = super().health()
        return {**health, 'revision_verified': health['status'] == 'AVAILABLE',
                'deployment_sha256': self.deployment_hash, 'pid': self.deployment['pid']}

    def lookup(self, request, request_hash):
        self._check_binding()
        path = self.root / (request.request_id + '.native.json')
        if not path.exists():
            raise ModelError('AMBIGUOUS_EFFECT', 'endpoint has no committed native completion; do not resend')
        try:
            value = json.loads(path.read_text())
            if (value['request_hash'] != request_hash or value['deployment_sha256'] != self.deployment_hash
                    or value['response_sha256'] != payload_sha256(value['response'])):
                raise ValueError('native endpoint completion changed')
            observation = value['response'].get('runtime', {}).get('video_observation')
            if observation is not None:
                self._verify_video_observation(request, observation)
            return value['response']
        except (OSError, KeyError, ValueError) as exc:
            raise ModelError('INVALID_RECEIPT', 'native endpoint completion is invalid') from exc

    def _media_content(self, request):
        if not any(ref.media_type.startswith('video/') for ref in request.artifacts):
            return super()._media_content(request)
        if (self.video_sampling is None or self.store is None or len(request.artifacts) != 1
                or request.artifacts[0].media_type != 'video/mp4'
                or request.role not in {'visual.inspect.video', 'visual.review.video'}):
            raise ModelError('UNSUPPORTED_MEDIA', 'video requires its bound sampled-frame transport and exact role')
        from PIL import Image
        from .local_visual_worker import _video_frames
        source = request.artifacts[0]
        frames = _video_frames(self.store.resolve(source), **self.video_sampling)
        indices = sorted({0, len(frames)//2, len(frames)-1})
        sizes = {frame.size for frame in frames}
        if len(sizes) != 1:
            raise ModelError('UNSUPPORTED_MEDIA', 'video frames have inconsistent dimensions')
        width, height = frames[0].size
        sheet = Image.new('RGB', (width*len(indices), height))
        for position, index in enumerate(indices):
            sheet.paste(frames[index], (position*width, 0))
        relative = Path('video-frames')/request.request_id/'contact-sheet.png'
        target = self.root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(target)
        observation = {'source': source.to_dict(), 'decoded_frame_count': len(frames), 'frame_indices': indices,
            'frame_pixel_sha256': [hashlib.sha256(frames[i].tobytes()).hexdigest() for i in indices],
            'frame_size': [width, height], 'scope': 'first_middle_last_frames_only',
            'layout': 'left_to_right_without_resizing', 'sampling': self.video_sampling,
            'contact_sheet': {'path': relative.as_posix(), 'sha256': file_sha256(target),
                              'size': [sheet.width, sheet.height]}}
        atomic_write_json(target.with_name('sampling.json'), observation)
        description = ('This single image is a contact sheet of video frames, left to right: '
            + json.dumps(indices) + f' out of {len(frames)} decoded frames. '
            'Each panel is a complete frame without resizing. Boxes must use coordinates within a frame, '
            'not within the whole contact sheet. Unobserved motion, timing and audio must remain uncertain.')
        return [{'type': 'text', 'text': description}, {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(target.read_bytes()).decode('ascii')}}]

    def _verify_video_observation(self, request, observation):
        count = observation['decoded_frame_count']
        expected = Path('video-frames')/request.request_id/'contact-sheet.png'
        if (self.video_sampling is None or len(request.artifacts) != 1
                or observation['source'] != request.artifacts[0].to_dict()
                or observation['sampling'] != self.video_sampling
                or type(count) is not int or not 1 <= count <= self.video_sampling['max_frames']
                or observation['frame_indices'] != sorted({0, count//2, count-1})
                or observation['contact_sheet']['path'] != expected.as_posix()
                or file_sha256(self.root/expected) != observation['contact_sheet']['sha256']):
            raise ValueError('sampled video evidence changed')
        self.store.resolve(request.artifacts[0])

    def native_invoke(self, request):
        response = super().invoke(request, '')
        if any(ref.media_type.startswith('video/') for ref in request.artifacts):
            observation = json.loads((self.root/'video-frames'/request.request_id/'sampling.json').read_text())
            self._verify_video_observation(request, observation)
            response['runtime'] = {'video_observation': observation}
        return response

    def invoke(self, request, request_hash):
        path = self.root / (request.request_id + '.native.json')
        if path.exists():
            return self.lookup(request, request_hash)
        self._check_binding()
        self.verify_deployment()
        intent = self.root / (request.request_id + '.native.intent.json')
        if intent.exists():
            raise ModelError('AMBIGUOUS_EFFECT', 'native HTTP dispatch already started; reconcile before retry')
        atomic_write_json(intent, {'request_hash': request_hash, 'deployment_sha256': self.deployment_hash,
                                  'request': request.to_dict(), 'state': 'STARTED'})
        response = self.native_invoke(request)
        if response['model'] != self.deployment['served_model']:
            raise ModelError('MODEL_MISMATCH', 'native endpoint returned a different served model')
        response = {**response, 'model': self.profile.model,
                    'revision': self.profile.revision, 'wire_model': response['model']}
        atomic_write_json(path, {'request_hash': request_hash, 'deployment_sha256': self.deployment_hash,
                                'response_sha256': payload_sha256(response), 'response': response})
        return response
