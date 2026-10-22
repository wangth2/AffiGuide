"""Pinned model acquisition; never deserialize a file before verifying its identity.

This module intentionally uses only Python's standard library. It is usable by
CPU setup checks before importing either scientific environment.
"""
import base64
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import zipfile


class ModelAssetError(RuntimeError):
    pass


def repository_root():
    return Path(__file__).resolve().parents[2]


def load_manifest(root=None, manifest_path=None):
    path = Path(manifest_path) if manifest_path else Path(root or repository_root()) / 'MODEL_MANIFEST.json'
    value = json.loads(path.read_text())
    if not isinstance(value.get('models'), dict):
        raise ModelAssetError('INVALID_MODEL_MANIFEST: expected models mapping in ' + str(path))
    for name, entry in value['models'].items():
        if len(entry.get('sha256', '')) != 64 or int(entry.get('size_bytes', 0)) <= 0:
            raise ModelAssetError('INVALID_MODEL_MANIFEST_IDENTITY: ' + name)
    return value


def _entry(name, root=None, manifest_path=None):
    manifest = load_manifest(root, manifest_path)
    try:
        return manifest['models'][name]
    except KeyError:
        raise ModelAssetError('UNKNOWN_MODEL: ' + str(name))


def _path(value, root):
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if '$' in expanded:
        raise ModelAssetError('UNRESOLVED_MODEL_PATH_ENVIRONMENT: ' + expanded)
    path = Path(expanded)
    return path if path.is_absolute() else Path(root) / path


def verify_model(name, path=None, *, root=None, manifest_path=None):
    """Return identity metadata after checking bytes and SHA256, without torch.load."""
    root = Path(root or repository_root()).expanduser().resolve()
    entry = _entry(name, root, manifest_path)
    path = _path(path if path is not None else entry['path'], root)
    if not path.is_file():
        raise ModelAssetError('MODEL_MISSING: %s at %s; run python scripts/download_weights.py --models %s' %
                              (name, path, name))
    observed_size = path.stat().st_size
    expected_size = int(entry['size_bytes'])
    if observed_size != expected_size:
        raise ModelAssetError('MODEL_SIZE_MISMATCH: %s at %s expected=%s observed=%s; file was not loaded or replaced' %
                              (name, path, expected_size, observed_size))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != entry['sha256']:
        raise ModelAssetError('MODEL_SHA256_MISMATCH: %s at %s expected=%s observed=%s; file was not loaded or replaced' %
                              (name, path, entry['sha256'], observed))
    return {'model': name, 'path': str(path.resolve()), 'sha256': observed,
            'size_bytes': observed_size, 'status': 'VERIFIED'}


@contextlib.contextmanager
def _download_lock(path):
    # The persistent lock inode must not be unlinked while other processes wait.
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _copy_stream(stream, output, expected_size):
    size = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        size += len(block)
        if size > expected_size:
            raise ModelAssetError('MODEL_DOWNLOAD_SIZE_EXCEEDED: expected %d bytes' % expected_size)
        output.write(block)


def _fetch_source(source, output, expected_size, timeout):
    request = Request(source['url'], headers={'User-Agent': 'AffiGuide-pinned-model-fetch/1'})
    with urlopen(request, timeout=timeout) as response:
        if source.get('format', 'raw') == 'raw':
            _copy_stream(response, output, expected_size)
        elif source['format'] == 'zip_member':
            with tempfile.TemporaryFile() as archive:
                _copy_stream(response, archive, int(source['archive_size_bytes']))
                if archive.tell() != int(source['archive_size_bytes']):
                    raise ModelAssetError('MODEL_ARCHIVE_SIZE_MISMATCH')
                archive.seek(0)
                digest = hashlib.sha256()
                for block in iter(lambda: archive.read(1024 * 1024), b''):
                    digest.update(block)
                if digest.hexdigest() != source['archive_sha256']:
                    raise ModelAssetError('MODEL_ARCHIVE_SHA256_MISMATCH')
                archive.seek(0)
                with zipfile.ZipFile(archive) as packed:
                    info = packed.getinfo(source['member'])
                    if info.file_size != expected_size:
                        raise ModelAssetError('MODEL_ARCHIVE_MEMBER_SIZE_MISMATCH')
                    # Read one named member; never extract archive paths to disk.
                    with packed.open(info) as stream:
                        _copy_stream(stream, output, expected_size)
        elif source['format'] == 'github_blob_base64':
            encoded = response.read(expected_size * 2 + 1048577)
            if len(encoded) > expected_size * 2 + 1048576:
                raise ModelAssetError('MODEL_ENCODED_RESPONSE_TOO_LARGE')
            body = json.loads(encoded)
            if body.get('encoding') != 'base64' or body.get('sha') != source['git_blob_sha1']:
                raise ModelAssetError('MODEL_GITHUB_BLOB_IDENTITY_MISMATCH')
            decoded = base64.b64decode(body['content'])
            if len(decoded) != expected_size:
                raise ModelAssetError('MODEL_DECODED_SIZE_MISMATCH')
            output.write(decoded)
        else:
            raise ModelAssetError('UNSUPPORTED_MODEL_SOURCE_FORMAT: ' + str(source.get('format')))


def ensure_model(name, path=None, *, root=None, offline=False, manifest_path=None,
                 source_dir=None, timeout=60):
    """Verify an existing file or atomically install its exact pinned bytes.

    Only pinned official sources in the manifest are used. ``source_dir`` is
    an explicit, optional offline asset directory (repository-relative paths or
    basenames); it is never an implicit cache. A corrupt existing target is an
    error and is never silently overwritten.
    """
    root = Path(root or repository_root()).expanduser().resolve()
    entry = _entry(name, root, manifest_path)
    path = _path(path if path is not None else entry['path'], root)
    if path.exists():
        verify_model(name, path, root=root, manifest_path=manifest_path)
        return path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(path.with_name(path.name + '.download.lock')):
        if path.exists():
            verify_model(name, path, root=root, manifest_path=manifest_path)
            return path.resolve()
        local_source = None
        if source_dir is not None:
            source_dir = Path(os.path.expandvars(str(source_dir))).expanduser().resolve()
            candidates = (Path(source_dir) / entry['path'], Path(source_dir) / Path(entry['path']).name)
            local_source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if local_source is None:
                raise ModelAssetError('EXPLICIT_MODEL_SOURCE_MISSING: %s in %s' % (name, source_dir))
            verify_model(name, local_source, root=root, manifest_path=manifest_path)
        sources = entry.get('verified_download_sources') or [
            {'url': url, 'format': 'raw'} for url in entry.get('verified_download_urls', [])]
        if local_source is None and offline:
            raise ModelAssetError('OFFLINE_MODEL_MISSING: %s at %s; supply matching bytes with --source-dir or pre-download before --offline' %
                                  (name, path))
        if local_source is None and not sources:
            raise ModelAssetError('EXACT_MODEL_PUBLIC_URL_UNAVAILABLE: %s; obtain the exact file identified in MODEL_MANIFEST.json and install via --source-dir; see the README weight instructions' % name)
        fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=str(path.parent))
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, 'wb') as output:
                if local_source is not None:
                    with local_source.open('rb') as stream:
                        _copy_stream(stream, output, int(entry['size_bytes']))
                else:
                    errors = []
                    for source in sources:
                        output.seek(0)
                        output.truncate()
                        try:
                            _fetch_source(source, output, int(entry['size_bytes']), timeout)
                            output.flush()
                            verify_model(name, temporary, root=root, manifest_path=manifest_path)
                            break
                        except Exception as exc:
                            errors.append('%s: %s' % (source['url'], exc))
                    else:
                        raise ModelAssetError('MODEL_DOWNLOAD_FAILED: %s; %s' % (name, '; '.join(errors)))
                output.flush()
                os.fsync(output.fileno())
            verify_model(name, temporary, root=root, manifest_path=manifest_path)
            os.replace(str(temporary), str(path))
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
    return path.resolve()


def required_models(config):
    """Identify models actually needed by the selected protocol, including legacy YAML."""
    names = ['DiffAb']
    guidance = config.get('guidance')
    if guidance is not None:
        from diffab.modules.diffusion.guidance_config import needs_gearbind
        names.append('D')
        if needs_gearbind(guidance):
            names.append('G')
    else:
        sampling = config.get('sampling', {})
        if sampling.get('sample_ddg', True):
            names.append('G' if sampling.get('ddg_guidance_model', 'ddgpred') in ('gearbind', 'gearb') else 'D')
    return tuple(names)


def configured_model_paths(config):
    model = config.get('model', {})
    scorers = config.get('scorers', {})
    sampling = config.get('sampling', {})
    return {'DiffAb': model.get('checkpoint'),
            'D': scorers.get('D', {}).get('checkpoint') or sampling.get('ddg_checkpoint'),
            'G': scorers.get('G', {}).get('checkpoint') or sampling.get('gearb_ckpt')}


def ensure_config_models(config, *, root=None, offline=False, manifest_path=None, source_dir=None):
    paths = configured_model_paths(config)
    return {name: ensure_model(name, paths.get(name), root=root, offline=offline,
                              manifest_path=manifest_path, source_dir=source_dir)
            for name in required_models(config)}
