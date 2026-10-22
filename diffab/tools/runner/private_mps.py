"""Isolated private MPS engineering adapter; importing performs no GPU operation.

The foreground control / private pipe / UID RPC sequence follows the audited
validated private MPS lifecycle.  No science code,
model, queue, precision, seed, or CUDA computation is implemented here.
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path


class MPSBlocked(RuntimeError):
    pass


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def command(argv, env=None, text=None, timeout=15):
    began = time.time()
    try:
        p = subprocess.run(argv, input=text, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
        return dict(command=list(argv), returncode=p.returncode, stdout=p.stdout,
                    stderr=p.stderr, unix_time=began)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=list(argv), error=repr(error), unix_time=began)


def process(pid):
    """Read-only best effort: exiting /proc entries are observations, not faults."""
    try:
        path = Path('/proc') / str(int(pid))
        before = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        identity = dict(pid=int(pid), uid=path.stat().st_uid,
                        start_ticks=int(before[19]),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        pid_namespace_inode=(path / 'ns/pid').stat().st_ino)
        result = dict(identity, state=before[0], ppid=int(before[1]),
                      user_ticks=int(before[11]), system_ticks=int(before[12]),
                      rss_bytes=int(before[21]) * os.sysconf('SC_PAGE_SIZE'))
        try:
            raw = (path / 'environ').read_bytes()
            env = dict(x.split(b'=', 1) for x in raw.split(b'\0') if b'=' in x)
            for name in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY',
                         'CUDA_VISIBLE_DEVICES'):
                result[name] = env.get(name.encode(), b'').decode(errors='replace')
        except OSError as error:
            result['environ_error'] = repr(error)
        try:
            result['exe'] = os.path.realpath(str(path / 'exe'))
        except OSError as error:
            result['exe_error'] = repr(error)
        try:
            result['CPU_affinity'] = sorted(os.sched_getaffinity(int(pid)))
        except OSError as error:
            result['affinity_error'] = repr(error)
        try:
            status = dict(x.split(':', 1) for x in (path / 'status').read_text().splitlines() if ':' in x)
            result['voluntary_context_switches'] = int(status.get('voluntary_ctxt_switches', 0))
            result['nonvoluntary_context_switches'] = int(status.get('nonvoluntary_ctxt_switches', 0))
        except (OSError, ValueError) as error:
            result['status_error'] = repr(error)
        try:
            result['PSS_bytes'] = int(next(x.split()[1] for x in
                (path / 'smaps_rollup').read_text().splitlines() if x.startswith('Pss:'))) * 1024
        except (OSError, StopIteration, ValueError) as error:
            result['PSS_bytes'] = None
            result['PSS_error'] = repr(error)
        after = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if int(after[19]) != identity['start_ticks'] or path.stat().st_uid != identity['uid']:
            return dict(pid=int(pid), observation='IDENTITY_CHANGED_DURING_READ')
        result['observation'] = 'READABLE'
        return result
    except (OSError, ValueError, IndexError) as error:
        return dict(pid=int(pid), observation='UNREADABLE_OR_EXITED', error=repr(error))


def existing_mps():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            comm = (path / 'comm').read_text().strip()
            exe = os.path.realpath(str(path / 'exe'))
            if 'nvidia-cuda-mps' in comm or Path(exe).name in (
                    'nvidia-cuda-mps-control', 'nvidia-cuda-mps-server'):
                found.append(process(int(path.name)))
        except OSError:
            continue
    return found


def numeric_pids(reply):
    if reply.get('returncode') != 0:
        return None
    # Some installed control UIs print the two characters backslash+n rather
    # than an actual newline. Accept only explicit decimal PID separators;
    # never extract digits from an arbitrary error message.
    text = reply.get('stdout', '').replace(chr(92) + 'n', chr(10))
    if any(not (x in '0123456789' or x.isspace()) for x in text):
        return None
    words = text.split()
    return [int(word) for word in words]


class PrivateMPS:
    def __init__(self, root, gpu_uuid):
        self.root = Path(root).resolve()
        self.gpu_uuid = str(gpu_uuid)
        if not self.gpu_uuid.startswith('GPU-') or any(x in self.gpu_uuid for x in ('/', ',', ' ', '\n')):
            raise ValueError('EXPLICIT_SINGLE_PHYSICAL_GPU_UUID_REQUIRED')
        self.root.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self.env = {}
        self.daemon = None
        self.control_identity = None
        self.server_identity = None
        self.handle = None
        self.closed = False

    def _rpc(self, text):
        return command([self.control], dict(os.environ, **self.env), text + '\n')

    def _same_identity(self, expected):
        if not expected:
            return False, dict(observation='NO_SAVED_IDENTITY')
        observed = process(expected['pid'])
        same = (observed.get('observation') == 'READABLE' and
                all(observed.get(k) == expected.get(k) for k in
                    ('pid', 'uid', 'start_ticks', 'boot_id', 'pid_namespace_inode')) and
                observed.get('uid') == os.geteuid() and
                observed.get('CUDA_MPS_PIPE_DIRECTORY') == str(self.pipe) and
                observed.get('CUDA_VISIBLE_DEVICES') == self.gpu_uuid)
        return same, observed

    def _own_identity(self, pid, executable, timeout=0):
        # MPS get_server_list exposes its forked child before exec/environment
        # and CUDA server initialization finish. Wait briefly for the SAME
        # birth/UID; startup transience is not proof of unsupported hardware.
        deadline = time.monotonic() + timeout
        birth = None
        observations = []
        while True:
            observed = process(pid)
            observations.append(dict(unix_time=time.time(), process=observed))
            if observed.get('observation') == 'READABLE':
                keys = ('pid', 'uid', 'start_ticks', 'boot_id', 'pid_namespace_inode')
                current = {k: observed.get(k) for k in keys}
                if birth is None:
                    birth = current
                if current != birth:
                    raise MPSBlocked('PRIVATE_MPS_PID_BIRTH_CHANGED_DURING_STARTUP ' + json.dumps(observed))
                if observed.get('uid') != os.geteuid():
                    raise MPSBlocked('PRIVATE_MPS_UID_MISMATCH ' + json.dumps(observed))
            if (observed.get('observation') == 'READABLE' and observed.get('uid') == os.geteuid() and
                    observed.get('exe') == os.path.realpath(executable) and
                    observed.get('CUDA_MPS_PIPE_DIRECTORY') == str(self.pipe) and
                    observed.get('CUDA_VISIBLE_DEVICES') == self.gpu_uuid):
                atomic(self.root / ('STARTUP_IDENTITY_%s.json' % pid),
                       dict(status='VERIFIED_SAME_BIRTH', observations=observations))
                return observed
            if time.monotonic() >= deadline:
                atomic(self.root / ('STARTUP_IDENTITY_%s.json' % pid),
                       dict(status='UNPROVED', observations=observations))
                raise MPSBlocked('PRIVATE_MPS_PROCESS_IDENTITY_UNPROVEN ' + json.dumps(observed))
            time.sleep(0.05)

    def _device(self):
        gpu = command(['nvidia-smi', '-i', self.gpu_uuid,
            '--query-gpu=uuid,name,utilization.gpu,memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'])
        apps = command(['nvidia-smi', '-i', self.gpu_uuid,
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'])
        result = dict(gpu_query=gpu, compute_query=apps)
        if gpu.get('returncode') == 0:
            cells = [x.strip() for x in gpu['stdout'].strip().split(',')]
            try:
                if len(cells) != 6 or cells[0] != self.gpu_uuid:
                    raise ValueError('GPU_UUID_OR_SCHEMA_MISMATCH')
                result.update(GPU_UUID=cells[0], name=cells[1], utilization_percent=float(cells[2]),
                    total_mib=float(cells[3]), used_mib=float(cells[4]), free_mib=float(cells[5]))
            except ValueError as error:
                result['gpu_parse_error'] = repr(error)
        if apps.get('returncode') == 0:
            values = []
            try:
                for line in apps['stdout'].splitlines():
                    if line.strip():
                        values.append(int(line.split(',', 1)[0].strip()))
                result['compute_PIDs'] = values
            except ValueError as error:
                result['compute_parse_error'] = repr(error)
        return result

    def start(self):
        if self.daemon is not None or self.pipe is not None:
            raise MPSBlocked('PRIVATE_MPS_START_ONCE_ONLY')
        self.control = shutil.which('nvidia-cuda-mps-control')
        self.server = shutil.which('nvidia-cuda-mps-server')
        device = self._device()
        prior = existing_mps()
        report = dict(schema='PRIVATE_MPS_PREFLIGHT_V1', GPU_UUID=self.gpu_uuid,
                      device=device, preexisting_MPS=prior, host=socket.gethostname(),
                      uid=os.geteuid(), unix_time=time.time(), compute_mode_changed=False)
        atomic(self.root / 'PREFLIGHT.json', report)
        if not self.control or not self.server or os.geteuid() == 0:
            raise MPSBlocked('UNPRIVILEGED_INSTALLED_PRIVATE_MPS_REQUIRED')
        if device.get('GPU_UUID') != self.gpu_uuid or 'compute_PIDs' not in device:
            raise MPSBlocked('ASSIGNED_GPU_IDENTITY_OR_COMPUTE_QUERY_UNPROVEN')
        if device['compute_PIDs']:
            raise MPSBlocked('ASSIGNED_GPU_ALREADY_HAS_COMPUTE_CLIENTS')
        # Existing services with an explicit, different physical UUID may run
        # undisturbed. Unknown / ordinal scopes cannot safely be attributed.
        for value in prior:
            visible = value.get('CUDA_VISIBLE_DEVICES', '')
            pieces = [x.strip() for x in visible.split(',') if x.strip()]
            if (value.get('observation') != 'READABLE' or not pieces or
                    any(not x.startswith('GPU-') for x in pieces) or self.gpu_uuid in pieces):
                raise MPSBlocked('EXISTING_SAME_GPU_OR_UNKNOWN_MPS_SCOPE_NO_ACTION ' + json.dumps(value))
        self.pipe = Path(tempfile.mkdtemp(prefix='ag_dmps_', dir='/tmp'))
        os.chmod(str(self.pipe), 0o700)
        st = self.pipe.lstat()
        self.pipe_identity = dict(path=str(self.pipe), inode=st.st_ino, device=st.st_dev, uid=st.st_uid)
        logs = self.root / 'mps_logs'
        logs.mkdir(mode=0o700, exist_ok=False)
        self.env = dict(CUDA_VISIBLE_DEVICES=self.gpu_uuid,
                        CUDA_MPS_PIPE_DIRECTORY=str(self.pipe), CUDA_MPS_LOG_DIRECTORY=str(logs))
        environment = dict(os.environ, **self.env)
        environment.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)
        self.handle = (self.root / 'MPS_CONTROL.log').open('x')
        self.daemon = subprocess.Popen([self.control, '-f'], env=environment,
            stdout=self.handle, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(0.3)
        if self.daemon.poll() is not None:
            raise MPSBlocked('OWN_FOREGROUND_MPS_CONTROL_EXITED')
        self.control_identity = self._own_identity(self.daemon.pid, self.control, timeout=5)
        first = self._rpc('get_server_list')
        if numeric_pids(first) != []:
            raise MPSBlocked('UNEXPECTED_SERVER_OR_RPC_IN_NEW_PRIVATE_PIPE ' + json.dumps(first))
        rpc = self._rpc('start_server -uid ' + str(os.geteuid()))
        atomic(self.root / 'START_SERVER_RPC.json', rpc)
        deadline = time.monotonic() + 30
        servers = None
        while time.monotonic() < deadline:
            listing = self._rpc('get_server_list')
            servers = numeric_pids(listing)
            if servers:
                break
            time.sleep(0.1)
        if servers is None or len(servers) != 1:
            raise MPSBlocked('OWN_PRIVATE_MPS_SERVER_NOT_READY ' + json.dumps(listing))
        self.server_identity = self._own_identity(servers[0], self.server, timeout=10)
        scope = dict(schema='PRIVATE_MPS_SCOPE_V1', GPU_UUID=self.gpu_uuid,
            environment=self.env, pipe_identity=self.pipe_identity,
            control=self.control_identity, server=self.server_identity,
            compute_mode_changed=False, active_thread_percentage='UNSET',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inherited_other_services_modified=False, unix_time=time.time())
        atomic(self.root / 'PRIVATE_MPS_SCOPE.json', scope)
        return dict(self.env)

    def sample(self, expected_pairs=None):
        """Observe real membership; transient reads never terminate model work.

        expected_pairs maps main PID to its G PID (or a list containing main/G).
        A failed observation is recorded as unproved, never silently PASS.
        """
        expected_pairs = expected_pairs or {}
        listing = self._rpc('get_server_list') if self.daemon is not None else {}
        servers = numeric_pids(listing)
        raw_clients = {}
        clients = []
        rpc_complete = servers is not None
        if servers is not None:
            for pid in servers:
                reply = self._rpc('get_client_list ' + str(pid))
                raw_clients[str(pid)] = reply
                ids = numeric_pids(reply)
                if ids is None:
                    rpc_complete = False
                else:
                    clients.extend(ids)
        clients = sorted(set(clients))
        pairs = {}
        for main, other in expected_pairs.items():
            values = list(other) if isinstance(other, (tuple, list)) else [int(main), int(other)]
            values = sorted(set([int(main)] + [int(x) for x in values]))
            pairs[str(main)] = dict(expected=values, attached=all(x in clients for x in values),
                                    observed_private_clients=[x for x in values if x in clients])
        pids = set(clients)
        for main, value in pairs.items():
            pids.update(value['expected'])
        if self.control_identity:
            pids.add(self.control_identity['pid'])
        if self.server_identity:
            pids.add(self.server_identity['pid'])
        resources = [process(pid) for pid in sorted(pids)]
        value = dict(schema='PRIVATE_MPS_OBSERVATION_V1', unix_time=time.time(),
            GPU_UUID=self.gpu_uuid, environment=self.env, server_list_reply=listing,
            client_list_replies=raw_clients, private_clients=clients,
            expected_pairs=pairs, RPC_complete=rpc_complete,
            all_expected_pairs_attached=bool(pairs) and rpc_complete and all(x['attached'] for x in pairs.values()),
            device=self._device(), processes=resources,
            policy='OBSERVATION_ONLY_NO_MODEL_STOP_ON_TRANSIENT_READ',
            memory_scope='whole GPU NVML instantaneous; RSS includes shared pages; PSS separately; no client/server memory summing')
        atomic(self.root / 'LATEST_SAMPLE.json', value)
        with (self.root / 'SAMPLES.jsonl').open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        return value

    def close(self, timeout=60):
        if self.closed:
            return self.close_result
        result = dict(status='NO_STARTED_PRIVATE_MPS', foreign_services_modified=False)
        if self.daemon is None:
            return result
        try:
            if self.control_identity is None and self.daemon.poll() is None:
                self.control_identity = self._own_identity(self.daemon.pid, self.control, timeout=2)
            valid, observed = self._same_identity(self.control_identity)
            if not valid:
                raise MPSBlocked('OWN_CONTROL_IDENTITY_NOT_CURRENTLY_PROVED ' + json.dumps(observed))
            deadline = time.monotonic() + timeout
            while True:
                listing = self._rpc('get_server_list')
                ids = numeric_pids(listing)
                if ids is not None and len(ids) == 1 and self.server_identity is None:
                    # Only the sole server reported by OUR current verified
                    # control through OUR private pipe can recover identity.
                    self.server_identity = self._own_identity(ids[0], self.server, timeout=2)
                if ids is None or any(pid != (self.server_identity or {}).get('pid') for pid in ids):
                    raise MPSBlocked('PRIVATE_SERVER_LIST_UNPROVEN_NO_QUIT ' + json.dumps(listing))
                replies = {}
                empty = True
                for pid in ids:
                    valid, observed = self._same_identity(self.server_identity)
                    if not valid:
                        raise MPSBlocked('OWN_SERVER_IDENTITY_UNPROVEN_NO_QUIT ' + json.dumps(observed))
                    reply = self._rpc('get_client_list ' + str(pid))
                    replies[str(pid)] = reply
                    clients = numeric_pids(reply)
                    if clients is None:
                        raise MPSBlocked('PRIVATE_CLIENT_LIST_UNPROVEN_NO_QUIT ' + json.dumps(reply))
                    empty = empty and not clients
                if empty:
                    break
                if time.monotonic() >= deadline:
                    raise MPSBlocked('OWN_PRIVATE_CLIENTS_STILL_LIVE_NO_QUIT ' + json.dumps(replies))
                time.sleep(0.25)
            quit_reply = self._rpc('quit')
            if quit_reply.get('returncode') != 0:
                raise MPSBlocked('OWN_PRIVATE_QUIT_FAILED ' + json.dumps(quit_reply))
            code = self.daemon.wait(timeout=30)
            if code != 0:
                raise MPSBlocked('OWN_PRIVATE_CONTROL_NONZERO ' + str(code))
            st = self.pipe.lstat()
            if (self.pipe.is_symlink() or st.st_uid != os.geteuid() or
                    st.st_ino != self.pipe_identity['inode'] or st.st_dev != self.pipe_identity['device'] or
                    self.pipe.parent != Path('/tmp') or
                    any(x.lstat().st_uid != os.geteuid() for x in self.pipe.rglob('*'))):
                raise MPSBlocked('PRIVATE_PIPE_IDENTITY_CHANGED_NO_REMOVAL')
            shutil.rmtree(str(self.pipe))
            result = dict(status='OWN_PRIVATE_QUIT_AND_PIPE_REMOVAL_COMPLETE',
                quit_reply=quit_reply, control_exitcode=code, owned_pipe_removed=True,
                foreign_services_modified=False, unix_time=time.time())
            self.closed = True
            self.close_result = result
        except (OSError, ValueError, subprocess.TimeoutExpired, MPSBlocked) as error:
            result = dict(status='CLEANUP_BLOCKED_NO_FORCE_ACTION', error=repr(error),
                          foreign_services_modified=False, unix_time=time.time())
        finally:
            if self.handle is not None:
                self.handle.close()
            atomic(self.root / 'MPS_CLEANUP.json', result)
        return result
