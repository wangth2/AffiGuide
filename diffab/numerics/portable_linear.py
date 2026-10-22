"""Isolated fixed-order FP32 CUDA Linear, first-order autograd; no automatic build/fallback.

Only explicit target nn.Linear.forward methods are replaced by install(). Main science,
weights and all other modules remain unchanged. A different summation order changes
FP32 rounding relative to cuBLAS; cross-device bitwise pass must be actually tested.
"""
import contextlib,ctypes,hashlib,json,math,threading
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable

_CACHE={}
_LOCK=threading.Lock()
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _ptr(t):return ctypes.c_void_p(0 if t is None else t.data_ptr())

class Backend:
 def __init__(self,path=None):
  root=Path(__file__).resolve().parent;self.path=Path(path or root/'build/portable_linear.fatbin').resolve()
  if not self.path.is_file():raise RuntimeError('Portable Linear binary missing; restore the matching runtime distribution')
  lock=self.path.parent/'PORTABLE_LINEAR_BUILD.json'
  if not lock.is_file():raise RuntimeError('PORTABLE_LINEAR_BUILD_LOCK_MISSING')
  self.build=json.loads(lock.read_text())
  if _sha(self.path)!=self.build['binary_sha256']:raise RuntimeError('PORTABLE_LINEAR_BINARY_HASH_CONFLICT')
  source=root/'kernels/portable_linear.cu'
  if not source.is_file()or _sha(source)!=self.build['source_sha256']:raise RuntimeError('PORTABLE_LINEAR_SOURCE_HASH_CONFLICT')
  if self.build.get('schema')!='PORTABLE_LINEAR_FIXED_ORDER_FP32_BUILD_V2_DRIVER_API':raise RuntimeError('PORTABLE_LINEAR_DEVICE_ONLY_ARTIFACT_REQUIRED')
  self.lib=ctypes.CDLL('libcuda.so.1');P=ctypes.c_void_p;U=ctypes.c_uint
  for name,args in [('cuInit',[U]),('cuCtxGetCurrent',[ctypes.POINTER(P)]),('cuCtxGetDevice',[ctypes.POINTER(ctypes.c_int)]),('cuPointerGetAttribute',[P,ctypes.c_int,ctypes.c_uint64]),('cuCtxPushCurrent_v2',[P]),('cuCtxPopCurrent_v2',[ctypes.POINTER(P)]),('cuDeviceGetUuid',[P,ctypes.c_int]),('cuModuleLoadData',[ctypes.POINTER(P),P]),('cuModuleGetFunction',[ctypes.POINTER(P),P,ctypes.c_char_p]),('cuLaunchKernel',[P,U,U,U,U,U,U,U,P,ctypes.POINTER(P),ctypes.POINTER(P)]),('cuDriverGetVersion',[ctypes.POINTER(ctypes.c_int)]),('cuGetErrorString',[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)])]:
   fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=ctypes.c_int
  self.counts=dict(forward=0,backward=0,dX=0,dW=0,db=0);self.shapes={};self.context_modules={};self.module_lock=threading.Lock()
 def check(self,e):
  if e:
   msg=ctypes.c_char_p();self.lib.cuGetErrorString(int(e),ctypes.byref(msg))
   raise RuntimeError('PORTABLE_LINEAR_DRIVER_ERROR %d '%e+(msg.value.decode(errors='replace')if msg.value else 'UNKNOWN'))
 @contextlib.contextmanager
 def using_context(self,value):
  context=ctypes.c_void_p(int(value));self.check(self.lib.cuCtxPushCurrent_v2(context))
  try:yield int(value)
  finally:
   popped=ctypes.c_void_p();self.check(self.lib.cuCtxPopCurrent_v2(ctypes.byref(popped)))
   if popped.value!=context.value:raise RuntimeError('PORTABLE_LINEAR_CUDA_CONTEXT_STACK_CONFLICT')
 @contextlib.contextmanager
 def tensor_context(self,weight):
  self.check(self.lib.cuInit(0));context=ctypes.c_void_p()
  # CUDA Driver enum CU_POINTER_ATTRIBUTE_CONTEXT=1; read the actual allocation's
  # owner, not rank/device-number guesses. Autograd threads may have no current context.
  self.check(self.lib.cuPointerGetAttribute(ctypes.byref(context),1,ctypes.c_uint64(weight.data_ptr())))
  if not context.value:raise RuntimeError('PORTABLE_LINEAR_WEIGHT_ALLOCATION_CONTEXT_MISSING')
  with self.using_context(context.value)as value:yield value
 def _functions(self):
  self.check(self.lib.cuInit(0));context=ctypes.c_void_p();self.check(self.lib.cuCtxGetCurrent(ctypes.byref(context)))
  if not context.value:raise RuntimeError('PORTABLE_LINEAR_REQUIRES_ACTUAL_TORCH_CUDA_CONTEXT')
  with self.module_lock:
   if context.value not in self.context_modules:
    image=ctypes.create_string_buffer(self.path.read_bytes());module=ctypes.c_void_p();self.check(self.lib.cuModuleLoadData(ctypes.byref(module),ctypes.cast(image,ctypes.c_void_p)));functions={}
    for name in ['forward','dx','dw','db']:
     fn=ctypes.c_void_p();self.check(self.lib.cuModuleGetFunction(ctypes.byref(fn),module,('ag_fixed_linear_'+name).encode()));functions[name]=fn
    self.context_modules[context.value]=dict(image=image,module=module,functions=functions)
   return self.context_modules[context.value]['functions']
 def launch(self,name,size,values,stream):
  if not size:return
  if size>2147483647:raise ValueError('PORTABLE_LINEAR_KERNEL_GRID_SIZE_EXCEEDS_BOUND')
  f=self._functions()[name];arguments=(ctypes.c_void_p*len(values))(*[ctypes.cast(ctypes.pointer(v),ctypes.c_void_p)for v in values])
  self.check(self.lib.cuLaunchKernel(f,(size+127)//128,1,1,128,1,1,0,ctypes.c_void_p(int(stream)),arguments,None))
 def forward(self,x,w,b,y,M,N,K,stream):self.launch('forward',M*N,[_ptr(x),_ptr(w),_ptr(b),_ptr(y),ctypes.c_int64(M),ctypes.c_int64(N),ctypes.c_int64(K)],stream)
 def backward(self,x,w,g,dx,dw,db,M,N,K,stream):
  dims=[ctypes.c_int64(M),ctypes.c_int64(N),ctypes.c_int64(K)]
  if dx is not None:self.launch('dx',M*K,[_ptr(g),_ptr(w),_ptr(dx)]+dims,stream)
  if dw is not None:self.launch('dw',N*K,[_ptr(x),_ptr(g),_ptr(dw)]+dims,stream)
  if db is not None:self.launch('db',N,[_ptr(g),_ptr(db),ctypes.c_int64(M),ctypes.c_int64(N)],stream)
 def summary(self):
  version=ctypes.c_int();self.check(self.lib.cuDriverGetVersion(ctypes.byref(version)))
  actual_uuid=None;owned_UUIDs=[]
  current=ctypes.c_void_p();self.check(self.lib.cuCtxGetCurrent(ctypes.byref(current)))
  for value in self.context_modules:
   with self.using_context(value):
    device=ctypes.c_int();self.check(self.lib.cuCtxGetDevice(ctypes.byref(device)));raw=(ctypes.c_ubyte*16)();self.check(self.lib.cuDeviceGetUuid(ctypes.cast(raw,ctypes.c_void_p),device.value));h=bytes(raw).hex();uid='GPU-'+h[:8]+'-'+h[8:12]+'-'+h[12:16]+'-'+h[16:20]+'-'+h[20:];owned_UUIDs.append(uid)
    if value==current.value:actual_uuid=uid
  return dict(schema='PORTABLE_LINEAR_FIXED_ORDER_FP32_V2_DRIVER_API',binary_path=str(self.path),binary_sha256=self.build['binary_sha256'],source_sha256=self.build['source_sha256'],execution_backend='Existing libcuda Driver API + device-only fatbin; no new host runtime library',CUDA_driver_API_version=version.value,actual_current_context_GPU_UUID=actual_uuid,actual_kernel_context_GPU_UUIDs=owned_UUIDs,context_binding='Actual weight allocation context from cuPointerGetAttribute, scoped Push/Pop per call; preserves autograd worker thread prior context',torch_version=torch.__version__,torch_CUDA_runtime=torch.version.cuda,counts=dict(self.counts),observed_shapes=list(self.shapes.values()),CPU_fallback=False,atomic=False,float64=False,FMA=False,math_order=self.build['math'],cross_GPU_architecture_acceptance='NOT_PROVED_BY_INSTALLATION')
 def validate(self,x,w,b):
  if not x.is_cuda or not w.is_cuda or x.dtype!=torch.float32 or w.dtype!=torch.float32:raise TypeError('PORTABLE_LINEAR_REQUIRES_CUDA_FLOAT32')
  if x.device!=w.device or x.dim()<1 or w.dim()!=2 or x.shape[-1]!=w.shape[1]:raise ValueError('PORTABLE_LINEAR_INPUT_WEIGHT_SHAPE_OR_DEVICE_CONFLICT')
  if b is not None and (not b.is_cuda or b.dtype!=torch.float32 or b.device!=x.device or b.dim()!=1 or b.shape[0]!=w.shape[0]):raise ValueError('PORTABLE_LINEAR_BIAS_SHAPE_DTYPE_DEVICE_CONFLICT')
  arch=10*torch.cuda.get_device_capability(x.device)[0]+torch.cuda.get_device_capability(x.device)[1]
  if arch not in self.build['compiled_compute_capabilities']:raise RuntimeError('PORTABLE_LINEAR_ARCH_NOT_COMPILED '+str(arch))
  k=(tuple(x.shape),tuple(x.stride()),tuple(w.shape),tuple(w.stride()),None if b is None else tuple(b.stride()))
  if k not in self.shapes:self.shapes[k]=dict(input_shape=list(x.shape),original_input_stride=list(x.stride()),weight_shape=list(w.shape),original_weight_stride=list(w.stride()),original_bias_stride=None if b is None else list(b.stride()),GPU_contiguous_copy_if_needed=True,compute_capability=arch)

def prepare_backend(library_path=None):
 path=str(Path(library_path or Path(__file__).resolve().parent/'build/portable_linear.fatbin').resolve())
 with _LOCK:
  if path not in _CACHE:_CACHE[path]=Backend(path)
  return _CACHE[path]

class _Linear(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,b,backend):
  backend.validate(x,w,b);ctx.backend=backend;ctx.input_shape=tuple(x.shape);ctx.has_bias=b is not None
  M=x.numel()//x.shape[-1];N=w.shape[0];K=w.shape[1]
  if N<=0 or K<=0:raise ValueError('PORTABLE_LINEAR_NONEMPTY_FEATURE_DIMENSIONS_REQUIRED')
  with torch.cuda.device(x.device), backend.tensor_context(w)as context:
   ctx.driver_context=context
   xx=x.contiguous();ww=w.contiguous();bb=None if b is None else b.contiguous();y=torch.empty((*x.shape[:-1],N),dtype=x.dtype,device=x.device)
   backend.forward(xx,ww,bb,y,M,N,K,int(torch.cuda.current_stream(x.device).cuda_stream))
   ctx.save_for_backward(xx,ww);ctx.M=M;ctx.N=N;ctx.K=K;backend.counts['forward']+=1
  return y
 @staticmethod
 @once_differentiable
 def backward(ctx,grad):
  x,w=ctx.saved_tensors;b=ctx.backend;M,N,K=ctx.M,ctx.N,ctx.K
  with torch.cuda.device(x.device), b.using_context(ctx.driver_context):
   g=grad.contiguous();dx=torch.empty_like(x)if ctx.needs_input_grad[0]else None;dw=torch.empty_like(w)if ctx.needs_input_grad[1]else None;db=torch.empty((N,),dtype=x.dtype,device=x.device)if ctx.has_bias and ctx.needs_input_grad[2]else None
   if M:
    b.backward(x,w,g,dx,dw,db,M,N,K,int(torch.cuda.current_stream(x.device).cuda_stream))
   else:
    for t in (dx,dw,db):
     if t is not None:t.zero_()
   b.counts['backward']+=1
   for n,t in [('dX',dx),('dW',dw),('db',db)]:b.counts[n]+=int(t is not None)
  return dx,dw,db,None

def linear(x,weight,bias=None,library_path=None):return _Linear.apply(x,weight,bias,prepare_backend(library_path))
