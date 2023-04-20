import datetime
import linecache
import os

os.environ['CUDA_LAUNCH_BLOCKING']='1'

#import pynvml3
from py3nvml import py3nvml
import torch
import socket

# different settings
print_tensor_sizes = False
use_incremental = True
empty_cache = True


if 'GPU_DEBUG' in os.environ:
    gpu_profile_fn = f"HOST_{socket.gethostname()}_gpu{os.environ['GPU_DEBUG']}_mem_prof-{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
    print('profiling gpu usage to ', gpu_profile_fn)


## Global variables
gpu_id = int(os.environ['GPU_DEBUG'])
last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None
guid = 0

def gpu_eval():
    global last_tensor_sizes
    torch.cuda.empty_cache()

    with open(gpu_profile_fn, 'a+') as f:
        new_tensor_sizes = {(type(tensor), tuple(tensor.size()), tensor.dbg_alloc_where) 
                            for tensor in get_tensors() if hasattr(tensor, 'dbg_alloc_where')}
        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
            f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')
        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
            f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')
        last_tensor_sizes = new_tensor_sizes

        for tensor in get_tensors():
            if hasattr(tensor, 'dbg_age'):
                tensor.dbg_age += 1

                if tensor.dbg_age > 20:
                    f.write(f'stale tensor: {tensor.dbg_alloc_where} {tensor.dbg_age} {tensor.size()} {tensor.dtype}\n')

    

def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_meminfo_used
    global lineno, func_name, filename, module_name, guid

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+'.'+func_name+':'+str(lineno)

                new_meminfo_used = meminfo.used
                mem_display = new_meminfo_used-last_meminfo_used if use_incremental else new_meminfo_used
                if mem_display != 0:
                    for tensor in get_tensors():
                        if not hasattr(tensor, 'dbg_alloc_where'):
                            tensor.dbg_alloc_where = f"{where_str}-{guid}"
                            tensor.dbg_age = 0
                            guid += 1
                        
                        #new_tensor_sizes.add((type(tensor), tuple(tensor.size()), tensor.dbg_alloc_where))

                    with open(gpu_profile_fn, 'a+') as f:
                        f.write(f"{where_str:<50}"
                                f":{(mem_display)/1024**2:<7.1f}Mb "
                                f"{line.rstrip()}\n")

                    
                    last_meminfo_used = new_meminfo_used

                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno
            
            #only profile codes within the parenet folder, otherwise there are too many function calls into other pytorch scripts
            #need to modify the key words below to suit your case.
            dirname = os.path.dirname(os.path.abspath(filename))
            if ('openspeech' not in dirname) and ('lightning' not in dirname):
                lineno = None  # skip current line evaluation

            if ('openspeech.data' in module_name
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip othe unnecessary lines
            
            return gpu_profile

        except (KeyError, AttributeError) as e:
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass