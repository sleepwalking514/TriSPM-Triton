import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_AOT_MODE = os.getenv("TRITON_CPU_AOT", "0") != "0"

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++"]
ccflags = []

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries, ccflags)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0, 0)

    def unload_module(self, mod):
        # TODO: implement
        pass

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format

    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")

    arg_ptrs_list = ', '.join(f"&arg{i}" for i in signature.keys())
    kernel_fn_args = [i for i, ty in signature.items() if i not in constants and ty != "constexpr"]
    signature_without_constexprs = {i: ty for i, ty in signature.items() if ty != "constexpr"}
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)

    # generate glue code
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

static void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;
  if (N == 1) {{
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} 0, 0, 0, 1, 1, 1);
      return;
  }}

  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  // Don't pay OMP overhead price when a single thread is used.
  if (max_threads == 1) {{
    for (size_t i = 0; i < N; ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
    return;
  }}

  // For now, use the default chunk size, total iterations / max_threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // Extract num_threads metadata.
  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_kernels(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


def make_aot_launcher(constants, signature, ids, kernel_name):
    """Generate a standalone C launcher (header + source) for AOT cross-compilation.

    Instead of a Python C extension, this produces plain C files that can be
    compiled with any C toolchain (including RISC-V cross-compilers) and linked
    with a test harness.

    The launcher calls the Triton-generated kernel symbol directly (no function
    pointer) and dispatches over a sequential 3-D grid — suitable for
    single-threaded gem5 SE mode.
    """
    # Flatten and filter signature
    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    signature_str = ','.join(map(_serialize_signature, signature.values()))
    signature_list = list(filter(bool, signature_str.split(',')))
    signature_flat = {i: s for i, s in enumerate(signature_list)}

    kernel_fn_args = [i for i, ty in signature_flat.items() if i not in constants and ty != "constexpr"]
    arg_decls = ', '.join(f"{ty_to_cpp(signature_flat[i])} arg{i}" for i in kernel_fn_args)
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    flash_head_resident = (
        kernel_name == "flash_attention"
        and os.getenv("TRITON_DISABLE_SPM", "0") != "1"
        and os.getenv("TRITON_SPM_FLASH_HEAD_RESIDENT", "1") != "0"
        and all(i in kernel_fn_args for i in (0, 1, 2, 3, 4))
    )
    flash_head_resident_k_env = os.getenv("TRITON_SPM_FLASH_HEAD_RESIDENT_K")
    flash_head_resident_v_env = os.getenv("TRITON_SPM_FLASH_HEAD_RESIDENT_V")
    flash_head_resident_k = (
        "1" if flash_head_resident_k_env is not None and
        flash_head_resident_k_env != "0" else "0"
    )
    flash_head_resident_v = (
        "1" if flash_head_resident_v_env is not None and
        flash_head_resident_v_env != "0" else "0"
    )
    flash_head_resident_chunk_rows = int(
        os.getenv("TRITON_SPM_FLASH_HEAD_RESIDENT_CHUNK_ROWS", "128"), 0)
    if flash_head_resident_chunk_rows <= 0:
        flash_head_resident_chunk_rows = 1

    alloc_cases = f"    default: return {kernel_name}_arg_malloc(arg_index, nbytes);"

    # Extern declaration for the Triton-generated kernel symbol.
    # Triton's LLVM lowering appends 6 i32 args: pid_x/y/z, gridX/Y/Z.
    extern_arg_types = ', '.join(
        [ty_to_cpp(signature_flat[i]) for i in kernel_fn_args] + ["int32_t"] * 6
    )

    guard = f"{kernel_name.upper()}_LAUNCHER_H"

    extra_helpers = ""
    launch_body = f"""\
    for (int32_t z = 0; z < gridZ; ++z)
        for (int32_t y = 0; y < gridY; ++y)
            for (int32_t x = 0; x < gridX; ++x)
                {kernel_name}({kernel_fn_args_list + ', ' if kernel_fn_args_list else ''}x, y, z, gridX, gridY, gridZ);
"""

    if flash_head_resident:
        extra_helpers = f"""\
static size_t {kernel_name}_align_up_size(size_t value, size_t alignment)
{{
    return (value + alignment - 1) & ~(alignment - 1);
}}

static inline void {kernel_name}_head_dma_enqueue_2d(void *dst,
                                                     const void *src,
                                                     size_t width,
                                                     size_t height,
                                                     size_t src_stride,
                                                     size_t dst_stride)
{{
#ifdef USE_XSPM_INSN
    xspm_dma_stride((uint64_t)src_stride, (uint64_t)dst_stride);
    xspm_dma_2d((uintptr_t)dst, (uintptr_t)src, (uint32_t)width,
                (uint32_t)height);
#else
    spm_dma_enqueue_2d(dst, src, width, height, src_stride, dst_stride);
#endif
}}

static inline void {kernel_name}_head_dma_wait(void)
{{
#ifdef USE_XSPM_INSN
    xspm_dma_wait();
#else
    (void)spm_dma_wait();
#endif
}}

"""
        launch_body = f"""\
    if (gridX > 0 && gridY > 0 &&
        {kernel_name}_arg_bytes[0] != 0 &&
        {kernel_name}_arg_bytes[1] != 0 &&
        {kernel_name}_arg_bytes[1] == {kernel_name}_arg_bytes[2] &&
        ({kernel_name}_arg_bytes[1] % (size_t)gridY) == 0 &&
        ({kernel_name}_arg_bytes[0] % ((size_t)gridX * (size_t)gridY)) == 0) {{
        size_t head_bytes = {kernel_name}_arg_bytes[1] / (size_t)gridY;
        size_t q_tile_bytes =
            {kernel_name}_arg_bytes[0] / ((size_t)gridX * (size_t)gridY);
        size_t row_bytes = q_tile_bytes / 16;
        size_t seq_rows = row_bytes != 0 ? head_bytes / row_bytes : 0;
        size_t head_dma_chunk_rows = (size_t){flash_head_resident_chunk_rows};

        if (head_bytes >= 8192 && gridX >= 2 &&
            row_bytes != 0 && seq_rows != 0 && head_dma_chunk_rows != 0 &&
            row_bytes * seq_rows == head_bytes) {{
            int resident_k = {flash_head_resident_k};
            int resident_v = {flash_head_resident_v};
            uintptr_t spm_base = (uintptr_t)SPM_BASE;
            size_t spm_size = get_spm_size();
            uintptr_t next_spm =
                spm_base + {kernel_name}_align_up_size(q_tile_bytes, 64);
            uintptr_t k_spm = 0;
            uintptr_t v_spm = 0;
            if (resident_k) {{
                k_spm = (uintptr_t){kernel_name}_align_up_size((size_t)next_spm, 64);
                next_spm = k_spm + head_bytes;
            }}
            if (resident_v) {{
                v_spm = (uintptr_t){kernel_name}_align_up_size((size_t)next_spm, 64);
                next_spm = v_spm + head_bytes;
            }}

            if ((resident_k || resident_v) &&
                next_spm >= spm_base && next_spm <= spm_base + spm_size) {{
                for (int32_t z = 0; z < gridZ; ++z) {{
                    for (int32_t y = 0; y < gridY; ++y) {{
                        size_t head_offset = (size_t)y * head_bytes;
                        for (size_t row = 0; row < seq_rows;
                             row += head_dma_chunk_rows) {{
                            size_t rows = seq_rows - row;
                            if (rows > head_dma_chunk_rows)
                                rows = head_dma_chunk_rows;
                            size_t byte_offset = row * row_bytes;
                            if (resident_k)
                                {kernel_name}_head_dma_enqueue_2d(
                                    (void *)(k_spm + byte_offset),
                                    (const void *)((const char *)arg1 +
                                                   head_offset + byte_offset),
                                    row_bytes, rows, row_bytes, row_bytes);
                            if (resident_v)
                                {kernel_name}_head_dma_enqueue_2d(
                                    (void *)(v_spm + byte_offset),
                                    (const void *)((const char *)arg2 +
                                                   head_offset + byte_offset),
                                    row_bytes, rows, row_bytes, row_bytes);
                            {kernel_name}_head_dma_wait();
                        }}

                        void *k_arg = resident_k ? (void *)(k_spm - head_offset) : arg1;
                        void *v_arg = resident_v ? (void *)(v_spm - head_offset) : arg2;
                        for (int32_t x = 0; x < gridX; ++x)
                            {kernel_name}(arg0, k_arg, v_arg, arg3, arg4,
                                          x, y, z, gridX, gridY, gridZ);
                    }}
                }}
                return;
            }}
        }}
    }}

    for (int32_t z = 0; z < gridZ; ++z)
        for (int32_t y = 0; y < gridY; ++y)
            for (int32_t x = 0; x < gridX; ++x)
                {kernel_name}({kernel_fn_args_list + ', ' if kernel_fn_args_list else ''}x, y, z, gridX, gridY, gridZ);
"""

    # --- Header (C-compatible) ---
    header = f"""\
#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <stddef.h>

/* Launch the Triton kernel over a 3-D grid (sequential, for gem5 SE mode). */
void {kernel_name}_launch(
    int32_t gridX, int32_t gridY, int32_t gridZ
    {(', ' + arg_decls) if arg_decls else ''});

void *{kernel_name}_alloc(int arg_index, size_t nbytes);
void {kernel_name}_free_all(void);

#endif /* {guard} */
"""

    # --- Source (C-compatible) ---
    real_hw_launch_init = ""
    if (
        os.getenv("TRISPM_REAL_HW", "0") == "1"
        and os.getenv("TRITON_DISABLE_SPM", "0") != "1"
    ):
        real_hw_launch_init = """\
#ifdef TRISPM_REAL_HW
    if (trispm_real_hw_init_spm() != 0)
        return;
#endif
"""

    source = f"""\
#include "{kernel_name}_launcher.h"

#include <stdlib.h>

#include "libspm.h"

/* Triton-generated kernel symbol (from the cross-compiled .s file). */
extern void {kernel_name}({extern_arg_types});

static void *{kernel_name}_malloc_ptrs[64];
static int {kernel_name}_malloc_count = 0;
static size_t {kernel_name}_arg_bytes[64];

static void *{kernel_name}_record_malloc(void *ptr)
{{
    if (ptr && {kernel_name}_malloc_count < 64)
        {kernel_name}_malloc_ptrs[{kernel_name}_malloc_count++] = ptr;
    return ptr;
}}

static int {kernel_name}_arg_uses_dma_buf(int arg_index)
{{
#ifdef TRISPM_REAL_HW
    const char *env = getenv("TRISPM_REAL_HW_UDMA_ALLOC_ARGS");
    if (!env || !env[0])
        return 0;

    const char *p = env;
    while (*p) {{
        char *end = NULL;
        long value = strtol(p, &end, 10);
        if (end != p && value == arg_index)
            return 1;
        if (end == p)
            ++p;
        else
            p = end;
        while (*p == ',' || *p == ' ' || *p == '\\t')
            ++p;
    }}
#else
    (void)arg_index;
#endif
    return 0;
}}

static void *{kernel_name}_arg_malloc(int arg_index, size_t nbytes)
{{
    if ({kernel_name}_arg_uses_dma_buf(arg_index))
        return dma_buf_malloc(nbytes);
    return {kernel_name}_record_malloc(malloc(nbytes));
}}

{extra_helpers}

void *{kernel_name}_alloc(int arg_index, size_t nbytes)
{{
    if (arg_index >= 0 && arg_index < 64)
        {kernel_name}_arg_bytes[arg_index] = nbytes;
    switch (arg_index) {{
{alloc_cases}
    }}
}}

void {kernel_name}_free_all(void)
{{
    for (int i = 0; i < {kernel_name}_malloc_count; ++i)
        free({kernel_name}_malloc_ptrs[i]);
    {kernel_name}_malloc_count = 0;
    spm_free_all();
    dma_buf_free_all();
}}

void {kernel_name}_launch(
    int32_t gridX, int32_t gridY, int32_t gridZ
    {(', ' + arg_decls) if arg_decls else ''})
{{
{real_hw_launch_init}
{launch_body}
}}
"""

    return header, source


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}

        if _AOT_MODE:
            kernel_name = metadata.name if hasattr(metadata, 'name') else metadata.get("name", "kernel")
            header, source = make_aot_launcher(constants, signature, ids, kernel_name)
            launcher_dir = os.getenv("KERNEL_AUX_FILE_DIR",
                                     os.getenv("KERNEL_LAUNCHER_DIR", "."))
            os.makedirs(launcher_dir, exist_ok=True)
            header_path = os.path.join(launcher_dir, f"{kernel_name}_launcher.h")
            source_path = os.path.join(launcher_dir, f"{kernel_name}_launcher.c")
            with open(header_path, "w") as f:
                f.write(header)
            with open(source_path, "w") as f:
                f.write(source)
            print(f"[AOT] Wrote {header_path}")
            print(f"[AOT] Wrote {source_path}")
            self.launch = lambda *args, **kwargs: None  # no-op in AOT mode
        else:
            src_code = make_launcher(constants, signature, ids)
            mod = compile_module_from_src(src_code, "__triton_cpu_launcher")
            self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.knobs.runtime.launch_enter_hook = None
        triton.knobs.runtime.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.knobs.runtime.launch_enter_hook = lambda arg: self._enter_hook()
        triton.knobs.runtime.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    # TODO maybe CPU should do anything here
    def clear_cache(self, cache):
        cache.zero_()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)
