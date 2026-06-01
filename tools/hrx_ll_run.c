// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Configurable low-level NPU run harness for the xrt-lite HAL driver.
//
// Generalizes hrx_npu_matmul.c into a parametrized runner: it dispatches an
// arbitrary precompiled HAL executable (amdaie-pdi-fb / PDIR flatbuffer) with a
// caller-described set of buffer bindings, then dumps the output buffers.
//
// Drives the AMD NPU through libhrx (no IREE VM / vmfb):
//   1. HRX_GPU_DRIVER=xrt-lite -> create + enumerate the NPU device.
//   2. Load the PDIR executable from disk.
//   3. Allocate one DEVICE_LOCAL|HOST_VISIBLE buffer per binding, upload
//   inputs.
//   4. Dispatch entry |ordinal| with workgroup |wg| (default 1x1x1).
//   5. Read back and dump output buffers.
//
// The whole create/dispatch/destroy cycle can be repeated --iters times in a
// single process to exercise device teardown (regression cover for the
// HRX/xrt-lite shutdown heap bug).
//
// Usage:
//   HRX_GPU_DRIVER=xrt-lite HRX_XRT_LITE_N_CORE_ROWS=4
//   HRX_XRT_LITE_N_CORE_COLS=1 \
//     hrx-ll-run --pdi=<exe.amdaie-pdi-fb> \
//       --constant=u32:123 \
//       --binding=in:i32:4096:fill=1 \
//       --binding=in:i32:4096:fill=2 \
//       --binding=out:i32:1024:dump=out0.bin \
//       [--ordinal=0] [--wg=1,1,1] [--iters=1]
//
// Binding spec (repeatable, order = dispatch binding order):
//   <kind>:<dtype>:<count>[:<source>]
//     kind   = in | out
//     dtype  = i8|i16|i32|i64|f32  (used for element size + dump preview)
//     count  = number of elements
//     source = fill=<int>     fill input with a constant (in only)
//              file=<path>    read raw input bytes from file (in only)
//              dump=<path>    write raw output bytes to file (out only)
//     (in bindings default to fill=0; out bindings default to dumping nothing)
//
// Constants:
//   --constant=<u32> or --constant=u32:<u32>
//     Append one 32-bit dispatch constant. The xrt-lite HAL can patch
//     amdaie.npu.write32 values marked as 0xA1EC0000 | constant_index from
//     this block.
//
// Returns 0 on success, non-zero otherwise.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hrx_runtime.h"

#define MAX_BINDINGS 32
#define MAX_CONSTANT_WORDS 256

#define CHECK_OK(expr, label)                                                  \
  do {                                                                         \
    hrx_status_t st__ = (expr);                                                \
    if (!hrx_status_is_ok(st__)) {                                             \
      char *msg__ = NULL;                                                      \
      size_t len__ = 0;                                                        \
      hrx_status_to_string(st__, &msg__, &len__);                              \
      fprintf(stderr, "FAIL %s: %s\n", label, msg__ ? msg__ : "(no message)"); \
      hrx_status_free_message(msg__);                                          \
      hrx_status_ignore(st__);                                                 \
      return 2;                                                                \
    }                                                                          \
  } while (0)

typedef enum { KIND_IN, KIND_OUT } binding_kind_t;
typedef enum { SRC_NONE, SRC_FILL, SRC_FILE, SRC_DUMP } source_kind_t;

typedef struct {
  binding_kind_t kind;
  size_t elem_size;     // bytes per element
  char dtype[8];        // e.g. "i32"
  size_t count;         // element count
  source_kind_t source; // how to init (in) or capture (out)
  int64_t fill_value;   // for SRC_FILL
  char path[512];       // for SRC_FILE / SRC_DUMP
} binding_spec_t;

static size_t dtype_size(const char *dtype) {
  if (!strcmp(dtype, "i8") || !strcmp(dtype, "u8"))
    return 1;
  if (!strcmp(dtype, "i16") || !strcmp(dtype, "u16") || !strcmp(dtype, "f16"))
    return 2;
  if (!strcmp(dtype, "i32") || !strcmp(dtype, "u32") || !strcmp(dtype, "f32"))
    return 4;
  if (!strcmp(dtype, "i64") || !strcmp(dtype, "u64") || !strcmp(dtype, "f64"))
    return 8;
  return 0;
}

// Parse "<kind>:<dtype>:<count>[:<source>]" into |out|. Returns 0 on success.
static int parse_binding(const char *spec, binding_spec_t *out) {
  memset(out, 0, sizeof(*out));
  char buf[640];
  strncpy(buf, spec, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  char *save = NULL;
  char *kind = strtok_r(buf, ":", &save);
  char *dtype = strtok_r(NULL, ":", &save);
  char *count = strtok_r(NULL, ":", &save);
  char *source = strtok_r(NULL, "", &save); // rest of string (may contain '=')
  if (!kind || !dtype || !count) {
    fprintf(stderr, "bad --binding=%s (need kind:dtype:count[:source])\n",
            spec);
    return 1;
  }

  if (!strcmp(kind, "in")) {
    out->kind = KIND_IN;
  } else if (!strcmp(kind, "out")) {
    out->kind = KIND_OUT;
  } else {
    fprintf(stderr, "bad binding kind '%s' (use in|out)\n", kind);
    return 1;
  }

  out->elem_size = dtype_size(dtype);
  if (out->elem_size == 0) {
    fprintf(stderr, "unknown dtype '%s'\n", dtype);
    return 1;
  }
  strncpy(out->dtype, dtype, sizeof(out->dtype) - 1);
  out->count = (size_t)strtoull(count, NULL, 0);
  if (out->count == 0) {
    fprintf(stderr, "binding count must be > 0\n");
    return 1;
  }

  out->source = SRC_NONE;
  if (source && *source) {
    if (!strncmp(source, "fill=", 5)) {
      out->source = SRC_FILL;
      out->fill_value = (int64_t)strtoll(source + 5, NULL, 0);
    } else if (!strncmp(source, "file=", 5)) {
      out->source = SRC_FILE;
      strncpy(out->path, source + 5, sizeof(out->path) - 1);
    } else if (!strncmp(source, "dump=", 5)) {
      out->source = SRC_DUMP;
      strncpy(out->path, source + 5, sizeof(out->path) - 1);
    } else {
      fprintf(stderr, "bad binding source '%s'\n", source);
      return 1;
    }
  }
  if (out->kind == KIND_IN && out->source == SRC_DUMP) {
    fprintf(stderr, "input bindings cannot use dump=\n");
    return 1;
  }
  if (out->kind == KIND_OUT &&
      (out->source == SRC_FILL || out->source == SRC_FILE)) {
    fprintf(stderr, "output bindings cannot use fill=/file=\n");
    return 1;
  }
  return 0;
}

// Fill a host buffer with the integer fill_value, replicated per element.
static void fill_host(void *dst, const binding_spec_t *b) {
  for (size_t i = 0; i < b->count; ++i) {
    void *e = (char *)dst + i * b->elem_size;
    switch (b->elem_size) {
    case 1:
      *(int8_t *)e = (int8_t)b->fill_value;
      break;
    case 2:
      *(int16_t *)e = (int16_t)b->fill_value;
      break;
    case 4:
      *(int32_t *)e = (int32_t)b->fill_value;
      break;
    case 8:
      *(int64_t *)e = (int64_t)b->fill_value;
      break;
    default:
      break;
    }
  }
}

// Print a small preview of an output buffer for quick eyeballing.
static void preview_output(int idx, const binding_spec_t *b, const void *data) {
  size_t n = b->count < 8 ? b->count : 8;
  printf("  out[%d] %s x%zu preview:", idx, b->dtype, b->count);
  for (size_t i = 0; i < n; ++i) {
    const void *e = (const char *)data + i * b->elem_size;
    long long v = 0;
    switch (b->elem_size) {
    case 1:
      v = *(const int8_t *)e;
      break;
    case 2:
      v = *(const int16_t *)e;
      break;
    case 4:
      v = *(const int32_t *)e;
      break;
    case 8:
      v = *(const int64_t *)e;
      break;
    }
    printf(" %lld", v);
  }
  printf("%s\n", b->count > n ? " ..." : "");
}

static int run_once(const char *exe_path, uint32_t ordinal,
                    const uint32_t wg[3], const binding_spec_t *bindings,
                    int binding_count, const uint32_t *constants,
                    int constant_count, int iter, int verbose) {
  int device_count = 0;
  CHECK_OK(hrx_gpu_device_count(&device_count), "hrx_gpu_device_count");
  if (device_count < 1) {
    fprintf(stderr, "FAIL: no NPU device enumerated\n");
    return 3;
  }

  hrx_device_t device = NULL;
  CHECK_OK(hrx_gpu_device_get(0, &device), "hrx_gpu_device_get");

  hrx_executable_t executable = NULL;
  CHECK_OK(
      hrx_executable_load_file(device, exe_path, "amdaie-pdi-fb", &executable),
      "hrx_executable_load_file");

  hrx_allocator_t allocator = hrx_device_allocator(device);

  hrx_buffer_params_t params = {0};
  params.type = HRX_MEMORY_TYPE_DEVICE_LOCAL | HRX_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;
  params.queue_affinity = (hrx_queue_affinity_t)~0ull;

  hrx_buffer_t bufs[MAX_BINDINGS] = {0};
  hrx_buffer_ref_t refs[MAX_BINDINGS];
  for (int i = 0; i < binding_count; ++i) {
    size_t bytes = bindings[i].count * bindings[i].elem_size;
    CHECK_OK(hrx_allocator_allocate_buffer(allocator, params, bytes, &bufs[i]),
             "allocate binding");
    refs[i].buffer = bufs[i];
    refs[i].offset = 0;
    refs[i].length = bytes;

    if (bindings[i].kind == KIND_IN) {
      void *host = malloc(bytes);
      if (!host) {
        fprintf(stderr, "FAIL: host alloc for input %d\n", i);
        return 2;
      }
      if (bindings[i].source == SRC_FILE) {
        FILE *f = fopen(bindings[i].path, "rb");
        if (!f) {
          fprintf(stderr, "FAIL: open input file %s\n", bindings[i].path);
          free(host);
          return 2;
        }
        size_t rd = fread(host, 1, bytes, f);
        fclose(f);
        if (rd != bytes) {
          fprintf(stderr, "FAIL: short read on %s (%zu/%zu)\n",
                  bindings[i].path, rd, bytes);
          free(host);
          return 2;
        }
      } else {
        // SRC_FILL or SRC_NONE (default 0).
        fill_host(host, &bindings[i]);
      }
      CHECK_OK(hrx_synchronous_h2d(device, host, bufs[i], 0, bytes), "h2d");
      free(host);
    }
  }

  hrx_dispatch_config_t config = {0};
  config.workgroup_count[0] = wg[0];
  config.workgroup_count[1] = wg[1];
  config.workgroup_count[2] = wg[2];

  hrx_semaphore_t sem = NULL;
  CHECK_OK(hrx_semaphore_create(device, 0, &sem), "semaphore_create");
  hrx_semaphore_t sig_sems[1] = {sem};
  uint64_t sig_vals[1] = {1};
  hrx_semaphore_list_t signal_list = {sig_sems, sig_vals, 1};

  CHECK_OK(hrx_queue_dispatch(device, /*affinity=*/0, /*wait=*/NULL,
                              &signal_list, executable, ordinal, &config,
                              constants,
                              (size_t)constant_count * sizeof(uint32_t), refs,
                              (size_t)binding_count, HRX_DISPATCH_FLAG_NONE),
           "hrx_queue_dispatch");
  CHECK_OK(hrx_semaphore_wait(sem, 1, UINT64_MAX), "semaphore_wait");

  // Read back + dump outputs.
  for (int i = 0; i < binding_count; ++i) {
    if (bindings[i].kind != KIND_OUT)
      continue;
    size_t bytes = bindings[i].count * bindings[i].elem_size;
    void *host = malloc(bytes);
    if (!host) {
      fprintf(stderr, "FAIL: host alloc for output %d\n", i);
      return 2;
    }
    CHECK_OK(hrx_synchronous_d2h(device, bufs[i], 0, host, bytes), "d2h");
    if (verbose)
      preview_output(i, &bindings[i], host);
    if (bindings[i].source == SRC_DUMP) {
      FILE *f = fopen(bindings[i].path, "wb");
      if (!f) {
        fprintf(stderr, "FAIL: open dump file %s\n", bindings[i].path);
        free(host);
        return 2;
      }
      size_t wr = fwrite(host, 1, bytes, f);
      fclose(f);
      if (wr != bytes) {
        fprintf(stderr, "FAIL: short write on %s\n", bindings[i].path);
        free(host);
        return 2;
      }
      if (verbose)
        printf("  wrote %zu bytes to %s\n", bytes, bindings[i].path);
    }
    free(host);
  }

  // Tear everything down (exercises the device-destroy path on the last iter
  // and every iter when --iters>1).
  hrx_semaphore_release(sem);
  for (int i = 0; i < binding_count; ++i) {
    if (bufs[i])
      hrx_buffer_release(bufs[i]);
  }
  hrx_executable_release(executable);

  if (verbose)
    printf("iter %d: dispatch OK\n", iter);
  return 0;
}

int main(int argc, char **argv) {
  const char *exe_path = NULL;
  uint32_t ordinal = 0;
  uint32_t wg[3] = {1, 1, 1};
  int iters = 1;
  binding_spec_t bindings[MAX_BINDINGS];
  int binding_count = 0;
  uint32_t constants[MAX_CONSTANT_WORDS];
  int constant_count = 0;

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strncmp(a, "--pdi=", 6)) {
      exe_path = a + 6;
    } else if (!strncmp(a, "--ordinal=", 10)) {
      ordinal = (uint32_t)strtoul(a + 10, NULL, 0);
    } else if (!strncmp(a, "--iters=", 8)) {
      iters = (int)strtol(a + 8, NULL, 0);
    } else if (!strncmp(a, "--wg=", 5)) {
      if (sscanf(a + 5, "%u,%u,%u", &wg[0], &wg[1], &wg[2]) != 3) {
        fprintf(stderr, "bad --wg=%s (need x,y,z)\n", a + 5);
        return 64;
      }
    } else if (!strncmp(a, "--binding=", 10)) {
      if (binding_count >= MAX_BINDINGS) {
        fprintf(stderr, "too many bindings (max %d)\n", MAX_BINDINGS);
        return 64;
      }
      if (parse_binding(a + 10, &bindings[binding_count]) != 0)
        return 64;
      binding_count++;
    } else if (!strncmp(a, "--constant=", 11)) {
      if (constant_count >= MAX_CONSTANT_WORDS) {
        fprintf(stderr, "too many constants (max %d)\n", MAX_CONSTANT_WORDS);
        return 64;
      }
      const char *value = a + 11;
      if (!strncmp(value, "u32:", 4))
        value += 4;
      constants[constant_count++] = (uint32_t)strtoul(value, NULL, 0);
    } else {
      fprintf(stderr, "unknown arg: %s\n", a);
      return 64;
    }
  }

  if (!exe_path || binding_count == 0) {
    fprintf(stderr,
            "usage: %s --pdi=<exe.amdaie-pdi-fb> "
            "--binding=<kind:dtype:count[:source]> ... "
            "[--constant=u32:V] [--ordinal=N] [--wg=x,y,z] [--iters=N]\n",
            argv[0]);
    return 64;
  }
  if (iters < 1)
    iters = 1;

  // Each iteration does a full gpu_initialize (creates the NPU device) ->
  // dispatch -> gpu_shutdown (destroys the device). Looping init/shutdown is
  // what exercises the device-teardown path repeatedly in one process, which
  // is the regression cover for the HRX/xrt-lite shutdown heap bug.
  int rc = 0;
  for (int it = 0; it < iters && rc == 0; ++it) {
    CHECK_OK(hrx_gpu_initialize(0), "hrx_gpu_initialize");
    rc = run_once(exe_path, ordinal, wg, bindings, binding_count, constants,
                  constant_count, it, /*verbose=*/1);
    // Always shut down (the whole point is to validate clean teardown).
    hrx_status_ignore(hrx_gpu_shutdown());
  }

  if (rc == 0) {
    printf("hrx-ll-run: OK (%d iter%s)\n", iters, iters == 1 ? "" : "s");
  } else {
    fprintf(stderr, "hrx-ll-run: FAILED (rc=%d)\n", rc);
  }
  return rc;
}
