// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "context_blob.h"
#include "kmt_api.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace mcdm = iree::hal::amdxdna::mcdm;

namespace {

enum class Stage {
  blob,
  discover,
  device,
  host_bo,
  cacheable_bo,
  execbuf_bo,
  all_bos,
  context,
  aperture,
  submit,
  ctxcmd,
};

std::string NarrowArg(const wchar_t* arg) {
  std::string result;
  for (const wchar_t* p = arg; p && *p; ++p) {
    result.push_back(static_cast<char>(*p));
  }
  return result;
}

bool ParseStage(const std::string& text, Stage* out_stage) {
  if (text == "blob") {
    *out_stage = Stage::blob;
  } else if (text == "discover") {
    *out_stage = Stage::discover;
  } else if (text == "device") {
    *out_stage = Stage::device;
  } else if (text == "host-bo") {
    *out_stage = Stage::host_bo;
  } else if (text == "cacheable-bo") {
    *out_stage = Stage::cacheable_bo;
  } else if (text == "execbuf-bo") {
    *out_stage = Stage::execbuf_bo;
  } else if (text == "all-bos") {
    *out_stage = Stage::all_bos;
  } else if (text == "context") {
    *out_stage = Stage::context;
  } else if (text == "aperture") {
    *out_stage = Stage::aperture;
  } else if (text == "submit") {
    *out_stage = Stage::submit;
  } else if (text == "ctxcmd") {
    *out_stage = Stage::ctxcmd;
  } else {
    return false;
  }
  return true;
}

bool ParseU64(const std::string& text, uint64_t* out_value) {
  char* end = nullptr;
  uint64_t value = _strtoui64(text.c_str(), &end, 0);
  if (!end || *end != '\0') return false;
  *out_value = value;
  return true;
}

bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out_bytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  file.seekg(0, std::ios::end);
  std::streamoff size = file.tellg();
  if (size < 0) return false;
  file.seekg(0, std::ios::beg);
  out_bytes->resize(static_cast<size_t>(size));
  if (size == 0) return true;
  file.read(reinterpret_cast<char*>(out_bytes->data()), size);
  return file.good();
}

void PrintContextBlobInfo(size_t private_data_size, size_t xclbin_size,
                          const mcdm::ContextBlobInfo& info) {
  std::cout << "context.private_size=" << private_data_size
            << " xclbin_size=" << xclbin_size
            << " kernel=\"" << info.kernel_name << "\""
            << " pdi=\"" << info.pdi_name << "\""
            << " column_width=" << info.column_width
            << " start_column=" << info.start_column
            << " dpu_kernel_id=0x" << std::hex << info.dpu_kernel_id
            << std::dec << "\n";
  if (!info.kernel_names.empty()) {
    std::cout << "context.kernels=[";
    for (size_t i = 0; i < info.kernel_names.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << info.kernel_names[i];
    }
    std::cout << "]\n";
  }
  if (!info.pdi_names.empty()) {
    std::cout << "context.pdis=[";
    for (size_t i = 0; i < info.pdi_names.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << info.pdi_names[i] << ":0x" << std::hex
                << (i < info.dpu_kernel_ids.size() ? info.dpu_kernel_ids[i]
                                                    : 0)
                << std::dec;
    }
    std::cout << "]\n";
  }
}

bool BuildContextBlobFromXclbinPath(const std::string& xclbin_path,
                                    std::vector<uint8_t>* out_xclbin,
                                    std::vector<uint8_t>* out_private_data,
                                    mcdm::ContextBlobInfo* out_info) {
  if (xclbin_path.empty()) {
    std::cerr << "--xclbin=PATH is required for blob, context, aperture, and "
                 "submit stages\n";
    return false;
  }

  if (!ReadFileBytes(xclbin_path, out_xclbin)) {
    std::cerr << "failed to read xclbin: " << xclbin_path << "\n";
    return false;
  }

  std::string error;
  if (!mcdm::BuildContextPrivateDataFromXclbin(
          out_xclbin->data(), out_xclbin->size(), GetCurrentProcessId(),
          out_private_data, out_info, &error)) {
    std::cerr << "BuildContextPrivateDataFromXclbin failed: " << error << "\n";
    return false;
  }
  return true;
}

bool RunBlobProbe(const std::string& xclbin_path) {
  std::vector<uint8_t> xclbin;
  std::vector<uint8_t> private_data;
  mcdm::ContextBlobInfo info;
  if (!BuildContextBlobFromXclbinPath(xclbin_path, &xclbin, &private_data,
                                      &info)) {
    return false;
  }
  PrintContextBlobInfo(private_data.size(), xclbin.size(), info);
  return true;
}

bool RunBufferProbe(const mcdm::KmtApi& api, const mcdm::Device& device,
                    mcdm::BufferKind kind, uint64_t size) {
  mcdm::BufferKindInfo kind_info = mcdm::GetBufferKindInfo(kind);
  std::string error;
  std::cout << "bo.kind=" << kind_info.name << " private_type=0x" << std::hex
            << kind_info.private_type << " xcl_flags=0x"
            << kind_info.xcl_flags << " size=0x" << size << std::dec << "\n";

  mcdm::Buffer buffer;
  if (!mcdm::CreateBuffer(api, device, kind, size, &buffer, &error)) {
    std::cerr << "CreateBuffer failed: " << error << "\n";
    return false;
  }
  std::cout << "bo.allocation=" << buffer.allocation << " gpu_va=0x"
            << std::hex << buffer.gpu_va << " cpu_ptr=" << buffer.cpu_ptr
            << std::dec << "\n";

  if (!mcdm::SyncBuffer(api, device, buffer, 0, buffer.size, &error)) {
    std::cerr << "SyncBuffer failed: " << error << "\n";
    mcdm::DestroyBuffer(api, device, &buffer);
    return false;
  }
  std::cout << "bo.sync=ok length=0x" << std::hex << buffer.size << std::dec
            << "\n";

  mcdm::DestroyBuffer(api, device, &buffer);
  std::cout << "bo.destroy=ok\n";
  return true;
}

bool RunContextProbe(const mcdm::KmtApi& api, const mcdm::Device& device,
                     const std::string& xclbin_path, bool create_aperture,
                     bool submit, bool allow_unsafe_submit, bool run_ctxcmd) {
  std::vector<uint8_t> xclbin;
  std::vector<uint8_t> private_data;
  mcdm::ContextBlobInfo info;
  if (!BuildContextBlobFromXclbinPath(xclbin_path, &xclbin, &private_data,
                                      &info)) {
    return false;
  }

  if (const char* dump = std::getenv("PROBE_DUMP_BLOB")) {
    FILE* f = nullptr;
    fopen_s(&f, dump, "wb");
    if (f) {
      fwrite(private_data.data(), 1, private_data.size(), f);
      fclose(f);
      std::cout << "blob_dumped=" << dump << " bytes=" << private_data.size()
                << "\n";
    }
  }
  PrintContextBlobInfo(private_data.size(), xclbin.size(), info);

  std::string error;
  mcdm::Context context;
  if (!mcdm::CreateContext(api, device, private_data, &context, &error)) {
    std::cerr << "CreateContext failed: " << error << "\n";
    return false;
  }
  std::cout << "context.handle=" << context.context
            << " hwqueue=" << context.hw_queue
            << " progress_fence=" << context.progress_fence
            << " progress_fence_cpu=" << context.progress_fence_cpu
            << " progress_fence_gpu=0x" << std::hex
            << context.progress_fence_gpu << std::dec << "\n";

  if (run_ctxcmd) {
    // Fire CREATE_AIE4_CTX with the sentinel-initialized status ring and report
    // whether the firmware wrote the completion slot (set TRACE_QHDL=1 to see
    // the slot_state). If the firmware processes the queue, the slot changes
    // from the 0xCC sentinel; if not, it stays 0xCCCCCCCC.
    bool ok = mcdm::SubmitAndWaitCreateAie4Ctx(api, device, &context, &error);
    std::cout << "create_aie4_ctx.result=" << (ok ? "ok" : "fail");
    if (!ok) std::cout << " error=" << error;
    std::cout << "\n";
    // Safe teardown: destroying a context whose CREATE_AIE4_CTX the firmware
    // never drained HANGS the KMD (UPDATE 8 wedge). When PROBE_NO_TEARDOWN is
    // set, leak the context and exit instead of risking the wedge - use this
    // until the firmware is confirmed to process the command (slot written).
    if (std::getenv("PROBE_NO_TEARDOWN")) {
      std::cout << "teardown.skipped=1 (leak to avoid firmware-drain wedge)\n";
      std::cout.flush();
      return ok;
    }
    mcdm::DestroyContext(api, &context);
    return ok;
  }

  if (submit && !allow_unsafe_submit) {
    mcdm::DestroyContext(api, &context);
    std::cout << "submit.skipped=1 reason=unsafe-submit-requires-"
                 "--allow-unsafe-submit\n";
    return true;
  }

  if (create_aperture || submit) {
    mcdm::CommandAperture aperture;
    if (!mcdm::CreateCommandAperture(api, device, context, &aperture, &error)) {
      std::cerr << "CreateCommandAperture failed: " << error << "\n";
      mcdm::DestroyContext(api, &context);
      return false;
    }
    std::cout << "command_aperture.allocation=" << aperture.allocation
              << " gpu_allocation=" << aperture.gpu_allocation
              << " cleanup_allocation=" << aperture.cleanup_allocation
              << " resource=" << aperture.resource
              << " gpu_resource=" << aperture.gpu_resource
              << " gpu_va=0x" << std::hex << aperture.gpu_va
              << " gpu_va_size=0x" << aperture.gpu_va_size
              << " cpu_ptr=" << aperture.cpu_ptr << std::dec << "\n";

    if (!submit) {
      mcdm::DestroyCommandAperture(api, device, &aperture);
      mcdm::DestroyContext(api, &context);
      std::cout << "aperture.destroy=ok\n";
      return true;
    }

    // ISOLATION: allocate IREE-like dispatch buffers before the kick to test
    // whether buffer allocation/residency causes the IREE path's 0xc01e0200.
    if (const char* pb = std::getenv("PROBE_BUFFERS")) {
      int n = atoi(pb);
      if (n <= 0) n = 1;
      for (int i = 0; i < n; ++i) {
        mcdm::Buffer b;
        if (mcdm::CreateBuffer(api, device, mcdm::BufferKind::host_only, 4096,
                               &b, &error)) {
          std::cout << "probe_bo[" << i << "] gpu_va=0x" << std::hex << b.gpu_va
                    << std::dec << "\n";
        } else {
          std::cout << "probe_bo[" << i << "] FAIL: " << error << "\n";
        }
      }
    }

    bool ok =
        mcdm::SubmitAndWaitCommandAperture(api, device, &context, &aperture,
                                           &error);
    if (!ok) {
      std::cerr << "SubmitAndWaitCommandAperture failed: " << error << "\n";
    } else {
      std::cout << "submit.wait=ok\n";
    }
    if (std::getenv("PROBE_NO_TEARDOWN")) {
      std::cout << "teardown.skipped=1\n";
      std::cout.flush();
      return ok;
    }
    mcdm::DestroyCommandAperture(api, device, &aperture);
    mcdm::DestroyContext(api, &context);
    return ok;
  }

  mcdm::DestroyContext(api, &context);
  std::cout << "context.destroy=ok\n";
  return true;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  if (std::getenv("PROBE_WARMUP")) {
    HMODULE xrt = LoadLibraryW(L"xrt_coreutil.dll");
    if (!xrt) {
      xrt = LoadLibraryW(
          L"C:\\Windows\\System32\\DriverStore\\FileRepository\\"
          L"kipudrv.inf_amd64_b3e90d6455884a5f\\xrt_coreutil.dll");
    }
    if (xrt) {
      using OpenFn = void*(__cdecl*)(unsigned);
      auto open = reinterpret_cast<OpenFn>(GetProcAddress(xrt, "xrtDeviceOpen"));
      if (open) {
        void* d = open(0);
        std::cout << "probe_warmup_device=" << d << "\n";
      }
    }
  }
  Stage stage = Stage::all_bos;
  uint64_t size = 4096;
  std::string xclbin_path;
  bool allow_unsafe_submit = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = NarrowArg(argv[i]);
    if (arg.rfind("--stage=", 0) == 0) {
      if (!ParseStage(arg.substr(strlen("--stage=")), &stage)) {
        std::cerr << "unknown stage: " << arg << "\n";
        return 2;
      }
    } else if (arg.rfind("--size=", 0) == 0) {
      if (!ParseU64(arg.substr(strlen("--size=")), &size)) {
        std::cerr << "invalid size: " << arg << "\n";
        return 2;
      }
    } else if (arg.rfind("--xclbin=", 0) == 0) {
      xclbin_path = arg.substr(strlen("--xclbin="));
    } else if (arg == "--allow-unsafe-submit") {
      allow_unsafe_submit = true;
    } else {
      std::cerr << "usage: mcdm_probe.exe "
                   "[--stage=blob|discover|device|host-bo|cacheable-bo|"
                   "execbuf-bo|all-bos|context|aperture|submit|ctxcmd] "
                   "[--size=N] "
                   "[--xclbin=PATH] [--allow-unsafe-submit]\n";
      return 2;
    }
  }

  if (stage == Stage::blob) {
    return RunBlobProbe(xclbin_path) ? 0 : 1;
  }

  std::string error;
  mcdm::KmtApi api;
  if (!api.Load(&error)) {
    std::cerr << error << "\n";
    return 1;
  }
  std::cout << "kmt.load=ok\n";

  mcdm::Adapter adapter;
  if (!mcdm::FindNpuAdapter(api, &adapter, &error)) {
    std::cerr << "FindNpuAdapter failed: " << error << "\n";
    return 1;
  }
  std::wcout << L"adapter.handle=" << adapter.handle << L" desc=\""
             << adapter.description << L"\"\n";
  if (stage == Stage::discover) {
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.handle;
    api.close_adapter(&close);
    return 0;
  }

  mcdm::Device device;
  if (!mcdm::CreateDevice(api, adapter, &device, &error)) {
    std::cerr << "CreateDevice failed: " << error << "\n";
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.handle;
    api.close_adapter(&close);
    return 1;
  }
  std::cout << "device.handle=" << device.device
            << " paging_queue=" << device.paging_queue
            << " paging_sync=" << device.paging_sync_object
            << " paging_fence_cpu=" << device.paging_fence_cpu << "\n";
  if (stage == Stage::device) {
    mcdm::DestroyDevice(api, &device);
    return 0;
  }

  if (stage == Stage::context || stage == Stage::aperture ||
      stage == Stage::submit || stage == Stage::ctxcmd) {
    bool ok = RunContextProbe(api, device, xclbin_path,
                              stage == Stage::aperture ||
                                  stage == Stage::submit,
                              stage == Stage::submit, allow_unsafe_submit,
                              stage == Stage::ctxcmd);
    mcdm::DestroyDevice(api, &device);
    return ok ? 0 : 1;
  }

  bool ok = true;
  if (stage == Stage::host_bo || stage == Stage::all_bos) {
    ok &= RunBufferProbe(api, device, mcdm::BufferKind::host_only, size);
  }
  if (stage == Stage::cacheable_bo || stage == Stage::all_bos) {
    ok &= RunBufferProbe(api, device, mcdm::BufferKind::cacheable, size);
  }
  if (stage == Stage::execbuf_bo || stage == Stage::all_bos) {
    ok &= RunBufferProbe(api, device, mcdm::BufferKind::execbuf, size);
  }

  mcdm::DestroyDevice(api, &device);
  return ok ? 0 : 1;
}
