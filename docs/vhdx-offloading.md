# VHDX Virtual Disk Memory Offloading

Ollama can use a **VHDX virtual hard drive** as a third memory tier behind
VRAM and system RAM. This lets Windows hosts load and run models that
are larger than the combined VRAM + RAM available on the machine by
streaming "cold" layers from disk while keeping "hot" layers resident in
system memory.

> **Status:** the storage abstraction, tiered cache, prefetcher, and
> `ollama ps` reporting ship by default. The CGO bridge that lets
> `llama.cpp` read weights directly from the storage provider is built
> with the `ollama_vhdx_bridge` build tag and is currently marked
> experimental. Default builds behave exactly as before.

---

## Why VHDX?

- **Resiliency and large capacity.** VHDX supports files up to 64 TiB
  and is resilient to mid-write power loss, making it a safer choice
  than raw files for model-sized offload regions.
- **Allocation-unit control.** You can format the virtual volume with a
  **64 KiB** cluster size, which matches the block size Ollama uses for
  every on-disk layer slot and minimizes read amplification.
- **Portability.** The VHDX file is a single, movable artifact you can
  place on any NTFS or ReFS volume.

## Hardware guidance

| Aspect            | Recommendation                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------ |
| Host file system  | **NTFS** or **ReFS** (robust large-file handling, efficient metadata).                     |
| Host allocation   | Format the host volume with a 64 KiB allocation unit (`format X: /FS:NTFS /A:64K /Q`).     |
| Host storage      | **NVMe SSD** strongly preferred; SATA SSD acceptable; avoid HDD for interactive inference. |
| VHDX format       | **Fixed** provisioning (consistent latency; no runtime growth).                            |
| VHDX cluster      | 64 KiB block size (`-BlockSizeBytes 65536` for `New-VHD`).                                 |
| VHDX file system  | NTFS or ReFS with a 64 KiB allocation unit (same as the host guidance).                    |

## Creating a VHDX

Ollama ships a helper at `llm/vhdx.Provision` that drives PowerShell's
`New-VHD` cmdlet (and degrades gracefully to a plain preallocated file
on editions without Hyper-V tooling, so developer laptops still work).
You can also create one manually:

```powershell
# Open an elevated PowerShell prompt.
New-VHD -Path 'D:\ollama\offload.vhdx' -SizeBytes 64GB -Fixed -BlockSizeBytes 65536
Mount-VHD -Path 'D:\ollama\offload.vhdx'
# ...then, in Disk Management or diskpart, initialize the disk, create a
# partition, and format it NTFS with /A:64K.
```

## Enabling VHDX offloading

Set the `OLLAMA_VHDX_OFFLOAD_PATH` environment variable to the directory
containing the VHDX-hosted file system before starting `ollama serve`:

```powershell
$env:OLLAMA_VHDX_OFFLOAD_PATH = 'V:\ollama-offload'
ollama serve
```

When the variable is empty (the default), VHDX offloading is disabled
and Ollama behaves exactly as before. The variable is only surfaced on
Windows; `AsMap()` on other platforms does not include it.

## Design in one picture

```
                      +--------------------------+
  inference engine -> |   StorageProvider (IF)   |
                      +--------------------------+
                         ^              ^
                         |              |
                 +---------------+  +--------------------+
                 |  RAMStorage   |  |   VHDXStorage      |
                 | (hot, bounded)|  | (cold, file-backed)|
                 +---------------+  +--------------------+
                         \              /
                          \            /
                           +----------+
                           |  Tiered  |  LRU hot + cold fallback
                           +----------+
                                |
                           +----------+
                           |Prefetcher|  async warm-up
                           +----------+
```

The public interface is `vhdx.StorageProvider` in
[`llm/vhdx`](../llm/vhdx). All inference code depends on the interface,
never on a concrete RAM or VHDX type (**Dependency Inversion**), and any
implementation can be substituted without behavioral change (**Liskov
Substitution**).

### Behavioral guarantees

- **Layer framing.** Every layer on disk is serialized as an 8-byte
  little-endian size header followed by the layer bytes, padded up to
  the next 64 KiB boundary.
- **Concurrency.** All providers are safe for concurrent use.
- **Fail-open.** A fatal I/O error latches the VHDX provider as failed
  (`VHDXStorage.Failed()`). `Tiered` then falls back to the RAM tier,
  and `ollama ps` continues to report the live state.
- **Memory preservation.** The RAM tier is bounded via
  `Tiered.HotCapacity` (LRU). System RAM is **not** sized against VHDX
  capacity — the cold tier absorbs overflow instead.

## Visibility: `ollama ps`

When VHDX offloading is active the `PROCESSOR` column shows an
additional suffix so you can see at a glance what fraction of the model
is coming off disk:

```
NAME         ID            SIZE     PROCESSOR                    CONTEXT    UNTIL
big:70b      abcdef012345  42 GB    25%/50% CPU/GPU +25% VHDX    4096       4 minutes from now
```

The `/api/ps` HTTP endpoint exposes the same information as a new
`size_vhdx` field on `ProcessModelResponse`. The field is omitted from
the JSON body when zero, so existing clients are unaffected.

## Inference engine bridge (experimental)

The package `llm/vhdx` exports two C-callable symbols for future
integration with `llama.cpp` / `ggml`:

```c
int64_t ollama_vhdx_read_layer(uint64_t handle, int index,
                               int64_t offset, char *dst,
                               int64_t size);
int64_t ollama_vhdx_layer_size(uint64_t handle, int index);
```

The `handle` is the value returned by `ml.RegisterWeightLoader` on the
Go side. These symbols are **only** compiled into builds produced with

```sh
go build -tags ollama_vhdx_bridge ./...
```

Default builds do not expose the symbols and are byte-for-byte
equivalent to pre-enhancement builds.

## Testing

The `llm/vhdx` and `ml` packages ship with tests covering:

- RAM and VHDX round-trip, defensive copies, invalid inputs, overwrites.
- 64 KiB alignment, I/O failure latching, concurrent access.
- Hot-tier preference, cold-miss promotion, LRU eviction, cold-failure
  fallback in `Tiered`.
- Prefetch deduplication, worker-pool cap, and context cancellation.
- Weight loader registry isolation and lookup after unregister.

Run them with:

```sh
go test ./llm/vhdx/... ./ml/... -race
```

## Troubleshooting

- **`OLLAMA_VHDX_OFFLOAD_PATH` does not appear in `ollama serve --help`.**
  The option is Windows-only.
- **`Provision` reports `"file"` format instead of `"vhdx"`.** Hyper-V
  tooling (`New-VHD`) was not available on this edition; a plain
  preallocated file is being used instead. This is a supported
  configuration; only the management ergonomics differ.
- **`ollama ps` never shows a VHDX percentage.** The field is only
  emitted when the runtime has actually attached a VHDX provider for
  this model. The default Ollama engine does not yet attach one — see
  the experimental CGO bridge section above.

## See also

- [`docs/gpu.mdx`](./gpu.mdx) – companion guidance for VRAM sizing.
- [`docs/faq.mdx`](./faq.mdx) – general performance tuning.
