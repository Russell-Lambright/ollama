/*
 * vhdx_bridge.h - C-facing declarations for the Ollama VHDX weight
 * loader bridge.
 *
 * The Go side (llm/vhdx/cgo_bridge.go) exports two symbols:
 *
 *   int64_t ollama_vhdx_read_layer(uint64_t handle, int index,
 *                                  int64_t offset, char *dst,
 *                                  int64_t size);
 *   int64_t ollama_vhdx_layer_size(uint64_t handle, int index);
 *
 * Both are safe to call from any thread. `handle` is the value returned
 * by ml.RegisterWeightLoader on the Go side and passed down to the
 * inference engine when the model is loaded.
 *
 * ollama_vhdx_read_layer semantics:
 *
 *   > 0  number of bytes copied into dst
 *   = 0  layer index is not known to the loader; callers should fall
 *        back to their default (mmap / file) path
 *   < 0  fatal error — the VHDX medium is unavailable and callers
 *        should abort the current load
 *
 * These symbols are only present in builds compiled with
 *   go build -tags ollama_vhdx_bridge ...
 */
#ifndef OLLAMA_VHDX_BRIDGE_H
#define OLLAMA_VHDX_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int64_t ollama_vhdx_read_layer(uint64_t handle, int index,
                               int64_t offset, char *dst,
                               int64_t size);

int64_t ollama_vhdx_layer_size(uint64_t handle, int index);

#ifdef __cplusplus
}
#endif

#endif /* OLLAMA_VHDX_BRIDGE_H */
