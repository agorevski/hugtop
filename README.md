# hugtop

`hugtop` is a Rust console app for browsing Hugging Face models already
downloaded to your local cache. It presents model storage, metadata, and cache
locations in a responsive, btop-inspired interface built with Ratatui.

```text
+ hugtop --------------------------------------- models: 18 -- cache: 42.7 GiB -+
| Models                              | Details                                |
|-------------------------------------+----------------------------------------|
| > meta-llama/Llama-3.2-3B-Instruct  | Repository: meta-llama/Llama-3.2...   |
|   Qwen/Qwen2.5-Coder-7B-Instruct    | Revisions:  2                         |
|   sentence-transformers/all-MiniLM  | Size:       6.4 GiB                   |
|   openai/whisper-small              | Modified:   2026-08-28                |
|                                     | Path:       ~/.cache/huggingface/...   |
+-------------------------------------+----------------------------------------+
| Filter: llama          Sort: size desc          [r] refresh  [?] help [q] quit |
+------------------------------------------------------------------------------+
```

## Features

- Scans the local Hugging Face Hub cache without downloading models
- Uses size-descending (largest-first) ordering by default, with filtering and
  alternate sort modes
- Separately reports cache disk size, estimated model-weight size, and estimated
  runtime VRAM
- Detects local NVIDIA GPUs through `nvidia-smi` and plans the minimum viable
  GPU set, with one assigned/total allocation bar per selected GPU
- Reports explicit unknown, no-GPU, and insufficient-installed-VRAM states
- Summarizes repositories, revisions, paths, and modification times
- Uses a compact, colorful terminal layout inspired by system monitors such as
  btop
- Discovers standard Hugging Face and XDG cache locations automatically

## GPU sizing

`hugtop` derives model-weight size from usable local weight artifacts; cache
disk usage is a separate value and may include other files or revisions.
Runtime VRAM adds the implemented 20% allowance to the weight estimate. This is
a planning estimate, not a guarantee: activations, KV cache, context length,
allocators, and framework/runtime needs can require more memory.

GPU planning uses total installed VRAM and chooses the smallest fitting set of
detected cards, then balances the assignment where possible. For example, a
60 GiB requirement on two equal 50 GiB cards is shown as about 30 GiB assigned
to each, with a per-GPU bar such as `30/50 GiB`. Current free memory is checked
separately and produces a warning when it is below an assigned share; it does
not change the installed-capacity plan. Missing weight artifacts or unavailable
GPU detection are shown as unknown, while inadequate aggregate installed VRAM
is shown as insufficient.

## Prerequisites

- A stable Rust toolchain with Cargo ([rustup](https://rustup.rs/) is recommended)
- A terminal with color and alternate-screen support
- At least one model downloaded through Hugging Face Hub tooling

## Build, install, and run

```sh
# Development build
cargo build

# Optimized build
cargo build --release

# Install hugtop from this checkout
cargo install --path .

# Launch the installed application
hugtop
```

## Cache discovery

The cache is selected from the first configured location in this order:

1. `HF_HUB_CACHE`
2. `$HF_HOME/hub`
3. `$XDG_CACHE_HOME/huggingface/hub`
4. `~/.cache/huggingface/hub`

Examples:

```sh
HF_HUB_CACHE=/mnt/models/hub hugtop
HF_HOME="$HOME/.local/share/huggingface" hugtop
```

## Keyboard controls

| Key | Action |
| --- | --- |
| `Up` / `k`, `Down` / `j` | Move through models |
| `PageUp` / `PageDown`, `Home` / `End` | Navigate by page or jump to an edge |
| `/` | Start filtering |
| `Enter` | Apply the filter |
| `Esc` | Cancel filter input or clear the active filter |
| `s` / `S` | Cycle sort field / reverse sort direction |
| `r` | Rescan and refresh cache data |
| `d` | Delete the selected cached model after confirmation |
| `?` | Toggle help |
| `q` or `Ctrl-C` | Quit |

Deletion is permanent, does not use Trash, and requires downloading the model
again to restore it. At the confirmation prompt, press `Y` (uppercase) or `y`
(lowercase) to permanently delete; every other key cancels.

## Development

```sh
cargo test
cargo fmt --all -- --check
```

Run `cargo fmt --all` to apply Rust formatting.

## License

MIT. See [LICENSE](LICENSE).
