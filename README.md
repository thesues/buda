# buda

An inference + retrieval service built **on top of** autumn-rs. It is a *user* of
autumn, not part of it — which is why it lives in its own repository even though
every piece of it talks to an autumn cluster.

```
buda/
├── docker/
│   ├── Dockerfile.freetoken     FreeToken serving image (+ the autumn client)
│   ├── Dockerfile.webui         hermes webui (no autumn client — see below)
│   └── README.freetoken.md      why the image is shaped the way it is
├── webui/                       the webui itself + ARCHITECTURE.md
└── k8s/
    ├── freetoken.yaml           MoE serving on one RTX 4090
    ├── upload-models.yaml       one-shot: load a checkpoint into autumn fs/
    ├── memory-mcp.yaml          MCP/HTTP retrieval over the document corpus
    └── webui.yaml               chat + session management
```

## What it does

**Serving** — FreeToken runs a large MoE model on a single consumer GPU by
keeping the routed experts in host RAM and computing part of them on the CPU.
Host memory, not VRAM, sets the model ceiling; the weights become a cold-start
streaming problem rather than a resident one, which is what makes it sensible to
keep them in autumn and read them through a FUSE mount.

**Retrieval** — `memory-mcp` (an autumn tool) indexes a document corpus stored in
autumn and exposes it over MCP and HTTP.

**Chat** — `webui` is a hermes front end: session management and a chat box. It
reaches retrieval through `memory-mcp`'s **HTTP** MCP transport rather than
spawning it, so it holds no autumn credential and is the one workload here NOT
bound by the WIRE lockstep below. See `webui/ARCHITECTURE.md`.

Both follow the same shape: a privileged `autumn-fuse` sidecar mounts the `fs/`
namespace, the app reads files from the mount, and anything the app *writes*
goes to its own namespace with its own credential.

## What lives here vs in autumn-rs

Here: the FreeToken image, and the manifests for these workloads.

In autumn-rs: the cluster itself (manager / extent-node / partition-server /
etcd / dashboard), the all-roles image and its entrypoint, `autumn-fuse`, and
`memory-mcp`'s source. This repo consumes those as binaries in a published
image — it does not fork or vendor them.

The dividing question is "would this exist if the workload went away?" The
`fuse` entrypoint role and the `memory-mcp` binary would: they are capabilities
of the storage system. These manifests would not.

## Dependency on the autumn image

`Dockerfile.freetoken` lifts `autumn-fuse`, `autumnfs` and `autumn-op` out of the
autumn-rs image via a multi-stage `COPY --from`, rather than rebuilding them:
one Rust build, one binary, and it cannot drift from the cluster it dials.

That makes the **WIRE lockstep** rule this repo's problem too. The image embeds
autumn's PyO3 client, and rkyv has no cross-version compatibility — a mismatched
client is refused at the handshake, and worse, a matching wire fingerprint across
a layout drift decodes garbage silently. So `AUTUMN_IMAGE` must name the same
commit the cluster is running:

```
--build-arg AUTUMN_IMAGE=<cr>/autumn-rs:${SCM_COMMIT_ID}
```

with `${SCM_COMMIT_ID}` resolving to a commit whose autumn-rs image **already
exists** — build autumn-rs first, then this. Getting that order wrong fails at
`resolveBaseImage`, which is at least loud.

## Cluster facts these manifests assume

- Namespace `autumn`, so pods can dial manager/PS pod IPs on flat pod networking.
- Node `192.168.3.2` for serving: FreeToken needs driver r580+ (CUDA 13), and it
  is the only node that has it. The other 4090 box is on 550.
- Secret `autumn-credential` with `fs.cred` / `kvc.cred` / `mem.cred` —
  per-family least privilege. autumn protects **every** key once authz is on
  (the partition server's `protected_prefixes` list is retired), so a workload
  without the right credential does not degrade, it is denied.
- `fs/` presplit before any data was written. Splitting a populated partition
  mostly fails on `has_overlap`.

## Building

Built by a Volcengine CP pipeline (workspace `dongmao-lerobot`), from this repo,
Dockerfile `docker/Dockerfile.freetoken`, context the repo root.

```
--build-arg BASE_REGISTRY=hub-cache-cn-beijing.cr.volces.com/
--build-arg APT_MIRROR=https://mirrors.aliyun.com
--build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
--build-arg RUSTUP_DIST_SERVER=https://rsproxy.cn
--build-arg RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
--build-arg CARGO_MIRROR=sparse+https://rsproxy.cn/index/
--build-arg AUTUMN_IMAGE=<cr>/autumn-rs:<autumn-commit-sha>
```

Two of those are not optional and not obvious:

`PIP_INDEX_URL` — the default index resolves pypi.org but pulls wheels from
files.pythonhosted.org, which stalls from the CN build pool: the build hangs with
no output rather than failing. Note `freetoken` itself does **not** use this arg
— it has its own `FREETOKEN_INDEX_URL`, because not every mirror carries it (the
ivolces mirror answers `from versions: none` for that one package, which is how
this was found).

`AUTUMN_IMAGE` — see the lockstep note above. Unlike the other args this one is a
correctness constraint, not a speed one.

## Deploying

```bash
kubectl -n autumn apply -f k8s/memory-mcp.yaml          # small, exercises the
                                                        # same fuse-sidecar shape
kubectl -n autumn apply -f k8s/upload-model-minimax.yaml
kubectl -n autumn apply -f k8s/freetoken.yaml
```

Fill the `IMAGE_*` placeholders first. Deploy `memory-mcp` before the serving
pod if you can: it uses the same sidecar pattern, the same mountPropagation
pairing and the same credential mounts, but needs no GPU and no 130 GiB
download — so a mistake in the pod shape surfaces cheaply.

## Status

Deployed and serving. FreeToken runs DeepSeek-V4-Flash reading its weights from
an autumn-fuse mount; `memory-mcp` is up; the webui is built but not yet rolled
out. Verified against the live cluster:

- `O_DIRECT` reads work on an autumn-fuse mount at 4 KiB and 8 MiB — the load
  path this whole design rests on, since FreeToken probes `O_DIRECT` once and
  otherwise falls back to a `MAP_SHARED` mmap that a FUSE `direct_io` file
  refuses outright. `--expert-load parallel` therefore works and is what the
  manifest uses.
- Reads from the mount sustain ~2.4 GB/s at 24-way concurrency (~97 MiB/s
  single-stream — this path scales with concurrency, not with one reader, so a
  serial measurement understates it by an order of magnitude).
