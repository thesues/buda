# buda

An inference + retrieval service built **on top of** autumn-rs. It is a *user* of
autumn, not part of it — which is why it lives in its own repository even though
every piece of it talks to an autumn cluster.

```
buda/
├── docker/
│   ├── Dockerfile.freetoken     FreeToken serving image (+ the autumn client)
│   └── README.freetoken.md      why the image is shaped the way it is
└── k8s/
    ├── freetoken.yaml           MoE serving on one RTX 4090
    ├── upload-model-minimax.yaml   one-shot: load the checkpoint into autumn fs/
    └── memory-mcp.yaml          MCP/HTTP retrieval over the document corpus
```

## What it does

**Serving** — FreeToken runs a large MoE model on a single consumer GPU by
keeping the routed experts in host RAM and computing part of them on the CPU.
Host memory, not VRAM, sets the model ceiling; the weights become a cold-start
streaming problem rather than a resident one, which is what makes it sensible to
keep them in autumn and read them through a FUSE mount.

**Retrieval** — `memory-mcp` (an autumn tool) indexes a document corpus stored in
autumn and exposes it over MCP and HTTP.

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

## Status

Not yet deployed. The image tags in the manifests are placeholders; the
model has not been uploaded. See the git log for what has been verified against
the live cluster (the FUSE `O_DIRECT` path has been).
