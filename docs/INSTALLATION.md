# Installation

Install the package into the Python environment used by vLLM. For a container,
install it into the runtime image. Installing it only on the host does not make
it available inside a container.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.10 or newer; CI tests 3.10–3.12. |
| Storage | Writable local storage supporting aligned `O_DIRECT`; NVMe is the intended medium. |
| Serving | Linux, CUDA and a working vLLM installation. |
| Models | Weights and revisions managed by the operator. |
| Distributed serving | The same SpoolCache wheel on every required PP/TP participant. |

SpoolCache does not bundle vLLM, Torch, CUDA or model weights. Package installation
is separate from runtime qualification. Startup checks the installed vLLM public
contracts; [Compatibility](COMPATIBILITY.md) defines the scope of that check.

## Published 0.1.0

The first recorded release is available from
[PyPI](https://pypi.org/project/spoolcache/0.1.0/) and
[GitHub](https://github.com/xudongcc/spoolcache/releases/tag/v0.1.0).

```bash
python -m pip install spoolcache==0.1.0
python -c 'import spoolcache; print(spoolcache.__version__)'
spoolcache-maintenance --help
```

That artifact uses the old interface. To render a configuration with a chosen
cache directory under 0.1.0:

```bash
export SPOOLCACHE_CONTAINER_ROOT="$HOME/.cache/spoolcache"
export SPOOLCACHE_DIRECT_IO=required
python -m spoolcache.vllm.config_json
```

Use the documentation from the release tag when operating that artifact.
The [migration table](MIGRATION.md) lists the next interface, which is currently
unreleased. Do not assume a local source change updated an installed wheel.

## Current checkout

From a checkout containing the current changes, in the intended Python environment:

```bash
python -m pip install -e .
spoolcache --help
spoolcache config
```

Editable installation is for local development. It does not create a release
artifact, and its version string alone does not identify the edited bytes.
For the lightweight CPU test environment, use the
[development setup](DEVELOPMENT.md#cpu-development).

A clone obtains the remote repository state; local uncommitted changes are
available only in the checkout where they were made.

## Serving images

Build from an existing, qualified vLLM runtime and install a single authenticated
wheel with `--no-deps`. The [deployment guide](DEPLOYMENT.md) explains the image
build and mount configuration; the [release guide](RELEASE.md) explains artifact
provenance and reproducible builds. Serving images import the installed wheel.

## Check the installation

```bash
python -c 'import spoolcache; print(spoolcache.__version__); print(spoolcache.__file__)'
```

Check that the import path belongs to the intended environment. For a release
installation, run `benchmarks/verify_release_install.py` with the retained wheel
and its trusted SHA-256. It checks installed package bytes and stale modules;
printing a version does not provide that assurance.
