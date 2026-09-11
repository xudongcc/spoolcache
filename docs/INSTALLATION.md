# Installation

Install the package into the Python environment used by vLLM. For a container,
install it into the runtime image. Installing it only on the host does not make
it available inside a container.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.11 or newer; CI tests 3.11–3.12. |
| Storage | Writable local Linux storage supporting buffered I/O, durable file/directory fsync and OFD locks; NVMe is the intended medium. |
| Serving | Linux, CUDA and a working vLLM installation. |
| Models | Weights and revisions managed by the operator. |
| Distributed serving | The same SpoolCache wheel on every required PP/TP participant. |

SpoolCache does not bundle vLLM, Torch, CUDA or model weights. Package installation
is separate from runtime qualification. Startup checks the installed vLLM public
contracts; [Compatibility](COMPATIBILITY.md) defines the scope of that check.

## Published package

Install or upgrade from [PyPI](https://pypi.org/project/spoolcache/).
Published versions and release notes are listed on
[GitHub Releases](https://github.com/xudongcc/spoolcache/releases).

```bash
python -m pip install --upgrade spoolcache
python -c 'import spoolcache; print(spoolcache.__version__)'
spoolcache --help
spoolcache config
```

For an older installation, use the documentation from its release tag.
The [migration table](MIGRATION.md) describes how to update legacy configuration.
Do not assume a local source change updated an installed wheel.

## Current checkout

For local development, run from the repository root in the intended Python environment:

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
