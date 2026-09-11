# CHANGELOG

<!-- version list -->

## v0.3.0 (2026-09-11)

### Documentation

- Keep installation and release guidance version independent
  ([`0e125de`](https://github.com/xudongcc/spoolcache/commit/0e125dec6e5327cac21d4782e7368e8bea2a4114))

### Features

- Replace snapshot storage with runtime-aligned token files
  ([#1](https://github.com/xudongcc/spoolcache/pull/1),
  [`84923f1`](https://github.com/xudongcc/spoolcache/commit/84923f1feec24212363b84c285211d77d6785e51))

### Breaking Changes

- Require Python 3.11+ and use token files as the only backend; remove snapshot/slot storage, native
  I/O, SQLite indexes and snapshot request/status commands. Existing snapshot roots are preserved
  without conversion.


## v0.2.0 (2026-09-08)

### Documentation

- Explain installation usage and development [skip ci]
  ([`220ec81`](https://github.com/xudongcc/spoolcache/commit/220ec8158e74ee80c8ed7a56165cb9b2a66a2051))

- Record G6 release and deployment receipts [skip ci]
  ([`6ed1cb7`](https://github.com/xudongcc/spoolcache/commit/6ed1cb753e1f960a9c8eedcaebfb267e2df18337))

### Features

- Add config command and Gemma end-to-end qualification
  ([`fcdb5ee`](https://github.com/xudongcc/spoolcache/commit/fcdb5ee32a9d8031cfa51b72098d18e06585f00f))

- Simplify cache configuration and request controls
  ([`7695390`](https://github.com/xudongcc/spoolcache/commit/76953902aa3fae7444c2404f62bf7454a39fdf9b))


## v0.1.0 (2026-09-08)

- Initial Release
