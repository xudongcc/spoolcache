# Code review backlog — 2026-09-06

本轮先提交 cache-spec 语义发现改造 `07ae286`，再对该提交做独立 review。下列问题均由
`38fa442` 修复；对修复提交的复审没有发现新的未关闭 P1/P2。

## CR-019 — P1 — 启动兼容 gate 未验证 semantic resolver 签名

Status: resolved. `verify_vllm_runtime()` 现在要求 public
`get_kv_cache_spec_kind()` 存在且可调用，并用签名 bind 证明它能以一个 cache spec 位置参数
调用；新增必需参数、零参数或 keyword-only 漂移都会在启动时拒绝。connector 把同一个已
验证 callable 交给 `build_hma_layout()`，receipt 也显式记录该 capability。

原实现只在 connector 中检查 resolver 是否 callable，却没有把它纳入统一启动兼容检查。
未来 vLLM 若改变必需参数，preflight 仍可能声称兼容，直到布局发现阶段才异常，不符合
fail-closed 的启动承诺。

Evidence: `src/spoolcache/vllm/compat.py`, `src/spoolcache/vllm/connector.py`,
`tests/test_vllm_contract.py`。

Acceptance criteria:

- resolver 缺失、不可调用或不能接受一个位置 cache spec 时启动拒绝；
- 合法的未来可选扩展继续通过；
- 实际布局发现使用 gate 验证过的同一 callable；
- 三套目标镜像执行默认 resolver preflight 并在 receipt 中看到该 capability。

## CR-020 — P2 — 忽略 public aggregate group semantic declaration

Status: resolved. 对含 `kv_cache_specs` 的 group，布局发现现在优先采用 vLLM public
resolver 的整体 semantic kind；只有整体为 `unknown` 时才回退到 members。CPU 正例使用
没有已知类名、member 也为 unknown 的对象，仅由 group public 声明成功归类。三套真实
runtime 还验证了任意改名的 full-attention subclass，无 SpoolCache 类名分支。

原实现只对聚合 group 的每个 member 调用 resolver。vLLM registry 可以在 member concrete
types 不同或单独为 unknown 时，依据已注册的 uniform base 对整个 group 给出有效语义；忽略
该公开声明会错误拒绝未来兼容 wrapper，削弱零白名单目标。

Evidence: `src/spoolcache/hma.py`, `tests/test_hma.py`,
`tests/test_vllm_cache_semantics_runtime.py`。

Acceptance criteria:

- 有效 aggregate declaration 能归类任意 concrete member 名称；
- aggregate `unknown` 才逐 member 回退；
- 尚未实现的 aggregate semantic kind 仍 fail closed；
- 不增加模型、架构、模态、版本或 concrete class gate。

## CR-021 — P1 — 聚合 scratch 未证明共享 allocator 的页所有权

Status: resolved. 最终 member policy 为 `circular_one` 的任何 packed group，现在都必须在
真正提供 block table 的 group 对象上再次证明 `max_num_blocks_per_req(...) == 1` 且
`max_memory_usage_bytes(...) == page_size_bytes`。group 可以省略重复布尔标记以兼容未来
wrapper，但若声明则必须与 members 一致。正负测试覆盖缺失 group contract、多页 group、
双向 prefix-sharing 冲突及无组标记但所有权充分的合法 wrapper；Qwen/GLM 的真实
`UniformTypeKVCacheSpecs` 聚合 scratch 均通过。

原实现只验证 member spec。members 各自一页并不能单独证明共享 allocator 暴露的 request
block table 也是一页；错误 group 可能通过启动检查，随后在页选择或 restore 阶段失败，甚至
在 admission 后形成跨 rank 不一致。

Evidence: `src/spoolcache/hma.py`, `tests/test_hma.py`,
`tests/test_vllm_cache_semantics_runtime.py`。

Acceptance criteria:

- member 与 group 的 prefix-sharing 声明不能矛盾；
- packed scratch 同时证明每个 member 和实际 group 的单页所有权；
- 缺失、异常、布尔或多页 group contract 在 persistent I/O 前拒绝；
- Qwen/GLM 真实聚合 scratch contract 通过同一通用路径。

## Final review disposition

- Host: `104 passed / 10 skipped`；`compileall`、diff whitespace、核心源码模型名与 concrete
  cache-spec 类名扫描通过。
- DeepSeek/Qwen/GLM 目标镜像：各运行 109 tests，分别 skip 6/5/5；三套默认启动 preflight
  均为 `automatic-contract` 并包含 `get_kv_cache_spec_kind`。
- skip 为无 GPU 的 CUDA mover 测试，DeepSeek 另有一个 runtime 未暴露 non-prefix scratch
  的环境跳过；本轮没有重跑 CUDA 或端到端内容 oracle，也没有停止或修改线上 GLM 服务。
- 协议继续使用 `spoolcache-coordination/v1`，布局 schema 继续使用
  `spoolcache-hma-layout/v3`；当前已验证部署的 reuse policy 与持久化解释没有变化。

## G3a post-implementation review

Review scope：SpoolCache `4a06e99`、DeepSeek `a237706`、Qwen `3af1c92`、GLM
`f92cf33`。先冻结实现提交，再从 transport trust boundary、single-owner lifecycle、启动失败
清理、worker source publication、readiness parser 和真实双机恢复逐项复审。以下 P1/P2 全部由
SpoolCache `35abc08`、DeepSeek `35804f3`、Qwen `b00792b`、GLM `5340ddf` 修复；对修复
提交的再次复审没有未关闭 P1/P2。

### CR-022 — P1 — inventory/stats transport 未在全部入口保持固定上界

Status: resolved。原实现的 reporter held state、delta、stats report count、pending rolling
checkpoint 累积和 startup iterable materialization 并非全部有界；新 sequence checkpoint 也可能
在漏收 delta 后短暂保留 stale offer。现在 local catalog=100,000、单 report=64、rank/report=
4,096，identity/counter/span 都有固定边界；重复 rank、超界页/组合 delta、checkpoint 累积溢出、
丢序和 generation change 都撤销 rank，只有完整 checkpoint 可恢复。启动 512-entry 子集是安全
假阴性，以 sequence=0 rolling image 补齐，不造成无必要的 readiness 退场。

Evidence: `src/spoolcache/quorum.py`, `src/spoolcache/vllm/connector.py`,
`tests/test_quorum.py`, `tests/test_vllm_contract.py`。

### CR-023 — P1 — stop 结果不能证明旧 supervisor owner 已退出

Status: resolved。原 wrapper 忽略 module stop 失败，且 state 缺失/损坏会被当作 stop 成功，
可能让下一次启动与旧 owner 竞争甚至并行管理 ranks。现在 stop request 后必须同时证明 `/proc`
没有 exact `-m spoolcache.supervisor run --config <path>` argv 且 `supervisor.lock` 可独占；等待上限
覆盖多 rank 顺序清理。runtime/state 缺失会重建窄 runtime 后继续扫描，而不是成功返回；procfs
除进程自然消失外任何读取错误都 fail closed。start 在该证明前不做 container mutation，stop
仍尽力清理 ranks，但不能证明 owner 时非零返回。

修复复审还发现部署 fallback 的 runtime-missing 路径与 GLM heredoc import 段有误；动态抽取并
执行每个内嵌 fallback 的测试现覆盖该路径，不再只依赖 shell syntax/static string assertion。

Evidence: `src/spoolcache/supervisor.py`, `tests/test_supervisor.py`，三个部署的 start/stop 与
launcher contract tests。

### CR-024 — P2 — GLM/Qwen 启动异常可能遗留 partial TP group

Status: resolved。GLM 过去只在部分显式失败路径 stop，worker 启动后发生任意 shell error 可能
留下远端 headless rank；Qwen trap 安装顺序也不能覆盖刚启动的 supervisor。现在 strict old-owner
stop 位于第一项 container mutation 之前，随后设置 `TP_GROUP_STARTING` EXIT trap；直到 API、
supervisor、rank identity/quorum 和 warmup 全部成功才解除。异常清理同时停止 supervisor、head、
所有 worker；显式 stop 即使 owner proof 失败也继续尝试移除 ranks，并最终返回非零。

Evidence: Qwen/GLM launcher and contract tests。

### CR-025 — P2 — readiness Prometheus label parser 接受尾随逗号

Status: resolved。`{condition="fatal_clear",}` 过去可被解析为合法 labels，削弱严格机器可读
readiness。parser 现在拒绝尾随逗号，负向测试覆盖。

Evidence: `src/spoolcache/readiness.py`, `tests/test_readiness.py`。

### CR-026 — P2 — supervisor probe/config/state 边界不完整

Status: resolved。非法/越界 URL port 可能在 probe 阶段抛出未处理 `ValueError`，atomic state
writer 也没有写入侧 byte bound。配置现在在启动时拒绝非法 port；health/metrics probe 把 URL
构造错误视为 not-live；读写 state 都限制 1 MiB 并拒绝 symlink/non-regular target。负向测试覆盖
port 0/65536/non-numeric、probe exception 和 oversized state。

Evidence: `src/spoolcache/supervisor.py`, `tests/test_supervisor.py`。

### CR-027 — P2 — worker source overlay/先删目录会保留旧 module 或破坏 live mount

Status: resolved。tar overlay 不删除本地已移除 module；直接 `rm -rf` stable source 又会在旧
service 尚未停止时破坏其 bind source。三个部署现在先向经过路径验证的 private `mktemp`
snapshot 传输 exact tree，排除 VCS/venv/build/bytecode，再用 stable symlink 原子发布；只有新
worker container 成功获得新树后才清理非活动 snapshot。DeepSeek 额外使用 raw remote path 加
shell-safe quoting，并以 `--force-recreate` 保证 Compose 重新解析 mount。

Evidence: three deployment launchers and launcher contract tests。

## G3a final review disposition

- Host: `156 passed / 10 skipped`；compileall 和 diff whitespace 通过。
- DeepSeek/Qwen/GLM target images: 同一最终源码各 `161 tests`，skip 依次 6/5/5；实际 vLLM
  preflight 均为 `automatic-contract`、19 capabilities，build SHA-256 前缀依次
  `e24d925d2e31`、`fa1e7da550af`、`89945da00615`。
- 部署 launcher tests：DeepSeek 3/3、Qwen 3/3、GLM 4/4；所有相关 shell `bash -n` 通过。
- 核心源码扫描只有阐明“不得使用 allowlist”的注释/文档，没有模型、架构、模态或 concrete
  cache-spec 分支；协议保持 `spoolcache-coordination/v1`。
- GLM 最终运行字节：本地/head/worker 对实际可导入的 `src/` 24 files / 332,528 bytes 的
  source manifest 都为 `7032f16388c7dbb3c15f63a60824afdd695065dbcf3efa0535c1fd4ae4cf778f`。真实 worker-stop
  注入观察到 readiness false、API false、0/2 ranks、整组停止，再于全部契约成立后 ready；
  supervisor restart count=6。
- 最终 persistent hit 为 3,584 tokens；两 rank 同 entry/span restore；精确输出 oracle
  `3141592653589` 及 SHA-256 与四个 cold controls 一致；两 rank 77-object verifier 均返回
  `all-payloads-authenticated`。API health=200、2/2 ranks、两端 `VLLM_SERVER_DEV_MODE=0`。

Disposition: approved。没有未关闭 P1/P2；G3a 范围内指标、readiness、完整 TP 恢复、真实损坏
闭环和内容正确性证据齐全。PP>1、layerwise restore、shared staging、深度 scrub/长期 GC 和 0.1
release 仍按 TODO 独立推进，不把本次结果外推为已验证。

## G3b post-implementation review

Review scope：`a633f75`、`7a2a35e`、`e7920a4`、`800ae13`、`e102c0d`、
`7510fda`、`761aaea`、`26cb7e7`、`3e15600`、`4af10b2`、`5f66d79`、`905c48a`、
`cdfa84c`、`c6f61fb`、`bb81636`。实现冻结后逐轮从
filesystem visibility、offer withdrawal、generation ordering、跨进程 owner、容量恢复和
soak 证据边界复审；每轮新发现先修复并新增反例门禁，再进入下一轮。

### CR-028 — P1 — 旧 generation 包可回滚 catalog；wall clock 回拨可保留死 worker quorum

Status: resolved。最初 startup/report 把任意不同 generation 当作新 image，延迟旧包可回滚；
第一次修复只比较 epoch 后，又会在系统时钟回拨时把真正的新 worker 当 stale。最终每个 rank
在 maintenance lock 下 crash-consistently 持久化 generation；旧 raw-clock epoch 首次升级到
`2^62` 以上的独立域，并以 `generation.required.json` 区分首次迁移与初始化后的 state 丢失。
scheduler 仅忽略自己有界历史中精确匹配的旧 UUID/epoch；未知较小 UUID、同 epoch 不同 UUID
及合法 rank 携带的 malformed identity 都撤销当前 rank image，严格更大 epoch 才切换。测试覆盖 delayed-old handshake/report、滚动
升级+时钟回拨、state 删除、未知回拨 UUID、同 epoch 冲突和有界历史淘汰。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/quorum.py`,
`tests/test_manifest_store.py`, `tests/test_quorum.py`。

### CR-029 — P1 — namespace mutation、reporter add/replace 与 quarantine receipt 存在竞态

Status: resolved。manifest rename/unlink 已发生后，目录 fsync 或 callback 失败过去会留下旧
offer；scan 完成到 `reporter.replace()`、manifest publication 到 `reporter.add()` 之间也可被
并发 remove 反向覆盖。现在 live pathname 一旦变化就始终尝试撤销，失败强制 empty+rescan；
scan+replace+ack 与 publication+add 分别共享一个 rank maintenance 临界区。已知坏 entry 在
rename 前写入持久 tombstone，shallow scan 不得清除；target/周期 scrub 在最后一个 object 后
都锁内复核 manifest digest。reporter 每次新 mutation 的首份 report 优先运输最新 delta；若旧
delta 真丢失，sequence gap 会立即撤销 rank，而不是让 retained history 延迟 withdrawal。
barrier 测试覆盖两个 TOCTOU 交错、callback 与每个目录 fsync
边界、rename 前失败、manifest concurrent replace/delete 和 forced-rescan 多轮重准入。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/maintenance.py`,
`src/spoolcache/vllm/connector.py`, `tests/test_storage_maintenance.py`,
`tests/test_vllm_contract.py`。

### CR-030 — P1 — corrupt content-address collision 可留下断链 manifest 与幽灵 offer

Status: resolved。旧实现先把 corrupt live object `rename` 到 quarantine，再 link replacement；
第二步失败时 manifest 仍 live、object 已 absent、reporter 仍可形成 quorum。现在发现碰撞后先
在遍历全部引用 manifest 前先持久写 digest-level object fence；lookup、startup scan 和
reporter reconciliation 都拒绝 fenced digest，逐引用 entry marker/撤销过程保持流式有界。
随后用 hardlink 保存坏 inode 证据，最后以 `os.replace()` 把已 fsync 的临时对象原子切换到
live name 并 fsync shard。任一 evidence link、quarantine fsync、replace、object-dir fsync、
object fence 后或首个共享引用后的真实 `os._exit` 边界都保留 fail-closed fence/marker；
成功切换没有缺失窗口，失败后只有完整 deep-scrub payload authentication 加最终 manifest
版本线性化点并再次 fsync object shard directory，才能释放 live marker。共享引用和 scheduler
quorum 反例均纳入测试。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/maintenance.py`,
`tests/test_manifest_store.py`, `tests/test_storage_maintenance.py`。

### CR-031 — P1 — 同 rank 两个 worker generation/独立 maintenance 可留下跨实例旧 offer

Status: resolved。maintenance lock 只串行单次磁盘操作，不能同步两个实例各自的内存 reporter。
每个 vLLM rank 现在必须在 startup scan 前非阻塞取得 `inventory-owner.lock` lifetime lease，并
持有到 store close；第二个 generation 无法重叠启动，进程退出由 kernel 释放 flock。独立
maintenance 不占 lease，且成功 quarantine 不再清除只通知了自己实例的 marker；唯一 owner
在构造下一份 stats report 的同一 maintenance lock 内，从已受 catalog 上界约束的 held set
过滤 marker、执行 reporter remove，再确认已 absent manifest 的信号。此外 owner 以固定页和
轮转游标扫描完整 marker namespace，因此 worker 离线期间已经删除的 manifest marker 也会最终
清理，不会因不在 held set 而永久积累。测试覆盖双 owner 拒绝、lease 释放接管、standalone
store 删除后 live reporter/quorum 撤销和离线 7-marker 分页收敛。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/vllm/connector.py`,
`tests/test_manifest_store.py`。

### CR-032 — P2 — scrub/soak 峰值收据可能只观察低水位或被继承 ru_maxrss 遮蔽

Status: resolved。SQLite 现在在 `start_cycle()` 建立 work queue 后及每个 step 后统计 main、
WAL、SHM 全部 artifacts；receipt 回归使用足够大的真实 work queue，强制
`max_state_database_bytes > final_state_database_bytes`。RSS 通过判据不再使用 Linux 可跨
fork/exec 继承、不可重置的 `ru_maxrss` 差值；独立 sampler process 每 2 ms 读取 workload
`/proc/<pid>/status` 的 current VmRSS，以 sampler ready 后的 baseline 计算 peak growth。
native mmap 负测先制造更高历史 ru_maxrss，再证明较小瞬时 resident burst 仍被 sampler 捕获。

Evidence: `benchmarks/soak_storage_maintenance.py`, `tests/test_storage_soak.py`。

### CR-033 — P2 — deep scrub 的边界恢复与 namespace 特殊项缺少 fail-closed 覆盖

Status: resolved。target 与周期 scrub 在成功尾部都复核 manifest 版本；POSIX surrogateescape
文件名在进入 SQLite TEXT 前安全隔离；所有 enabled access mode 启动时先做容量 pass 再形成
唯一 startup image；deep-scrub request 与 event journal 各自只清理精确 UUID 临时文件名。
对象引用读取 unknown 时保守视为 referenced，inventory scan 不改变 LRU mtime。测试还覆盖
restore-only 超水位启动、source/quarantine dir fsync、state-temp、manifests/objects/tmp 字节名
和最终 object 读取期间的 manifest replacement/delete。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/maintenance.py`,
`src/spoolcache/event_journal.py`, `src/spoolcache/vllm/connector.py` 及对应测试。

### CR-034 — P1/P2 — 共享对象逐引用撤销非崩溃原子，离线 marker 与引用列表无界

Status: resolved。第四轮复审证明，旧实现虽然在触碰 object 前逐个撤销引用，但进程若在第一个
共享引用后退出，后续 manifest 没有 marker，重启 shallow scan 仍可能 offer 同长度坏 payload；
同时引用 ID 被累计进无界 list，standalone maintenance 在无 worker 时删除 manifest 后的 marker
永远不在 held set，无法确认。`26cb7e7` 增加先行持久 object fence，并让 lookup、startup scan、
worker stats reconciliation 全部遵守；引用处理只保留常数计数。entry marker namespace 通过固定
page 和 lexicographic cursor 轮转，wire report 构造后才确认 absent marker；object fence 也以
固定页复核无引用项。子进程测试在 fence 后、首个 collision/quarantine 引用后退出，三份共享
manifest 均不被重启扫描准入；129 共享引用路径保持流式，离线 marker 和 live-marker 饥饿反例
均收敛。deep scrub 释放 fence 前在最终 manifest 锁内重新 hash，防止晚到 fence 被旧认证误清。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/vllm/connector.py`,
`tests/test_manifest_store.py`, `tests/test_storage_maintenance.py`,
`tests/test_vllm_contract.py`。

### CR-035 — P1 — 单 object 修复误清 entry 标记，marker/absence receipt 可在断电后回滚

Status: resolved。第五轮复审构造出双对象 manifest：A、B 均损坏时，修复 A 的旧代码会遍历
引用并清掉整个 entry marker，使只做 metadata probe 的 scan 重新广告仍引用坏 B 的 entry。
同轮还证明，manifest unlink 的 shard fsync 失败后，absent-marker ack 只看当前 `lstat` 就持久
删除 marker，断电可让旧 manifest 回滚而 marker 不回滚；marker/withdrawal namespace 首次
`mkdir` 成功但 parent fsync 失败后，幂等重试也曾跳过 ancestor receipt。

`3e15600` 删除 object repair 对引用 entry marker 的推断式释放：repair 只清对应 digest fence，
无 provenance 的 entry marker 只能由完整 replacement commit 或最终锁内全 manifest payload
认证清除；后者通过显式 `inventory_released` 信号触发 reporter rescan。absent entry 确认现在先
fsync 精确 manifest shard 并锁内重查；无引用 object fence 确认先 fsync 固定 256 个 canonical
manifest shard namespace，再重扫引用。所有 marker 幂等 mark 都重新 fsync parent。
`4af10b2` 进一步把同一规则扩展到 managed root、withdrawal/lock namespace 与 object/manifest
shard：重开并验证 existing directory 后仍重新取得 parent/ancestor receipt。负向测试覆盖
双坏对象只修 A、unlink/marker/namespace 首次 fsync EIO 后重试、ack 收据顺序、namespace
fsync 失败保留 fence，以及完整认证解除后产生 rescan epoch。

Evidence: `src/spoolcache/store.py`, `src/spoolcache/maintenance.py`,
`tests/test_manifest_store.py`, `tests/test_storage_maintenance.py`。

### CR-036 — P2 — shard receipt 持续失败会无界累积完整 object 临时文件

Status: resolved。`4af10b2` 让 existing shard 每次重新 fsync parent 后，复审发现
`_ensure_shard_directory()` 位于 object temporary 的清理 `try/finally` 之外：若 parent fsync
持续 EIO，每次调用都先写满并 fsync 一个 `.part`，随后异常绕过 unlink，长寿命 worker 会把
磁盘填满。`5f66d79` 把 shard 初始化移入同一 cleanup scope；连续三次注入相同 EIO 后，
`tmp/` 每轮都保持为空。随后复审发现 cleanup 自己的 unlink/tmp-fsync 异常会覆盖 primary
durability error；`cdfa84c` 让 object/manifest publication 共用 cleanup helper：仍尝试两项
清理，无 primary 时传播 cleanup error，双失败时保留 primary traceback 并附加 bounded note。
object shard+tmp-fsync 与 manifest link+tmp-fsync 两组双故障测试都验证原错误不被覆盖。

Evidence: `src/spoolcache/store.py`, `tests/test_manifest_store.py`。

### CR-037 — P2 — withdrawn/corrupt manifest 可堵塞有界 startup offer 候选窗

Status: resolved。旧 `scan_offers(limit)` 先按 mtime 选最新 `limit` 个原始 manifest，之后才
过滤 marker、fence 或损坏项。`limit=1` 时一个较新的 withdrawn manifest 会让较旧健康 entry
得到空结果；大量 tombstone 可让 startup/rescan 长期报告空 catalog。`905c48a` 改成单次流式
读取时先做 metadata/marker/fence 验证，只有健康 offer 才进入 O(limit) 最新项 heap；仍不更新
manifest LRU mtime。负测固定 `limit=1`、新 withdrawn/旧 healthy，证明返回健康 entry。
跨 POSIX 文件系统继续复核后，`bb81636` 又消除了 active `scandir` 内 rename 当前目录项的
可移植性依赖：scan 只暂存固定 64 个坏 path，所有目录流关闭后才 quarantine，其余留给后续
scan/scrub。65 个 hostile manifest 的回归证明首轮恰好处理固定批次、第二轮收敛，健康 heap
仍为 O(limit)。

Evidence: `src/spoolcache/store.py`, `tests/test_manifest_store.py`。

### CR-038 — P1 — JSON resource-limit 异常可让单一 manifest 阻断 worker startup

Status: resolved。`json.loads()` 对 4 MiB 上限内的 5,000 位整数会抛裸 `ValueError`，深嵌套
输入可抛 `RecursionError`；旧 decoder 只归一 Unicode/JSON syntax error。`lookup()` 因而不能
把该文件转为 quarantine miss，`scan_offers()` 流式验证后更会让任意一个这种 manifest 阻断
整个 worker startup。`c6f61fb` 在 manifest trust boundary 归一 JSON parse、canonicalization
与 payload materialization 的数据驱动 `UnicodeDecodeError`、`ValueError`、`TypeError`、
`OverflowError`、`RecursionError` 为 `ManifestError`，明确不吞 `MemoryError` 或系统 I/O。
测试覆盖 5,000 位整数的 lookup/scan 隔离与健康 offer 保留，以及 JSON NaN 在 canonicalization
阶段的 clean manifest error。

Evidence: `src/spoolcache/manifest.py`, `src/spoolcache/store.py`,
`tests/test_manifest_store.py`。

## G3b review disposition

Disposition: approved。第六轮独立复审冻结在 `bb81636`，没有未关闭 P1/P2；复审与主线
分别重跑完整宿主 suite，均为 pytest 247 passed / 10 skipped，标准库入口为
252 tests / 10 skipped / OK。最终源码在 DeepSeek、Qwen、GLM 三个真实目标镜像各运行
252 tests，分别 skipped 6/5/5，结果全部 OK；三个镜像又各自在真实 CUDA 上通过 5/5 mover。
三套实际 vLLM preflight 均为 `automatic-contract`、19 capabilities，且构造器、HMA 与
18 个 hook 的签名可替代性检查通过。

GLM 最终生产恢复使用同一 `bb81636` 源码：host/head/worker 按 launcher 快照排除规则均为
25 files / 502,590 bytes / SHA-256
`95b1f599c4ab88f84b28ddd6f0795efcb803c49edee4ccdadd7c7cfbec14341f`。最终
`spoolcache-readiness/v1` 为 `ready=true`、API live、2/2 ranks、supervisor
`phase=ready`，rank identity、inventory quorum、fatal-clear 三项均为 1；两个容器均确认
`VLLM_SERVER_DEV_MODE=0`。当前容器时间窗没有 SpoolCache/CUDA/NCCL/traceback error，
严格 64-token bypass control 返回 cached=0、完整 8-token 输出。此前同一生产部署完成的
单字节 payload 损坏、定点 scrub、quorum 撤销、完整 TP 重启、admission 前 miss 和 bypass
oracle 一致收据继续作为 destructive qualification，不为最终健康核验重复破坏缓存。

当前协议仍为 `spoolcache-coordination/v1`，核心源码扫描未发现模型、架构或模态白名单。
保留的 residual risk 是 fault injection 不能等价于真实断电文件系统 replay、2 ms VmRSS
轮询可能漏掉更短瞬时峰值，以及不遵守 maintenance lock 的第三方磁盘修改只能在 restore
或下轮 scrub 被捕获；超大 namespace 的分步扫描/有界 shutdown 已明确进入 G3d，不把它们
误写成 G3b 已验证范围。

## G3c post-commit review disposition

Review range: SpoolCache `2922773..51ec9a6` 中的 G3c changeset，以及 DeepSeek `6c3f680`、
Qwen `3cb6744`、GLM `2fb1046`。Disposition: approved；未发现未关闭 P1/P2。

审查重新追踪了 namespace 从 vLLM `ModelConfig` 到 coordination digest、role-local deployment
identity、rank root、manifest 和 scheduler quorum 的完整数据流。三个目标镜像的实际构造器都
公开 `model: str`、`model_weights: str`、`revision: str | None`；新类名和 served alias 不参与
namespace，而 locator、revision、tenant namespace、runtime/build、dtype、topology、execution
config 或 layout 任一变化都会隔离。上游 `ModelConfig.compute_hash()` 同样明确忽略 served alias，
所以 role-local execution identity 不会把 alias 偷渡回来。字段缺失或类型漂移在模型分配前
fail closed，不会降级成无模型身份的共享缓存。

源码和三个 launcher 的 production-path 搜索只在迁移说明/拒绝测试中保留旧 checkpoint 名称；
`src/` 没有模型、架构或模态名称分支，唯一协调 schema 仍为
`spoolcache-coordination/v1`。旧 deployment/v1 digest 与新 v2 root 自然不同，启动、GC 和
迁移路径都没有删除旧缓存。Qwen 只把已解析的标准 Hub revision 传给 vLLM，DeepSeek/GLM
继续使用原本的 locator/revision 路径；三个 launcher 均不生成 cache profile 或私有 digest。

复审证据包括 244 passed / 9 skipped 的宿主全量、三个目标镜像 248 tests（skipped 6/5/5）、
各镜像 CUDA mover 5/5、19-capability automatic contract、GLM TP=2 的四次 cold oracle、
同 entry/span 双 rank 跨完整重启 restore、两端 payload authentication，以及三端 source digest
一致。当前 residual risk 仅是已声明的 artifact 发布信任边界：同一 locator/revision 被运维
原地替换不由 KV connector 检测；应通过不可变 artifact 或显式 tenant namespace 轮换处理，
不得重新引入模型仓库扫描或模型特例。

## G3d implementation-stage review

Review range: `5d58785` 的增量 namespace snapshot、scheduled scrub 有界关闭和长稳
benchmark 实现。本阶段 review 发现并关闭以下 P1；G3d 仍需完成真实最大上下文与
24 小时资格，本节不提前给出整个 Goal 的最终 disposition。

### CR-039 — P1 — shutdown timeout 后的持久计数器 fsync 会重新引入无界主关闭

Status: resolved。`5d58785` 已把 scrub thread join 限制为 5 秒，但 connector 在观测到
`thread_alive=true` 后，仍然在调用者线程同步写入并 fsync
`spoolcache_scrub_shutdown_failures_total`。如果导致 scrub timeout 的原因本身就是慢盘或
卡住的文件系统，这个 journal fsync 可让主 shutdown 再次无界阻塞，与机器可读
timeout 承诺冲突。

修复后，主路径只同步输出 `spoolcache-scrub-shutdown/v1` 结构化日志收据、移交对
scheduler/mover/store/journal 的所有权并立即返回。daemon finalizer 先 best-effort 持久计数，
再等待 scrub 真实停止，最后释放 mover/store；它的 journal fsync 或 scrub reader 卡住都
不会阻止进程退出，同时也不会在 reader 存活时提前关闭存储。并发回归让假 journal
主动卡住，验证 connector shutdown 在 100 ms 内返回，并在两个阻塞点都保持
mover/store 未关闭。

Evidence: `src/spoolcache/vllm/connector.py`, `tests/test_vllm_contract.py`。聚焦回归
65 passed；全量宿主回归 252 passed / 9 skipped，`compileall` 和 `git diff --check`
通过。核心源码、测试和 benchmark 仍无模型/架构/模态白名单，coordination schema 仍唯一为
`spoolcache-coordination/v1`。
