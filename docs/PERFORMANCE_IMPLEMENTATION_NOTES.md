# SpoolCache 无补丁性能实施笔记

本文记录在不修改 vLLM 的前提下，SpoolCache 性能优化的假设、测试方法、实现决定、
原始结果和回退条件。设计目标与正式验收矩阵仍以
[`SPOOLCACHE_DESIGN.md`](SPOOLCACHE_DESIGN.md) 为准；本文是可持续追加的工程工作日志。

## 记录规则

- 每项优化先写可证伪假设，再写测试，最后改实现。
- 同时保留正确性结果和性能结果；性能提高但 KV/输出不正确视为失败。
- 明确区分 CPU-only、单 rank CUDA microbenchmark 和双机服务测试。
- 报告原始样本、中位数和测试环境，不用单个最好值代替结果。
- 任何依赖固定 vLLM/HMA 行为的结论都进入 capability/profile gate。
- 测试失败时记录原因和回退方案，不通过扩大内存或修改 vLLM 隐藏问题。

## 环境

| 项目 | 当前值 |
|---|---|
| 日期 | 2026-09-04 UTC |
| SpoolCache | `0.1.0a0`，开发工作树 |
| 目标容器 | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| 模型 | `DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9` |
| topology | 双 DGX Spark，TP=2，PP=1 |
| 当前 staging/rank | `2 × 64 MiB` pinned，加 `2 × 64 MiB` aligned I/O |
| payload I/O | rank-local NVMe，`O_DIRECT=required` |

宿主 Python 由 PEP 668 管理，pytest 安装在仓库 `.venv`：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'pytest>=8,<9'
PYTHONPATH=src .venv/bin/python -m pytest -q
```

目标镜像没有 pytest，CUDA 测试暂用标准库 runner：

```bash
docker exec \
  -e PYTHONPATH=/opt/spoolcache/src:/opt/spoolcache \
  deepseek-v4-flash-vllm-dspark-1 \
  python3 -m unittest -v tests.test_gpu_mover
```

## P1：单次认证恢复

### 假设

旧路径在 `lookup(verify_payloads=True)` 中读取并哈希所有对象，真正 restore 又读取并哈希
相同对象。若把 lookup 限制为 manifest、identity、对象类型和大小 probe，并在实际搬运时
完成一次 payload SHA-256/padding 验证，物理读取量可从约 2 倍降到约 1 倍。

### 正确性边界

- 一个对象必须在其 SHA-256 和 padding 验证通过后才写入 GPU。
- 固定 staging 不允许为保存整个 entry 而增长。
- 多对象 entry 的后续对象损坏时，前序对象可能已经写入请求私有 blocks；这属于既定的
  post-admission fatal，engine 必须停止；完整组恢复由外部部署编排器负责。
- 不能把 payload 校验失败改成成功命中，也不能继续 attention。

### 测试

`test_restore_probe_and_stream_read_and_hash_payload_once` 对底层读取字节和 SHA-256 输入字节
计数，要求：

```text
physical_read_bytes == descriptor.stored_length
hashed_bytes == descriptor.byte_length
hash_object_count == 1
```

CUDA/HMA round-trip 与首对象损坏测试在目标容器中继续通过。

### 结果

| 测试 | 结果 |
|---|---|
| 宿主 pytest | 31 passed，3 CUDA skipped |
| 目标容器 GPU mover | 3 passed |
| 31,744-token 视觉 entry，rank 0，旧双读 I/O 中位数 | 约 0.307 s |
| 同 entry，单次认证读取，5 次中位数 | 0.142086 s |
| 新逻辑吞吐 | 941.17 MiB/s |
| physical/logical read ratio | 1.0 |

本项证明的是单 rank manifest/object I/O，不等价于完整双机 API TTFT。服务进程需要重启后
才会加载修改后的 connector，端到端复测在全部低层回归通过后进行。

## P2：双 slot restore 流水线

### 假设

当前目标 entry 有 170 个对象。每个对象后调用一次 `cudaStreamSynchronize` 会把 NVMe/CPU
准备下一对象与前一对象 H2D/scatter 完全串行化。两个固定 slot 可以交替使用：CUDA 消费
slot 0 时，CPU 准备 slot 1；重新使用 slot 0 前只等待该 slot 的 completion event；整个
entry 最后只做一次 terminal stream synchronize。

### 所有权规则

- completion event 完成前，slot 不得重新交给 CPU 写入。
- pool 使用 FIFO；LIFO 会立刻取回刚释放的 slot，使双缓冲退化为单缓冲。
- connector 返回 vLLM 前仍执行 terminal synchronize，因此不改变 vLLM block ownership。
- 异常退出也必须等待已经提交的 CUDA 工作，不允许关闭或复用仍被读取的 pinned memory。

### 已加入的测试

- HMA 全组 round-trip 只允许一次 `_synchronize_restore_stream()`。
- 连续对象必须交替取得不同 slot。
- terminal synchronize 后所有 slot 的 `pending` 状态必须清空。
- 首对象损坏仍须在任何 GPU page placement 前失败。
- 最后一个对象损坏时，已提交的 CUDA 工作仍须执行一次 terminal synchronize，且所有
  slot 必须恢复为可复用状态。由于早先对象可能已经写入请求私有 blocks，此后 engine 必须
  fail-stop，不能继续 attention。

### 实施中发现的问题

独立的 Torch pinned allocation 并不保证每次返回 4096-byte aligned 地址。旧代码把
O_DIRECT 的要求错误施加到独立 CUDA pinned pool，连续创建 mover 时可触发启动失败。
独立 pinned pool 只要求 page-locked；4096 对齐属于 aligned I/O pool。该限制已经移除，
未来只有共享 staging fast path 同时满足并验证两种要求。

### 实机结果

目标容器执行：

```bash
docker exec \
  -e PYTHONPATH=/opt/spoolcache/src:/opt/spoolcache \
  deepseek-v4-flash-vllm-dspark-1 \
  python3 /opt/spoolcache/benchmarks/bench_restore_pipeline.py
```

输入为 170 个 1 MiB 对象、预热 1 次、记录 7 次：

| 指标 | 逐对象同步 | 双 slot/event |
|---|---:|---:|
| stream synchronize 次数 | 170 | 1 次 terminal sync |
| 中位数 | 4.647030 ms | 3.696252 ms |
| 相对速度 | 1.00× | 1.257× |

宿主 pytest 为 31 passed、4 CUDA skipped；目标容器 GPU mover 为 4 passed。新增的晚期损坏
测试证明异常退出会排空已提交的 CUDA 工作并清除 slot ownership。

这个 microbenchmark 只隔离 H2D 与同步开销，没有模拟真实 NVMe 读取、对象哈希、HMA
scatter、双 rank barrier 或 vLLM 调度，因此不能把 1.257× 直接解释为 warm TTFT 提升。

### 尚待端到端验证

- 在目标服务完整重启后比较相同持久 entry 的 warm TTFT。
- 连续多次 restore 的 soak 测试确认 slot/event 状态和 pinned memory 均不增长。

## 后续队列

| 优先级 | 项目 | 当前状态 |
|---:|---|---|
| 3 | `O_DIRECT + cudaHostRegister` 统一 128 MiB staging 可行性 | 探针通过 |
| 4 | Store admission/skip 策略与收益矩阵 | 已实现首轮保守策略 |
| 5 | layer-wise restore 预取契约 | hook/映射契约通过，生产预取未启用 |
| 6 | profile-gated 增量/page-tail publication | 安全规划测试通过，生产增量写入未启用 |
| 7 | 真正异步 Store ownership/fence/preemption | 仅保留设计要求 |

## P3：统一 aligned+pinned staging 可行性

### 要回答的问题

当前每 rank 同时持有 `2 × 64 MiB` Torch pinned pool 和 `2 × 64 MiB` mmap aligned I/O
pool，共约 256 MiB。只有同一块内存能同时被 NVMe `O_DIRECT` 和 CUDA 异步 H2D 使用，才
可以把它降为一套固定 128 MiB pool。

### 探针约束

`benchmarks/probe_shared_staging.py` 不允许 buffered-I/O fallback，也不允许创建第二个 pinned
handoff buffer。它执行以下闭环：

1. 创建两个 64 MiB anonymous mmap，并验证实际地址 4096-byte aligned；
2. 通过 CUDA Runtime `cudaHostRegister` 原地 page-lock 两个 mmap；
3. Torch `frombuffer` 对同一地址建零拷贝 CPU tensor view；
4. 使用 `O_DIRECT` 把两个 slot 写入并读回 rank-local NVMe；
5. 对全部 128 MiB 做 SHA-256，防止只检查头尾造成假阳性；
6. 通过 Torch `non_blocking=True` 搬到 CUDA，并用全量求和验证 GPU 数据；
7. 无论成功失败都删除临时文件、unregister 并关闭 mmap。

运行命令：

```bash
docker exec \
  -e PYTHONPATH=/opt/spoolcache/src:/opt/spoolcache \
  deepseek-v4-flash-vllm-dspark-1 \
  python3 /opt/spoolcache/benchmarks/probe_shared_staging.py \
    --root /var/lib/spoolcache/cf8eabb85fc91ffbab8bc1a1cc32f0ff795dacd899022eab365bfaec7852678a/rank-0000 \
    --slot-mib 64 --slot-count 2
```

### 结果

| 检查 | 结果 |
|---|---|
| 总 host allocation | 128 MiB |
| slot 地址对齐 | 两个地址均为 4096-byte aligned |
| `cudaHostRegister` | 成功 |
| `Tensor.is_pinned()` | 两个 slot 均为 `true` |
| `O_DIRECT` 写 + fsync | 128 MiB / 15.387 ms，约 8318.5 MiB/s |
| `O_DIRECT` 读 + SHA-256 | 128 MiB / 64.376 ms，约 1988.3 MiB/s |
| H2D | 64 MiB / 1.254、1.278 ms，约 51.0、50.1 GiB/s |
| 全 payload SHA-256 | 通过 |
| 全 GPU payload 校验 | 通过 |

结论：当前 GB10/CUDA 13/Torch 2.11/ext4/NVMe 环境具备统一 128 MiB pool 的必要条件。
这还不是生产切换批准：实际实现需要让 Store 与 Torch mover 共享 slot ownership、CUDA
completion event 和关闭顺序，并重跑 crash/corruption/并发/soak 测试。写入带宽受设备缓存、
文件系统及单次样本影响，只作为兼容性证据，不作为稳态 NVMe 性能结论。

## P4：Store admission 与 skip

### 已有规则与缺口

此前已有三项安全过滤：小于 `min_span_tokens` 不产生候选、完整 rank quorum 已存在的 entry
不重复保存、`restore-only`/`disabled` 不保存。但配置中的 `max_pending_stores` 没有接入
调度路径；同一个 scheduler step 如果产生多个候选，worker 会在 forward 前同步串行执行
全部 Store，造成无上限的 TTFT 放大。

### 已实现规则

新增纯函数 `admit_store_plans()`，不依赖 vLLM，规则如下：

- 同批相同 `entry_id` 只保存一次；相同 ID 却声明不同 span 时 fail closed；
- 最多接纳 `max_pending_stores` 个候选；默认值 1 现在真正生效；
- `max_pending_stores=0` 明确表示跳过所有可选 Store，不影响 restore；
- 候选竞争时优先 span 更长者，同长度按输入顺序稳定选择；
- 日志只记录 admitted/duplicate/budget 计数，不记录 prompt、token 或 salt。

这不是后台异步队列：当前 Store 仍在公开 hook 内同步完成。“pending”预算在本阶段表示每个
scheduler metadata 批次允许进入 worker 的 Store 数量，以保护前台延迟。

### 经济性测试

`StoreEconomics` 与 `benchmarks/analyze_store_economics.py` 把配对的 disabled cold、
miss+Store、cross-restart hit TTFT 转成可审查的盈亏点：

```bash
PYTHONPATH=src python3 benchmarks/analyze_store_economics.py \
  --case 8K,4.910,6.474,0.918 \
  --case 32K,19.216,20.822,1.036 \
  --expected-reuses 0 0.1 0.25 0.5 1 2
```

| 上下文 | Store 额外成本 | 每次 persistent hit 节省 | 盈亏平衡的预期未来复用 | 恰好复用 1 次净节省 |
|---|---:|---:|---:|---:|
| 8K | 1.564 s | 3.992 s | 0.3918 | 2.428 s |
| 32K | 1.606 s | 18.180 s | 0.0883 | 16.574 s |

因此，已测的 8K/32K 前缀在“至少再用一次”时 Store 明显合算；但 1K 尚无数据，未来复用
概率也不是 connector 能可靠预知的事实。当前不硬编码“见到第二次才存”或把默认
`min_span_tokens` 提升到 8K。低复用工作负载可使用 `restore-only`。早期实验使用过
`SPOOLCACHE_MAX_PENDING_STORES=0`；该低层环境变量现已删除，pending/store/span 上限均为
内部有界默认值，待补齐 1K/并发/写放大矩阵后再评估是否需要新的高层策略。

### 测试结果

- 宿主 pytest：37 passed、4 CUDA skipped；
- 目标容器完整 unittest：41 passed（包含 4 个 CUDA/HMA mover 测试）；
- admission 覆盖去重、最长优先、稳定 tie-break、零预算、identity 冲突和 TTFT 盈亏计算。

## P5：layer-wise restore 预取契约

### 固定 vLLM 源码证据

目标镜像中的 vLLM 为 `0.25.2.dev0+g752a3a504.d20260714`。真实
`maybe_transfer_kv_layer` wrapper 在 connector metadata 与 attention metadata 存在时执行：

```text
connector.wait_for_layer_load(layer_name)
attention(...)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

DeepSeek 的 `unified_mla_attention_with_output` 装饰顺序为外层
`@eager_break_during_capture`、内层 `@maybe_transfer_kv_layer`。vLLM 自身说明，如果 connector
在这些 hook 做异步逐层同步，必须通过 `requires_piecewise_for_cudagraph()` 请求 PIECEWISE
CUDA graph，否则 replay 会跳过 Python side effect 并产生 data race。

### DeepSeek HMA 特殊点

现有 rank-0 manifest 的 170 个注册 cache tensor 只对应 46 个实际 attention hook：

- layer 0、1、43、44、45 各只有一个 SWA cache tensor；
- layer 2–42 的普通层通常对应 3 个 tensor；
- layer 2–42 的偶数层对应 5 个 tensor，还包括 indexer cache/state。

所以 `wait_for_layer_load("model.layers.2.attn")` 返回前，不能只恢复同名 `.attn` tensor；它
必须保证该层的 `.swa_cache`、`.compressor.state_cache`、
`.indexer.k_cache` 和 `.indexer.compressor.state_cache` 全部就绪。

新增 `build_layerwise_stages()` 以 profile/name gate 构建严格映射。它要求层号从 0 连续、
每个 cache 名称能映射到唯一 attention hook、每个 HMA tensor 恰好出现一次。任何未来模型
命名或深度漂移都会 fail closed，而不是把辅助状态挂到错误 barrier。

### 测试与决定

- CPU/profile 测试：170 个 tensor 恰好映射为 46 个 stage，未识别名字 fail closed；
- 目标 vLLM 真实 decorator 测试：实际事件顺序严格为 wait → attention → save；
- 目标 MLA 源码测试：cudagraph break/transfer 装饰顺序正确；
- 第一版 runtime 测试曾因 `get_attention_context` 在 decorator 构造时捕获而 patch 太晚，
  修正为先 patch、再构造 wrapper；这属于测试隔离问题，已保留注释避免回归。

结论是“公开 hook 足以承载未来 layer-wise 实验”，不是“现在已经安全启用”。生产实现仍需
后台 worker/CUDA stream、批内所有 request 的逐层 barrier、异常跨线程传播、取消与 shutdown
排空，并评估 PIECEWISE CUDA graph 对原始推理性能的影响；这些证据齐全前维持整 entry
同步 restore。

## P6：增量/page-tail publication 安全规划

### 安全边界

当前生产路径仍写完整快照。新增的 `plan_incremental_publication()` 只是把未来增量实现必须
遵守的不可变性规则变成可执行测试：

- 只有非 EAGLE 的 `reuse_policy=full`、已经完整封口的旧前缀页可以引用旧 immutable
  objects；
- full group 在旧 span 之后的新页必须重新捕获；
- sliding 与 recurrent state 即使新旧 selected page count 相同，也必须全部重新捕获；
- EAGLE group 在模型版本专门证明前全部重新捕获；
- 两个 span 都必须按整个 HMA layout 的 LCM 对齐，partial page tail 直接拒绝；
- caller 必须先证明 token、media geometry、namespace 与 salt 的前缀连续性，不能仅凭两个
  不同 span 的 entry ID 猜测。

### 8K → 32K 规划样例

在实机 `62/23/23/42/20` 个 cache tensor 的五组目标几何下：

- full group：旧 32 页可以复用，新增长的 96 页/层重新捕获；
- 其余四组：2、2、2、16 个 boundary-relative page/层全部重新捕获；
- 可复用 1,984 个 page-layer pair；需新捕获 6,448 个；
- 新 32K 完整快照共有 8,432 个 page-layer pair，因此这个样例理论上减少约 23.5% 的
  page capture/object publication 操作，实际节省字节仍应按各层 `page_size_bytes` 计算。

测试同时覆盖 EAGLE 保守路径、partial tail、未证明 prefix continuity 和非增长 span。此阶段
不修改 manifest schema，也不接入 connector；生产化前还需要验证旧 object descriptor 的
分片边界、引用计数/淘汰、跨崩溃恢复、两 rank 原子 quorum 和真实写放大收益。

## 优化后端到端加载验证

本节不是新的 enabled/disabled A/B，而是确认 P1–P6 代码已经被实际服务加载，并且没有破坏
此前建立的文字、视觉和短请求基线。服务绑定端口为 `8888`；第一次误用 `8000` 得到连接拒绝，
请求没有进入引擎，因此不计入任何缓存或性能样本。

### 文字持久命中

先以唯一 nonce 发送约 8K 的文字请求，再完整停止并启动两个 TP rank，最后逐字节重放：

| 阶段 | prompt tokens | restored tokens | TTFT | wall time |
|---|---:|---:|---:|---:|
| 首次 miss + Store | 8,205 | 0 | 13.598785 s | 13.625637 s |
| 完整重启后命中 | 8,205 | 7,168 | 0.892959 s | 1.145024 s |

两个 rank 均记录相同 entry `dabf9a006bbc` 的 Store/restore，API 输出正确，未出现
CUDA、NCCL 或 payload 校验错误。首个 cold 样本受到启动后状态和单次采样噪声影响，约
15.2× TTFT 只作为这对请求的功能收据，不替代正式 A/B 中的 5.35× 结论。

### 视觉持久命中

视觉探针使用内嵌 1×1 白色 PNG 和固定约 5K 文本。复原旧测试文本的第一次请求比旧记录少
2 个模板 token，因此按正确语义发生 miss；它随后在两个 rank 发布了新的 entry。完整重启后
用完全相同的图片字节和文字重放：

| 阶段 | prompt tokens | restored tokens | wall time | 输出检查 |
|---|---:|---:|---:|---|
| 首次 miss + Store | 5,136 | 0 | 8.688073 s | 正确识别白色方块 |
| 完整重启后命中 | 5,136 | 4,096 | 3.164754 s | 同样正确识别白色方块 |

rank 0 和 rank 1 均恢复 entry `f263e410577f`。API 的 `cached_tokens=4096`，且日志没有
partial-rank hit、CUDA/NCCL 或校验错误。因为这里使用非流式请求，记录的是 wall time，不把
它冒充 TTFT。可复现脚本为 `benchmarks/bench_visual_prefix_e2e.py`。持久层验证必须在两次
执行之间完整重启全部 rank，或者在隔离实验环境用官方开发接口只清 GPU prefix cache；
否则可能只测到 vLLM 自己的进程内命中。

### 运行时布局与无重启 GPU reset 资格收据（2026-09-04）

启用 `vllm-runtime-kv-v1` 后，实机发现 scheduler 与两个 worker 的 group 顺序、170 个
layer、block/window、EAGLE 和 layer ownership 完全一致。三者的 `logical_digest` 均为
`cad6a42db130`；worker 的完整物理摘要为 `00d41a39ecda`，scheduler 的简化物理视图为
`eb5418d7d2d4`。这验证了“逻辑语义跨角色一致、page 字节几何留在 rank-local manifest”
的设计，而不是继续依赖写死的模型尺寸表。

第一版启动握手错误地要求 scheduler/worker 的完整 `DeploymentIdentity` 相同，实机按设计
fail closed 并拒绝建立 quorum，错误为 `scheduler/worker deployment identities differ`。
检查表明 vLLM 在模型 materialize 前后给各角色的 `ModelConfig` 视图不同。修正后，握手只
比较 checkpoint、runtime、topology、逻辑布局等真正跨角色的 `coordination_digest`；完整
deployment identity 仍分别隔离 scheduler 的前缀 namespace 和 worker 的持久目录。随后
两节点服务正常完成握手并通过 47/47 启动预热。CPU 契约测试也固定了“角色本地 deployment
不同、coordination 相同”这一行为，防止以后回归。

开发环境打开 vLLM 官方 `POST /reset_prefix_cache`，调用时显式指定
`reset_running_requests=false&reset_external=false`。这只使 GPU 中的 prefix blocks 失效，
不会删除 SpoolCache NVMe entry。无重启、逐字节重放结果：

| 请求 | 冷请求 | GPU reset 后 SpoolCache 命中 | 证据 |
|---|---:|---:|---|
| 8,205-token 文字 | cached 0，TTFT 6.772627 s | cached 7,168，TTFT 0.974652 s | entry `569cd7d9c089`；两 rank restore；输出 SHA-256 相同 |
| 5,146-token 白色 PNG + 文字 | cached 0，wall 6.495118 s | cached 4,096，wall 1.365422 s | entry `60e5938514f9`；两 rank restore；两次均正确识别纯白空白图 |

因此这两组热请求不是 vLLM 进程内 prefix cache 命中。文字测试约 6.9 倍 TTFT 差异、视觉
测试约 4.8 倍 wall-time 差异只作为单对功能收据，不替代重复采样的正式性能结论。视觉
探针未把非流式 wall time 冒充 TTFT；其生成措辞有轻微差异，但语义 oracle 一致，KV
payload 的字节正确性另由 page checksum/mover 测试覆盖。

开发模式同时暴露权重更新、RPC 和调试路由，仅允许在隔离实验机使用；示例和生产默认值
仍为关闭。复现 GPU-only reset 使用 `benchmarks/reset_gpu_prefix_cache.py`，不要把
`reset_external` 改为 `true`。本次变更后的宿主测试为 56 passed、6 个环境性 skipped；
在线目标镜像的真实 vLLM/HMA 契约子集为 16/16 passed；部署仓库 CPU recipe CI 通过。

### 短提示空路径回归

`bench_decode.py` 使用 270 个实际 prompt token，低于 1,024-token Store 门槛，因此不会把
同步落盘成本混入 decode 测量。P1–P6 加载后的 5 次 C1/128-token 结果为：

| 指标 | 优化后复测 | 历史 enabled 基线 | 差异 |
|---|---:|---:|---:|
| median decode | 80.5425 tok/s | 80.65 tok/s | 约 -0.14% |
| median aggregate/wall | 66.2755 tok/s | 66.12 tok/s | 约 +0.24% |
| median TTFT | 0.342081 s | 0.348 s | 约 -1.70% |

差异远小于逐波运行波动，支持“本轮没有发现短请求性能回归”，不支持宣称 SpoolCache 让
decode 变快。五次单独 decode 样本为 80.5425、80.5892、80.6114、77.1730、77.4708 tok/s。

### C6 配对复测（2026-09-04）

由于早期 C6 只有 5 轮且出现过 193.77–212.99 tok/s 的大幅度差异，本轮用部署仓库原生
`scripts/bench-miaai.py` 重做 10 + 10 轮配对对照。两侧都完整重启两个 TP rank、通过
47/47 boot-shape warmup，且仅改变 `SPOOLCACHE_ENABLE`。为了避免修改私有 `.env.dspark`，
启动器新增了仅对本次启动生效的 `--spoolcache-enable 0|1`；基准脚本也新增从
`VLLM_API_KEY`/`DSPARK_API_KEYS` 构造 Authorization header 的支持，不会打印密钥。

```bash
python3 scripts/bench-miaai.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model "$SERVED_MODEL_NAME" --prompt 256 --concurrency 6 --repeat 10
```

| 指标（10 轮中位数） | SpoolCache 关闭 | SpoolCache 开启 | 开启相对关闭 |
|---|---:|---:|---:|
| 每路 decode | 30.0 tok/s | 29.9 tok/s | 约 -0.3% |
| aggregate/wall | 143.8 tok/s | 143.7 tok/s | 约 -0.1% |
| TTFT | 881 ms | 896 ms | 约 +1.7% |

原始 aggregate 样本：

- 关闭：136.8、145.3、143.5、140.9、139.3、147.6、144.1、150.0、135.4、149.3 tok/s；
- 开启：137.0、147.6、145.4、147.0、133.2、147.5、142.9、143.9、133.3、143.5 tok/s。

三个指标的差异都远小于轮间波动，所以这组证据支持“SpoolCache 没有导致 C6
回归”，不支持宣称有性能提升。这些请求实际 prompt 长度远低于 1,024-token Store
门槛；启动后日志中的 2,048/9,216-token Store 来自 boot-shape 长上下文预热，不是
C6 请求。因此 P1 的单次读取和 P2 的 restore 流水线在本测试中都不会进入数据路径，
理论上也不应改善短提示 C6。

本轮两侧的 aggregate 都比部署项目 `ab-measure.sh` 记录的 156–162 tok/s 参考带低约
8–11%，说明用户关心的 C6 偏低现象仍然存在，但与 SpoolCache 开关无关。启用服务的
speculative-decoding 累计 accepted/draft 为 6,564/13,584，约 48.3%，处于项目标注的
45–51% 参考带。所以当前更值得单独调查的是 DSpark/vLLM 的 C6 scheduler/fairness 和
运行时波动；“不是 acceptance 异常”只是基于本次累计窗口的推断，不是已完成的瓶颈定位。

## 128K 满 GPU KV / NVMe restore 干扰测试

### 为什么使用 128K

本部署一次启动实际报告 2,447,611-token GPU KV 池。测试用 20 个彼此从首个 hash block
开始就不同的 131,072-token 前缀构造 2,621,440-token 工作集，即池容量的 107.1%。相比
约 80 个 32K 前缀，128K 显著减少请求、manifest 和对象调度碎片，也让一次 NVMe restore
形成更容易观察的连续读取。工作集超过容量后，较早的三个 128K 前缀由 vLLM 自然淘汰；
正式过程不通过服务重启制造 miss。

计时阶段固定为五路 256-token 前台提示、每路生成 1,024 tokens，并把第六个并发槽用于
三次 128K 注入。记录前台 decode throughput、全部非空 SSE event gap、注入窗口内 gap、
running/waiting/KV usage、local/external hit tokens 和 preemption。SSE chunk 可能包含多个
speculative token，因此 event gap 不能冒充逐 token TPOT。

### 首轮收据与无效对照

原始收据为 `results/restore-interference-c5-128k-20260904b.json`。20/20 工作集和三个
NVMe 请求均成功、两 rank 日志一致、无抢占或请求错误。NVMe 组每次恢复 130,048 tokens，
合计 external hit 390,144、local hit 0，证明磁盘数据路径确实进入。与无注入基线相比，
前台全部 event-gap P99 从 0.179 s 增至 1.347 s，最大值从 0.824 s 增至 1.501 s；前台
aggregate decode 从 143.50 增至 156.14 tok/s，因此这一轮没有观察到总吞吐下降，但观察到
客户端可见的秒级尾部停顿。

这份收据仍由严格 validator 标为 `INVALID`，不能把尾部停顿全部归因于 NVMe：原计划的
“GPU-local”组虽然 metrics 只把 9,216 / 390,144 tokens（2.36%）记为 external，两个
worker 的日志却证明三次都执行了完整 130,048-token SpoolCache restore。它不是纯本地
对照，只能说明“加入第六个长请求”的调度影响与磁盘影响尚未拆开。原始文件和失败结论均
保留，不把昂贵但有缺口的样本改写成通过。

为补齐对照，connector 新增请求级
`kv_transfer_params.spoolcache_bypass=true`：它同时跳过该请求的持久读写，仍允许 vLLM
进程内 GPU prefix cache 命中。benchmark 先用 bypass 冷建一个独立 128K 本地前缀，随后
所有 local-control 注入也携带 bypass，并要求 external-hit delta 严格为 0。已落盘的首轮
restore/eviction fixtures 可以通过 `--reuse-persisted-fixtures-from` 重放，避免再次冷计算
2.62M tokens；重放后仍在同一进程内用超容量工作集自然淘汰，正式对比不使用重启。

### 统一内存观察

模型日志记录每 rank 约 80.12 GiB model allocation，KV 预算为 rank 0 18.4 GiB、rank 1
16.48 GiB。SpoolCache 当前生产实现每 rank 为 128 MiB pinned pool 加 128 MiB aligned-I/O
pool；共享 128 MiB 单池只有探针通过，尚未接入生产。两个并行 128K cold prefill 时，head
一度只剩约 0.15 GiB MemAvailable、swap 峰值约 13.6/16 GiB；一批结束后 swap 曾回落到约
6.4 GiB。该锯齿与活跃 cold prefill 同步，而不是随 NVMe entry 数线性增长，说明主压力
来自模型、vLLM GPU KV 和超长 prefill 的临时工作区，不是 SpoolCache 把持久 payload 留在
RAM。生产资格应同时限制 `gpu_memory_utilization` 和长 prefill 并发，不能只依据理论 KV
token 容量。

复现命令：

```bash
python3 benchmarks/bench_visual_prefix_e2e.py \
  --model "$SERVED_MODEL_NAME" --nonce stable-visual-id

python3 benchmarks/bench_decode.py \
  --model "$SERVED_MODEL_NAME" --prompt-tokens 256 --output-tokens 128 \
  --concurrency 1 --repetitions 5 --nonce unique-short-regression-id
```

### 本轮测试总账

- 宿主 `.venv` pytest：43 passed、6 skipped；skip 均为没有本机 CUDA/vLLM runtime 的
  环境性跳过，不是失败；
- 目标镜像离线完整 unittest：49 passed，包含真实 CUDA/HMA mover 与固定 vLLM runtime
  契约。最初尝试在在线模型旁启动第二个 CUDA context 时，head 只有约 3.5 GiB available
  unified memory，测试进程连最小 CUDA allocation 都报 OOM；停止两 rank 后，以相同固定
  image 和只读挂载的当前源码重跑为 49/49。这个资源型失败不计为代码通过，也不能靠跳过
  CUDA 测试隐藏；以后完整 GPU 测试应安排在服务停止时执行；
- DSpark 仓库 `scripts/ci-validate.sh`：通过，覆盖 shell/Python 语法、CPU 单元测试、
  compose/启动配方 gate 和已知回归 guard；脚本中的若干 `[FAIL]` 字样来自故意注入错误的
  fail-closed 单测，其 unittest 结果为 `OK`，最终总状态为 `CI validate passed`；
- 最终目标服务重新启动并完成 `47/47` boot-shape warmup：两 rank inventory 各 29
  entries，`direct_io=True`，每 rank `pinned_bytes=134217728`，API health 为 HTTP 200；
- 当前生产实现仍是两套池：128 MiB pinned + 128 MiB aligned I/O / rank。P3 只证明未来
  合并成一套 128 MiB shared pool 可行，尚未切换，不能把“探针通过”写成“已经节省内存”。

## Qwen3.8-Flash-Next 自动兼容资格（2026-09-05 UTC）

本轮在两台 DGX Spark（TP=2、EP、MTP=3）上使用
`RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594` 和镜像
`vllm/vllm-openai:qwen38-flash-next`（image config digest
`d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`）完成。
镜像内 vLLM 为 `0.1.dev20073+g8e685d198`，运行时 public-API source fingerprint 为
`032f5527b32f5cc47864dc6c8a7e11a9b73bafa0`。

启动没有模型特定 profile。当时的 `SPOOLCACHE_PROFILE=auto` 选择器从实际
`KVCacheConfig` 自动发现
6 groups / 76 layers：26 层 full attention、13 层单请求 circular buffer，以及
12+12+12+1 层 align-mode recurrent state；共同安全边界为 1,600 tokens。运行时兼容
选择器 `auto` 通过 public API gate，v3 logical layout digest 为 `36dccf844d36…`。
未知 cache spec、接口漂移、非法页或 tensor ownership 不一致仍会在启动时 fail closed。
后续零特例清理删除了该冗余环境变量；同一发现路径现为无条件启动行为。

首轮测试暴露了一个真实正确性问题：vLLM 在 connector hook 前把 align-mode recurrent
边界状态预拷贝到下一 active page，而旧选择仍恢复历史页。命中虽然是 12,800 tokens，
确定性输出却不同，因此该结果被判失败。修复改为按运行时 block table 选择 active page，
并升级 layout schema，使失败运行写出的旧条目不可能被新服务复用；没有加入 Qwen 名称、
层数表或逐模型配置。

修复后的冷写入与恢复收据如下（entry `36b868f6e693…`）：

| 请求 | prompt | external cached | TTFT | elapsed | 输出 SHA-256 |
|---|---:|---:|---:|---:|---|
| producer/cold | 12,800 | 0 | 6.629 s | 7.617 s | `eb62f2aa7a27…` |
| 12,801 control（bypass） | 12,801 | 0 | 4.305 s | 4.427 s | `00c9f2fd9176…` |
| 12,801 SpoolCache restore | 12,801 | 12,800 | 0.337 s | 0.458 s | `00c9f2fd9176…` |

control 与 restore 使用相同精确 token prefix、salt、seed、`temperature=0`，中间两次均通过
开发端点仅清 GPU prefix cache 并明确保留 external cache。两 rank 均记录同一 entry 的
12,800-token store 和 restore；scheduler 记录同 entry hit。输出哈希完全一致，TTFT 相对
bypass 约 12.8x，且最终日志没有 CUDA、NCCL、checksum、layout 或 fatal restore 错误。
再次清空 GPU cache 后复验仍命中 12,800 tokens、输出哈希相同，TTFT 为 0.242 s。
目标镜像内 HMA/合同测试为 12/12 和 4/4；宿主回归为 62 passed、6 个环境性 skipped。

### Qwen 图片/视频内容校验（2026-09-05 UTC）

多模态启动发现通过 vLLM 公共 registry 得到 `image`、`video`；这两个名字来自当前模型
能力和部署 limit，而不是 SpoolCache 白名单。图片固定为
[COCO](https://cocodataset.org/#termsofuse) `val2017/000000039769.jpg`（173,131 bytes，
SHA-256
`dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`）；视频固定为
[PyTorchVideo `archery.mp4`](https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4)
（549,197 bytes，SHA-256
`8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562`）。请求发送本地固定
bytes 的 data URL，避免 producer 与 consumer 下载到变化的远端内容。媒体只作为临时
资格输入，不提交到仓库；COCO 图片继续受其来源图片条款约束，PyTorchVideo 仓库本身为
[Apache-2.0](https://github.com/facebookresearch/pytorchvideo/blob/main/LICENSE)，重新分发前仍
应核对该视频素材自身的权利信息。

图片和视频各先用 12,800-token producer 在两个 rank 写入，再构造与 producer 前
12,800 tokens 完全一致的 12,801-token consumer。每个 consumer 先携带
`spoolcache_bypass=true` 生成冷对照，然后清除 GPU prefix、encoder 和 multimodal cache，
明确保留 external store，再重放同一请求。受约束三选一语义 oracle 的结果为：

| 模态 | 冷对照 | SpoolCache 命中 | 恢复 tokens | 完整输出 SHA-256 |
|---|---|---|---:|---|
| image | `CATS` | `CATS` | 12,800 | `90a4171e6918b5dc2d62d01b658933b06bfd4614881bf459da63a493b9657aff` |
| video | `ARCHERY` | `ARCHERY` | 12,800 | `37cad37e8e0ef0d3dea1d7c8540184cc2a17becf9a9556bac1c299e6e5ea1e5b` |

两组命中的规范化完整消息逐字节一致，scheduler 与两个 TP worker 均记录同一个 entry 和
12,800-token span。自由生成旁路也分别识别为 `CATS` 和 `ARCHERY`；图片自由生成的一次
冷/热比较只差句末句点，所以没有把它冒充成逐字节确定性证据，而是使用同一候选集合的
受约束生成完成严格比较。当前 checkpoint 没有通过 registry 声明 audio，因此本轮没有
audio KV 收据；这不会在 SpoolCache 中形成 audio 或模型白名单。

### Qwen 最终 review 回归与故障注入（2026-09-05 UTC）

零模型特例、统一自动合同检查和强 identity 绑定完成后，又在同一目标镜像执行了一轮最终
回归。启动时从实际可导入 vLLM 包计算出的完整 SHA-256 为
`0b2c8c058b981d91b5ea728104f2f810aa43fab6a1e35f8ddd4acee040f83089`；registry 仍只声明
`image`、`video`，没有声明 audio。运行时自动发现 6 groups / 76 layers、1,600-token 对齐，
logical layout 为 `36dccf844d36…`，两个 worker 的 HMA layout 为 `60368d3e62db…`。

最终新建的 text/image/video entries 分别为 `fc7eb2b9397c…`、`b4b4797c0727…` 和
`926ed2769a21…`。所有 consumer 都先清除 GPU prefix、encoder 和 multimodal cache且保留
external store；结果如下：

| 输入 | bypass cached | restore cached | bypass/restore oracle | 完整输出 SHA-256 |
|---|---:|---:|---|---|
| text | 0 | 12,800 | 逐字节相同 | `7b91622137f9860f7196875095d7e2d8601a9a4e8fb4e4588304bce2439f3a25` |
| COCO cats image | 0 | 12,800 | `CATS` / `CATS` | `f43c3a6a804ebd0b94e1d1384cb4680f7a24a9d2558076cfee5fee173e0836f8` |
| PyTorchVideo archery | 0 | 12,800 | `ARCHERY` / `ARCHERY` | `9041bdc879c87001e177bc15e8eca4e2de5d1aabe1038204e1f68d8d631e64dd` |

相同 padding/nonce 下替换为 street image 或 kitchen video 均为 0 external cached tokens，并
分别严格输出 `STREET`、`KITCHEN`，证明媒体 bytes 参与 prefix identity。scheduler、rank 0
和 rank 1 对三条 entry 均记录相同 12,800-token hit/restore。

身份绑定版离线 verifier 在两个 rank 上逐条验证了这三条 entry：每条均为 6 groups、76
layers、76 objects，页覆盖 `8,1,1,1,1,1`，每 rank logical/stored bytes 均为
242,995,200，最终状态都是 `all-payloads-authenticated`。验证参数明确绑定各 rank 的
deployment identity、rank identity、physical rank、topology 和 HMA layout。

另对旧 text entry `879cac187ffc…` 的 rank-0 object
`f786779416ca…` 翻转首字节。已 admission 的 restore 检出 SHA-256 不一致后，严格 benchmark
在约 3.47 秒内以不完整 stream 失败，故障 worker 通过 `os._exit(70)` 退出，head API 随即
不可用；恢复原始字节并完整重启后，上述最终矩阵全部通过。此次实验也发现当前部署脚本
没有让远端 headless rank 随 head rank 自动退出，必须由 `stop.sh` 清理。该 supervisor
整组生命周期缺口已列入 `TODO_GOALS.md` 的 G3，不影响 fail-stop 对请求返回伪成功的封堵，
但在生产运维闭环完成前仍是明确限制。

## GLM-5.3-Flash EXL3 自动兼容与全内容正确性（2026-09-05 UTC）

本轮使用 `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` immutable revision
`25a44fdbf16862a46b7cc9921142c6c81350af2f`，目标镜像 digest
`ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263`，镜像内
vLLM `0.1.dev20051+g487ecf187`，public-API source fingerprint
`efc7dc774062bd9453ac463ed1627ffe9d0f60fb`。两台 DGX Spark 以 TP=2、DFlash2
k=7、`fp8_ds_mla` target KV 启动，SpoolCache 仍为只读源码挂载，没有向镜像或 vLLM
写入任何连接器补丁。

当时的 `SPOOLCACHE_PROFILE=auto` 从真实 `KVCacheConfig` 发现 7 groups / 72 layers，安全对齐为
3,584 tokens：22 层 full MLA、11 层请求级 circular K-pool、四个 recurrent-align 组
（9+9+8+8 层）和 5 层 DFlash sliding window。K-pool 的 cache spec 继承
`SlidingWindowSpec`，但公开契约声明每请求一个循环 scratch page 且
`participates_in_prefix_caching=False`；通用 inspector 据此识别 `circular-one`，没有读取
GLM 模型名或层数表。
后续零特例清理删除了该冗余环境变量；同一发现路径现为无条件启动行为。

首次启动还发现 scheduler 的逻辑 dtype 标签为 `fp8`，worker materialize 后标签为
`fp8_ds_mla`。旧 coordination receipt 错把该角色本地标签当成跨角色恒等事实，因而在建
quorum 前 fail closed。修复后的 coordination schema 不再包含物理 dtype 标签；它仍由
每个 role-local deployment identity 和 rank manifest 绑定。新增回归测试证明
ModelConfig/dtype 的合法 materialize 差异不会破坏握手，而 checkpoint、runtime、topology
和 logical layout 差异仍会拒绝。

端到端收据使用 entry `848b8605a843…`：

| 请求 | prompt | external cached | TTFT | elapsed | 输出 SHA-256 |
|---|---:|---:|---:|---:|---|
| producer/cold | 7,168 | 0 | 10.378 s | 11.553 s | `9b2fab355d67…` |
| 7,169 control（bypass） | 7,169 | 0 | 8.321 s | 8.432 s | `95b2dd251d37…` |
| 7,169 SpoolCache restore | 7,169 | 7,168 | 0.422 s | 0.533 s | `95b2dd251d37…` |
| 第二次 GPU-reset restore | 7,169 | 7,168 | 0.432 s | 0.540 s | `95b2dd251d37…` |
| 生产模式重启后 restore | 7,169 | 7,168 | 0.511 s | 0.613 s | `95b2dd251d37…` |

control 与两次 restore 使用相同 token prefix、salt、seed、`temperature=0`，每次 restore
前都只清除 vLLM GPU prefix cache 并保留 external store。scheduler 两次均命中相同
7,168-token entry；两个 TP rank 的 store/restore 日志也逐次一致。恢复输出与 bypass
对照的完整 SHA-256 相同。

资格测试完成后，两节点以 `VLLM_SERVER_DEV_MODE=0` 重新启动；两个 rank 的 startup
inventory 均发现 1 个旧 entry。新进程 GPU cache 为空时直接恢复同一 7,168-token entry，
输出哈希仍与 bypass 对照一致，从而同时验证了跨进程持久性。最终容器环境确认两个
expandable allocator 变量为空、`PYTHONPATH=/opt/spoolcache/src`，生产服务不暴露开发重置
端点。

额外运行 `benchmarks/verify_entry_content.py` 对两个 rank 的 manifest 和所有 payload 做
离线全量认证：每个 rank 都覆盖 7 groups / 72 layers / 77 objects，按组页覆盖严格为
`2,1,1,1,1,1,32`，逻辑 payload 511,741,440 bytes、落盘 511,758,336 bytes；所有文件类型、
长度、零 padding 和逻辑内容 SHA-256 均通过。最终 API health 为 HTTP 200，日志中没有
CUDA/NCCL/checksum/layout/corruption/partial-rank 错误。宿主 SpoolCache 回归 63 passed、
6 个环境性 skipped；GLM 目标镜像内部署仓 tests 为 68 passed；Qwen 目标镜像上更新后的
HMA/connector 回归为 17/17。

review 修复后使用身份绑定版 `verify_entry_content.py` 复核当前 14,336-token entry
`66c81caf5f2b…`。调用时分别提供两个 rank 的 deployment/rank identity、physical rank、
topology 和 layout 预期，并由工具校验内部 layout protocol；身份比较完成后才逐一读取 payload。rank 0/1 均
覆盖 7 groups / 72 layers / 77 objects，逻辑 payload 各 566,067,712 bytes，最终返回
`all-payloads-authenticated`。两端 deployment、topology、layout 一致，physical rank 与
rank identity 按 rank 区分；负向 CPU 测试确认错误 deployment、rank 或 layout 在 payload
I/O 前拒绝。复核期间生产 health 保持 HTTP 200。

### GLM 图片/视频内容校验（2026-09-05 UTC）

GLM 启动时同样由 vLLM registry 自动发现 `image`、`video`。测试复用上节 Qwen 收据中的
固定 COCO 图片和 PyTorchVideo 视频 bytes。两种模态分别由 14,336-token producer 在两个
rank 落盘，再构造 14,337-token consumer；其前 14,336 tokens 已通过 `/tokenize` 逐 token
确认与 producer 完全一致。每组都依次执行本地缓存 reset、
`spoolcache_bypass=true` 冷对照、再次只清本地 cache、SpoolCache restore。

| 模态 | 冷对照 | SpoolCache 命中 | 恢复 tokens | 完整输出 SHA-256 |
|---|---|---|---:|---|
| image | `CATS` | `CATS` | 14,336 | `90a4171e6918b5dc2d62d01b658933b06bfd4614881bf459da63a493b9657aff` |
| video | `ARCHERY` | `ARCHERY` | 14,336 | `37cad37e8e0ef0d3dea1d7c8540184cc2a17becf9a9556bac1c299e6e5ea1e5b` |

两个 TP worker 对每种模态都记录了相同 entry/span 的 store 和 restore。两组规范化完整消息
逐字节相同；自由生成的 producer 也分别回答 `CATS` 和 `ARCHERY`。验证后服务重新以
`VLLM_SERVER_DEV_MODE=0` 启动，避免在生产状态保留 reset 等开发端点。

### GLM 最终自动契约与故障闭环复验（2026-09-05 UTC）

零模型特例清理后的 GLM runtime 自动报告 `image`、`video`，没有声明 audio。运行时实际
installed-package SHA-256 前缀为 `74b8b658f0e8`；TP=2、PP=1，7 groups / 72 layers，
alignment=3,584，logical layout 为 `30d35dbc680d…`，worker HMA layout 为
`d8311f3f9860…`。topology digest 为 `748cd94a00db…`；rank 0/1 deployment identity
分别为 `533e0903c9b4…` / `b041dbe51327…`，rank identity 分别为
`51e49a2b9cf4…` / `bd84e0e81728…`。这些值在多次完整进程重启后保持稳定，且启动日志现
输出完整 digest，供离线 verifier 独立绑定。

最终条目和内容 oracle：

| 输入 | entry | persistent span | bypass/restore 完整输出 SHA-256 | 不同输入隔离 |
|---|---|---:|---|---|
| text | `01502cde8f19…` | 14,336 | `7a7bcd4680eb64f51f1f69c732303943f6a0a7b662088fddc6ca7ce5336add03` | synthetic nonce/salt 固定 |
| COCO cats image | `9d716a181b66…` | 14,336 | `5fce21e0a589c04f609eae65c7c12f014b6e6e835396f8bf261654f887d81c5b` | COCO street image cached=0 |
| PyTorchVideo archery | `9a8f4cfd43cb…` | 14,336 | `ff7fa9d9441b7443ee4682da6fa9044d0be683a6c9ae5005d4be2fca4916ccf7` | generated kitchen video cached=0 |

图片和视频 consumer 的前 14,336 个 token ID 均与各自 producer 完全一致。每种输入在
完整两节点重启后仍由 scheduler 命中，rank 0/1 恢复同一 entry/span；text 的恢复后输出
hash 也与故障注入前一致。图片 fixture 为 173,131 bytes、SHA-256
`dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`；视频为
549,197 bytes、SHA-256
`8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562`。

身份绑定 verifier 对 text/image/video 的两个 rank 均返回
`all-payloads-authenticated`。每份 manifest 覆盖 77 objects / 72 layers / 7 groups，页数
为 `4,1,1,1,1,1,32`，logical/stored bytes 分别为 566,067,712 / 566,095,872；rank 0/1
使用各自 deployment/rank identity，topology/profile/HMA layout 也在读取第一个 payload
前核对。

第一次 rank-0 单字节损坏注入证明 SHA 校验会发现问题，但同时暴露普通
`FatalRestoreError` 会被 vLLM worker RPC 捕获，令另一 TP rank 和 HTTP stream 悬挂；该
失败记录为 CR-004。修复后对同一 text object 再次从首字节 `242` 改为 `0`，scheduler 已
承诺 entry 后 rank 0 报 `ObjectCorruptionError`，随后以固定退出码 70 终止。vLLM worker
monitor 在约 3 秒内关闭完整 executor，EngineCore/API 均停止，没有返回部分生成结果。
恢复备份并验证原 SHA-256 `d6e057647054…` 后，完整两节点重启、text 双 rank restore 和
77-object verifier 全部再次通过。开发端点仍只为后续 Qwen/DeepSeek 资格过程临时启用；
全部实验结束后必须按 `TODO_GOALS.md` 恢复 GLM 生产模式。

## DeepSeek-V4 Flash 最终自动契约与故障闭环复验（2026-09-05 UTC）

当前部署实际加载 `orcarouter/DeepSeek-V4-Flash-Vision-Uncensored` revision
`2ef3d5c2bb7d9ccba6ab66314ed9e63bd52ac2a6`，目标镜像 digest 为
`a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`。镜像内 vLLM 为
`0.25.2.dev0+g752a3a504.d20260714`，SpoolCache 对安装包实际内容计算的 SHA-256 为
`dbdf0b80a80233fb1551b268361e96c8a66ad2756e2cd59a5733f7d671f1753f`。vLLM registry
只声明 `image`，没有声明 `video` 或 `audio`；后两者因此记为未验证，未形成运行时拒绝规则。

TP=2、PP=1 的自动 layout 发现得到 5 groups / 170 layers、alignment=256：组层数为
`62,23,23,42,20`，页覆盖为 `48,2,2,2,16`。logical layout digest 为
`5da2e5801a1f…`，worker HMA layout 为 `8f4cb0e6086c…`，topology digest 为
`748cd94a00db…`。rank 0/1 deployment identity 分别为 `f0244f658f8c…` /
`ab26dc79f757…`，rank identity 分别为 `470645d941a8…` / `fbaf1613cf21…`；完整重启前后
保持一致。

最终内容收据：

| 输入 | entry | bypass cached | restore cached | bypass/restore oracle | 完整输出 SHA-256 |
|---|---|---:|---:|---|---|
| text | `e00f9de1f1b3…` | 0 | 12,288 | 逐字节相同 | `2f43521c4cb3c84362542f89a6b68ce1b874a4ef473a268b45deb54a942c4993` |
| COCO cats image | `cb1f8fc9e8bc…` | 0 | 12,288 | `CATS` / `CATS` | `f43c3a6a804ebd0b94e1d1384cb4680f7a24a9d2558076cfee5fee173e0836f8` |

图片 fixture 是 173,131-byte COCO `000000039769.jpg`，SHA-256
`dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`；替换为
`000000252219.jpg`（SHA-256
`1cf48bddb1bcab80d9f4d18514ae438c3e4f57d18e80c4fbea9d5d22396f514a`）后 external
cached tokens 为 0 且输出 `STREET`，证明媒体 bytes 参与 identity。12,801-token 请求只恢复
12,288 tokens，是 chunked prefill 在跨过目标前公开的最后一个完整持久化边界；收据没有把它
误报成 12,800 或 12,801。

完整停止并重新启动两节点后，startup inventory 保留已有 entry。只清理 GPU prefix、encoder
和 multimodal cache 后，scheduler 再次命中 text entry `e00f9de1f1b3…`，rank 0/1 均恢复
12,288 tokens，完整输出 SHA-256 仍为 `2f43521c…`。身份绑定 verifier 随后在两个 rank 上
分别认证 5 groups / 170 layers / 170 objects，logical/stored bytes 为
64,904,192 / 65,200,128，最终都返回 `all-payloads-authenticated`。

故障测试对 rank-0 text entry 的首个 content-addressed object 翻转一个字节。scheduler 已
admit 后，worker 检出 payload SHA-256 不一致并以固定退出码 70 fail-stop；严格 benchmark
在约 4.27 秒内因不完整响应失败，API 随即不可用，没有把部分输出当成功。还原原字节并核对
原对象 SHA-256 后，执行上述完整重启、双 rank restore、输出 oracle 和 verifier，结果全部
恢复。与 Qwen 相同，现有部署 supervisor 只自动拉起本地容器，远端 headless rank 不会随
整个 TP group 自动退出；这项生产生命周期缺口保留在 `TODO_GOALS.md` 的 G3。本轮结束后已
按最初要求停止 DeepSeek 两端服务。

最终实现提交后的独立 code review 又关闭 CR-011 至 CR-017：固定 DeepSeek 派生 checkpoint
identity、严格多模态 capability/usage/离线 verifier 输入、拒绝 rank 强制转换，并让 restore
admission 上限覆盖已分配计划；CR-018 又统一收紧无重启 interference 与 decode benchmark
的响应证据。最终源码的 host pytest 为 93 passed / 7 个环境性 skipped，
`compileall`、diff whitespace 和模型名称源码扫描通过；DeepSeek、Qwen、GLM 三个目标镜像
各自以真实 CUDA 和镜像内 vLLM 执行 95/95 tests。DeepSeek 部署仓 `ci-validate.sh` 全部通过，
并新增 unpinned identity 与双 worker `.venv` 排除门禁；GLM 部署仓此前在目标镜像中为 68/68，
Qwen launcher 通过 shell syntax 和 diff whitespace 检查。目标镜像第一次 discovery 暴露的
顶层 `benchmarks` 导入阴影及后续修复记录为 CR-010。

## Cache-spec 语义发现通用化（2026-09-06 UTC）

复核发现旧 HMA inspector 虽然没有模型名白名单，却仍按 concrete/MRO 类名识别四类 cache
spec。该逻辑会拒绝 vLLM 新增或重命名但语义等价的实现类，也可能让仅伪装成旧类名的对象
绕过启动语义检查。本轮改为由 vLLM 公开的 `get_kv_cache_spec_kind()` 声明可复用状态语义，
SpoolCache 只实现与验证通用 physical-page selection；模型名、架构配置、层数表和 concrete
cache class 均不参与准入。

non-prefix scratch 不依赖 resolver 能否识别其类型。只有公开严格布尔 capability 明确禁止
prefix sharing，`max_num_blocks_per_req()` 在本次部署上界证明 request block table 恰好一页，
且 `max_memory_usage_bytes()` 严格等于一页时才使用 `circular_one`。若 runtime 公开 admission
上界方法，它也必须在真实 `max_in_flight_tokens/max_model_len` 上返回非布尔整数 1。矛盾标记、
缺失/抛错 contract、多页或布尔结果全部在启动时拒绝。未实现 page-selection 的公开 semantic
kind 继续 fail closed，不能靠猜测扩大支持范围。

提交后 review 又发现三处缺口并由 `38fa442` 关闭：启动 contract gate 现已验证 resolver
存在、可调用且能接受单个 cache spec，并把同一个 callable 交给布局发现；聚合 group 优先
使用 resolver 的整体声明，只有 `unknown` 才回退到 members；聚合 scratch 除逐 member 证明
外，还必须在实际共享 allocator/group 上证明单页 block table 与一个 packed physical page，
并拒绝 group/member prefix-sharing 冲突。详情见 `CODE_REVIEW_2026-09-06.md` 的 CR-019 至
CR-021。

验证结果：

| 环境 | vLLM | 结果 | 关键覆盖 |
|---|---|---:|---|
| host Python | 未安装 | 104 passed / 10 skipped | 任意类名正例、类名伪装、未知 kind、聚合语义、group/member 冲突和 scratch 所有权负例 |
| DeepSeek 目标镜像 | `0.25.2.dev0+g752a3a504.d20260714` | 109 tests，OK，6 skipped | 真实 resolver/启动 preflight；任意改名 subclass；镜像没有 non-prefix scratch 类型 |
| Qwen 目标镜像 | `0.1.dev20073+g8e685d198` | 109 tests，OK，5 skipped | 真实 resolver/启动 preflight；任意改名 subclass；聚合单页 scratch 通过 |
| GLM 目标镜像 | `0.1.dev20051+g487ecf187` | 109 tests，OK，5 skipped | 真实 resolver/启动 preflight；任意改名 subclass；聚合单页 scratch 通过 |

三套镜像均以当前源码只读挂载、无 GPU 的临时容器运行；skip 是 CUDA mover，DeepSeek 另有
一项“镜像未暴露 non-prefix scratch”的环境性跳过。`compileall`、diff whitespace 和
`src/` 模型名/具体 cache-spec 类名扫描通过。该变更不触碰 mover、payload 或线上 cache，
因此没有停止、重启或修改当前 GLM 服务；本轮不把既有 CUDA/端到端收据冒充成新实测。
真实默认启动 preflight 三套均返回 `automatic-contract` 且 receipt 包含
`get_kv_cache_spec_kind`；build SHA-256 前缀依次为 DeepSeek `e24d925d2e31`、Qwen
`fa1e7da550af`、GLM `89945da00615`。开发前已同步三个 MiaAI-Lab 仓库，三者当前
`origin/main` 都是目标分支 HEAD 的 ancestor；Qwen 合并上游的审计 commit 为 `cff9ee2`，
三个部署仓工作树均为空。
输出 reuse policy 与三部署既有布局语义不变，所以保持 `spoolcache-coordination/v1` 和
`spoolcache-hma-layout/v3`，不新增运维选项或环境变量。

## G3a 生产可观测性与完整 TP 恢复收据（2026-09-06 UTC）

本轮保持 vLLM 与镜像只读、不修改 vLLM，并继续使用
`spoolcache-coordination/v1`。核心没有模型、架构或模态白名单；18 个公开 connector hook、
构造器、HMA completion hook 和 public cache semantic resolver 都由启动时参数签名与
override substitutability 自动检查。新增指标只使用固定名称和有限 label tuple，worker
通过 vLLM 原生 stats channel 上报有界 delta/gauge；post-admission fatal 与 quarantine
计数先写 rank-local crash-consistent journal，替代进程可在重启后继续导出。

最终 review 修复后，CPU 回归在宿主为 156 passed / 10 skipped；skip 仅因没有真实
CUDA/vLLM 环境。DeepSeek、Qwen、GLM 三个目标镜像分别运行同一 161-test 源码套件，
结果依次为 6、5、5 项环境性跳过，均为 OK。三个部署的 shell syntax、diff whitespace、
内嵌 stop fallback 动态执行和完整 TP supervisor launcher contract 也通过。目标镜像和
vLLM 分别为：

| 部署 | 目标镜像 | vLLM | 测试结果 |
|---|---|---|---:|
| DeepSeek | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | `0.25.2.dev0+g752a3a504.d20260714` | 161 tests / 6 skipped |
| Qwen | `vllm/vllm-openai:qwen38-flash-next` | `0.1.dev20073+g8e685d198` | 161 tests / 5 skipped |
| GLM | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | `0.1.dev20051+g487ecf187` | 161 tests / 5 skipped |

真实故障闭环在 GLM 双机 TP=2、PP=1、`VLLM_SERVER_DEV_MODE=0` 上完成。scheduler
deployment 为 `2d30c245b573894719dd99411639a73b383dc90fdc5579b3023019880c88e765`；
rank 0/1 deployment 分别为 `533e0903c9b42dbe0610b27809a7eb5b0a45e87e4f58c9da8254eeb80cb3514c`
和 `b041dbe51327eaf6d6b10691acc21d02c1271701d92826bd1627a0b1694535dc`，rank identity
分别为 `51e49a2b9cf49f1a1fee5fa46bfd48316a4c8bc62442a11592c169290cd69f36` 与
`bd84e0e81728f31effcc46b70817eba94ec1b23af24c3997f4c9802de04dfb9a`。topology 为
`748cd94a00db81944cadb80b8dd27eff5e51bd76036299762161e1570f225dbe`，worker layout 为
`d8311f3f9860494ea0adb2ed7f9f6e0fbf665a2ac655732f43ab708bbe280c37`。

选择的已认证 entry 是
`66f35feb70438492ee3ca9b78e26665ebb89aa5967dbc6d23a990c56e297009e`，span 3,584。
只修改 rank 0 中一个 2,351,104-byte immutable object
`objects/dc/dc1fd1aff2f53168231fb0cf0d619f4639e48645d6368091de36d37ebad9a04e.spool`：
先核对其逻辑 SHA-256 为 `dc1fd1aff2f53168231fb0cf0d619f4639e48645d6368091de36d37ebad9a04e`，
创建并 fsync 窄范围备份，再把首字节从 242 改为 243；损坏后 digest 为
`e12cec50…`。没有删除、重置或清空任何 persistent cache root。

scheduler 已命中并 admission 3,584 tokens 后，rank 1 完成 restore，rank 0 报
`SPOOLCACHE_POST_ADMISSION_RESTORE_FAILED` 并触发 worker 内部固定 exit 70。严格 streaming
client 约 3 秒内因响应不完整非零退出，没有伪成功或永久悬挂。容器最外层 vLLM 进程随后
执行自身 graceful shutdown，所以 Docker 记录的最终父进程码为 0；supervisor 同时从 fatal/API
丢失与 rank epoch 变化检测到故障，先把 readiness 降为 false（API false、0/2 ranks、phase
`restarting`），确认完整组停止，再按 worker→API owner 顺序启动原容器。不能把 Docker
父进程码 0 误写成“未执行 exit 70”；CPU process-boundary contract 和 worker 日志直接证明
固定 fail-stop 路径，supervisor 则不依赖单一外层退出码才能闭环。

旧组完全停止后，备份以 atomic replace 恢复，恢复 digest 再次等于 descriptor；两个 rank
分别运行身份绑定的完整 verifier，最终均为 77 objects / 72 layers / 7 groups，页覆盖
`1,1,1,1,1,1,32`，logical/stored bytes 为 484,578,304 / 484,589,568，并返回
`all-payloads-authenticated`。认证完成后只删除了已冗余的临时备份及其空临时目录，persistent
object 与其他 cache 内容保持不变。

GLM/DFlash 的普通自由生成在数次 cold bypass 中产生不同文本，因此最初的单次 completion
hash 被明确拒绝为内容 oracle。新的 consumer 保持前 3,584 个已缓存 tokens 不变，在其后追加
约束 continuation：`VERIFICATION SEQUENCE ... A=`。四次使用互不相同 salt 的独立 cold bypass
均为 cached=0，且精确输出 `3141592653589`，完整输出 SHA-256 都为
`eb0fe2753ed6596aa3ff1fcf7b7cbf9426a8a9a7b2fd16aac290cfcbed25a680`。再次完整监督重启以排除
GPU APC 后，persistent consumer 的 prompt 为 3,624 tokens、external cached 为 3,584、
completion 为 8；scheduler 命中上述同一 entry，rank 0/1 均恢复同一 3,584 span，精确输出和
SHA-256 与四个 cold controls 一致。这一强 oracle，而不是 HTTP 200 或单次自由输出，作为
恢复内容正确的验收依据。

故障后的 journal 导出 `spoolcache_post_admission_failure_total{phase="payload"}=1`。首轮恢复
采样观察到 lookup hit/ready-entry=1、hit tokens=3,584、restore bytes=969,156,608、两 rank
restore histogram count=2/sum≈0.688339 秒、required/ready ranks=2/2、三项 readiness=1、
pinned pool=268,435,456 bytes、disk=4,355,748,006 bytes、quarantine 所有 reason=0。当时
machine-readable health 为 `ready=true`、API live、2/2 ranks、supervisor phase `ready`；两个
容器都确认 `VLLM_SERVER_DEV_MODE=0`。

实现提交为 SpoolCache `4a06e99`、DeepSeek `a237706`、Qwen `3af1c92`、GLM `f92cf33`。
提交后 review 的 CR-022 至 CR-027 由 `35abc08`、`35804f3`、`b00792b`、`5340ddf`
关闭：inventory/report/pending state 全链路有界且 gap fail closed；stop 不再信任 state file，
而以 exact argv 加 advisory lock 证明单 owner；GLM 启动失败会清理完整组；Prometheus label
parser 拒绝尾随逗号；probe/port/state 写入有界；三个部署都以私有临时目录加原子 symlink
同步源码，不会把已删除 module 留在 worker，也不会先删除 live bind mount。修复复审没有
未关闭 P1/P2。三套真实 vLLM preflight 均返回 `automatic-contract`、19 capabilities，build
SHA-256 前缀分别为 `e24d925d2e31`、`fa1e7da550af`、`89945da00615`。

最终 exact-source 验收把 worker stable source 从旧目录原子切换到 `35abc08` 对应 snapshot。
本地、head 容器和 worker 容器对实际可导入的 `src/`（排除 bytecode）24 files / 332,528
bytes 都得到内容摘要 `7032f16388c7dbb3c15f63a60824afdd695065dbcf3efa0535c1fd4ae4cf778f`。
停止 worker 后，supervisor 先记录 `restarting`、API false、0/2 live ranks，并确认 head/worker
均退出；随后两容器于 05:17:31 UTC 重新启动，05:26:06 才在 API、2/2 ranks、identity、
inventory quorum、fatal-clear 全部成立后进入 `ready`，restart count=6。只在新容器内容摘要
通过后删除旧的非活动源码 snapshot；没有触碰 persistent cache。

在这套最终字节上再次请求同一个 3,624-token strong-oracle consumer：external cached=3,584、
completion=8，scheduler 命中 `66f35feb7043…`，rank 0/1 都恢复同一 3,584-token entry，输出
仍精确为 `3141592653589`，SHA-256 仍为 `eb0fe2753ed6596aa3ff1fcf7b7cbf9426a8a9a7b2fd16aac290cfcbed25a680`。
两个 rank 的离线 verifier 再次分别认证 77 objects / 72 layers / 7 groups、页覆盖
`1,1,1,1,1,1,32`，logical/stored bytes 都为 484,578,304 / 484,589,568，并返回
`all-payloads-authenticated`。最终 metrics 为 hit=1、hit tokens=3,584、restore bytes=
969,156,608、restore count=2/sum≈2.583953 秒、payload fatal journal=1、quorum entries=4、
required/ready ranks=2/2、三项 readiness=1、pinned pool=268,435,456 bytes、delayed store=0、
disk=4,355,748,006 bytes、quarantine=0；API health=200，两端 `VLLM_SERVER_DEV_MODE=0`。
coordination schema 始终保持 `spoolcache-coordination/v1`，核心仍无任何模型、架构或模态特例。

## G3b 持久化存储自愈与容量故障收据（2026-09-06 UTC）

实现提交 `a633f75` 增加 rank-local 可恢复 deep scrub、定点请求/状态 CLI、
损坏对象与共享引用 manifest 隔离、worker offer 撤销、临时文件/孤儿对象安全
清理、managed/quarantine 容量核算和高低水位 GC。scrub 的 64 MiB/s、64 MiB step、
64 items/step、60 秒首轮延迟和 6 小时周期都是固定内部上界，没有增加模型
profile、白名单、模型名分支或环境变量。协调格式继续是
`spoolcache-coordination/v1`。首轮 review 修复提交 `7a2a35e`，第二轮 review 修复提交
`e7920a4`；后者增加 rank-local 持久单调 generation epoch、rename 前 durable withdrawal
marker、manifest publication 与 reporter admission 同锁线性化，以及 SQLite 峰值采样。
第三轮修复提交 `800ae13` 再加入 legacy epoch 高位迁移域、generation 初始化哨兵、unknown-lower
scheduler 冲突语义、rank-root lifetime inventory owner、standalone maintenance 跨实例 marker
消费、corrupt content-address object 的 evidence-hardlink + atomic replace，以及独立进程 VmRSS
连续采样。随后的 `e102c0d` 在释放 repaired-object tombstone 前补齐 shard directory fsync，
`7510fda` 让新 withdrawal delta 优先于 retained history 发出，`761aaea` 则确保合法 rank 的
malformed generation identity 先撤销旧 image 再报错。第四轮复审修复提交 `26cb7e7` 在共享
引用遍历前写入 digest-level object fence，让 lookup/startup scan/reporter reconciliation 均
fail closed；引用撤销改为流式计数，并让 owner 以固定页和轮转游标收敛离线 absent markers。
第五、六轮随后由 `3e15600`、`4af10b2`、`5f66d79`、`905c48a`、`cdfa84c`、
`c6f61fb` 和 `bb81636` 收口 marker provenance、namespace/shard durability receipt、失败
publication 临时文件上界、健康 offer 的 filter-before-heap、primary error 保留、hostile JSON
解码，以及关闭 `scandir` 后再执行固定批 quarantine。

CPU 故障注入覆盖了每个 object/manifest write、`fsync`、link 和目录 `fsync`
边界的 ENOSPC 与 `os._exit`，以及 1/2/3 个对象后的进程丢失、幂等重试、异内容
冲突、并发 maintenance、scrub 实进程中断/恢复、损坏 SQLite/请求、全部身份负向、
共享损坏、symlink/未识别路径、限速/取消、rescan epoch 竞态、quorum 退出、rank reopen
和 40 轮容量回归。最终修复提交的宿主为 252 tests / 10 skipped / OK，pytest 为
247 passed / 10 skipped，skip 仅为需要 CUDA
或真实 vLLM 的环境项。三个目标运行时镜像使用同一源码套件：

| 目标镜像 | vLLM | 结果 |
|---|---|---:|
| `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | `0.25.2.dev0+g752a3a504.d20260714` | 252 tests / 6 skipped / OK |
| `vllm/vllm-openai:qwen38-flash-next` | `0.1.dev20073+g8e685d198` | 252 tests / 5 skipped / OK |
| `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | `0.1.dev20051+g487ecf187` | 252 tests / 5 skipped / OK |

停止服务后，三个镜像还分别在真实 CUDA 上通过 5/5 mover 测试：全 HMA group
往返、损坏在任何 GPU page 放置前拒绝、晚期损坏排空已提交 CUDA 工作、manager
split rows/physical tail 与 rank geometry identity。构建的 wheel 包含
`maintenance.py` 和 `spoolcache-maintenance` entry point。三套实际 vLLM preflight 都返回
`automatic-contract` 和 19 capabilities；installed-package content SHA-256 分别为
`e24d925d2e31333b…`、`fa1e7da550af91ee…`、`89945da006159b8a…`。
最终 wheel `spoolcache-0.1.0a0-py3-none-any.whl` 的 SHA-256 为
`7b3bd33634481554453ca7254703b75e8311eee5611dfdefcf44654c707c35b1`，并已核对
`store.py`、`maintenance.py`、`vllm/connector.py` 和 console entry point 均在包内。

1,000 轮有界 soak 收据为：

```json
{"baseline_process_peak_rss_kib":176132,"baseline_rss_kib":24880,"boundary_max_rss_kib":27032,"capacity_runs":20,"elapsed_seconds":343.55418121488765,"final_cache_bytes":481111,"final_managed_bytes":513911,"final_manifests":158,"final_objects":158,"final_quarantine_bytes":0,"final_state_database_bytes":32768,"final_temporary_files":0,"iterations":1000,"low_watermark_bytes":393216,"max_cache_bytes":524288,"max_cache_observed_bytes":527095,"max_rss_kib":28720,"max_state_database_bytes":86016,"orphans_created":143,"payload_bytes":2048,"payload_pattern":"shake256-indexed-v1","peak_tracemalloc_bytes":2509766,"process_peak_rss_growth_kib":0,"process_peak_rss_kib":176132,"rank_reopens":10,"result":"passed","rss_growth_bound_kib":131072,"rss_growth_kib":3840,"rss_sample_count":143448,"rss_sampling":"external-process-vmrss-poll-v1","sampled_rss_growth_kib":3840,"schema":"spoolcache-storage-soak/v1","scrub_cycles":21,"scrub_steps":298,"state_database_bound_bytes":2097152,"state_database_sampling":"cycle-start-and-every-step-v1","temporary_files_created":91,"tracemalloc_bound_bytes":16777216,"withdrawn_entries":842}
```

这次收据来自独立 Python 进程，并在每次 `start_cycle()` 建立 work queue 后及每个 step
后采样全部 SQLite artifacts；另一个采样进程每 2 ms 读取 workload 的 current VmRSS，共取得
143,448 个样本。实际 database 峰值
86,016 bytes 明显高于 cycle 完成并 FULL auto-vacuum 后的 32,768 bytes；旧收据只看到
后者，不能证明运行期上界，已由本收据替换。外部 RSS 峰值增长 3,840 KiB，tracemalloc
峰值 2,509,766 bytes，分别低于 131,072 KiB 与 16,777,216-byte 内部资格上界。进程历史
`ru_maxrss` 在开始前已为 176,132 KiB 且本轮增量为 0，明确只保留作诊断；对应 native mmap
负测证明低于该历史高位的 24 MiB resident burst 仍被外部 sampler 捕获。

在 GLM-5.3-Flash-EXL3 TP=2、PP=1、`VLLM_SERVER_DEV_MODE=0` 生产路径上，旧 store
首轮自动 scrub 在两个 rank 分别认证 4 manifests / 308 objects /
2,182,877,184 payload bytes，无孤儿、临时文件或失败。新样本使用精确 3,584-token
边界和 25-token 验证后缀。验证后缀没有生成文案中请求的数字，而是稳定输出
`1 B=2 C=3 D`；因此没有伪造预期值，而是用四个互不相同 salt 的 cold bypass
建立实测 oracle。四次均为 prompt=3,609、cached=0、completion=8，完整输出 SHA-256
都是 `fc24f73bf1f741e80639f764c0abd7db21a9862373af7dafd36228898b182e18`。带后缀的 producer
被 HMA 正确以 `unsafe_boundary` 跳过 Store；改为单独提交精确边界后，两 rank 存入同一
entry `83ac019aa857a32c880bda4cbab1052791607a61fe1b3fd9f9f8f8c5fd481107`。

第一次完整 TP 重启清空 APC 后，consumer 为 cached=3,584、TTFT=0.634106 秒、
elapsed=0.856041 秒；scheduler 与 rank 0/1 均记录同一 entry/span，external restore 为
969,156,608 bytes，两 rank 耗时合计约 0.713716 秒。输出与四个 cold controls 完全一致。
两 rank 的基线 verifier 均返回 77 objects / 72 layers / 7 groups、页覆盖
`1,1,1,1,1,1,32`、logical/stored bytes=484,578,304/484,589,568 和
`all-payloads-authenticated`。

故障注入只选择 rank 0 中 reference count=1 的
`objects/da/da75138902a718f86e3664b0bc466e11af69cbd55abc9b003f01264643dc32d5.spool`，
长度为 2,351,104 bytes。它的 logical/full SHA-256 在备份前均为 descriptor 中的
`da751389…32d5`；窄范围 object+manifest 备份已 `fsync` 并独立复核。在 maintenance
lock 内把首字节从 242 变为 243 并 `fsync` 后，hash 变为 `f832defb…510a4`，长度
不变。定点 CLI 约 1 秒后持久记录 `last_request_status=quarantined`，live object/manifest
都消失，quarantine 保留 2,351,104-byte object 和 26,132-byte manifest。

纯空闲 90 秒内 vLLM 原生 worker stats channel 没有运输新 report，因此本地 scrub
收据不被误写为 scheduler 已收敛。一个与目标无关的 64-token
`spoolcache_bypass=true` 请求作为显式 report barrier 后，quorum entries 从 5 降为 4，
`payload_checksum=1`、scrub authenticated bytes=2,351,104、scrub quarantined objects=1、
quarantine bytes=2,377,236，scrub failures=0。在 quorum 下降前没有回放受影响请求。

第二次完整 TP 重启后，rank 0/1 inventory 分别为 4/5，quorum=4。同一 consumer
为 cached=0、TTFT=4.104378 秒、elapsed=4.317982 秒，`rank_quorum` miss 从 0 变为 1，
external hit/restore 仍为 0，两 rank 无 restore 日志，既有 G3a payload fatal counter 也没有增加。
冷重算输出仍与 oracle 一致。这是损坏数据在 admission 前被拒绝的在线证明。

随后重新提交同一安全边界：rank 0 重建缺失 manifest，rank 1 正确跳过已有副本，
report barrier 后 quorum 回到 5，两 rank 再次完整认证。同进程 consumer 虽报
cached=3,584，但 SpoolCache hit/restore 均为 0，因此该 APC 结果被明确拒绝为持久
证据。第三次完整 TP 重启后，最终 consumer 为 cached=3,584、TTFT=0.660490 秒、
elapsed=0.880626 秒；SpoolCache hit=1、hit tokens=3,584、restore bytes=969,156,608、
restore count=2/sum=0.645632 秒，scheduler/rank 0/rank 1 均为 entry `83ac019aa857…` / span 3,584，
完整输出再次匹配 oracle。

最后两 rank verifier 仍分别返回 77 objects / 72 layers / 7 groups、
484,578,304/484,589,568 bytes 和 `all-payloads-authenticated`。只在这些检查完成后才再次
认证并删除临时备份；quarantine 证据未删除。当前 API 和 2/2 ranks 均 live，
supervisor phase=`ready`，两容器 `VLLM_SERVER_DEV_MODE=0`。宿主、head 容器和 worker
容器按 launcher 排除 `.egg-info`/bytecode 的快照规则核对 `src/`，均为 25 files /
502,590 bytes / 内容摘要
`95b1f599c4ab88f84b28ddd6f0795efcb803c49edee4ccdadd7c7cfbec14341f`。

在最终 correctness 提交 `bb81636` 上重新收口全部非破坏性资格：宿主 pytest 为
247 passed / 10 skipped，unittest 为 252 tests / 10 skipped / OK；DeepSeek、Qwen、GLM
目标镜像各运行 252 tests，分别 skipped 6/5/5，全部 OK，且各自真实 CUDA mover 都为 5/5。
三套实际 vLLM preflight 仍为 `automatic-contract`、19 capabilities；build SHA-256
分别为 `e24d925d2e31333b9af6e383becf6f8e4ad5d4712e6b3bbe71b396fe40eaa191`、
`fa1e7da550af91ee9a16b3e3c777bae22fb56528ad84f2ea1f9251d9031137bd` 和
`89945da006159b8a278fb2ae191ee80fdf4b34f81ba474ea6060a209ef4a50e5`。
GLM 重启后的最终 `spoolcache-readiness/v1` 为 ready=true、API live、2/2 ranks、
supervisor ready、三项 SpoolCache readiness 均为 1；本次容器时间窗无
SpoolCache/CUDA/NCCL/traceback error。额外 64-token 严格流式 bypass control 为
cached=0、completion=8，完整输出 SHA-256
`9b2fab355d67353084be1a5efb007737c4d68b773b5f880c7a46c4075dbb8765`。

## 2026-09-06：roadmap 重写与过渡设计审计

本轮没有改变生产 cache 语义，也没有运行新的服务资格。曾在未提交工作树中试作的完整
checkpoint 文件 inventory、逐文件 SHA-256、rank receipt、symlink-free 模型副本和 launcher
模型路径改写已全部撤回；SpoolCache、DeepSeek、Qwen、GLM 分别回到 `38b2423`、`35804f3`、
`b00792b`、`5340ddf` 的干净代码基线后才重写规划。该实验没有 commit，不能作为已实现能力
或性能收据引用。

提交历史审计另外确认了三项未进入生产调用链的 runtime 脚手架：`layerwise.py`、
`publication.py` 和只被 benchmark 使用的 `StoreEconomics`。前两项始于首个功能提交
`c5c1dab`，生产 connector 从未导入；它们的单测只证明规划函数自身，不证明 layerwise restore
或增量 publication 已启用。G3c 已删除/移出这些内容，而不是沿用早期命名和 profile 假设。

当时审计列出的生产路径包括开发态 source snapshot/`PYTHONPATH` bind mount、PP=1 guard、
同步完整 Store/Restore、分离 pinned/aligned pools、一次性 deep-scrub namespace snapshot、
无界 scrub join、legacy generation sentry，以及 launcher 中的 dev-mode qualification 开关。
后续 deep-scrub snapshot/join 已被有界实现替换；当前 roadmap 又明确把同步 Store/Restore 和
分离 pools 定为 0.1 正式实现，撤销原 G5，只有 benchmark 证明需要时才重新立项。现存 source
同步、PP guard、dev-mode launcher glue 和 pre-0.1 migration 审计分别由 G4/G6 收口。
post-admission TP fail-stop、allocator gate 和 idle stats barrier 是上游契约不足时的条件性保护，
不作为 0.1 必须“优化掉”的功能。

新的信任边界参考 LMCache 基础 KV 路径：由 runtime 的模型标识和结构化 KV 事实隔离缓存，
SpoolCache 只认证自己的 manifest/payload，不复制或逐字节认证整个模型仓库。同一 locator/
revision 下权重被原地替换不在本项目保证内，应由不可变 artifact 或 namespace 轮换解决。
详细对账与下一目标验收条件以 `TODO_GOALS.md` 为准。

## 2026-09-06：G3c runtime 模型 namespace 与脚手架清理

G3c 将误导性的 launcher checkpoint pin 替换成 connector 内的通用 runtime 派生：优先使用
vLLM 公共 `ModelConfig.model_weights` 的非空原始 locator，否则使用 `model`，并同时绑定
`revision`。scheduler 与 worker 可能持有不同的 materialized `model` 路径，但只要
`model_weights`/`revision` 相同，就得到同一 `spoolcache-model-namespace/v1` 摘要；served
alias 和模型配置类名不参与。不同 locator/revision、operator namespace、runtime/build、
topology、dtype 或实际 layout 仍由现有通用 identity 层隔离。

公开 connector 配置和 DeepSeek/Qwen/GLM launcher 已不再接受、生成或传递
`SPOOLCACHE_CHECKPOINT_SHA256`。deployment identity 因字段形状变化使用
`spoolcache-deployment/v2`，而 scheduler/worker 协调协议保持
`spoolcache-coordination/v1`。旧 v1 digest 目录不会命中，也不会被启动路径删除；这是
非破坏性 clean-miss 迁移，是否归档由运维在回滚窗口后决定。SpoolCache 继续认证自己写出的
manifest/payload，但不扫描、复制或逐文件认证模型仓库。

未进入生产调用链的 `src/spoolcache/layerwise.py`、`publication.py` 及其专属假能力测试已
删除；真实 vLLM layer-hook 顺序证据移到 `benchmarks/probe_vllm_layerwise_contract.py`，明确
不代表产品支持。`StoreEconomics` 也从 runtime API 移入
`benchmarks/analyze_store_economics.py`。

提交前回归使用同一工作树：宿主 pytest 为 244 passed / 9 skipped；DeepSeek、Qwen、GLM
三个目标镜像各运行 248 tests，分别 skipped 6/5/5，均为 OK。三个镜像的实际 vLLM
preflight 都返回 `automatic-contract` 和 19 capabilities，installed-package content SHA-256
分别为 `e24d925d2e31333b9af6e383becf6f8e4ad5d4712e6b3bbe71b396fe40eaa191`、
`fa1e7da550af91ee9a16b3e3c777bae22fb56528ad84f2ea1f9251d9031137bd` 与
`89945da006159b8a278fb2ae191ee80fdf4b34f81ba474ea6060a209ef4a50e5`；服务停机后各自
真实 CUDA mover 均为 5/5。宿主 `compileall`、三个 launcher shell 语法和 launcher
contract tests 也全部通过。

真实跨重启资格在 GLM-5.3-Flash-EXL3、TP=2、PP=1、`VLLM_SERVER_DEV_MODE=0` 上完成。
首次用上游镜像启动后，launcher 按其正常 recipe 检测重建了本地镜像；runtime/build identity
变化使上一轮 v2 目录自然 clean miss，没有删除或重置任何 cache。随后固定该镜像和源码，使用
`SKIP_PULL=1 SKIP_BUILD=1` 完整停止并重启。最终 live runtime content SHA-256 在两个节点都为
`74b8b658f0e8cef81712e2eebe8b867bf88d6d49a43ffae6614ed58558970eba`；scheduler、rank 0、
rank 1 的 model namespace 均为
`43158592c9fc9d8d369035303873fd13460f4c3cd85eb95834ff2f480b766234`。scheduler deployment
为 `8f5aeea8ca48cb4d8cadf4195e6d3018a02fdf56220640e95f583107b92fac20`；rank 0/1 deployment
分别为 `0c4d7764f279abb2730e7d680871e9011a20f9d0c35fd78d3e9b04dcaac5e6ec` 与
`614a21c29657ebedb87644593c4d677dd69a854dca30689f41460c01ea5c3e9b`，rank identity 分别为
`69418af49271803d932810a32bf991b8830280bd587729dcf6dce8fa03e2f35d` 与
`4ad427c4c770080b7609347be118744a04978d39b669e06aa68e3c03cca89a12`。topology 为
`748cd94a00db81944cadb80b8dd27eff5e51bd76036299762161e1570f225dbe`，worker HMA layout 为
`d8311f3f9860494ea0adb2ed7f9f6e0fbf665a2ac655732f43ab708bbe280c37`。

四个不同 salt 的 3,609-token cold bypass controls 都返回 cached=0、completion=8，完整输出
SHA-256 都是 `95b2dd251d37978ee11420d4be9d96b1c3f54b6ca24b8cbff7763a321b65cac7`。
精确 3,584-token producer 在两个 rank 写出同一 entry
`da044b970903544e59a8314ab5bb8540b9534dbb6d0937cb90a228fd47166526`。停止两个 rank 并以
相同镜像/源码完整启动后，consumer 返回 cached=3,584、completion=8、TTFT=0.799483 秒、
elapsed=0.900284 秒，输出 hash 与四个独立 controls 完全一致。scheduler 与 rank 0/1 日志均为
同一 entry/span；metrics 为 hit tokens=3,584、restore bytes=969,156,608、restore count=2、
restore sum=0.659478 秒、quorum entries=1、quarantine bytes=0，所有 quarantine reason 均为 0。

重启前后各自运行身份绑定 verifier，两 rank 都返回 77 objects / 72 layers / 7 groups、页覆盖
`1,1,1,1,1,1,32`、logical/stored bytes=484,578,304/484,589,568 和
`all-payloads-authenticated`。宿主、head 容器、worker 容器对实际可导入 `src/` 的摘要也完全
一致：23 files / 493,673 bytes /
`4e1c41b331e3eb8c6cf4233f5d3fa766e87565d8e368d776df5b3f23f62c7bf0`。最终 readiness 为 API
live、2/2 ranks、supervisor ready、rank identity/inventory quorum/fatal-clear 全部成立；当前
容器时间窗没有 SpoolCache/CUDA/NCCL/traceback 错误。全过程未删除 persistent cache，协调
协议保持 `spoolcache-coordination/v1`，核心仍无模型、架构或模态特例。实现提交为
SpoolCache `51ec9a6`；三个 launcher 提交分别为 DeepSeek `6c3f680`、Qwen `3cb6744`、
GLM `2fb1046`。post-commit review 又逐项核对三个真实 `ModelConfig` 的公共
`model`/`model_weights`/`revision` 字段、served-alias 排除、角色收敛、迁移边界和 launcher
失败路径，未发现未关闭 P1/P2。移动 locator 或同一 locator/revision 下原地替换 artifact
仍是已声明的运维信任边界；它不通过模型特例、仓库扫描或新增 connector 选项解决。

## 2026-09-06：G3d 增量 namespace 与有界关闭（实现阶段）

deep scrub 的 cycle 初始化不再在 maintenance lock 内同步扫描完整目录。新的
`snapshot_manifests`、`snapshot_objects`、`snapshot_tmp` phase 每个 step 最多读取固定 64 个
原始目录项，把安全路径用幂等 insert 提交到原有 SQLite work table。目录 iterator 不进入持久
schema；进程重启或另一个 scrub 实例推进 cycle 后从当前 namespace 开头重扫，已提交项去重。
因此没有引入不稳定的 filesystem offset、模型配置或新环境变量。orphan 删除仍必须等完整
manifest pass，并在 unlink 前锁内重扫 live reference。

scheduled scrub 会在 namespace 条目边界检查 cancellation，`close()` 最多 join 固定 5 秒并返回
`spoolcache-scrub-shutdown/v1`。timeout 在主路径输出结构化日志收据，daemon finalizer 再 best-effort 持久累计
`spoolcache_scrub_shutdown_failures_total`；计数器 fsync 不参与主关闭上界。connector 将仍被 reader 使用的
mover/store 交给 daemon finalizer，线程真正结束后才关闭，主进程退出不再等待无界 join。宿主实现阶段全量测试为
252 passed / 9 skipped，新增测试覆盖固定 item budget、取消回滚、进程重开、多个 scrubber cycle
切换、新增/删除交错、timeout 收据和 connector 延迟释放。

默认百万 namespace 资格真实创建 1,000,000 个 canonical manifest 路径（共 1,000,257
namespace inodes）。为避免创建/计数百万路径造成的 Python allocator 预热掩盖被测增长，文件
创建与 inode 统计在独立 spawn 子进程完成，maintenance 父进程才建立 VmRSS baseline。最终
结果为：population 14.407003 秒、`start_cycle` 0.005529 秒、固定 64 项 step 0.010433 秒、
scheduler close 0.003493 秒、state SQLite 45,056 bytes、外部 VmRSS 采样增长 984 KiB；全部
低于固定 5/5/6 秒与 128 MiB 上界。完整进程树 wall time 25.89 秒，其中包含临时百万文件清理；
机器可读 schema 为 `spoolcache-namespace-scaling/v1`。原始收据归档在
`docs/receipts/2026-09-06-g3d-namespace-million.json`。

第一次 1,000 轮扩大 storage soak 虽通过已有上界（20 次 capacity、21 次 scrub、10 次重开，
外部 VmRSS 增长 2,112 KiB，SQLite 峰值 86,016 bytes），但新 state/inode 采样揭示 benchmark
没有模拟 production inventory owner 的 marker acknowledgement：842 个已撤销 entry marker
使 final state tree 达到 3,579,904 bytes。该收据只作为诊断，不作为 G3d 通过证据。benchmark
随后改为像 connector 一样每轮消费固定 marker page，并要求最终 state/quarantine/inode 峰值；
修订后的 1,000 轮独立进程已通过：20 次 capacity、21 次 scrub、529 个 bounded step、10 次
rank-root 重开、842 次 withdrawal 和 20 个 inventory-marker page；final manifest/object 均为
158，全部 final offer 再做完整 payload authentication。SQLite peak/final 为 86,016/32,768
bytes，state tree peak/final 为 98,304/45,056 bytes，quarantine tree peak/final 均为 4,096
bytes，managed inode peak/final 为 864/834，temporary file 最终为 0。外部进程取得 152,797 个
VmRSS 样本，增长 2,364 KiB；tracemalloc peak 2,136,928 bytes；wall time 364.67 秒。所有数值
均低于 receipt 内固定上界，该收据可作为 G3d 扩大 storage soak 证据。
原始收据归档在 `docs/receipts/2026-09-06-g3d-storage-soak-1000.json`。

## 2026-09-07：G3d 最大上下文现场问题与 recurrent 运行页修复

GLM-5.3-Flash-EXL3 的 992,769-token cold control 明确携带
`SpoolCacheMetadata(loads=[], stores=[])`，因此没有进入 SpoolCache 数据路径；请求推进到
144,832 tokens 后停止。两台 Spark 硬重启后读取上一 boot 的 kernel journal，Spark-1/2
均在同一窗口连续报告 NVIDIA `NV_ERR_NO_MEMORY`，随后出现 `jbd2` 和系统服务 hung task；
Spark-1 进一步触发 global OOM。事故因此归类为最大上下文 prefill 引发的双机统一/系统
内存耗尽，而不是 cache hit/store 故障。该 99 万 token 请求禁止直接重放；后续最大上下文
资格必须有界递增，并在每步检查 OS reserve、SSH、API 和两 rank readiness 后才能继续。
机器可读证据位于
`docs/receipts/2026-09-07-g3d-glm-text-max-cold-bypass-failure.json`。

Qwen3.8-Flash-Next 随后的 160,000-token entry 在 208,000-token consumer 上暴露了独立 P1：
vLLM V2 runner 以 8,192-token chunk 调度时，align-mode recurrent block table 会在 external
hit 边界与本轮 running-state page 之间填入 null blocks。旧 v3 选择器固定读取
`boundary_pages`，因此在 scheduler `update_state_after_alloc()` fail-stop；两台宿主始终可达，
TP supervisor 正确重启整组。vLLM 的公开 cache spec 同时声明
`num_speculative_blocks`，其 manager 契约把当前运行状态保存在 block table 尾部、位于这些
speculative pages 之前。修复据此把选择规则改为运行态尾部，并把 tail 数量纳入 logical/
physical layout identity；内部 layout schema 升为 `spoolcache-hma-layout/v4`，coordination
仍为 `spoolcache-coordination/v1`。实现不读取模型、架构或模态名称，也没有新增选项或环境变量。

实现阶段证据：宿主完整套件 254 passed / 10 skipped；Qwen 目标镜像完整套件 264 passed、
273 subtests，包含真实 CUDA mover 与真实 vLLM cache semantic tests；新增 tail identity/
selection 聚焦回归在目标镜像为 32 passed / 11 subtests。live 资格在
`GPU_MEMORY_UTILIZATION=0.75` 下使用固定双机安全门完成：208,000-token cold bypass 的
TTFT 为 84.575 秒；160,000-token producer 在两 rank 各持久化并逐字节认证
2,324,992,000 bytes / 102 objects，随后完整停止并重启 TP 组。新 generation 两 rank 各从
磁盘扫描到该 entry，208,000-token consumer 命中 160,000 tokens，两 rank 共恢复
4,649,984,000 bytes，TTFT 为 23.843 秒，输出 SHA-256 与 cold bypass 完全相同。最终
quorum=1、LayoutError=0、所有 post-admission failure=0、fatal clear，故该 P1 已关闭。
机器可读证据为 `2026-09-07-g3d-qwen-text-208k-bypass-v4.json`、
`2026-09-07-g3d-qwen-text-160k-v4-producer.json`、两份 payload verify receipt、
`2026-09-07-g3d-qwen-text-208k-v4-restore.json` 和
`2026-09-07-g3d-qwen-hma-v4-cross-restart-summary.json`。

### Qwen 260.8K 最大请求矩阵与主机安全门

Qwen runtime 公开的 `max_model_len` 为 262,144；资格请求选用 260,800 tokens，为输出和
chat template 留出余量。先用 232,000、再用 260,800-token text cold bypass 递增确认容量，
TTFT 分别为 95.065 秒和 108.863 秒，cached 均为 0。所有长请求由独立 shell guard 每 2 秒
同时读取 Spark-1/2 的 `MemAvailable`、`SwapFree` 与 SSH 可达性；固定提前终止阈值为
3 GiB available memory 和 4 GiB free swap。Qwen 本轮没有触发 guard。

使用已经跨完整 TP 重启并认证的 160,000-token text entry 运行 260,800-token consumer，
两 rank 均恢复同一 `6f6297f2f503…`，cached=160,000，TTFT 从同请求 bypass 的 109.303 秒
降至 49.117 秒。完整输出 SHA-256 都是
`9b2fab355d67353084be1a5efb007737c4d68b773b5f880c7a46c4075dbb8765`。

图片和视频使用固定 COCO/PyTorchVideo bytes，在 producer 与最大 consumer 之间逐 token
证明精确前缀关系。producer span 均为 160,000，consumer 总 prompt 均为 260,800；每个
consumer 都先清除 GPU prefix、encoder、multimodal cache 后运行 exact bypass，再重复清除
本地 cache 并运行 SpoolCache restore：

| 输入 | entry | bypass cached / wall | restore cached / wall | bypass/restore oracle |
|---|---|---:|---:|---|
| text | `6f6297f2f503…` | 0 / 109.336 s | 160,000 / 49.155 s | 完整输出 SHA-256 相同 |
| COCO cats image | `9bf713021181…` | 0 / 109.479 s | 160,000 / 49.613 s | `CATS`，完整输出 SHA-256 相同 |
| PyTorchVideo archery | `4fe4472d7ff6…` | 0 / 109.387 s | 160,000 / 50.094 s | `ARCHERY`，完整输出 SHA-256 相同 |

图片/视频 entry 在两个 rank 上各自是 102 objects / 2,324,992,000 logical bytes，绑定当前
deployment、physical rank、topology、profile 和 `spoolcache-hma-layout/v4`，四份 verifier
全部返回 `all-payloads-authenticated`。scheduler 与两 worker 对每次 restore 的 entry/span
一致；最终 quorum=3、累计 restore=18,599,936,000 bytes，全部 post-admission failure=0。

随后完整停止并重启两个 TP rank；新 generation 的 rank0/rank1 各扫描到 3 个 entry，使用一条
无关的 3,200-token bypass 作为首个 stats report barrier 后 scheduler quorum=3。重启后只重放
160,055-token image/video consumer，分别在两 rank 恢复同一 160,000-token entry 并返回
`CATS`/`ARCHERY`；本 generation 共恢复 9,299,968,000 bytes / 4 rank operations。再次运行四份
身份绑定 verifier 仍全部认证。汇总收据为
`2026-09-07-g3d-qwen-max-multimodal-cross-restart-summary.json`。

另对精确 260,800-token 图片 producer 做了两次尝试，第二次在显式清空全部本地 cache 后
仍使 `spoolcache_store_skipped_total{reason="unsafe_boundary"}` 从 1 增至 2；该请求前后的
store count 增量为 0、quorum 不变，没有发布 manifest。这是 recurrent/HMA 正确性门的预期
fail-closed：当前 scheduler 状态越过目标时，SpoolCache 不猜测旧 running state。资格因此采用 160K 已认证
前缀服务最大 consumer，而不是加入 Qwen 特例或放宽通用边界。

本轮最低 `MemAvailable` 出现在视频 exact bypass：Spark-1/2 分别为 6,389,060 KiB 和
20,180,928 KiB，最低 `SwapFree` 分别为 13,557,856 KiB 和 14,012,204 KiB，仍高于 guard。
清除本地 GPU/encoder/multimodal cache 后 Spark-1 available memory 从约 6.3M KiB 回升到约
11.9M KiB，服务最终 API live、2/2 ranks、supervisor ready、identity/quorum/fatal-clear 全部
成立。当时 GLM/DeepSeek 的安全最大长度仍待完成；下节已收口 DeepSeek。后续 roadmap
把剩余 GLM 安全长度归入非阻塞 Q1，并按用户要求把不少于 24 小时 RW/RO 长稳从验收目标
删除；本段保留当时的实测状态，不再定义当前 Goal。

### DeepSeek 196.6K 安全最大请求与跨重启恢复

DeepSeek 资格使用独立启动形状 `max_model_len=524288`、GPU memory utilization `0.75`；启动
进程读取参数后，私有 launcher 配置立即恢复原有 `1048576`/`0.835`，因此没有把资格参数留成
生产默认。runtime 仍通过 vLLM 公共数据自动发现 5 个 HMA cache group、170 个 layer 和
`image` 模态；SpoolCache 核心没有 DeepSeek 名称、架构或模态分支。

在每 2 秒双机 watchdog 下先运行 131,072、再运行 196,608-token cold bypass。watchdog 同时
要求两端 `MemAvailable >= 3 GiB`、`SwapFree >= 4 GiB`、SSH/API 可达、2/2 ranks 以及
rank-identity/inventory-quorum/fatal-clear 三项 readiness。两次 cold control 的 TTFT 分别为
185.284 和 310.024 秒。196,608-token control 的 Spark-1 最低 MemAvailable 为 4,091,724 KiB，
只比终止线多 945,996 KiB；因此它被确定为当前安全最高资格长度，没有继续尝试 262K。

producer 与 consumer 都从同一个确定性 196,608-token token stream 切片，避免两个独立构造的
文本只凭描述假定前缀相同。131,072-token producer 发布安全 span 130,048 的 entry
`588b599e004d…`。两 rank 各保存 170 objects / 520,782,592 logical bytes，5 组页覆盖为
`508,2,2,2,16`；身份绑定 verifier 在完整 TP 重启前后均逐对象校验 type、length、logical
SHA-256 和 zero padding，结果都是 `all-payloads-authenticated`，且两 rank 的 content index
SHA-256 同为 `914986fad4b1…`。

完整停止并重启后，新 worker generation 各自扫描到 5 个启动 entry，scheduler readiness
恢复为 2/2 ranks。精确 196,608-token consumer 命中 130,048 tokens，两 rank 共恢复
1,041,565,184 bytes；TTFT 为 126.057 秒。完整输出 SHA-256
`2f43521c4cb3c…c4993` 与独立 cold control 完全一致。恢复期间 55 个 watchdog 样本全部健康，
Spark-1/2 最低 MemAvailable 为 11,291,836/15,549,576 KiB，最低 SwapFree 为
11,607,580/14,569,884 KiB。最终 post-admission failure、quarantine、scrub failure、store
error 和 telemetry drop 均为 0；协调协议仍为 `spoolcache-coordination/v1`。

原始请求、双 rank payload 以及汇总收据位于
`docs/receipts/2026-09-07-g3d-deepseek-*`。不少于 24 小时的 read-write/restore-only 双机
长稳已按用户要求从当前验收目标删除；将来若重新需要，必须作为新的 qualification task
单独立项。DeepSeek 最大上下文子项的完成不依赖该长稳收据。

## 2026-09-07：统一功能开发模型决策

后续需要真实 vLLM 模型的测试分为两个角色：兼容性开发与回归继续使用 DeepSeek、Qwen、
GLM，覆盖不同目标 build、架构、模态声明和 runtime-discovered layout；除此之外的功能、
正确性、故障注入、PP、异步、常规性能、长稳和发布安装测试统一使用
`google/gemma-4-E2B-it`。初始固定 Hugging Face revision 为
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`。本节记录模型选择当时的测试基线；后续取得的
功能收据记录在下文，不改写前述 DeepSeek/Qwen/GLM 历史资格结果。

该模型选择不得进入 `src/spoolcache`、connector 配置或 identity 准入逻辑。纯 CPU 的存储、
quorum、identity、scrub 和故障契约仍使用无模型 fixture；新的模型继续由 vLLM 公共接口和
cache semantics 自动发现。

## 2026-09-07：官方 vLLM 功能开发容器

为统一功能开发环境，仓库根目录新增 development-only `Dockerfile` 与 `compose.yaml`。
基础镜像使用当日官方 `vllm/vllm-openai:latest` 所解析的稳定版本 `v0.28.0`，固定多架构
manifest digest 为
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`；其 arm64 manifest
为 `sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce`，upstream build commit
为 `2cf0a6915ce544dc493a0990f2ea38d81601128a`。upstream registry 在该 commit 声明
`Gemma4ForConditionalGeneration`，但这只证明模型实现存在，不代替 SpoolCache contract、
CUDA mover 或内容正确性资格。

宿主使用 `uv sync --group dev` 安装 pytest/Hugging Face CLI；vLLM、Torch 和 CUDA 由官方
镜像提供，避免在 DGX Spark aarch64 宿主 `.venv` 中形成不可复现的 ABI 组合。Compose 固定
`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`，以 TP=1/PP=1、16K
context、read-write SpoolCache 作为日常功能基线。工作树和 HF cache 都只读挂载，SpoolCache
数据使用独立 named volume；运行该服务前必须先停止占用同一 GPU 的现有模型服务。
开发 Compose 固定设置 `VLLM_LOGGING_LEVEL=DEBUG` 与 `VLLM_SERVER_DEV_MODE=1`，便于显式
清除进程内 prefix/encoder/multimodal cache 而保留 SpoolCache 数据。生产部署不继承这两个
开发态默认值。

构建收据：删除 package 内 supervisor/readiness 后重建的 arm64 本地镜像
`sha256:8c85f0695d13724b4d393cb15ec69ba1b969b9045ccb70e5ab5cdda1d2de1ed9`
包含 vLLM 0.28.0、Transformers 5.15.1 和 SpoolCache 0.1.0a0；安装 wheel SHA-256 为
`459c112528cf9be4d7203a37aa44149ff818f7c5548bf95b8d0dbc948b8f590f`，且镜像内
`spoolcache.supervisor`、`spoolcache.readiness` 均不可导入。真实 vLLM runtime contract
subset 为 46 tests、45 passed、1 skipped，且 registry 确认
`Gemma4ForConditionalGeneration`。随后本地配置核验确认固定 revision 的 architecture 为
`Gemma4ForConditionalGeneration`、`audio_config` 非空且 `audio_token_id=258881`；同一镜像的
`Gemma4ProcessingInfo.get_supported_mm_limits()` 根据该配置公开返回 image/audio/video。
一次 GPU 启动已完成完整 checkpoint load，并在日志中看到 vision/audio tower 权重；服务在
编译阶段因开发目标切换而主动停止，尚未到 API-ready，所以仍不构成 CUDA mover、SpoolCache
命中内容或多模态正确性收据。DeepSeek 也保持停止以释放资源。

## 2026-09-07：LMCache-style 编排边界收敛

通用性复核后删除包内 `spoolcache.supervisor`、`spoolcache.readiness` 及其专属测试。
connector 继续导出 rank/quorum/fatal metrics 和持久 event journal；post-admission 恢复错误
仍以固定 exit 70 fail-stop，避免部分写入 KV 被继续使用。Docker/SSH 生命周期、readiness
聚合和完整 TP/PP 组恢复改由外部部署编排器承担，不进入 SpoolCache 配置或运行时依赖。
此前 G3a/故障注入段落记录的是当时实现与实测历史，不再代表当前发行包包含 supervisor。

三个 Mia-Lab 仓库先同步 `origin/main`，然后把本地 SpoolCache 集成各压成一个上游之上的
提交：DeepSeek `54e8d35`、Qwen `d8ec9d1`、GLM `3dfda93`。launcher 只保留启动失败时清理
本次创建 ranks 的有限逻辑，health 只报告 API/container liveness，不冒充 connector 或全局
部署 readiness。验证结果为 SpoolCache host 237 passed / 10 skipped / 249 subtests、重建后的
官方 vLLM 0.28.0 镜像 runtime contract 45 passed / 1 skipped、三个 launcher 各 3/3 静态契约
和全部 shell 语法检查通过。GLM 上游当前仍有一个既有断言不一致：测试要求 `stock`，上游 launcher 已使用
`rightsize`；本次没有借 SpoolCache 提交改写该上游行为。

## 2026-09-07：Gemma 4 text/audio/image/video 与混合模态功能资格

本地 DGX Spark 使用固定
`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`、官方
`vllm/vllm-openai:v0.28.0` 多架构 digest、TP=1/PP=1 和 16K context。官方 serving image
虽然由 registry 声明 `audio/image/video`，但默认没有音频解码依赖；直接提交 OGG 的首次
`/tokenize` 返回 “Please install vllm[audio]”。开发 Dockerfile 最终只用 `--no-deps` 加入
av 18.1.0、scipy 1.18.1、soundfile 0.14.0、soxr 1.1.0，避免普通 extras 解析把基础镜像的
NCCL 2.30.7 降级到 2.29.7。最终开发 image ID 为
`sha256:da350b96a30ef010b91265475d530bf61112d39529d2f14c5600ba447400b6e4`，NCCL 仍为 2.30.7。

首次完整启动又暴露一个通用 KV-sharing contract：vLLM 的公开 `kv_cache_groups` 只有 15 个
拥有 block table 的 owner，但模型初始化后注册表包含 35 个名称，其中 20 个是跨 layer
共享同一 tensor 的 alias。旧的严格集合相等检查因此拒绝启动。修复后只选择 group owners；
额外名称必须与 owner 的 device/dtype/shape/stride/storage offset/data pointer/storage bytes
全部相同才可排除，并把 alias 映射绑定进 rank identity。独立 extra tensor 或缺失 owner 仍
fail closed。运行时收据为 alias count 20、digest `cf276f024859`，核心代码没有模型名称分支。

图片使用 COCO `000000039769.jpg`（173,131 bytes，SHA-256
`dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`）；音频使用 vLLM public
`mary_had_lamb.ogg`（65,449 bytes，SHA-256
`c8f0a87f8d7e44f2d6e0f88ec63f6401b4f153f53fd14a9d730a5d1ba9927c4e`）；视频使用
PyTorchVideo `archery.mp4`（549,197 bytes，SHA-256
`8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562`）。多模态 benchmark 增加
可重复的 `--media-kind/--media-file`，使混合请求仍由同一个通用路径构造，单媒体 CLI 保持
兼容。

所有 case 先产生固定 4,096-token producer，再清除 vLLM prefix/encoder/multimodal cache，
同时显式保持 `reset_external=false`。冷 bypass 和 restore 使用完全相同的渲染请求、salt、
seed 和结构化语义 oracle。完整 engine 重启后再次发现 5 个 entries 并完成以下复验：

| case | entry 前缀 | bypass / restore cached tokens | 内容 oracle | 完整输出 SHA-256 |
|---|---|---:|---|---|
| text | `a73e28946582` | 0 / 2,048 | deterministic continuation | `84e527be21a47227ac003589ef02c90a5fada9e27a21491e5bdb77275705a249` |
| image | `5a30e8979f4d` | 0 / 2,048 | `CATS` | `f43c3a6a804ebd0b94e1d1384cb4680f7a24a9d2558076cfee5fee173e0836f8` |
| audio | `0e2bd54c1bd9` | 0 / 2,048 | `MARY_HAD_A_LITTLE_LAMB` | `f58effbcb9b1dd8008724ac940fb853b8962ab7ee9f016a963402f2e089c1f60` |
| video | `726aff999dc4` | 0 / 2,048 | `ARCHERY` | `9041bdc879c87001e177bc15e8eca4e2de5d1aabe1038204e1f68d8d631e64dd` |
| image+audio+video | `941178983e71` | 0 / 2,560 | `CATS_MARY_ARCHERY` | `a68325f169555a2158ea6bd3f942260a8849335d8755429318118d7794a5e307` |

离线 verifier 对每个 entry 全量读取 payload，绑定 deployment/rank/topology/profile/layout，
并验证 5 groups、15 owners、连续页覆盖、对象 length/SHA-256/zero padding，全部返回
`all-payloads-authenticated`。机器可读汇总为
`docs/receipts/2026-09-07-gemma4-text-multimodal-cross-restart-summary.json`。这是本地日常功能
收据，不代替 G4 PP=2、多节点或 release artifact 资格；SpoolCache 只持久化语言模型 KV，
不持久化 vLLM encoder 输出。

## 2026-09-07：G4 通用 PP=2 拓扑与双机全模态资格

G4 删除 PP=1 启动拒绝，但没有增加模型、架构、layer 或拓扑白名单。connector 只从 vLLM
public `ParallelConfig` 与 PP/TP/DCP process groups 派生 worker coordinate；DCP 被视为 TP
subdivision，不额外增加进程。scheduler 的 required participants 是 PP×TP global ranks，
PP>1 只接受 public PP-aware handshake，且 `(pp_rank,tp_rank)`、inventory global rank、
coordination identity 和完整 participant set 必须同时一致。PP=1 仍走原 handshake，原
coordination/layout/rank-ownership 摘要字段保持不变；协调 schema 继续是
`spoolcache-coordination/v1`。

vLLM 对 PP worker 的 `kv_cache_groups` 保留全局 group 顺序，但 stage-local group 可以没有
layer。HMA builder 因此允许单个空 group，并从 group-level public semantic contract 读取页选择
语义；整个 worker 仍必须至少持有一个 KV layer。跨 stage coordination digest 只绑定有序
group 的 block/reuse/page-selection 语义，stage-local layer、shared aliases 和物理 byte geometry
继续严格进入各自 rank identity/layout/manifest。离线 verifier 新增 PP/TP/DCP coordinate 与
每 group stage-local layer count，空 group 必须显式给出 0 pages；错误 deployment、rank、
topology、profile、layout、coordinate、ownership coverage 或页数不能输出认证成功。

参考边界复核了本机 LMCache `dev@dfc2720bb8aaf3edc6b61173018f6aad772c9369`。采用的是三项
基础原则：public vLLM parallel config 作为 rank 事实来源、`register_kv_caches()` 最终映射作为
worker tensor 事实来源、多模态 identifier 参与 key。未采用 LMCache 多节点工具对 TP/PP
物理放置的固定假设、窄整数 media projection、独立 encoder cache engine，或缺少持久
manifest/global quorum 的宽松处理。

真实资格在两台 DGX Spark 上使用 CX-7 地址 `10.100.216.1/10.100.216.2`，NCCL 日志确认
`NET/IB` 同时使用 `rocep1s0f1` 与 `roceP2p1s0f1`。模型固定为
`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`，镜像为官方
vLLM 0.28.0 派生开发镜像 `sha256:da350b96a30ef010b91265475d530bf61112d39529d2f14c5600ba447400b6e4`。
同一精确源码 snapshot 共 150 files，SHA-256
`eb2557b3a18e7ffc658d528d9485286225956fdfb044f86b2d32a9455763e4ed`。

上游 vLLM 默认 Gemma PP partition 18/17 在 SpoolCache 初始化前因 shared-KV owner 跨 stage
而失败；13/22 可以启动，但即使请求明确 `spoolcache_bypass=true`，图片/天空内容 oracle 也与
独立 PP=1 baseline 不一致。12/23 的无缓存 PP=2 baseline 正确返回 `CATS`、`BLUE` 和 `4`，
因此资格 harness 使用 `VLLM_PP_LAYER_PARTITION=12,23`。这只是固定模型/固定 vLLM 的测试
前提，不进入 SpoolCache 源码或配置。

12/23 下，PP0 最终拥有 12 个 tensor（group layers 3/3/2/2/2），PP1 拥有 3 个 tensor
（0/0/1/1/1）；20 个 shared layer 均由 vLLM 最终映射证明为 exact view 后去重。两 rank 的
topology digest 同为 `740cb6c328aa…`，但 deployment/rank/layout identity 按 stage-local 事实
隔离。五个资格 case 均先 cold producer，再做相同请求的 bypass control、清空 vLLM
prefix/encoder/multimodal cache 后 restore，最后停止并重启完整 PP group 后复验：

| case | entry 前缀 | bypass / restore / restart cached tokens | 内容 oracle | 完整输出 SHA-256 |
|---|---|---:|---|---|
| text | `6c2e07744af6` | 0 / 2,048 / 2,048 | deterministic byte equality | `2e28226806679bf8aaa59b0eff6db5a5a91ba3d5fe39ac4b89ea0aa66edf588e` |
| image | `35dc738dca25` | 0 / 2,048 / 2,048 | `CATS` | `f43c3a6a804ebd0b94e1d1384cb4680f7a24a9d2558076cfee5fee173e0836f8` |
| audio | `721af4f9d166` | 0 / 2,048 / 2,048 | `MARY_HAD_A_LITTLE_LAMB` | `f58effbcb9b1dd8008724ac940fb853b8962ab7ee9f016a963402f2e089c1f60` |
| video | `b25b031ba972` | 0 / 2,048 / 2,048 | `ARCHERY` | `9041bdc879c87001e177bc15e8eca4e2de5d1aabe1038204e1f68d8d631e64dd` |
| video+image+audio | `046f68ef531b` | 0 / 2,560 / 2,560 | `CATS_MARY_ARCHERY` | `a68325f169555a2158ea6bd3f942260a8849335d8755429318118d7794a5e307` |

重启前后 engine host PID 均改变；重启后两个 stage 对五个 entry 全部记录相同 span 的 restore。
离线 verifier 对整个六-entry namespace 的 rank0 72 个和 rank1 18 个 manifest objects 全量读取，
总计 90 个对象均为 `all-payloads-authenticated`。额外第六项是早期 mixed prompt order 试验；
media 被放在 sliding window 之外导致内容 oracle 不成立，虽 cache 自身认证一致，但不计入功能
资格。最终机器收据为
`docs/receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json`。

通用变更完成后，同一源码在 DeepSeek、Qwen、GLM 目标镜像各运行 256 tests，分别 skipped
6/5/5，全部 OK；三个 MiaAI-Lab launcher contract 各 3/3。三个部署仓库保持 main、相对
origin/main 0 behind/1 ahead，且各只有一个 rebased 本地 SpoolCache 集成提交。宿主最终为
249 passed / 12 skipped / 256 subtests；官方 vLLM 0.28 开发镜像使用真实 CUDA 运行 256 tests，
1 个环境性 skip，结果 OK；compileall、JSON 解析与 diff whitespace 检查通过。

## 2026-09-07：仓库内 Gemma 双机开发 launcher

新增 `scripts/gemma-pp2-dev.sh`，把 G4 已通过的 Gemma/vLLM/TP=1/PP=2/`12,23` 组合固化为
双机日常开发 harness，而不是 connector 模型 profile。脚本默认通过
`10.100.216.1/10.100.216.2` CX-7 路径工作，NCCL 使用两个 interface/HCA selector；只有宿主
地址、fabric 名称、HF cache 和 NVMe root 可以在 ignored `.env.gemma-pp2` 中覆盖。它提供
独立 image/model sync、双机规范化 digest 相等的 clean source snapshot、worker-first start、
完整组 stop/restart、JSON status 和单侧 logs；没有 restart policy、daemon 或 supervisor。

实机 smoke 使用 image
`sha256:da350b96a30ef010b91265475d530bf61112d39529d2f14c5600ba447400b6e4`，双机源码 snapshot
digest 为 `2861464b0fe86d227c7850363fc47ba93352335a5d39e151fc918c6aa9712a26`。preflight 验证两侧
固定模型 revision、image ID、GPU idle、两个 CX-7 interface/HCA 与双向 route；API ready 后
SpoolCache 报告 required/ready ranks 2/2、三项 readiness 均为 1。4,096-token producer 建立
2,048-token 安全持久前缀，清除 vLLM prefix/encoder/multimodal cache 后，4,128-token consumer
恢复 2,048 tokens；相同 consumer 的 bypass 与 restore 完整输出 SHA-256 均为
`f77219433d9fc439468318f45e2e67b1c800884598978947768381271752c2c2`。最后通过 launcher stop
移除两侧容器，JSON status 为 head/worker absent、API false，持久 cache root 保留。


## 2026-09-08：G6 不可变 wheel 与发布资格

本地候选 `0.1.0` 由压缩历史前的 `93e7ad4` 构建，wheel SHA-256 为
`9ab91e0eb4782b723891a379a7fe634fa1fae08f0329fa2e622ee29bb2e9925b`。
独立重建完全一致；开发镜像和 DeepSeek/Qwen/GLM 三种 runtime 在两节点安装相同 wheel，
逐文件、来源和 import path 认证通过。Serving 的 source sync/PYTHONPATH/source mounts
已经删除，生产开发端点关闭，整组固定同一 image ID。Schema 不变，但 SpoolCache version
本身参与 identity，所以从 alpha 到 0.1.0 会安全切换 namespace。

宿主 258 passed / 12 skipped / 279 subtests；四个真实 CUDA/vLLM runtime 各 264 tests，
official/DeepSeek 各 skip 1 个不存在的 scratch contract，Qwen/GLM 无 skip，全部 OK。
三个部署 launcher 各 5 tests 通过。Gemma 固定 revision、TP=1/PP=1 与 TP=1/PP=2（12,23）
各完成 text/image/audio/video/mixed 的两次 disjoint cold bypass、producer、GPU/MM reset
restore、整组 restart，以及一字节损坏后的完整恢复。前四类均恢复 2,048 tokens，mixed
恢复 2,560；PP=1 的 5 manifests / 75 objects 与 PP=2 的 10 manifests / 75 objects 全量认证。
每个 participant 的 entry/span 与 scheduler 一致，完整输出与两份零命中冷对照相同。

必须保留两个限制：PP=2 长填充 image/mixed 在冷 bypass 下稳定返回 OTHER，因此这些 case
证明输出等价和载荷正确，不证明图片/混合语义分类正确；短 image 控制仍为 CATS。远端 PP
worker 的实际 exit_group(70) 不保证 head /health 自动失效；第一次请求在外部 90 秒期限到达
后退出，随后精确恢复对象。复测明确由部署端观测 worker 退出后停止整组，客户端拒绝不完整
stream、API false、两 rank 停止，再恢复原始对象并整组重启复验。无 upstream patch 或包内
supervisor。临时备份只在全量复验后移除；最终两节点停止，所有 cache roots 保留。

机器收据和负面观察见 [`2026-09-08-g6`](receipts/2026-09-08-g6/README.md)。没有新性能收益
声明，也未重复 DeepSeek/Qwen/GLM 的真实模型功能测试或已豁免的 24h soak。版本发布已改为
GitHub Actions + python-semantic-release + PyPI Trusted Publishing；最终公开 wheel 的 commit
和 SHA 单独记录，不能用相同 version 字符串冒充相同 artifact。主仓库首次推送前按用户要求
把所有开发历史压缩为一个 feat commit，旧 commit 只作为本地历史收据保留。


发布收尾：51 个旧开发提交压缩为 `80a19db` 一个 `feat:` 初始提交，PSR 自动生成
`40528e6` / `v0.1.0`。GitHub Actions run `34180558915` 的 Python 3.10/3.11/3.12、
独立重建、fresh-wheel 安装测试、GitHub Release 和 PyPI Trusted Publishing 均成功。
Python 3.10 先发现两处旧测试错误地强制要求 3.11 的 exception notes；已改为所有版本
认证 secondary-error 日志，仅支持 add_note 时额外检查 notes，未修改 runtime 源码。

公开 wheel SHA-256 为 `abaf71af05afd41c0d5a2873a003bc94af3a65b54dbe9db97c1015b2dd01bd28`。
GitHub Release、Actions artifact、PyPI 下载三者字节完全一致；全新环境从 PyPI 安装后
27 个文件与公开 wheel 一致，CLI 正常。公开 wheel 的 22 个 spoolcache package files 与
候选完全相同，METADATA headers 完全相同；差异仅 README description/RECORD/ZIP timestamps。
因此通过精确 payload 等价承接上述 runtime/live 资格，未把 version 相同当成 artifact 相同。
该公开 wheel 已装入两节点的开发/DeepSeek/Qwen/GLM 镜像，8 次安装认证全部通过；三个生产
env 已选择新 image ID，双机 preflight 通过，最后两节点维持停止且保留缓存。最终发布、安装
和等价证据在同目录 `publication.json`、`pypi-install.json`、`published-installations.json`。
