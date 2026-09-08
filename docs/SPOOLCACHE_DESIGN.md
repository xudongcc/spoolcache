# SpoolCache 设计与实施方案

| 项目 | 内容 |
|---|---|
| 状态 | Alpha 已实施；TP=1/PP=2 双机文本与全模态跨重启路径已验证 |
| 目标版本 | SpoolCache 0.1 |
| 项目名 | `SpoolCache` |
| Python 包名 | `spoolcache` |
| 核心约束 | **不修改、不覆盖、不热补丁 vLLM 源码** |
| 首个目标环境 | Linux、CUDA、DGX Spark/统一内存设备、vLLM V1、TP=2、DeepSeek-V4 Flash HMA |
| 更新日期 | 2026-09-07 |

## 1. 结论

SpoolCache 已实现为一个可由 vLLM 动态加载的、进程内的 `KVConnectorBase_V1`
连接器：每个 worker 只把自己 runtime-derived PP×TP global rank 的 KV 状态持久化到本机
NVMe，scheduler 只维护少量前缀索引和全 participant 一致性状态。缓存容量可以很大，但用户态常驻内存由固定
大小的 staging buffer 决定，不随上下文长度或磁盘缓存容量线性增长。

0.1 采用以下方案：

- 通过 `kv_connector_module_path` 加载 `SpoolCacheConnector`，不把任何文件写入
  vLLM 安装目录。
- 不启动独立缓存服务，不建设常驻 CPU L1；数据面是 rank-local NVMe。
- 使用固定大小、预分配的 pinned-memory 与 direct-I/O 双缓冲分片传输；当前每 rank
  各为 `2 × 64 MiB`，合计 256 MiB 固定 host staging。
- 使用精确前缀摘要、不可变对象、manifest-last 原子提交、全 rank quorum 和恢复时
  完整性校验。
- 启动时从 vLLM 公共多模态 registry 和本次部署 limit 自动发现所有已启用输入模态；
  SpoolCache 代码不维护模型、架构或模态白名单。
- 任何 HMA 布局都必须把所有 cache group 作为一个事务处理；缺少任何一组都视为
  miss，不能部分命中。
- 命中承诺之前的任何异常都退化为普通 miss；在当前固定 vLLM 上，命中承诺之后的
  HMA 恢复异常必须 fail-stop，由外部编排器按部署策略重启 engine，不能假装支持安全重算。
- 首版不实现共享 GPU 前缀、GDS/RDMA、远端缓存、P/D 解耦、增量 CoW 快照或跨
  topology 复用。

一句话定位：**A patch-free, bounded-memory NVMe KV cache for vLLM.**

## 2. 背景与设计依据

### 2.1 目标部署已经确认的问题

DeepSeek DSpark 目标部署的 LMCache 实验表明，持久缓存可以显著缩短长上下文重复 prefill：约 107K
token 的上下文从约 65 秒预填充降至约 1.8–1.9 秒恢复。但当前实验拓扑也暴露了两个
不适合统一内存设备的风险：

1. 独立 cache server 的生命周期独立于模型服务；任一 server 异常可能扩大为整组
   服务故障。
2. 常驻 CPU/L1 与模型冷启动峰值共享同一物理内存，已经出现过 kernel OOM kill。

这不是对 LMCache 通用能力的否定。LMCache 支持 CPU、磁盘和远端多层存储，也可以
关闭本地 CPU tier。SpoolCache 的差异是把“无大容量常驻 L1、无独立服务、rank-local
NVMe、内存硬上限”设为默认产品约束，而不是部署时的可选调优。

### 2.2 SparkCache 是主要参考实现

本方案以 `FujitsuPolycom/sparkcache` 为最重要的参考项目之一。初始实现研究基线为提交
`66057174301a4759ca3a45207ea41016689449cb`；2026-09-04 又复核了上游提交
`5c3bd3e` 的测试与部署工具。直接吸收的设计经验包括：

- scheduler 一次计算候选前缀摘要，选择所有物理 rank 都具备的最长前缀；
- worker 只读写自己 rank 的本地状态，正常缓存数据不跨网络；
- 启动握手只携带有界 manifest inventory，之后用有界 delta/checkpoint 报告补齐；
- 缓存对象不可变，先同步对象，最后原子发布 manifest；
- cache identity 绑定模型、checkpoint、KV 布局、topology、rank 和格式版本；
- HMA 模型按完整 manager page/cache group 保存，不能把多组状态当成普通单组 KV；
- 缓冲区或后台队列饱和时跳过可选 store，不能拖慢无关推理。

SpoolCache 0.1 明确不照搬以下部分：

- SparkCache 的 vLLM patch overlay；
- 依赖 patched scheduler 的 HMA `recompute`；
- `expandable_segments:True` 的 connector 白名单豁免；
- shared GPU prefix lease/attach；
- page-tail CoW、稀疏别名和原生 CUDA 快照库。

SparkCache 使用 Apache-2.0 许可证。后续若复用代码而不只是复用思想，必须记录文件级
来源、保留版权和许可证声明，并在发布前完成一次 provenance 审查。

### 2.3 LMCache 通用性参考边界

SpoolCache 的基础 KV 路径采用与 LMCache 相同方向的通用性边界：从 serving runtime 读取
model locator、rank/world、KV shape/dtype 和实际 cache tensor，不按模型名称选择 layer 表、
cache profile 或模态白名单。LMCache 的基础 cache key 也使用 model name、rank/world、
token chunk hash、dtype 和可选 tag，而不是在 connector 启动时遍历整个 checkpoint 文件树。

LMCache 的 vLLM adapter 直接接受 `register_kv_caches()` 最终 tensor 映射，并按 runtime
`kv_cache_groups` 区分真正拥有 block table 的 layer 与跨 layer 共享项。SpoolCache 沿用这个
原则：最终映射是 worker 物理事实来源；group owner 以外的注册项只有在运行时证明为完全相同
的 Torch storage view 后才去重。额外证明只用于防止持久 HMA manifest 把同一物理页写进错误
block-id 空间，不是模型 layer 白名单。

2026-09-07 再次复核 LMCache `dev@dfc2720b`：它同样从 vLLM public `ParallelConfig` 获取
world/rank/TP/PP/DCP，并让每个 PP stage 保存自己的 KV。SpoolCache 借鉴这一拓扑事实来源，
但不复制 LMCache 多节点工具里“TP 必定在节点内、PP 必定跨节点”的放置假设；rank 必须由
实际 public process group 坐标交叉验证。LMCache 以 vLLM 多模态 identifier 改写 placeholder
token 的思路也验证了 media identity 必须进 key；SpoolCache 继续保留完整 identifier、modality、
offset、length，而不采用其有碰撞可能的窄整数投影。LMCache 新增的 encoder cache 是独立可选
engine，不属于 SpoolCache 0.1 的语言模型 KV/HMA 数据面。

这里参考的是 LMCache 的基础结构化路径，不是照搬其所有可选优化。LMCache 的 CacheBlend、
CacheGen 等独立能力可以有模型 dispatch 或调优参数；SpoolCache 核心不得因此引入模型特例。
SpoolCache 额外保留 opaque HMA layout/rank identity、全 rank quorum 和持久 payload 认证，
因为这些是本项目 rank-local NVMe 格式的解释与正确性边界。

### 2.4 vLLM 能力与边界

当前部署固定为：

- image：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- vLLM：`0.25.2.dev0+g752a3a504.d20260714`
- 已安装 vLLM package content SHA-256：
  `e24d925d2e31333b9af6e383becf6f8e4ad5d4712e6b3bbe71b396fe40eaa191`
- KV layout：DeepSeek-V4 `nvfp4_ds_mla`，HMA 多 cache group
- topology：两台 DGX Spark，TP=2

该版本 V1 connector 已提供外部模块加载、scheduler/worker 元数据交换、启动握手、
load/store 完成通知和 `SupportsHMA.request_finished_all_groups()`。SpoolCache 0.1 在这些
hook 内使用同步数据路径；因此“写一个外部 connector”本身不需要修改 vLLM。

但需要明确两个限制：

1. `KVConnector V1` 有加载失败上报接口，vLLM 也有 `kv_load_failure_policy`；不能简单
   说“V1 没有 recompute”。问题在于当前固定版本的 scheduler 对 HMA 多组 block 的
   失败回滚不具备我们需要的安全语义，SparkCache 正是为此提供 scheduler patch。
2. 当前 vLLM 对任意 KV connector 与
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 的组合做保守拒绝。SpoolCache
   不打豁免补丁，因此启用时必须取消该配置，或在验证通过后使用 vLLM 认可的 cuMem
   allocator。**不改 vLLM 源码不等于完全不改启动配置。**

## 3. 目标与非目标

### 3.1 目标

- 在 vLLM、容器或主机重启后复用已持久化的精确 prompt 前缀。
- 缓存磁盘容量和最大上下文增大时，SpoolCache 的 staging 内存保持有界。
- 支持 vLLM 运行时公开且语义可验证的 HMA 布局；所有物理 rank、所有 cache group 原子命中。
- 支持 vLLM 为当前模型声明并启用的多模态 prompt KV 持久化；媒体内容身份、模态和
  placeholder 几何都进入缓存键。图片、视频和未来 registry 模态走同一条路径。
- 缓存关闭时，vLLM 行为与未安装 SpoolCache 时一致。
- miss、队列满、磁盘满、部分 rank 缺失等常见情况不影响普通模型计算。
- 正确性优先于命中率和可用性；绝不把部分或未经验证的 KV 暴露给 attention。
- 对版本、身份、存储和故障给出可审计的日志、指标和启动收据。

### 3.2 非目标

- macOS、Apple Silicon、MLX 或 llama.cpp。
- 跨 checkpoint、跨量化格式、跨 TP/PP/DCP topology 复用。
- prompt embedding、LoRA/prompt adapter（0.1 遇到这些请求直接 bypass）。
- vLLM 未声明/已通过 per-prompt limit 禁用的模态，或缺少稳定媒体 ID、placeholder
  offset/length 的请求；这些请求 fail closed 为 persistent-cache bypass。
- 非精确或语义缓存。
- 多实例远程共享、跨节点数据复制、P/D disaggregation。
- CPU 热缓存、独立 SpoolCache engine/server/controller、cache daemon、Redis/S3 等远端后端。
- 包内 Docker/SSH supervisor、服务 readiness endpoint 或自动 TP/PP 重启器。
- 恢复失败后的无中断透明重算。
- 对任意 vLLM 版本的兼容承诺。

### 3.3 模型 artifact 信任边界

SpoolCache 只认证自己发布的 KV manifest 和 payload，不充当模型仓库供应链验证器。基础
KV identity 使用 vLLM 公开的 model locator/revision namespace，并继续绑定 runtime、dtype、
topology 和实际 layout；生产运维应通过不可变镜像、不可变 revision 或显式 namespace 轮换
保证模型 artifact 发布一致。

因此系统不承诺发现“同一 locator/revision 下权重字节被原地替换”。曾试验的完整 checkpoint
文件树 inventory、逐文件 SHA-256、rank receipt、symlink-free 全量副本和 launcher 模型路径
改写已在提交前撤回，也不是后续启用 SpoolCache 的前提。G3c 已删除历史上命名错误的
`spoolcache_checkpoint_sha256` 部署 pin；connector 现在只使用下述公共 runtime namespace。

### 3.4 多模态能力发现原则

SpoolCache 不接受任何模型白名单、架构白名单或写死的模态集合。connector 初始化时先
读取 `ModelConfig.is_multimodal_model`，再通过 vLLM 公共
`MULTIMODAL_REGISTRY.get_processing_info(model_config).supported_mm_limits` 获取模型声明
能力，并用 `ModelConfig.get_multimodal_config().get_limit_per_prompt(modality)` 排除本次
部署显式设为 0 的输入。其余键无论名称是什么都自动启用。因此 vLLM 将来注册的新模态
不需要 SpoolCache 增加一条兼容配置。

运行时每个 `mm_feature` 必须同时具备：vLLM 声明且启用的 modality、非空稳定
identifier，以及合法且有序的 placeholder offset/length。缓存摘要绑定这四项事实；同
内容 ID 在 image 和 video 下也产生不同 key。任何事实无法证明时只 bypass 该请求的
持久缓存，不退化成 token-only key。下文“已验证模型”表只表达测试覆盖，不参与上述
启用判定。

| 已验证模型/部署 | vLLM 启用模态 | 多模态 KV 收据 |
|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp，TP=2 | image | image 双 rank 跨重启命中、换图隔离；text 在 196,608-token consumer 跨重启恢复 130,048 tokens |
| Qwen3.8-Flash-Next-NVFP4，TP=2 | image、video | 两者均通过 bypass 冷对照、进程内缓存清空、双 rank restore 和完整输出 hash 对比 |
| GLM-5.3-Flash-EXL3，TP=2 | image、video | 两者均跨进程恢复路径验证，并通过同样的完整输出 hash 对比 |
| google/gemma-4-E2B-it，TP=1 | audio、image、video | text、三个单模态及 image+audio+video 混合请求均通过 local-cache 清空、跨进程 restore、内容 oracle 和全 payload 认证 |

Qwen 与 GLM 当前 registry 都未声明 audio，所以没有 audio 资格收据。这只表示尚无测试
证据；若另一个 vLLM 模型把 audio 或未来新模态声明为 enabled，通用实现仍会自动接入，
不会查询本表。

### 3.5 统一功能开发测试模型

需要真实 vLLM 模型的测试分成两个互不混用的角色：

- **兼容性开发与资格**继续使用 DeepSeek、Qwen、GLM 三个部署，目的是覆盖不同 vLLM
  build、模型架构、模态声明和 runtime-discovered KV layout；这些结果只进入已验证模型表。
- **其他所有模型相关开发测试**统一使用
  [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)，包括功能、正确性、
  故障注入、PP、异步路径、常规性能和发布安装测试。初始固定 revision 为
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`，以后升级必须在收据中明确记录新 revision
  和重新资格结果。

Gemma 4 E2B-it 在同一个较小 checkpoint 中保留 full/sliding attention、KV sharing 和
文本/图片/视频/音频入口，适合作为统一功能基线。纯 CPU 的 identity、quorum、存储、scrub
和故障契约测试继续使用无模型 fixture，不为执行这些测试下载模型。历史 DeepSeek/Qwen/GLM
收据保持原样；最大上下文或部署专项若使用它们，应明确归类为兼容性/部署资格，而不是第二套
日常功能模型。

上述名称和 revision 只存在于测试、文档和资格收据。生产 `src/spoolcache`、connector 配置、
identity 和启动准入不得查询它们；新模型仍完全由 vLLM 公共 runtime contract 和 cache
semantics 自动发现。

开发环境按依赖边界分成两层。宿主 `.venv` 只安装 `pyproject.toml` 的 `dev` dependency
group，用于 CPU fixture、存储和协议测试；CUDA/Torch/vLLM 不进入该环境。真实模型测试由根
目录 `Dockerfile`/`compose.yaml` 提供，基于官方多架构
`vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`。
Dockerfile 以 `--no-deps` 安装 SpoolCache；由于官方 serving image 不带可选音频解码器，开发
镜像另以 `--no-deps` 固定安装 vLLM 0.28 audio extra 选择的 av/scipy/soundfile/soxr，并验证
Torch/CUDA/NCCL 仍保持基础镜像版本。Compose 把开发工作树只读挂载并通过公开
`kv_connector_module_path` 加载。该 Compose 是单机 TP=1/PP=1、16K context 的日常
功能基线，并固定开启 vLLM DEBUG 日志和 `VLLM_SERVER_DEV_MODE=1`，供外部缓存资格测试只
清进程内 GPU/encoder/multimodal cache；它不代替 Q1 最大上下文、G4 PP=2 或双机资格收据。

双机日常调试由仓库内 `scripts/gemma-pp2-dev.sh` 提供外部 launcher。它固定已资格的 Gemma
revision、vLLM 开发镜像、TP=1/PP=2、`12,23` upstream partition 和 SpoolCache 公开 connector
参数；只允许通过被忽略的 `.env.gemma-pp2` 调整两台宿主的地址、CX-7 interface/HCA、HF cache
和 NVMe root。启动前必须验证两边固定 snapshot 与 image ID 相同、GPU 空闲、CX-7 地址存在；
源码用 tracked/untracked non-ignored 文件构造新的私有 snapshot，比较双机规范化 tar digest 后
原子切换稳定 symlink，worker 容器成功创建并且 API ready 后才删除旧 snapshot。worker stage
先于 head stage 启动，失败时只移除未完整启动的容器组，不删除持久 cache root。镜像用
`docker save | ssh docker load`、模型 repo cache 用不带 `--delete` 的 `rsync` 走 worker CX-7
SSH 地址显式同步。该脚本不提供 restart policy、后台 daemon 或 SpoolCache supervisor；这些
部署动作不进入 connector identity，也不改变核心零模型特例要求。`12,23` 仍只是固定测试
harness 的上游事实，新模型必须重新证明其无缓存 PP baseline，但无需新增 SpoolCache 适配。

## 4. 总体架构

```text
OpenAI-compatible request
          │
          ▼
┌─────────────────────────────────────────────┐
│ vLLM scheduler process                     │
│ SpoolCacheConnector(SCHEDULER)              │
│ - exact-prefix digest                       │
│ - all-rank quorum catalog                   │
│ - load/store plan                           │
└───────────────────┬─────────────────────────┘
                    │ vLLM connector metadata / handshake
          ┌─────────┴─────────┐
          ▼                   ▼
┌───────────────────┐  ┌───────────────────┐
│ worker, rank 0    │  │ worker, rank 1    │
│ Connector(WORKER) │  │ Connector(WORKER) │
│ fixed pinned pool │  │ fixed pinned pool │
└─────────┬─────────┘  └─────────┬─────────┘
          │                      │
          ▼                      ▼
   rank-0 local NVMe       rank-1 local NVMe
   objects + manifests     objects + manifests
```

系统只有两个逻辑角色，均在 vLLM 自有进程内：

- scheduler connector：只持有摘要、rank generation、quorum 和请求计划，不持有 KV
  payload。
- worker connector：注册本 rank 的 KV tensors/HMA groups，通过固定缓冲池在 GPU 和
  本地 NVMe 之间分片传输。

不新增网络数据面和常驻服务。worker 状态通过 vLLM 已有的启动 handshake 与每步
connector stats 返回 scheduler。启动 inventory 默认最多报告 512 个近期 entry；其余
entry 在服务启动后通过小批量 delta/checkpoint 收敛。在 quorum 收敛前，相应请求只是
miss，不阻塞 API readiness。

inventory 传输本身也必须有界：worker 最多持有 100,000 个可报告 entry，单个 stats report
最多携带 64 个 checkpoint/delta 项，单次聚合最多接受 4,096 个互异 physical rank report；
rank、generation、sequence、cycle 和 token span 都有固定类型/数值上限。启动握手只建立其
512-entry 安全子集，其余 entry 以 sequence=0 的 rolling checkpoint 补齐，不把安全的假阴性
伪装成巨大 delta，也不撤销已同步 rank。每个 worker 在上报前从 rank-local
`state/generation.json` crash-consistently 预留严格递增 epoch；时钟回拨不能复用旧 epoch。
首次升级从高于旧版 wall-clock 值的独立高位域播种，并创建
`state/generation.required.json`；哨兵已存在但 generation state 缺失时启动失败。严格更大的
epoch 先撤销旧 rank image，再由完整 checkpoint 恢复；较小 epoch 只有在 UUID/epoch 精确命中
scheduler 有界历史时才是可忽略的延迟旧包，未知较小身份按 state rollback 冲突撤销当前
image；相同 epoch 配不同 UUID 也必须撤销并保持 conflict。当前
generation 内任何超界、重复 rank、sequence gap 或不完整 checkpoint 都立即撤销该 rank 的
catalog offer，只有完整有界 checkpoint 才能重新加入 quorum。

## 5. 核心协议

### 5.1 Cache identity

身份分为三层，避免“每个 rank 的键不同”与“所有 rank 对同一逻辑前缀达成 quorum”
相互冲突：

- `coordination_digest` 只包含所有角色必须相同的模型部署 namespace、runtime、topology 和
  有序 group page-selection/reuse 语义。PP=1 继续使用既有完整 logical layout digest；PP>1
  排除必然因 stage 投影而不同的 layer ownership 和物理 byte geometry。scheduler 在接受任一
  worker 的启动 inventory 前先校验
  它；不一致时启动失败，不建立错误 quorum。后端可能在 worker materialize 时把逻辑
  dtype 名称（例如 `fp8`）细化为物理格式名（例如 `fp8_ds_mla`），这个角色本地标签不进入
  coordination digest，而是继续由 deployment/rank manifest 严格绑定；
- `deployment_identity_digest` 是角色本地的完整部署视图。scheduler 用自己的稳定摘要
  生成逻辑 `entry_id`，worker 用自己的稳定摘要隔离本地存储 namespace。vLLM 在模型
  materialize 前后可能向 scheduler/worker 暴露不同的 `ModelConfig`，因此这里不能要求
  跨角色逐字相同；
- `rank_identity_digest` 由 worker deployment identity、PP×TP global rank、PP/TP/DCP 本地
  坐标、该 rank 的完整物理布局 digest 和 stage-local layer/shared-alias ownership 再次摘要
  得到，用于验证本地 manifest 和 payload。PP=1 保留原摘要字段和 namespace。

任何位于上述信任边界内、可从 vLLM 公共契约观察且会改变 KV 字节或解释方式的事实，
都必须进入其中一层：

- SpoolCache schema 和 profile 版本；
- 模型 locator/revision namespace、模型配置摘要、量化方式、KV dtype；
- vLLM 顶层及所有公开 component `compute_hash()`，并额外绑定 attention、cache、
  Mamba、量化、speculative 和多模态 processor 的公开配置；模型/HF 配置完整绑定
  position/rope、dtype、revision 与 architecture 等事实；
- 实际已安装 vLLM package 的逐文件内容 SHA-256 与 SpoolCache version；版本字符串只作
  诊断信息，绝不伪装成 source commit 或准入依据；
- cache group、layer、block/reuse 语义进入跨角色逻辑布局；worker 的 tensor
  shape/stride、page 字节数和压缩布局另进入 rank-local 物理布局；
- manager block 总数只代表当次启动的 GPU 容量，可能随空闲显存波动，明确不进入
  持久布局摘要；
- TP、PP、DCP、DP、DP rank、role-local world size 与跨 DP world size；worker global rank、
  PP/TP/DCP 坐标和 rank-specific ownership 只进入 rank identity；
- publication policy 和 chunk/alignment 规则；
- tenant namespace/`cache_salt`。

G3c 由 connector 直接从 vLLM 公共 `ModelConfig` 派生模型 namespace：非空
`model_weights` 优先保留 object-storage/HF 原始 locator，否则使用 `model`，并绑定可空的
`revision`。这个规范化输入使用 `spoolcache-model-namespace/v1` 摘要；served alias、模型
配置类名和 launcher 名称不参与。三个 launcher 不再生成或传递模型 digest。未知
runtime/layout 或不支持的 geometry 仍在启动时 fail-fast，但模型仓库内容认证不进入
connector 职责。

### 5.1.1 运行时布局发现

SpoolCache 不再要求每个模型维护 layer 数量和 tensor 字节数表。connector 构造时直接
接收 vLLM 的 `KVCacheConfig`，worker 随后从 `register_kv_caches()` 接收实际 CUDA
tensors。`vllm-runtime-kv-v1` profile 从这两个公开入口冻结一份规范化布局：

- group 顺序和数量、每组 layer ownership；
- manager block、storage block、每 layer/manager page 字节数；
- full/sliding/recurrent-align/circular-one 页选择语义和 window；
- DCP replication、EAGLE 标志及 TP/PP/DCP topology；
- 注册 tensor 集合，并由 mover 继续验证 shape、stride 和物理 page 边界。

`KVCacheConfig.kv_cache_groups` 定义拥有独立 block table 的物理 layer。vLLM 可能在模型加载后
向 `register_kv_caches()` 增加跨 layer KV-sharing 名称；这些名称不拥有另一套 block table，
不能重复写入 manifest。SpoolCache 只在额外名称与至少一个 group owner 是完全相同 Torch
storage view（device、dtype、shape、stride、storage offset、data pointer 和 storage bytes 均
一致）时把它记录为 alias 并排除；alias 映射同时进入 rank ownership identity。缺少 owner、
额外 tensor 独立或无法证明 storage identity 均在启动时 fail closed。该规则只读取公共
layout/tensor 事实，不识别 Gemma 或任何模型名称。

自动发现只消除“模型尺寸写死”，不放弃语义契约验证。对参与 prefix caching 的状态，
SpoolCache 调用 vLLM 公开的 `get_kv_cache_spec_kind()`；准入与页选择不比较 concrete class
或 MRO 名称：

- `full_attention`、`mla_attention` 使用完整历史页规则；
- `sliding_window`、`sliding_window_mla` 使用运行时 `sliding_window` 的尾部页规则；
- `mamba` 仅在公开 `mamba_cache_mode=align` 时使用 recurrent active-page 规则。

因此，新 cache-spec 实现类只要被 vLLM 归入已实现的相同语义，便会自动通过，不需要在
SpoolCache 增加类名或模型配置。`sink_full_attention`、`chunked_local_attention`、
`encoder_only_attention`、`cross_attention`、`unknown` 等尚未证明安全 physical-page
selection 的公开语义保持 fail closed；这是对页复用语义的实现边界，不是模型、架构或
模态白名单。

对于含 `kv_cache_specs` 的聚合 group，先采用同一个 public resolver 对 group 的整体语义
声明；只有整体返回 `unknown` 才逐 member 回退。这样 vLLM registry 新增或重命名统一包装
类型时不需要修改 SpoolCache。聚合语义、group 的 prefix-sharing capability 与 member
推导出的 reuse policy 必须一致，任何“group 可共享/member 不可共享”或反向矛盾都在启动
时拒绝。

具体类型名只保留为 rank-local physical layout 的诊断/change-detector 字段，不参与语义
准入，也不进入跨角色 `logical_digest`；伪装成已知类名但被 vLLM resolver 判为 `unknown`
的对象仍会拒绝。

请求私有的 non-prefix scratch 可以不属于已知 semantic kind，但必须通过公开契约完成更强
证明：`participates_in_prefix_caching` 或 `prefix_cacheable` 至少一个是严格布尔 `False` 且
互不矛盾；`max_num_blocks_per_req(vllm_config, max_model_len)` 在当前部署真实上界严格返回 1；
`max_memory_usage_bytes(vllm_config)` 严格等于一页 `page_size_bytes`。若 runtime 另外公开
`max_admission_blocks_per_request()`，则它在真实
`max_in_flight_tokens/max_model_len` 上也必须严格返回 1。这里不做长度抽样，`bool` 不视为
整数；缺少必要方法、属性读取或调用异常、标记矛盾、页数不为一时一律启动失败，也不会把
这种 scratch 状态误当滑窗历史。若 scratch members 被包装成聚合 group，还必须在真正提供
共享 block table/allocator 的 group 对象上再次证明 block table 恰好一页、总内存恰好一个
packed manager page；group 不强制暴露第二个布尔标记，但若暴露就必须与 members 一致。空
group、重复 layer、同组混合不兼容策略或 page 几何矛盾同样启动失败。布局规范化内容进入
`layout_sha256`，因此同一 checkpoint 在 vLLM 改变分组或物理解释后不会误用旧缓存。

recurrent-align 与 circular-one 都是请求当前边界的状态，而不是可任意回看的历史页。
connector 因此只在 scheduler 的 `before_tokens` 恰好等于目标边界时发布；越过边界则安全
放弃，不读取 null block，也不按模型名修改 batch size。Store 可以覆盖生产者完整的对齐
prompt；restore 仍至少保留一个 token 交给 vLLM forward，因此同一 entry 可服务带后续
token 的消费者。对公开的 align-mode recurrent 语义，vLLM input preparation 会在 connector hook
之前把已完成边界预拷贝到本轮 active running-state page；mover 因而从运行时 block table
的运行态尾部选择该页做 capture/restore，而不是写回已被预拷贝消费的历史页。chunked
scheduling 可以在命中边界与运行页之间留下 null blocks，运行时声明的 speculative state
pages 又位于运行页之后；选择器会从公开 cache spec 发现这个尾部数量并纳入 layout identity，
不再假设运行页与边界相邻。这个生命周期规则来自 cache spec 与公开 connector 调用顺序，
不依赖模型名称。

布局发现没有运维选择器，也不读取 model name 或逐模型 layer/page 表；启动时一律执行
上述 `KVCacheConfig` 发现，并把内部协议标识 `vllm-runtime-kv-v1` 写入布局和 manifest。
兼容检查没有 strict/auto 选择。启动时统一检查已安装 vLLM 的公共 connector/HMA
callback，并验证基类允许的每种调用形状都能由 SpoolCache override 接收；不维护模型到
vLLM 版本的映射。该启动 gate 也要求 public cache-spec resolver 存在、可调用且能以一个
cache spec 位置参数调用；connector 随后使用同一个已验证 callable 做布局发现。任何未知
spec、接口漂移或 tensor ownership 不一致都 fail closed。
模型 locator/revision 只用于缓存 namespace 隔离，不是兼容白名单或字节证明。

vLLM 给 scheduler 的 `KVCacheConfig` 是逻辑视图，给 worker 的则包含模型分配后的打包
物理 page；两边的 `page_size_bytes` 可以正常不同。SpoolCache 因此计算两个摘要：

- `logical_digest` 排除 worker-only 字节数，进入所有角色必须一致的 coordination
  identity，并进入各角色自己的 deployment identity；
- `physical_digest` 包含每 layer/manager page 字节数，用于 manifest coverage；worker
  注册 CUDA tensor 后再生成 rank-local geometry digest，完整绑定 dtype、除容量维以外的
  shape、规范化 byte stride、storage offset 与 manager page view，并进入 rank identity。

测试必须证明“只改变 scheduler/worker 表示方式时 logical 相同而 physical 不同”，并且
改变 block、window、reuse、layer ownership 或 EAGLE 语义时 logical 必须变化。实机启动
还必须证明各角色 `coordination_digest` 一致；完整 deployment digest 不作为跨角色握手
条件。

### 5.2 前缀键

0.1 使用 SHA-256 链式摘要，不使用 Python `hash()`：

```text
h0 = SHA256(deployment_identity_digest || tenant_namespace || cache_salt)
hi = SHA256(h(i-1) || canonical_token_chunk_i)
entry_id = SHA256(hi || media_identity_at_boundary)  # 有媒体时
```

候选边界必须同时满足 profile 的所有 HMA group 对齐条件。scheduler 对一个请求只做
一次正向计算，然后从最长边界向前查找 quorum。0.1 只保存 prompt 的已完成、完整对齐
前缀；生成 token 和不完整尾块不保存。

对视觉请求，每个候选边界还摘要 vLLM 提供的稳定媒体 `identifier`、模态、placeholder
offset/length。边界位于图片之前时仍可复用相同的纯文本前缀；边界触及图片后必须同时
匹配图片内容与 prompt 几何。媒体身份不完整、外部 embedding、adapter，或缺少要求的
tenant salt 时返回 0 个外部 token，由 vLLM 正常 prefill。

### 5.3 全 rank quorum

一个逻辑 `entry_id` 只有在每个预期物理 rank 都报告以下条件时才可见：

- generation 是当前 worker generation；
- identity/topology/profile 相符；
- 本地 manifest 结构、对象存在性和对象大小预检通过；
- token span 与其他 rank 一致。

各 rank 的 payload digest 可以不同，因为它们拥有不同 shard；共同的是逻辑 entry、
token span 和 topology。任一 rank 重启、撤回、淘汰或报告损坏时，scheduler 立即撤销
该 entry 的全局 quorum。新 generation 或当前 generation 的 sequence gap 先撤销旧确认，
再通过完整 checkpoint 恢复；带较小持久 epoch 的延迟包只能忽略，不能回滚 catalog；
相同 epoch 的不同 UUID 必须 fail closed，不能保留可能过期的“幽灵命中”。

### 5.4 Load 流程

1. scheduler 计算对齐前缀摘要并查找最长全-rank entry。
2. 没有 quorum 时返回 miss，vLLM 正常计算。
3. 有 quorum 时，scheduler 通过 `update_state_after_alloc()` 取得各 group 的私有目标
   blocks，并生成一次 HMA load transaction。
4. worker 在写入任何 GPU page 前验证完整 manifest、identity、descriptor chain，以及
   所有对象的存在性、类型和声明大小；该 metadata probe 不预读完整 payload。
5. 每个对象只从 NVMe 读取一次：分片读入固定 I/O staging 时流式验证 padding 和
   SHA-256；该对象认证通过后经 pinned slot scatter 到请求私有 blocks。已放置前序对象后
   才发现后续对象损坏属于 post-admission fatal，必须停止 engine，不能继续推理。
6. 所有对象都不允许出现“预验证一次、搬运时再读一次”的双读；固定 staging 也不能为
   保留整个 entry 而扩张。
7. `start_load_kv()` 在本步 forward 前同步完成；全部 group、全部 layer、全部 rank 都
   成功后模型才会继续执行。

目标 blocks 在完成前只属于该请求，因此校验失败时不会污染其他请求。但在当前未打补丁
的 vLLM 上，HMA post-admission 回滚不被视为安全能力，失败处理见第 8 节。

### 5.5 Store 流程

1. scheduler 在 prompt 对齐边界形成 store plan；已有 quorum 时跳过。
2. full-history group 可在下一个 forward 即将跨过目标时，选择该 forward **之前**已经完成的
   最长全组对齐边界。recurrent/circular group 只能在公开 scheduler 状态恰好位于目标边界时
   保存；一旦调度跨过目标，旧 running state 不可由 block table 反推，必须计为
   `unsafe_boundary` 并放弃 publication。最大上下文 consumer 可以恢复一个较短、已认证的
   安全前缀后继续本地计算；不得为追求更长 cached-token 数猜测已越过的状态。
3. worker 按 profile 顺序把 opaque pages 分片 D2H 到固定 pinned slots，再经固定
   direct-I/O slots 顺序写入临时不可变对象。一个 entry 可以经过任意多轮 slot，不要求
   把整个上下文放进内存。
4. 对象全部写入并 `fsync` 后，写 rank-local manifest 临时文件，再以不覆盖 link 加目录
   `fsync` 原子发布。
5. manifest 可见性与 worker `reporter.add()` 保持在同一个 rank maintenance 临界区；scrub
   只能在线性化点之前完成，或在 add 之后撤销，因此延迟 add 不能复活已隔离 entry。
6. 所有 rank commit 并上报后 scheduler 才建立 quorum。

0.1 用同步路径换取最小状态空间和无补丁正确性。异步 Store 不是当前 Goal；只有新的基准和
设计评审重新立项后，才需要证明 all-group ownership、backpressure、取消和 preemption，
不能只把写盘放进线程。

## 6. 内存与 I/O 设计

### 6.1 内存上限

SpoolCache 不分配与缓存容量成比例的 CPU L1。近似上限为：

```text
M_spoolcache ≈ 2 × 2 slots × 64 MiB
             + bounded_catalog
             + bounded_queues
             + small runtime overhead
```

初始配置假设：

| 项目 | 当前默认值（每 rank） | 性质 |
|---|---:|---|
| pinned D2H/H2D slot | `2 × 64 MiB` | 硬上限 |
| aligned direct-I/O slot | `2 × 64 MiB` | 硬上限 |
| synchronous store concurrency | 每 rank 1 | 由 model-runner 调用串行化 |
| pending restore | 2 | 满时 miss |
| startup digests | 512 | 有界握手 |
| in-memory catalog | 100,000 entries | LRU/有界 |

这些值是内部有界默认值，不是部署调优接口；已经通过短前缀功能资格，但不是已验证的
性能最优值。验收仍须证明长上下文从 32K 增加到目标上限
时，staging 内存不随 payload 总量增长。

### 6.2 避免 page cache 变成隐形 L1

普通 buffered I/O 会把 NVMe 文件放进 Linux page cache；在统一内存设备上，这仍可能
与模型争用物理内存。目标 profile 因而使用：

- `direct_io=required`：对象和 slot 按文件系统块大小对齐，使用 `O_DIRECT`；
- 不对 payload 使用 `mmap`；
- manifest 等小元数据可使用 buffered I/O；
- 其他平台可显式选择 `direct_io=best-effort`，失败时使用顺序 I/O并调用
  `posix_fadvise(..., DONTNEED)`，但该模式不满足“严格内存上限”的生产验收。

CUDA KV 内存本身不向 RDMA/GDS 等外部设备注册。首版使用 PyTorch/CUDA copy 与
gather/scatter 建立正确性基线；原生 CUDA 优化只能作为后续独立、可回退的 fast path。

### 6.3 性能边界与证据门

0.1 把当前同步 Store/Restore、双 slot 内部流水线以及分离的 pinned/direct-I/O pools 视为
正式实现，不把它们定义成等待异步 engine 替换的过渡路径。restore 已使用 metadata-only
manifest probe，并在传输时对每个 payload 完成唯一一次物理读取和 SHA-256/padding 认证；
不会先完整验证再完整读取第二次。

LMCache 的异步 engine、multiprocess server 和多层 backend 证明这些架构可以成立，但它们
不自动成为 SpoolCache 的需求。bounded async Store、layer-wise prefetch、incremental/page-tail
publication、aligned+pinned shared staging、native CUDA mover、压缩和 shared GPU prefix
都只保留为候选，不是活动 Goal。只有同时满足以下条件才单独立项：

- 同条件 benchmark 证明当前实现未达到已经冻结的延迟、吞吐、写放大或内存目标；
- 当前公开 vLLM hook 足以证明 page/block ownership、取消、preemption 和提交边界；
- 新路径可以一次性替代旧路径，不引入模型 profile、pipeline 选择器或两套长期实现。

证据不足时的结论是“不实现”。当前大容量 CPU L1、独立 cache engine/server 和通用 CPU
压缩均不进入 0.1；`restore-only` 可继续作为写入收益不足时的生产策略。

## 7. 持久化格式

当前目录：

```text
<root>/<deployment-identity-digest>/rank-0000/
├── objects/ab/<sha256>.spool
├── manifests/cd/<entry-id>.json
├── quarantine/
├── tmp/
└── state/
    ├── generation.json
    ├── generation.required.json
    ├── deep-scrub.sqlite3
    ├── deep-scrub-request.json  # 仅在定点请求待处理时存在
    ├── spoolcache-events.json
    ├── inventory-withdrawn/<entry-id>/  # 仅在隔离结果待收口时存在
    ├── object-withdrawn/<object-sha256>/ # 共享对象事务 fence
    ├── inventory-owner.lock
    └── maintenance.lock
```

rank-local manifest 至少包含：

```json
{
  "schema": "spoolcache-manifest/v1",
  "entry_id": "<logical-prefix-sha256>",
  "deployment_identity_digest": "<sha256>",
  "rank_identity_digest": "<sha256>",
  "span_tokens": 106752,
  "physical_rank": 0,
  "topology_digest": "<sha256>",
  "profile": "vllm-runtime-kv-v1",
  "layout_digest": "<sha256>",
  "objects": [
    {
      "group_index": 0,
      "layer_name": "<vllm-layer-name>",
      "page_start": 0,
      "page_count": 4,
      "byte_length": 3964160,
      "stored_length": 3964928,
      "sha256": "<sha256>",
      "relative_path": "objects/ab/<sha256>.spool"
    }
  ],
  "created_at_unix_ns": 0
}
```

持久文件禁止包含 CUDA pointer、vLLM allocator block ID、临时 slot index 或请求 ID。
这些值只属于当前进程生命周期，恢复时必须由 connector 根据本次分配重新生成映射。

对象是 content-addressed 且不可变。manifest 是磁盘可见性的唯一提交点。完整 object
写入、`fsync`、不覆盖 link 和目录 `fsync` 与 manifest-last publication 的整个生命期
都受同一 rank-local maintenance lock 保护，因此 scrub/GC 不会把正在发布的
临时文件或对象当成孤儿。connector 在同一锁内完成 manifest publication 与
`reporter.add()`；锁后的 scrub 撤销一定最后生效。
每个 vLLM physical rank 在扫描 startup inventory 前还必须以非阻塞 `flock` 持有
`inventory-owner.lock` 直到 store 关闭；第二个 generation 不能与旧 reporter 重叠。同 root 的
独立 maintenance 进程不占用 lifetime lease，而通过持久 withdrawal marker 把 namespace
变化交给唯一 owner 在下一份 stats report 前消费。owner 不只检查当前 held set，还按固定
页大小和游标轮转 `inventory-withdrawn` namespace；因此 worker 离线期间已删除 manifest
留下的 marker 也会最终确认，不会永久积累。每页内先修改 reporter 并构造 wire report，
再清除已证明 manifest absent 的 marker。startup/rescan 对完整 namespace 流式验证，只有健康
offer 才进入 O(catalog limit) 的 newest heap；不能先截取最新原始 paths 再过滤，否则较新的
tombstone/corrupt manifest 会让较旧健康 entry 饥饿。scan 不得在 active `scandir` stream 内
rename 同目录项；只延迟固定批坏路径到所有 stream 关闭后隔离，其余由后续 scan/scrub 收敛。
扫描不得更新 manifest LRU。

每个启用 SpoolCache 的 worker（`restore-only` 或 `read-write`）都启动一个低优先级
深度 scrubber。它使用固定、与模型无关的
64 MiB/s 物理读取上限、64 MiB step 和每 step 64 个 work item；新 store 首轮最早
在 60 秒后启动，完整周期间隔为 6 小时。这些是内部运维上界，不是模型 profile、
connector 选项或环境变量。

周期开始事务只记录 cutoff、清空旧 work table 并进入 `snapshot_manifests`，不遍历目录。
manifest/object/tmp 各自的 snapshot phase 在 maintenance lock 内每 step 最多流式读取 64 个
原始路径，并用 `INSERT OR IGNORE` 提交到 SQLite work queue；目录 iterator 只存在于当前
进程且可在条目边界取消。进程重启或另一个 maintenance 实例推进 cycle 后，从该 namespace
开头重扫，已提交路径幂等去重，不持久化不稳定的 filesystem directory offset。每个 object
完成后继续持久化 manifest digest、object cursor、引用集和计数器。重启后从已提交的 object
边界继续，不信任上次进程的部分 hash。当前 schema 的损坏状态会隔离并重建；未知/未来
schema 保留原文件并 fail closed，不得静默降级。

scheduled scrub 关闭时先发 cancellation，再最多等待固定 5 秒，并返回
`spoolcache-scrub-shutdown/v1` 的 `stopped`/`timeout` 收据。timeout 的结构化日志在主路径同步输出，
并由 daemon finalizer best-effort 持久累计 `spoolcache_scrub_shutdown_failures_total`；不允许该
counter 的 fsync 反向阻塞主关闭路径。connector 不会
关闭仍被 reader 使用的 store，而由 daemon finalizer 在 scrub 线程真实退出后释放资源。daemon 不阻止
进程退出，因此慢盘或卡住 syscall 不再造成无界 join，也不会换成 use-after-close。该超时是模型
无关内部常量，不增加配置或环境变量。

对每个 manifest，scrub 必须在打开 payload 前先匹配 deployment、rank identity、
physical rank、topology、profile 和 layout，然后验证每个对象是普通文件、
stored/logical 长度正确、logical SHA-256 匹配且对齐 padding 全零。损坏对象和
所有引用它的 manifest 在同一窄范围锁下移入 quarantine，并同步从 worker
reporter 撤销。已知坏 entry 在尝试 rename 前先持久创建
`state/inventory-withdrawn/<entry-id>/`；即使 rename 先失败，浅层 inventory scan 也只能
把它视为 miss。独立 maintenance 不能在只更新自己实例的 callback 后清除标记；唯一 inventory
owner 先从 reporter 移除对应 entry，manifest 已不存在时才确认该信号。live-manifest 标记只能
由重新捕获并成功发布的 replacement commit 或带最终 manifest-version 线性化点的全 payload
认证清除，不能由 metadata rescan 清除。entry marker 不推断或记录单个故障来源；因此修复
一个 object 只能清除该 digest 的 object fence，不能顺带清除引用 manifest 的 entry marker，
因为同一 manifest 可能还有另一损坏 object。全 manifest 认证确实清除 marker 时必须触发
inventory rescan，恢复可用性不能依赖另一次偶然 mutation。发现同 content address 的现存对象字节损坏时，先
durably 创建 `state/object-withdrawn/<sha256>/`。lookup、startup scan 和每次 reporter report
都必须遵守该 fence；随后流式遍历全部 manifest、逐项写 entry marker/撤销 offer，不能把磁盘上
无上限的共享引用积累为内存 list。这样进程在第一个引用后终止，尚未遍历的引用仍不能准入。
然后 hardlink 保存坏 inode，最后用已 fsync 的临时对象原子替换 live name 并 fsync shard；
任一边界失败都保留 fence/marker，绝不留下可报告的 manifest 指向缺失或已知坏对象。
只有完整 repair/quarantine receipt，或 deep scrub 在最终 manifest 版本锁内重新 hash fenced
object，才能释放 object fence。marker、managed namespace 与 shard 的 `mkdir` 即使在幂等
重试中看到已有合法目录，也必须重新 fsync parent/ancestor，补全上一次可能失败的 durability
receipt。确认 absent entry 前先
fsync 对应 manifest shard 并在锁内重查；确认无引用 object fence 前先 fsync 固定 256 个
canonical manifest shard namespace 并重扫引用，marker/fence 的删除不能比它所证明的 namespace
mutation 更持久。无引用 fence 仍按固定页大小复核并回收。
使用 vLLM 原生 stats channel 的部署还必须等待 scheduler 看到 quorum
下降；纯空闲期可用与目标无关的 bypass 请求作为 report barrier，不能直接回放
可能已损坏的 entry。

孤儿对象只在完整 manifest namespace pass 结束后才成为删除候选，真正 unlink 前还要
在 maintenance lock 内重建当前 live reference 并二次核对。临时文件必须同时是
SpoolCache 管理的窄路径、非 symlink 且超过安全年龄才能清理；未识别的 managed
路径移入 quarantine 而不跟随链接。容量管理超过 high watermark 时，先回收已证明的
crash orphan，再按 manifest LRU 逐出健康 entry，直到 low watermark；每次删除都保持
引用二次核对，quarantine 不会被容量 GC 当成可重用 cache。

离线深度校验不能只证明 manifest 与 payload 自洽。`verify_entry_content.py` 的调用者必须
同时提供预期 entry/span、deployment identity、rank identity、physical rank、topology
和 layout digest；工具自行强制内部 layout protocol，且在打开任何 payload 前逐项比较，只有全部一致后才允许输出
`all-payloads-authenticated`。因此来自其他部署或其他 rank 的完整 manifest 不能被误报为
当前目标的有效缓存。

## 8. 故障语义

| 故障时点/类型 | 0.1 行为 |
|---|---|
| entry 不存在、rank 不齐、身份不符 | 普通 miss |
| 启动 inventory 尚未收敛 | 普通 miss，后续报告完成后可命中 |
| inventory/pre-admission 检查发现 manifest 损坏或对象大小不符 | quarantine、撤销 rank offer、普通 miss |
| manifest JSON 触发整数/递归上限或非有限规范化错误 | 归一为数据损坏、quarantine 并继续扫描健康 offer；不吞系统/内存错误 |
| store 写失败或磁盘满 | 不发布 manifest；撤销本 rank commit |
| store 中进程崩溃 | 临时文件和孤儿对象下次 GC，旧 manifest 不受影响 |
| generation state 缺失（首次升级） | 进入高于旧 wall-clock epoch 的持久域并写 generation+sentry |
| generation state 在 sentry 已存在后缺失，或状态损坏/溢出 | worker 启动失败；scheduler 遇未知较小 UUID 也撤销旧 image |
| 同 rank 新旧 worker generation 重叠 | 新 worker 无法取得 lifetime inventory-owner lease，启动失败 |
| 一个 rank 重启或失联 | 持久 generation epoch 增大，立即失去旧 quorum |
| 定期或定点深度 scrub 发现 payload 损坏 | 对象与全部引用 manifest quarantine、撤销 offer；等 scheduler quorum 收敛后请求 miss |
| 已知损坏但 manifest quarantine rename 失败 | 持久 withdrawal marker 保持 miss，重试完成前 rescan 不得准入 |
| 共享损坏对象遍历引用时进程退出 | 预先持久化的 object fence 使 lookup/scan/report 全部拒绝该 digest；引用流式恢复，不依赖已遍历数量 |
| 损坏 content-address object 的 evidence/link/replace/fsync 失败 | live object 原子切换且 object fence/entry marker 保留到完整认证 |
| 修复一个多对象 manifest 中的单个 object | 只清对应 object fence；无 provenance 的 entry marker 保留到整 manifest 认证/替换 |
| marker `mkdir` 或 manifest unlink 的父目录 fsync 失败 | 幂等重试重新 fsync；absence/reference receipt 持久化并锁内复核后才清 marker/fence |
| object shard 初始化/parent fsync 持续失败 | publication 失败，完整 `.part` 仍在同一 finally 中删除并 fsync `tmp/`，重试不累积；cleanup 双失败保留 primary durable-state 原因 |
| scrub/GC 进程或 rank 在中途崩溃 | snapshot phase 从 namespace 开头幂等重扫，payload cursor 从最后完整 object 恢复；未知结果不进入 offer |
| scheduled scrub 在固定关闭超时后仍存活 | 返回结构化 timeout、持久累计失败；主退出不等待，store 延迟到 reader 真正结束后关闭 |
| 容量超过 high watermark | 先清理已证明 crash orphan，再按 LRU 清理到 low watermark |
| **命中承诺后**发生 read/hash/H2D/group 错误 | **抛出致命恢复异常并停止 engine；恢复策略由部署编排器负责** |
| 未知 vLLM/profile/layout | 启动失败，不能带风险继续运行 |

这里的“命中承诺”指 `get_num_new_matched_tokens()` 已向 scheduler 返回正 token 数并进入
同步加载。当前固定 vLLM 的 HMA 多组失败回滚没有满足本设计的证明条件，因此 0.1：

- 配置 `kv_load_failure_policy="fail"`；
- post-admission 失败不向 scheduler 宣称“已成功完成”；
- 不依赖 `get_block_ids_with_load_errors()` 触发 HMA recompute；
- 绝不继续使用可能被部分写入的 KV；
- SpoolCache 不实现进程/容器 supervisor；需要自动恢复的部署由 Docker、systemd、Kubernetes
  或其他外部编排器重建完整 TP 组。

这会牺牲罕见损坏场景的可用性，但不会牺牲推理正确性。只有未来 upstream vLLM 提供
经过 HMA、多 rank 和 speculative-output 测试的事务回滚能力后，SpoolCache 才能通过
compatibility gate 开启 transparent recompute；不会为此维护私有 vLLM patch。

## 9. vLLM 集成契约

### 9.1 Connector 回调映射

| vLLM V1 hook | SpoolCache 职责 |
|---|---|
| `register_kv_caches` | 发现 tensors、layers 和 HMA group geometry，验证 profile |
| `get_handshake_metadata` | 有界上报本 rank 的启动 inventory 与 generation |
| `set_xfer_handshake_metadata` | PP=1 scheduler 建立启动 quorum |
| `set_xfer_handshake_metadata_pp_aware` | PP>1 按 `(pp_rank,tp_rank)` 接收全部 worker，交叉核对 global rank 后建立启动 quorum |
| `get_num_new_matched_tokens` | 无副作用地选择当前确实可用的最长全-rank 前缀 |
| `update_state_after_alloc` | 记录本次 load 的所有 group 目标 blocks |
| `build_connector_meta` | 下发 load/store transaction |
| `start_load_kv` / layer hooks | forward 前同步完成 restore；并在滑窗页仍有效时完成 store |
| `wait_for_save` | store 已在 pre-forward 完成；此 hook 不再触碰可能已循环复用的页 |
| `request_finished_all_groups` | 同步实现不延迟 HMA block 释放 |
| `get_kv_connector_stats` | 上报 generation、inventory checkpoint/delta 和 commit |
| `update_connector_output` | 更新 quorum、撤销过期或失败 entry |

兼容层统一使用 automatic-contract gate，没有 strict/auto 或版本范围选项。构造器语义
前缀必须仍为 `(self, vllm_config, role, kv_cache_config)`，HMA 完成 hook 必须仍接收
`(self, request, block_ids)`；SpoolCache 使用或覆写的 18 个基础 public callback 都必须存在，
并逐个检查参数名、顺序、kind、必填性以及基类到 SpoolCache override 的调用可替代性。
PP>1 另外要求并检查 PP-aware handshake callback；缺少或签名漂移在模型分配前失败，PP=1
仍可运行没有该新增 callback 的兼容 runtime。

检查会构造基类允许的最小、全 positional、全 keyword、`*args` 和任意 `**kwargs` 调用形状；
只要 override 无法接收其中任一种，模型加载前就失败。因此新增可选参数也不会因“基类
仍兼容”而被误放行，除非 SpoolCache 已通过同名参数或对应 variadic 参数接住。版本字符串
只进入诊断和 deployment identity；真正的 build proof 是对实际 importable vLLM package
逐稳定文件计算 SHA-256，忽略 `.pyc`/`__pycache__` 等运行时缓存，并检测 hash 期间的文件
集合或内容变化。automatic-contract 通过只说明接口可以接入，不能代替保存/恢复、故障
注入和性能资格测试；未知布局仍由运行时语义检查独立 fail-close。

### 9.2 启动配置示例

```json
{
  "kv_connector": "SpoolCacheConnector",
  "kv_connector_module_path": "spoolcache.vllm.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "spoolcache_root": "/cache/spoolcache",
    "spoolcache_deployment_namespace": "production",
    "spoolcache_access_mode": "read-write",
    "spoolcache_direct_io": "required",
    "spoolcache_max_bytes": 214748364800
  }
}
```

布局和 API 兼容发现始终启用且不可按模型或版本配置。chunk/span、两个 64 MiB staging
slot、pending restore/store、startup inventory、report batch、catalog 上限和 90% GC low
watermark 都是经过验证的内部有界默认值，不再暴露环境变量或 connector JSON 选项。

模型 locator/revision namespace 在 connector 启动时自动生成，不接受 launcher 提供的
checkpoint digest。`spoolcache_deployment_namespace` 仍是 operator 控制的 tenant/发布边界，
不是模型兼容 profile。

旧版 `spoolcache_profile`、`spoolcache_expected_vllm_version`、
`spoolcache_qualified_gpu_mover`、`spoolcache_runtime_compatibility` 以及上述低层调优字段
已经删除；环境生成器也不再输出对应字段。
G3c 将 deployment identity 从 `spoolcache-deployment/v1` 提升到
`spoolcache-deployment/v2`，而跨角色 transport 仍严格保持
`spoolcache-coordination/v1`。旧 deployment/model-profile 目录的 digest 与 v2 不同，因而
自然 clean miss；启动和维护都不隐式删除旧目录。运维可在确认回滚窗口结束后通过正常流程
归档或清理。

启用前还必须满足其中一种 allocator 条件：

```bash
unset PYTORCH_CUDA_ALLOC_CONF
```

或使用经本部署验证且被 vLLM 接受的 cuMem allocator。当前 compose 默认设置
`expandable_segments:True`，所以需要一个显式的 SpoolCache 启动分支取消它；这是配置
变化，不是 vLLM 源码修改。关闭 SpoolCache 时保持现有 allocator 配置不变。

请求可通过 vLLM 已公开透传的 `kv_transfer_params` 显式设置
`{"spoolcache_bypass": true}`。只有 JSON 布尔值 `true` 生效；它让该请求同时跳过
SpoolCache restore 和 Store，但不关闭 vLLM 自己的进程内 GPU prefix cache。这个能力用于
一次性不持久化请求和性能测试的 GPU-local 对照，不需要 vLLM 补丁，也不会改变未携带该
字段的默认行为。

隔离开发环境还可以在启动前设置 `VLLM_SERVER_DEV_MODE=1`，然后调用
`POST /reset_prefix_cache?reset_running_requests=false&reset_external=false`。这会使 vLLM
遗忘进程内 GPU prefix cache（已分配 KV 内存仍留给 block pool 复用），但不调用
connector 的 `reset_cache()`，因此 NVMe entry 保留，可在不重启模型的情况下验证外部
restore。SpoolCache 提供 `benchmarks/reset_gpu_prefix_cache.py`，把两个布尔参数写死为安全
值，避免手工把 `reset_external` 写反；多模态资格使用 `--multimodal` 同时清理 processor
和 encoder cache。该 vLLM 开关同时暴露 RPC/debug 开发端点，正式部署必须保持关闭；
无法开放时继续使用超过 GPU KV 容量的独立工作集自然淘汰。

## 10. 当前包结构

```text
spoolcache/
├── pyproject.toml
├── README.md
├── docs/
│   └── SPOOLCACHE_DESIGN.md
├── src/spoolcache/
│   ├── config.py
│   ├── identity.py
│   ├── prefix.py
│   ├── manifest.py
│   ├── store.py
│   ├── buffers.py
│   ├── gpu.py
│   ├── hma.py
│   ├── admission.py
│   ├── quorum.py
│   └── vllm/
│       ├── connector.py
│       ├── compat.py
│       └── config_json.py
└── tests/
    ├── test_gpu_mover.py
    ├── test_manifest_store.py
    ├── test_quorum.py
    └── ...
```

公开命名约定：

- distribution/package：`spoolcache`
- connector class：`SpoolCacheConnector`
- 环境变量前缀：`SPOOLCACHE_`
- manifest schema：`spoolcache-manifest/v1`
- Prometheus 前缀：`spoolcache_`

## 11. 实施阶段

尚未完成工作的执行顺序、验收条件和完成记录统一维护在
[`TODO_GOALS.md`](TODO_GOALS.md)。本设计不再维护另一套 Phase 0–5 roadmap；已完成能力、
已提交过渡代码、退场条件和下一目标都以该文件的实现对账为唯一状态源。历史实验结果仍保留
在性能记录，不把 probe/planner 测试描述成已实现产品能力。

### 11.1 已完成的双机资格收据

| 项目 | 实测结果 |
|---|---|
| image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| model revision | `DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9` |
| topology/layout | TP=2、PP=1、5 HMA groups、170 layers、alignment=256 |
| 数据面 | 两 rank 均 `O_DIRECT=True`，每 rank 128 MiB pinned + 128 MiB aligned I/O slots |
| 文本跨重启 | 40,033-token 的 early-secret 提示在两 rank 写入 `554c68f71046`（38,912 cached tokens）；重启后 rank 0/1 均 restore，仍正确回答开头信息 |
| 视觉跨重启 | 32,637-token 四色图提示在两 rank 写入 `b3d93f0f6a66`（31,744 cached tokens）；完整双机进程重启后 rank 0/1 均 restore |
| 视觉进程内状态隔离 | 临时诊断中依次清空 vLLM prefix、encoder 和 multimodal cache，仍由 `b3d93f0f6a66` 正确恢复；诊断端点随后关闭并确认 404 |
| 图片隔离 | 相同文本、salt 和 placeholder 几何换另一张图，不命中 `b3d93f0f6a66`，两 rank 写入独立 entry `3d7d5e2777d5` |
| 输出一致性 | 文本 early-secret 输出一致；视觉冷/热均识别同一四色内容（greedy/speculative 路径存在无语义影响的标点差异） |
| 测试 | 固定目标容器内 49/49 tests 通过，含 4 个 CUDA page-mover/损坏测试和 2 个真实 vLLM layer-hook/runtime 契约测试 |

单请求观察到 40K 文本冷/跨重启热 25.30s/1.59s，以及 32K 视觉冷/跨重启热
21.26s/1.17s。不同图片的同文本请求为 21.27s 且生成不同 entry，证明没有误命中。
这些数字只证明路径和收益方向，不是正式 benchmark；样本数为 1，且包含 API、调度和
解码开销。

资格过程中曾发现并隔离一条由早期实验迭代生成、输出不正确的视觉 manifest
`6e72e954f5da`。它已在两个 rank 上移出可命中目录，payload 未删除并可供取证。当前收据
只以之后重新生成、通过本机三类缓存清空和完整双机进程重启的 `b3d93f0f6a66` 为依据。

G3c 已删除误导性的 checkpoint pin 接口和未接线 runtime planner，同时保持
`spoolcache-coordination/v1`；超大 namespace 的有界实现、Gemma TP=1 全模态功能和 G4
TP=1/PP=2 双机全模态资格均已完成。下一开发目标是 G6 不可变 artifact 发布。长上下文部署
资格独立记为 Q1，不阻塞开发；性能候选没有证据前不立 Goal。

## 12. 测试与验收

凡以下测试需要启动真实模型，除 DeepSeek/Qwen/GLM 兼容性矩阵外，统一使用
`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`。同一 Goal 不再另选
普通 attention 小模型、HMA 小模型或按功能维护模型 profile；普通/HMA 语义的细粒度反例由
model-independent CPU fixtures 覆盖，真实服务用统一模型完成端到端收据。若该 checkpoint
在目标 vLLM 上暴露未知语义，正确结果是 automatic gate fail closed 并修复通用契约，不能
加入 Gemma 特例。

### 12.1 正确性测试

- 同 token、同 identity 命中；任一 identity 字段改变必须 miss。
- 离线 payload 校验只有在 deployment/rank/topology/profile/layout 全部与调用者预期一致时
  才能成功，且身份不符必须先于 payload I/O 失败。
- 最长前缀选择正确，未对齐尾部由模型计算。
- hit 与无缓存 greedy 输出一致；抽样比较恢复前后 KV page digest。
- TP=2 两个 rank、全部 HMA group 都完成才可解除请求等待。
- 同图片内容与几何可命中；不同图片或 placeholder 几何不得串缓存。
- prompt embedding、adapter、缺 salt 等非支持请求必须 bypass。
- vLLM/container 重启后仍能从 NVMe 恢复。

### 12.2 崩溃与故障注入

- 在 object write、object fsync、manifest write、rename、directory fsync 各点 kill。
- 删除/截断/翻转 payload，破坏 manifest JSON 或摘要。
- 单 rank 丢 entry、重启、generation 变化、delta 丢包/乱序。
- staging slot 获取失败、NVMe 满、只读文件系统、I/O timeout。
- post-admission load error 必须导致明确的 engine fatal 和整组重启，不能 hang，也不能
  返回使用部分 KV 的结果。

### 12.3 内存与性能

候选验收线只有在另立性能 Goal 时，才由已归档收据和同条件 benchmark 校准后冻结：

- pinned pool 不超过配置值；connector 私有 RSS 峰值不超过已解释预算的 110%。
- 32K、107K 和最大资格长度之间，staging 内存无 payload-size 线性增长。
- `direct_io=required` 的 payload I/O 不造成持续增长的 page cache。
- miss lookup 的 scheduler 开销 p99 小于 2 ms。
- warm-hit TTFT 必须显著优于完整 prefill；至少对 32K 以上目标工作负载有净收益。
- 同步 store 对稳定 decode throughput 的影响目标不超过 5%，超出则默认
  `restore-only`。
- 最大上下文资格必须从已通过长度有界递增，每个请求同时监控全部参与主机的 SSH、API、
  rank readiness、`MemAvailable` 和 `SwapFree`，在固定收据阈值前主动终止。一次请求的
  `loads=[]/stores=[]` 只能证明没有走 SpoolCache 数据面，不能证明模型 prefill 不会耗尽
  统一/系统内存。曾导致主机硬重启的长度不得直接重放。

### 12.4 零补丁验收

- 发布物不包含 vLLM patch 文件或安装目录覆盖脚本。
- 对干净的目标 image 只执行 `pip install spoolcache` 和配置变更即可启动。
- 启动时记录 vLLM 版本、校验 installed-package content SHA-256 与 connector/override API
  签名；不匹配只允许 fail-fast，不能自动修改。
- CI 对目标 vLLM commit 做 API/ownership/failure contract tests。

### 12.5 参考项目测试吸收原则

2026-09-04 对以下上游版本完成了专项测试评估：

| 参考项目 | 复核提交 | 适合 SpoolCache 的主要能力 |
|---|---|---|
| DeepSeek-v4-Flash-DSpark-2x-DGX-Spark | `9923a9b` | 真实服务 A/B、精确 token 长度、TTFT/decode 分离、内存/swap/日志、长上下文与视觉稳定性 |
| LMCache | `59a56a2`（`dev`） | cold/store/warm/retrieve 正确性、目标缓冲区清零、逐操作延迟、缓存压力工作负载、内存池测试 |
| SparkCache | `5c3bd3e` | 崩溃一致性、单次读取/哈希、异步写回所有权、线上干扰测试和机器可验证资格收据 |

吸收时遵循以下原则：

- 复用测试思想和可观察契约，按 SpoolCache 的固定 staging、rank-local NVMe 和
  patch-free V1 connector 重新实现，不机械复制内部结构。
- 单元测试默认不要求 GPU、vLLM 服务或两台机器；真实 CUDA、双 rank 和服务重启放入
  integration/qualification 层。
- benchmark 必须同时验证正确性和性能。只有 TTFT 变快、但没有 cache state、checksum、
  token 数和输出证据的结果不算通过。
- cold、store、warm、restart-hit、corrupt 必须是显式状态，不能从延迟大小反推命中。
- LMCache 和 SparkCache 均为 Apache-2.0；若直接复用代码，必须保留文件级版权与许可证
  声明。SpoolCache 优先独立实现其测试契约，保持 provenance 清晰。

### 12.6 已完成的 P0 正确性与 I/O 回归基线

#### 单次认证恢复

SparkCache 明确测试“manifest probe 后每个对象只读一次”和“已认证对象只哈希一次”。
SpoolCache 已改为 `lookup(verify_payloads=False)` 的 metadata probe，restore 对同一 payload
只执行一次物理读取，并在这次传输中完成 SHA-256 与 padding 认证。现有回归契约覆盖：

- 统计每个对象的 `read`/`readv` 次数、实际物理读取字节和 SHA-256 调用；
- 正常 restore 的 `physical_read_bytes / logical_restore_bytes` 接近 1，而不是 2；
- 覆盖小对象、跨多轮 slot 的对象、对齐填充、短读和所有 HMA group；
- 对象读取到固定 I/O staging 时流式计算摘要，摘要与 padding 均验证通过后才对该对象
  执行 H2D/scatter；
- 如果后续对象在命中承诺后损坏，仍遵循第 8 节 fail-stop/整组重启契约，不能返回部分
  恢复结果。

后续优化不能取消该完整性验证，也不能为了保留整个 entry 而引入随上下文增长的内存。

#### 完整持久化故障矩阵

现有 manifest fault 测试已像 SparkCache 一样覆盖发布协议的可中断边界，并持续回归：

- 每个对象写入前后、对象 `fsync` 前后、link/rename 前后和对象目录 `fsync`；
- manifest 临时文件写入、`fsync`、link/rename 和 manifest 目录 `fsync`；
- 多对象 entry 在写完第 0 至第 N 个对象后的进程丢失；
- 临时文件、孤儿对象、已存在同内容对象和同 entry 并发发布；
- commit 前故障在重启后必须完全不可见；可见性点后的不确定提交只能是完整、可校验的
  hit，不能是 partial hit；
- 每个故障点都必须验证新进程可安全重试；相同内容重试幂等，不同内容冲突 fail closed。

#### Cold/warm 数据自检

采用 LMCache server-bench 的核心顺序，并验证所有物理 rank：

1. cold lookup 必须是 full miss；
2. 对源 KV 按 rank/group/layer 或对象计算 checksum；
3. store 必须在所有预期 rank 成功；
4. warm lookup 必须是 full hit；
5. **恢复前显式清零目标 blocks**，防止旧 KV 造成 checksum 假阳性；
6. retrieve 后重新计算 checksum，与 cold 源逐项比较；
7. 分别记录 `cold.lookup`、`cold.store`、`warm.lookup` 和 `warm.retrieve`，不能只记录总
   请求耗时。

该流程还要增加“完整 TP 组停止并重启后 warm hit”，证明结果来自持久存储而不是 vLLM
进程内 prefix/encoder/multimodal cache。

#### 机器可验证资格收据

正式 benchmark 统一输出带 schema 版本的 JSON receipt，并提供独立 validator。receipt
至少包含：

- SpoolCache/部署仓库 commit、vLLM installed-package SHA-256、容器 digest、模型不可变
  revision；
- profile、access mode、direct-I/O、固定 staging、容量和 TP/PP/DCP/DP/rank topology；
- 每请求原始样本、输入/输出 token、TTFT、decode 时长、完成时间和错误；
- cold/warm 状态、`cached_tokens`、每 rank lookup/store/restore 结果和 checksum/output
  SHA-256；
- p50/p95/p99、输入/输出吞吐、逻辑与物理 I/O 字节；
- 运行前后 RSS、MemAvailable、swap I/O、cache 指标及 engine/NCCL/JIT 错误计数。

validator 必须从原始 rank/request 样本重算汇总值，并拒绝缺 rank、样本重复、配置不一致、
汇总值低报或缺少重启证据的 receipt。

### 12.7 P1：双 DGX Spark 真实服务性能矩阵

DeepSeek DSpark 的评估方法作为端到端基线：prompt 长度必须经 `/tokenize` 验证；cold
请求使用唯一 nonce；避免会使 speculative acceptance 异常下降的机械重复 prompt；首轮
CUDA graph/编译/autotune 只用于 warmup；通过 API `usage.completion_tokens` 统计输出，
TTFT 与首 token 之后的 decode throughput 分开计算。

正式矩阵至少包含：

| 维度 | 用例 |
|---|---|
| 上下文 | 1K、8K、32K、128K；再加入当前部署资格最大长度 |
| 并发 | C1、C2、C6；容量/干扰测试可增加 C8 |
| SpoolCache 状态 | disabled、restore-only miss、read-write cold/store、同进程 hit、完整 TP 重启后 hit、corrupt |
| 前缀模式 | 完全相同前缀、共享前缀不同尾部、多轮增长会话、超过容量后的 LRU/thrash |
| 输出 | greedy 确定性短答案和固定长度 decode 两类 |

每个矩阵单元保留逐请求结果并报告 p50/p95/p99；A/B 两侧固定模型、revision、镜像、
sampling、prompt、salt 和启动配置。比较前先测量环境自然噪声，再冻结回归阈值，避免只凭
单次样本或任意百分比判定。

视觉 KV cache 增加独立 live 矩阵：

- 相同文字、相同图片、相同 placeholder 几何：应跨完整 TP 重启命中；
- 相同文字但图片内容改变：必须 miss；
- 图片相同但 placeholder offset/length、图片顺序或模态改变：必须 miss；
- 恢复前清零目标 KV，恢复后验证 page checksum、`cached_tokens` 和确定性颜色/对象答案；
- 在长文本 KV 存在时穿插视觉请求，再继续相同文本前缀，验证两条路径互不污染。

长稳与容量测试循环执行 context ladder、并发 soak、写入、恢复、淘汰、rank 重启和损坏
隔离；要求 staging/RSS 有界、swap I/O 无持续活动、quorum 最终收敛，并检查没有持续增长
的临时文件、孤儿对象、quarantine、worker report backlog 或中途 JIT/engine 错误。RSS
资格必须由独立采样进程以固定短间隔读取 workload 的 `/proc/<pid>/status` 当前 VmRSS，并以
workload 就绪后的 current baseline 计算峰值增长；Linux `ru_maxrss` 会继承父进程历史高位，
只能保留为诊断，不能参与通过判据。负向测试必须证明低于旧 `ru_maxrss` 的 native 瞬时分配
仍会被采到。scrub SQLite 必须在轻量 `start_cycle()` 后、每个增量 snapshot step 后和后续
每个处理 step 后采样，且 receipt 回归必须实际观察到 pre-vacuum 峰值高于完成事务删除
work table 并 FULL auto-vacuum 后的低水位。

### 12.8 未来异步 Store RFC 的强制证据（非当前 Goal）

当前没有异步 Store 开发目标。只有新的同条件 benchmark 先证明同步路径不满足已冻结目标，
并通过单独设计评审后，才可参考 SparkCache 的 async manager-page runtime 和 page-tail
interference 测试。届时不能把同步路径简单移进后台线程就宣称完成，至少必须证明：

- ring/slot 满时 admission 立即跳过可选 store，推理线程不等待容量；
- block/page ownership 从 reserve、CUDA submit、event fence、writer handoff 到 release
  的状态转换无重叠所有者；
- cancellation、preemption、CUDA query/sync error 和 shutdown 不会过早释放仍被读取的
  manager page；未知 ownership 不能报告 finished；
- 已完成请求的通知不等待慢磁盘 writer；但 writer 持有的数据在 durable commit 前不能
  被关闭或复用；
- 大上下文后台落盘期间，用同步 barrier 发起短请求 cohort，对比基线与 overlap 的
  p50/p95/max TTFT，并验证输出 sentinel；
- 多轮“大请求后接并发小请求”后，running/waiting、delayed request、retained page、
  uncertain rank 和 KV usage 必须在超时内全部归零。

### 12.9 明确不吸收的测试

- LMCache CacheBlend、任意位置 chunk reuse、共享 suffix recompute 和 long-document
  permutation 测试不属于 0.1 exact-prefix 语义。
- SparkCache 依赖私有 vLLM patch 的 HMA recompute/rollback、shared GPU prefix lease、
  page-tail CoW 测试不能作为当前 patch-free 实现的验收要求。
- DeepSeek 部署仓库中只针对特定 vLLM hotfix、XGrammar 或工具调用格式的测试继续留在
  部署级回归，不复制到 SpoolCache 核心包。
- 远端 backend、P/D disaggregation、跨节点复制和 operator 测试在相应能力进入产品
  目标前不引入，避免扩大 0.1 的状态空间。

## 13. 可观测性

SpoolCache 通过 vLLM 公开的 connector metrics hook 注册指标。worker 与 scheduler 之间只传
`spoolcache-metrics/v1` 的 JSON-compatible delta/gauge；API 进程预绑定全部允许的 label tuple，
运行时不能动态创建 label。histogram 每次 stats 周期最多保留 4,096 个 observation，超出部分
进入固定 label 的 dropped counter，不允许监控积压随请求数无限增长。

| 指标 | 类型 | SpoolCache 自定义 label / 聚合语义 |
|---|---|---|
| `spoolcache_lookup_total` | counter | `result,reason`，只允许代码中声明的有限组合 |
| `spoolcache_hit_tokens_total` | counter | 无；只计已 admission 的 external token 数 |
| `spoolcache_restore_bytes_total` / `spoolcache_store_bytes_total` | counter | 无；所有物理 rank 的逻辑 payload bytes |
| `spoolcache_restore_seconds` / `spoolcache_store_seconds` | histogram | 无；固定 buckets |
| `spoolcache_store_skipped_total` | counter | `reason`：duplicate/budget/busy/unsafe_boundary/error |
| `spoolcache_post_admission_failure_total` | counter | `phase`：lookup/span/manifest/payload/unknown |
| `spoolcache_rank_quorum_entries` | gauge | scheduler 当前全 rank 交集 |
| `spoolcache_rank_generation_changes_total` | counter | scheduler 观察到的 generation 替换 |
| `spoolcache_pinned_pool_bytes` / `spoolcache_disk_bytes` | gauge | 按 reporting physical rank 求和 |
| `spoolcache_managed_disk_bytes` / `spoolcache_quarantine_bytes` | gauge | 全部 managed tree 与不可准入 quarantine 字节，按 rank 求和 |
| `spoolcache_delayed_store_requests` | gauge | scheduler 尚未到安全同步边界的请求数 |
| `spoolcache_quarantined_entries_total` | counter | `reason`：manifest_io/manifest_validation/object_collision/payload_checksum/payload_size/unknown |
| `spoolcache_scrub_payload_bytes_total` / `spoolcache_scrub_objects_total` / `spoolcache_scrub_manifests_total` | counter | 实际认证的物理字节、对象和 manifest，所有 rank 求和 |
| `spoolcache_scrub_cycles_total` / `spoolcache_scrub_objects_quarantined_total` / `spoolcache_scrub_failures_total` | counter | 完整周期、scrub 隔离对象与非预期 step 失败 |
| `spoolcache_scrub_namespace_items_total` / `spoolcache_scrub_shutdown_failures_total` | counter | 增量 snapshot 实际检查的目录项；超过固定 join timeout 的关闭失败（后者跨 worker 重启持久） |
| `spoolcache_orphan_objects_removed_total` / `spoolcache_orphan_bytes_removed_total` / `spoolcache_temporary_files_removed_total` | counter | 只计通过完整 namespace 与最终锁内复核后的安全清理 |
| `spoolcache_required_ranks` / `spoolcache_ready_ranks` | gauge | 拓扑必需 rank 与当前 inventory generation ready rank |
| `spoolcache_readiness` | gauge | `condition`：rank_identity/inventory_quorum/fatal_clear |
| `spoolcache_telemetry_dropped_total` | counter | `kind=histogram` |

Prometheus client 自动产生的 `*_created`、histogram `*_bucket/_sum/_count` 是导出格式的一部分，
不是额外的动态 SpoolCache label。vLLM connector stats 在 scheduler iteration 后才运输；为了
让空闲的新服务也可判定 readiness，API exporter 只从已校验的 topology/access config 初始化
required/ready/identity/quorum/fatal 五项启动 gauge。这个初始化成立的前提是 EngineCore 构造时
`set_xfer_handshake_metadata` 已严格收到每个必需 transport rank 恰好一份 inventory，并核对
rank、`spoolcache-coordination/v1` identity 与完整 rank set；缺失、额外、错 rank 或错 identity
均使 API 启动失败。第一份正常 stats packet 会覆盖这些启动 gauge。

SpoolCache 的可观测性边界止于 connector metrics 和持久 event journal。包内不实现 Docker、
SSH、容器探测、服务 readiness endpoint 或 TP 组重启器；这与 LMCache connector 的职责边界
一致。部署如需 machine-readable readiness，应在外部组合 API `/health`、所有必需 rank 的
liveness 与上述 `spoolcache_readiness`/rank 指标。SpoolCache 的 active-process 指标不能单独
证明其他 rank 存活，也不能替外部编排器宣称整个服务已经恢复。

命中承诺后的不可恢复错误仍固定 fail-stop，避免部分写入 KV 被继续用于推理。外部编排器可按
部署策略重建完整 TP/PP 组；SpoolCache 不规定容器启动顺序、restart policy 或 SSH 拓扑，也
不把这些运维语义带入 connector 配置。G6 发布使用同一个 SHA-256 认证的不可变 wheel，安装到各自固定 runtime 镜像；
启动时不再同步或挂载 SpoolCache 源码。升级、回滚、namespace 兼容策略见 `RELEASE.md`。

功能开发的固定 `google/gemma-4-E2B-it` revision 包含 `audio_config`；当前 vLLM Gemma 4
processor 由该公开配置动态声明 `image`、`video`、`audio`。因此它可用于验证音频输入产生的
语言模型 KV 持久复用，但 SpoolCache 仍只缓存 vLLM 公共 KV layout 中的页，不另行缓存或识别
audio tower/encoder 输出，也不因这项开发基线增加模型或模态白名单。

本地 TP=1/PP=1 功能资格已经对 text、image、audio、video 和 image+audio+video 混合 prompt
完成冷 bypass、三类进程内缓存清空、完整 engine 重启后的 restore、受约束内容 oracle 和
rank-local payload 全量认证。三个单模态恢复 2,048 tokens，混合请求恢复 2,560 tokens；vLLM
注册的 20 个跨 layer KV-sharing 名称均通过 exact storage-view 证明后映射到 15 个真实 owner。
机器可读证据见
[`2026-09-07-gemma4-text-multimodal-cross-restart-summary.json`](receipts/2026-09-07-gemma4-text-multimodal-cross-restart-summary.json)。
这仍是日常单机功能收据。G4 随后在两台 DGX Spark 的 CX-7 双 rail 上，以 TP=1/PP=2
重复上述五条路径：两个 stage 都恢复同一 entry/span，stage-local owner 分别为 12/3，所有
payload 均认证，完整 engine group 重启后输出 hash/oracle 仍与 bypass control 相同。机器收据见
[`2026-09-07-g4-gemma4-pp2-cross-restart-summary.json`](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json)。
资格使用的 `VLLM_PP_LAYER_PARTITION=12,23` 是该固定 Gemma/vLLM 0.28 harness 的 upstream
partition 前提：默认 18/17 在 SpoolCache 初始化前触发跨 stage shared-KV owner 错误，13/22
则在明确 bypass SpoolCache 时也未通过内容 oracle。这个值不进入 SpoolCache 生产代码、配置
或准入规则；新模型仍由 vLLM 最终 tensor 映射和 public process groups 自动发现。

post-admission fatal 和 quarantine counter 先写入 rank-local、crash-consistent、固定 schema 的
小型 event journal，再由替代进程导出，避免 exit 70 前尚未发生下一次 stats drain 而丢失事件。
metrics 和 journal 都不接受 prompt、原始 token、API key、tenant salt、完整 entry ID
或其他请求级高基数字段。日志仍只记录 hit/store/restore 的 entry 前 12 位摘要、span、rank 和
request ID，不记录原始 tokens、prompt、API key 或 tenant salt。

## 14. 主要风险与缓解

| 风险 | 缓解 |
|---|---|
| KVConnector V1 是实验接口，版本漂移 | automatic base/override signature gate、实际 package build SHA-256、contract tests |
| 当前 HMA 恢复失败不能安全重算 | pre-admission 保守校验；post-admission fail-stop |
| 关闭 expandable segments 后模型更易碎片化/OOM | 单独做长稳测试；评估 cuMem；不靠白名单补丁 |
| 同步 store 增加请求延迟 | 最小保存阈值、完整快照测量；超标时默认 restore-only |
| buffered I/O 把 page cache 变成隐形 L1 | `direct_io=required` 强制 `O_DIRECT`，禁止 payload mmap |
| NVMe 耐久度和写放大 | 默认完整快照但带 admission/TTL；测量后再考虑 CoW |
| cache key 跨租户泄漏命中信息 | 强制 namespace/`cache_salt`，目录权限，日志脱敏 |
| 模型管理的 HMA 状态未被完整捕获 | runtime-discovered profile 契约、全组 opaque snapshot、页级 golden test |
| 没有 CPU hot tier 导致小前缀恢复不划算 | 最小 token/byte 阈值，小请求直接 prefill |

## 15. 非活动候选池

以下条目不是待办 Goal，也不计入 0.1 未完成项。只有出现新的产品需求、公共 runtime contract
和可证伪 benchmark 证据后，才允许通过独立设计评审重新立项：

- upstream vLLM 已具备安全 HMA rollback 时启用 transparent recompute；
- page-tail CoW，减少增长会话的 NVMe 写放大；
- 可选原生 CUDA gather/scatter；
- bounded shared GPU prefix；
- buddy replication 和跨节点修复；
- 独立的通用 multimodal encoder-output 缓存；
- 其他统一内存 CUDA 设备和普通离散 GPU。

在重新立项之前，结论统一为“不实现”，不能预先提交 planner、选项、环境变量或模型分支。

## 16. 决策记录

| 决策 | 选择 | 原因 |
|---|---|---|
| 集成方式 | out-of-tree KVConnector V1 | 无需修改 vLLM，部署可插拔 |
| 数据面 | rank-local NVMe | 避免网络传输和大容量 CPU L1 |
| 控制面 | vLLM handshake + connector stats | 无独立 daemon，生命周期跟随 engine |
| 内存模型 | 固定 pinned + aligned I/O slots | 上限与上下文/磁盘容量解耦 |
| 分布式一致性 | 全物理 rank quorum | 任一 shard 缺失都不能安全恢复 |
| HMA | 所有 group 单事务 | 防止部分状态进入 attention |
| 模型布局 | 无 profile 选项，始终运行时发现 | 不按模型名写配置；尺寸从 vLLM 读取，未知缓存语义仍拒绝 |
| vLLM 兼容 | 统一 automatic contract + installed-package SHA-256 | 无版本/模型白名单；base/override 不可替代即 fail-close |
| 模型 artifact | vLLM model locator/revision namespace；artifact 不可变性由发布系统负责 | connector 认证自己的 KV，不复制或逐文件认证 checkpoint |
| 缓存键 | SHA-256 exact prefix | 可重复、跨进程稳定、碰撞风险可忽略 |
| 视觉键 | 媒体 content ID + 模态 + placeholder 几何 | 允许图片 KV 复用且防止跨图片别名 |
| 提交 | immutable objects + manifest last | 崩溃后不会暴露半成品 |
| 存储自愈 | rank-local durable scrub + reporter offer 撤销 | 限速验证全 payload，损坏数据不再进入 quorum |
| scrub 配置 | 固定内部速率/step/周期 | 不增加模型 profile、白名单或运维环境变量 |
| 恢复故障 | pre-admission miss；post-admission fail-stop | 当前无补丁 HMA rollback 不足以证明安全 |
| allocator | 启用时 unset expandable segments 或验证 cuMem | 遵守当前 vLLM 启动检查，不打豁免补丁 |
| 0.1 publication | 完整对齐 prompt snapshot | 先降低状态空间，CoW 后置 |

## 17. 参考资料

- [SparkCache 仓库](https://github.com/FujitsuPolycom/sparkcache)
- [SparkCache 固定研究提交](https://github.com/FujitsuPolycom/sparkcache/tree/66057174301a4759ca3a45207ea41016689449cb)
- [SparkCache package 设计说明](https://github.com/FujitsuPolycom/sparkcache/blob/66057174301a4759ca3a45207ea41016689449cb/sparkcache/README.md)
- [SparkCache 对固定 vLLM 的 patch 说明](https://github.com/FujitsuPolycom/sparkcache/blob/66057174301a4759ca3a45207ea41016689449cb/patches/vllm-e2666d9a6/README.md)
- [SparkCache 单次读取/哈希恢复测试](https://github.com/FujitsuPolycom/sparkcache/blob/5c3bd3eb038cdf0b3ad48629112c1e49112a5d2b/sparkcache/persistent_context_cache/test_cache_manifest.py)
- [SparkCache manifest 事务故障测试](https://github.com/FujitsuPolycom/sparkcache/blob/5c3bd3eb038cdf0b3ad48629112c1e49112a5d2b/sparkcache/persistent_context_cache/test_manifest_transaction_faults.py)
- [SparkCache 异步 store 干扰测试](https://github.com/FujitsuPolycom/sparkcache/blob/5c3bd3eb038cdf0b3ad48629112c1e49112a5d2b/tools/measure_page_tail_interference.py)
- [vLLM 固定提交的 KVConnector V1 接口](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- [vLLM 固定提交的 scheduler](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py)
- [vLLM 固定提交的 allocator/connector 兼容检查](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/vllm.py)
- [LMCache 最终 KV tensor 注册映射（2026-09-07 上游 `dev`）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/integration/vllm/vllm_v1_adapter.py)
- [LMCache runtime KV group 与共享 layer 映射（同一提交）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/integration/vllm/kv_cache_groups.py)
- [LMCache multimodal placeholder identity（同一提交）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/integration/vllm/utils.py)
- [LMCache public vLLM parallel strategy（同一提交）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/integration/vllm/lmcache_mp_connector.py)
- [LMCache 官方架构说明](https://github.com/LMCache/LMCache/blob/dev/docs/source/developer_guide/architecture.rst)
- [LMCache 官方本地磁盘后端说明](https://github.com/LMCache/LMCache/blob/dev/docs/source/kv_cache/storage_backends/local_storage.rst)
- [LMCache 基础 cache key（固定复核提交）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/utils.py)
- [LMCache CacheBlend 可选模型 dispatch（固定复核提交）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/v1/compute/models/utils.py)
- [LMCache CacheGen 可选调优表（固定复核提交）](https://github.com/LMCache/LMCache/blob/dfc2720bb8aaf3edc6b61173018f6aad772c9369/lmcache/v1/storage_backend/naive_serde/cachegen_basics.py)
- [LMCache cold/store/warm/retrieve benchmark case](https://github.com/LMCache/LMCache/blob/59a56a2cb6cdf36ace6661d2ed557f435609969e/lmcache/cli/commands/bench/server_bench/cases/baseline.py)
- [LMCache engine benchmark 统计实现](https://github.com/LMCache/LMCache/blob/59a56a2cb6cdf36ace6661d2ed557f435609969e/lmcache/cli/commands/bench/engine_bench/stats.py)
- [DeepSeek DSpark 评估方法](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/9923a9b9d99a7632071820dc8ed87da309ab7bf5/scripts/EVAL.md)
- [DeepSeek DSpark 稳定性测试](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/9923a9b9d99a7632071820dc8ed87da309ab7bf5/scripts/stability-quick.py)
- [DeepSeek DSpark 部署中的 LMCache 实验说明](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/lmcache/README.md)
- [DeepSeek DSpark 部署的 patch 清单](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/PATCHES.md)
