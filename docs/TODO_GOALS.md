# SpoolCache Goals

本文是 SpoolCache 未完成工作的唯一状态源。设计原理见
[`SPOOLCACHE_DESIGN.md`](SPOOLCACHE_DESIGN.md)，历史实验和资格收据见
[`PERFORMANCE_IMPLEMENTATION_NOTES.md`](PERFORMANCE_IMPLEMENTATION_NOTES.md)。目标完成时必须
在这里记录 commit、测试环境和收据，不能只在对话中宣布完成。

状态约定：`[ ]` 未开始，`[-]` 进行中，`[x]` 已完成。只有全部验收条件满足且没有未关闭的
P1/P2 review 问题，才能标记完成。

## LMCache 参考边界

SpoolCache 参考 LMCache 的基础 vLLM KV 路径，不复制它的全部产品范围：

| LMCache 基础原则 | SpoolCache 决策 | 当前状态 |
|---|---|---|
| `register_kv_caches()` 接受 vLLM 最终 tensor 映射 | 不维护模型 layer 表或模型 profile；最终映射是 worker 物理事实来源 | 已完成 |
| 从 `KVCacheConfig.kv_cache_groups` 和实际 tensor 建立 engine group | group 内 layer 是 block-table owner；额外注册项只有在证明为相同 Torch storage view 后才作为共享别名排除 | 已完成，`eb1c1a1` |
| 用 vLLM 的 multimodal identifier/placeholder 参与 key | SpoolCache 绑定 identifier、modality、offset、length；缺失稳定事实时只 bypass persistent cache | 已完成 |
| connector/engine/storage 可以在 serving 进程内工作 | 维持 connector、store、mover 的进程内组合，不另建 SpoolCache cache-engine 服务 | 已完成 |
| MP server、controller、远端 backend、CacheBlend 等是可选能力 | 不自动变成 SpoolCache 0.1 目标，也不把其模型 dispatch 带入核心 | 不实施 |

SpoolCache 比 LMCache 基础路径多做的工作仅来自持久化格式本身：rank-local HMA manifest、
全 rank quorum、身份隔离、payload 认证、崩溃一致性和有界 NVMe 维护。这些不是模型适配层，
不能为了“更像 LMCache”而删除。

## 始终成立的约束

- 核心代码零模型、零架构、零模态特例；已验证模型表只作为证据，不参与运行时准入。
- 兼容性资格使用 DeepSeek、Qwen、GLM；除此之外，真实模型测试统一使用
  `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`。纯 CPU/存储测试使用
  model-independent fixture，测试模型不得进入生产 identity 或配置。
- 只使用公开 vLLM V1 connector/HMA 接口，不修改、覆盖或热补丁 vLLM。
- 未知 cache semantic、接口签名、identity、topology、page ownership 或共享关系必须
  fail closed；不能用模型白名单代替证明。
- 保持 `spoolcache-coordination/v1`。内部 manifest/layout schema 只有真实格式变化时才升级。
- persistent hit 必须同时证明 cached-token span、全部必要 rank/stage 的同 entry restore、
  payload authentication 和输出 oracle；延迟或 HTTP 200 不是正确性证据。
- 开发涉及三个 MiaAI-Lab 部署仓库前，先拉取各自 `main` 最新代码并检查工作树；每个仓库
  保持一个 rebased 的本地 SpoolCache 集成提交。

## 当前状态与执行顺序

```text
通用 PP=1 基线（G1/G2/G2.1/G3-core，已完成）
    -> G4：PP>1 通用拓扑（已完成）
         -> G6：0.1 可复现发布与开发态源码同步退场（已完成，PyPI 0.1.0 已发布）

Q1：长上下文/部署资格（独立进行，不阻塞 G4/G6）

性能候选池：无活动 Goal；只有基准证明当前路径不满足目标时才重新立项
```

G1–G4 和 **G6** 核心开发目标均已完成。Q1 是独立部署资格，不要求新增模型适配代码；
原 G5 已撤销为必做目标。

## 0.1 明确不实施

以下项目不计入“未完成目标”，也不得在没有新设计评审和可证伪收益证据时开始实现：

- 模型、架构、layer 名称、模态或 vLLM 版本白名单，以及按模型选择的 profile/fast path；
- 独立部署的 SpoolCache cache engine/server、controller、CPU L1、远端 backend 或后台
  supervisor/readiness 服务；进程/容器重建由外部编排器负责；
- vLLM patch/overlay、私有 scheduler hook 或按目标模型热补丁；
- checkpoint 全目录扫描、逐文件权重 hash、权重复制、模型路径改写或 symlink-free 模型副本；
- 多模态 encoder-output cache；0.1 只持久化 vLLM 公共 KV layout 中的语言模型状态；
- 跨 topology 复用、P/D disaggregation、跨节点缓存复制、RDMA/GDS、Redis/S3 数据面；
- 把异步 Store、layerwise restore、增量/page-tail CoW、shared GPU prefix、共享 staging、
  原生 CUDA mover 或压缩作为预定功能；
- 为 pre-0.1 cache 再建设自动迁移 daemon 或隐式删除旧缓存；只提供明确的兼容说明和人工
  盘点/归档步骤；
- 直接重放曾导致主机失联或硬重启的理论最大上下文；安全门停止本身可以形成资格结论；
- 不少于 24 小时的双机长稳作为 0.1/G4/G6 阻塞条件。用户已要求跳过；将来如需运行，另立
  qualification task，不恢复为核心开发 Goal。

## [x] G1–G3-core：通用 connector 与持久化正确性基线

以下能力已经实现，继续作为回归基线，不再拆成新的开发目标：

- 18 个公开 vLLM connector hook 的签名和 override 可替代性检查，以及实际 importable
  vLLM package 内容摘要；没有 strict/auto、版本或模型选择器。
- 通过公共 semantic-kind resolver、`KVCacheConfig` 和最终注册 tensor 自动发现 HMA
  group、复用语义、dtype、page geometry、rank ownership 与共享别名。
- 从 vLLM multimodal registry 和部署 limits 自动发现启用输入；任意模态走同一 identity 路径。
- 自动 model locator/revision namespace；SpoolCache 认证自己的 KV，不认证整个模型仓库。
- rank-local immutable object、manifest-last、全 rank quorum、generation ordering、容量/GC、
  scrub/quarantine、withdrawal fence、单次认证恢复和 post-admission fail-stop。
- 删除未接线的 `layerwise.py`、`publication.py`、`StoreEconomics` runtime helper，以及包内
  supervisor/readiness 实现；`spoolcache-coordination/v1` 保持不变。
- `google/gemma-4-E2B-it` TP=1/PP=1 的 text、image、audio、video 和三模态混合请求均完成
  本地 cache 清空、跨 engine restart restore、内容 oracle 与全 payload 认证；最终注册的
  20 个共享 KV layer 已映射到 15 个真实 owner。

主要完成证据：

| 范围 | Commit/收据 |
|---|---|
| 自动兼容与三部署资格 | `1d74877`, `40cdccd`, `71e391b`; `CODE_REVIEW_2026-09-05.md` |
| cache semantic 通用化 | `07ae286`, `38fa442`; `CODE_REVIEW_2026-09-06.md` |
| 故障、scrub、容量与身份简化 | G3a `4a06e99`/`35abc08`; G3b `a633f75`..`bb81636`; G3c `51ec9a6`/`581c55b`; G3e 见性能记录 |
| Gemma 共享 KV 与全模态功能 | `eb1c1a1`; `receipts/2026-09-07-gemma4-text-multimodal-cross-restart-summary.json` |

历史 G1/G2/G2.1/G3a/G3b/G3c/G3e 的逐项过程保留在 code-review 和 performance 文档；本文件
只维护当前状态，避免已完成过程淹没剩余工作。

## [x] G4：PP>1 通用拓扑支持

目标：取消当前 PP=1 限制，使 PP stage 与 TP/DCP rank 在不依赖模型 layer 表的情况下形成
正确、完整的持久化事务。

工作项：

- [x] 只从 vLLM 公共 parallel/topology config、`KVCacheConfig` 和最终 tensor 映射派生
  global rank、PP stage、TP/DCP rank、stage-local layer ownership 与 rank-local root。
- [x] 定义 scheduler 可见的 required participant 集合；所有必要 stage/rank 必须报告同一
  entry/span，任一缺失、重复、错 stage 或错 identity 都只能 miss/fail closed。
- [x] deployment/rank/layout identity、manifest 和离线 verifier 绑定 PP stage、全局/本地
  rank 与实际 stage ownership；PP=1 现有 namespace 和行为不得无理由变化。
- [x] 增加 model-independent PP=2 CPU 正负契约，覆盖 stage 缺失、重复 rank、错 ownership、
  部分 manifest、重启 generation 和共享 tensor alias。
- [x] 使用固定 Gemma revision 完成真实 PP=2 text、image、audio、video 与混合模态的 cold、
  bypass、process-local cache reset、跨完整 engine restart restore、payload verifier 和输出
  oracle；若当前硬件不能安全运行，记录明确的外部阻塞，不以模拟结果冒充实机资格。
- [x] 实现完成后再用 DeepSeek/Qwen/GLM 做兼容性回归；不得产生模型专用 stage mapping。
- [x] 删除 PP=1 启动拒绝，更新 README、设计、skill、收据并完成 code review。

验收条件：

- [x] PP=1 全量回归不变；PP=2 所有 stage/rank 才能命中，任一局部状态都不能进入推理。
- [x] 核心 `src/spoolcache` 仍无模型名称、固定 layer 表或 topology 白名单。
- [x] host 全量、目标 vLLM runtime tests、compileall 和 code review 通过，无 P1/P2。

完成证据：双 DGX Spark、CX-7 双 rail 上以 TP=1/PP=2 运行固定 Gemma revision；text、image、
audio、video 和三模态混合请求均完成 bypass=0、进程内 cache reset 后命中、完整双进程重启后
命中、双 rank restore、输出 oracle/hash 相等和全 payload 认证。两 stage 分别持有 12/3 个
最终注册 tensor owner，20 个共享 layer 只按 exact storage-view 关系去重。机器收据见
[`2026-09-07-g4-gemma4-pp2-cross-restart-summary.json`](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json)，
复审见 [`CODE_REVIEW_2026-09-07_G4.md`](CODE_REVIEW_2026-09-07_G4.md)。

后续双机开发入口已收敛到仓库内 `scripts/gemma-pp2-dev.sh`：它复用这份固定 Gemma/G4
基线，提供 preflight、CX-7 image/model sync、双机精确 source snapshot、worker-first
start、整组 restart/status/logs/stop。它只服务 G6 前的开发态源码同步，不恢复已删除的
supervisor 设计，也不把 Gemma/partition 写入核心代码；G6 仍须让生产部署改用不可变 artifact。

## [x] G6：0.1 可复现发布与开发态同步退场

依赖：G4。性能候选池和 Q1 的 24 小时长稳不阻塞 G6。

- [x] 构建一次不可变 wheel，并在开发镜像与三个 MiaAI-Lab 部署中安装同一 artifact；记录
  wheel SHA-256、SpoolCache commit、官方 vLLM image digest 和模型 revision。
- [x] 删除三个生产 launcher 的 `SPOOLCACHE_SOURCE_DIR`、源码 tar/snapshot/symlink、
  `PYTHONPATH`/source bind mount；把 `VLLM_SERVER_DEV_MODE` 留在独立 qualification harness。
- [x] 审计并删除只服务 pre-0.1 过渡状态且已不再需要的代码。不得为旧开发 cache 新增自动
  migration service；旧 namespace 保持不被隐式删除，并在文档中给出人工盘点/归档方法。
- [x] 冻结 manifest/deployment/rank identity 的 0.1 兼容策略、升级/回滚/显式清缓存说明，
  并保留 `spoolcache-coordination/v1`。
- [x] 精简生产配置和环境变量，只保留 root、deployment namespace、access mode、direct-I/O、
  capacity 等实际运维边界；不得加入模型 profile 或独立 engine 配置。
- [x] GitHub Actions + python-semantic-release 管理版本/changelog/tag/GitHub Release，
  经测试的同一 wheel 使用 PyPI Trusted Publishing 发布；记录首次发布所需的账号侧绑定。
- [x] 用固定 Gemma revision 完成 release-candidate 可复现安装、PP=1/PP=2 功能与故障回归；
  DeepSeek/Qwen/GLM 只完成兼容性矩阵。
- [x] 输出支持矩阵、性能/正确性收据、许可证和已知限制；完成最终 code review，无 P1/P2。

完成收据：`docs/receipts/2026-09-08-g6/README.md`。首次发布 workflow `34180558915`
通过，公开 wheel SHA-256 `abaf71af05afd41c0d5a2873a003bc94af3a65b54dbe9db97c1015b2dd01bd28`。
压缩后的初始提交为 `80a19db`；PSR 自动 release commit/tag 为 `40528e6` / `v0.1.0`。
公开 wheel 的 22 个包文件和 metadata headers 与完成实机资格的候选逐字节相同；最终两节点
8 份镜像安装认证通过，runtime/live 收据按这项精确 payload 等价证明承接，不以版本字符串
冒充同一 artifact。PP=2 长 image/mixed 的语义限制、远端故障需要部署端整组停止的限制已
明确记录；测试结束两节点停止，缓存根目录保留。

已有但不等于 release artifact 的开发证据：root `Dockerfile`/`compose.yaml` 使用官方
`vllm/vllm-openai:v0.28.0`，Gemma TP=1 全模态收据已通过。G6 不再重复开发这些功能。

## [-] Q1：长上下文与部署资格（非阻塞）

Q1 不改变通用代码范围，也不作为 G4/G6 的依赖。它只补充“哪些具体部署已验证到什么安全
长度”的证据。

- [x] 百万级 namespace 的分步、可取消、可恢复 scrub 与有界 shutdown/storage soak。
- [x] Qwen 260,800-token consumer 和 160,000-token external prefix restore；不安全的
  260,800-token producer 按 `unsafe_boundary` 拒绝。
- [x] DeepSeek 196,608-token consumer 和 130,048-token 跨重启 restore/payload 认证。
- [ ] 以有界递增长度完成 GLM 的最高安全资格；不得重放已造成两台主机硬重启的 992,769
  token 请求。若安全门先触发，记录最高通过长度和停止原因即可完成该子项。
- [ ] 归档 GLM 原始 receipt，完成该资格变更的 review，并恢复部署服务/仓库状态。

不少于 24 小时的 Gemma read-write/restore-only 双机长稳已从验收条件删除，不计为未完成。

## 性能候选池（不是 Goals）

同步 Store/Restore 和分离的 pinned/direct-I/O pools 是当前正式、已验证的有界实现，不是等待
删除的“临时代码”。以下候选只有同时满足三项条件才新建 Goal：同条件 benchmark 证明当前
路径未达目标；vLLM 公共 ownership contract 足够；替代路径能一次性取代旧路径而不增加模型
或 pipeline 选择器。

- bounded async Store；
- layerwise restore/prefetch；
- incremental/page-tail publication；
- aligned+pinned shared staging；
- native CUDA mover、压缩、shared GPU prefix。

若证据不足，决定是**不实现**，而不是保留两个长期可选实现。LMCache 的相应可选能力只提供
测试和接口设计参考，不构成 SpoolCache 的功能清单。

## 完成记录

| Goal | 状态 | 主要证据 | 备注 |
|---|---|---|---|
| G1–G3-core | 已完成 | 上述提交、code review 和 performance receipts | 通用 TP=1、TP=2(HMA) 基线与持久化闭环 |
| G4 | 已完成 | G4 Gemma PP=2 收据与 code review | PP×TP 全 participant quorum；PP=1 identity 稳定 |
| G5 | 已撤销 | 本文件“不实施/候选池” | 不再计入未完成目标 |
| G6 | 已完成 | `docs/receipts/2026-09-08-g6/` | PyPI/GitHub 0.1.0 已发布；两节点四种镜像安装同一公开 wheel |
| Q1 | 进行中、非阻塞 | Qwen/DeepSeek/G3d receipts | 仅剩 GLM 安全最大资格；24 小时项已删除 |
