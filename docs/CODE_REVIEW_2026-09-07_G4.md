# G4 PP>1 通用拓扑 Code Review

日期：2026-09-07
范围：G4 最终工作树（基线 parent `5dc7f61a`）

## 结论

未发现未关闭的 P1/P2。G4 可以提交。实现保持 `spoolcache-coordination/v1`，PP=1 的
coordination layout 与 rank-ownership payload 保持原字节形状；PP>1 的新增身份和 quorum
只来自 vLLM 公共 topology/process-group、`KVCacheConfig` 与最终 tensor 映射。生产源码没有
模型名称、固定 layer 表、partition 值或模态白名单。

## 复审重点

1. **rank 与 root**：worker 同时读取 PP/TP/DCP public process-group size/rank，和
   `ParallelConfig.rank` 交叉验证 `global_rank = pp_rank * tp_degree + tp_rank`；DCP 只作为 TP
   subdivision。任何 bool、负数、越界、size 不一致或 rank 关系不一致均在创建 store 前失败。
   rank-local root 使用 global rank，避免不同 PP stage 写入同一路径。
2. **stage-local HMA ownership**：所有 stage 保留相同有序 runtime groups；单个 stage 可以在
   某 group 中没有 layer，但整个 worker 必须拥有至少一个 layer。跨 stage coordination 只绑定
   block/reuse/page-selection 语义；layer、shared alias、tensor geometry 和 page bytes 继续绑定
   rank-local identity/layout/manifest。额外注册 tensor 仍只有在 exact Torch storage-view 证明后
   才能作为 alias 排除。
3. **启动与滚动 quorum**：scheduler required set 为完整 PP×TP global rank。PP>1 只接受
   PP-aware `(pp_rank,tp_rank)` handshake；missing/extra coordinate、错 global rank、重复 inventory
   rank、错 coordination identity、错类型或超界 inventory 均使启动失败。运行期任一 stage
   generation 更换先撤销该 rank image，完整 checkpoint 后才重新形成 entry quorum。
4. **兼容 gate**：PP-aware hook 只有 PP>1 必须存在；一旦 runtime 暴露该 hook，基类和 override
   的参数调用可替代性都与其余 connector hooks 一样在模型分配前检查。没有新增版本或模型
   selector。
5. **离线认证**：verifier 在读取 payload 路径前核对 deployment/rank/topology/profile/layout
   digest、global rank 与 PP/TP/DCP coordinate；逐 group 核对 stage-local layer count。空 group
   必须且只能声明 0 pages，非空 group 必须覆盖连续且精确的所有层/页。
6. **PP=1 回归**：PP=1 仍使用 legacy handshake，coordination digest 仍引用原
   `logical_digest`，rank ownership 不加入 worker-coordinate 字段，required rank 数仍为 TP。

## 复审中确认的边界

- Gemma/vLLM 0.28 的 `VLLM_PP_LAYER_PARTITION=12,23` 只记录在资格文档和 receipt；它不在
  `src/spoolcache`、connector 配置或 runtime identity 中。默认 18/17 的 shared-KV 跨 stage
  失败发生在 SpoolCache 初始化前；13/22 在显式 bypass SpoolCache 时也未通过内容 oracle。
- PP worker 的 role-local deployment identity 可以不同；scheduler/worker 只对真正必须跨角色
  一致的 coordination identity 达成一致。把 stage-local layer digest 强行设为全局相同会造成
  正常 PP false negative，因此没有这样做。
- LMCache 的 public rank/final tensor mapping 原则被采用；其物理节点放置假设、独立 encoder
  engine、窄 media hash 投影和非持久 global quorum 不适合作为 SpoolCache 正确性边界。

## 验证证据

- host：249 passed，12 skipped，256 subtests passed；`compileall` 与 `git diff --check` 通过；
- 官方 vLLM 0.28 开发镜像、真实 CUDA：256 tests，1 skipped，OK；
- DeepSeek/Qwen/GLM 目标镜像：各 256 tests，skip 依次为 6/5/5，全部 OK；
- 三个 MiaAI-Lab launcher contract：各 3/3；三个仓库均 main、0 behind/1 ahead、一个本地提交；
- 两台 DGX Spark、CX-7、TP=1/PP=2：text/image/audio/video/mixed 均完成 cold、bypass、三类
  process-local cache reset、restore、完整 PP group restart restore、双 rank payload authentication
  和输出 oracle；详见
  [`receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json`](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json)。

## Review findings

无 P1/P2。未把上游模型 partition 行为误判为 SpoolCache 兼容逻辑，也未为提高当前模型命中率
引入任何模型定制。
