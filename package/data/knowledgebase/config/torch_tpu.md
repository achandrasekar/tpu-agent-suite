---
slug: torch_tpu
repo: https://github.com/google-pytorch/torch_tpu
# Pin (release/0.1.2 branch head, 2026-10-12); also recorded in wiki/codebases/torch_tpu/index.md `commit:`.
# `prepare`, `finalize` and `verify` all re-acquire this ref (a `git checkout` on a local
# checkout), so when ingesting from a developer working copy comment it out until the run
# is finished, then restore it.
ref: 6d2a45754bdb52743883f75fbf753408345f6d1b
acquire: clone
wiki_subdir: codebases
source_type: code
# wikify 0.3.0 catalog tier: the symbol index (catalog/symbols/*.tsv, edges/*.tsv, index.md)
# instead of one page per module; `full` profile keeps signatures and doc lines in the rows
# because customers have no C++ source on disk.
catalog: index
index_profile: full
index_shard_depth: 0
languages: [cpp, python]
# The C++ backend is indexed via a bazel-derived compile DB (wikify/bazel_cc.py): `prepare`
# runs `bazel build` + `aquery` on these targets, converts to compile_commands.json, and
# runs scip-clang. First run on a cold bazel cache is slow; cached after.
bazel_targets: "//csrc/..."
index_shards: ["src/torch_tpu/*"]
tests: ["tests/**/*.py"]
docs: ["**/README*.md", "docs/**/*.md"]
# Private repo: the wiki ships without the source. Catalog links, OKF `sources` and change
# pages point at GitHub at the pinned commit; the local checkout (raw/code/torch_tpu) is
# gitignored and excluded from the dist.
source_url: "https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b"
# Keep the shipped catalog focused: tests and vendored code keep a citeable page but drop
# the member body (coverage_collapse); nothing is excluded, so every citation resolves.
coverage_collapse: ["tests/**", "**/*_test.*", "third_party/**"]
agenda: subsystems
agenda_max: 24
# The flat common/, ops/ and eager/ directories split into reference communities named after
# utility hubs (error macros, op names); the python-side _internal/ tree is many small dirs.
# Those planner units are dropped here and replaced by the topic-named units under ## Concepts;
# every module still gets a catalog page (coverage is unaffected by the agenda).
agenda_exclude:
  - csrc/common
  - csrc/ops
  - csrc/eager
  - csrc/internal
  - csrc/experimental
  - csrc/api
  - src/torch_tpu        # the Python package (its units are pinned below); catalog-only otherwise
synthesis_focus: >-
  What a customer onboarding PyTorch code to TPU, or optimizing it there, needs: how an ATen
  op is dispatched and lowered, eager deferred execution and materialization, the
  compilation pipeline and its cache tiers, dynamic shapes, the torch.compile backend,
  distributed collectives and DTensor, precision/dtype handling, Pallas custom kernels,
  and the env vars and APIs that control them.
---

# torch_tpu - knowledge base ingest config

The TorchTPU runtime (PyTorch on Cloud TPU via OpenXLA/PjRt), ingested as the grounded code
silo behind [`wiki/codebases/torch-tpu.md`](../wiki/codebases/torch-tpu.md).

## Concepts
<!-- Discovery-driven: the subsystem planner proposes the units; the entries below pin, rename and add. -->
- **eager-op-dispatch** - seeds: `DispatchOp`, `DispatchOpOptions`, `PromoteScalar`, `MakeTensor`, `DeviceBufferRefArray`
- **deferred-execution-and-materialization** - seeds: (subsystem: csrc/eager::device_buffer)
- **compilation-pipeline** - seeds: `LoadedExecutableWithMetadata`, `CompiledKernel`, `DynamicKernelAdapter`, `ContextedModule`, `MlirComputationBuilderToExecutableBuilder`, `MakeMlirContext`, `Compile`, `GetCompilationSpec`, `MakeCompilationSpecs`, `MakeCompilerOptionOverrides`, `ApplyCompilerOptionOverrides`, `PushCompilerOptionOverrides`, `PopCompilerOptionOverrides`, `MakeCompileOptionsKey`, `SetTpuOptions`, `UpdateFromXlaFlags`, `SetDefaultDeviceAssignment`, `GetCompilerOptionOverridesFromEnvVar`, `ParseEffortLevel`, `CompilationSpec`, `CompilationContext`, `CompilationMode`
- **compilation-cache** - seeds: `CompilationCache`, `GetOrCompile`, `EnqueueCompilation`, `GetOrCreateCacheEntry`, `GetFromTier2OrCompile`, `GetFromTier3OrCompile`, `SetExecutable`, `CacheEntry`, `CacheEntryStats`, `BoundedDynamicCacheEntry`, `CreateDynamicKernelAdapter`, `CreatePaddingKernel`, `CreateSlicingKernel`, `PerfStats`, `CompilationCacheInitializationOptions`, `EvictAll`, `SetCacheOnlyMode`, `SetAllowCacheMode`, `IsExecutableReady`, `GetCacheStats`, `GetNumCompilationThreads`, `SetDumpOnCacheMissMode`, `HbmUsageSummary`, `EnsureInitialized`, `ShouldUseTier2Cache`, `Tier2CacheEntryStats`, `Tier2CacheEntryLock`, `LoadSerializedExecutable`, `AtomicWriteToCacheFile`, `GetTorchTpuBinaryFingerprint`
- **compilation-cache-key** - seeds: `OpParamCacheKeys`, `MakeOpParamCacheKeys`, `EnforceSetParamType`, `PromotedTensorStates`, `CompilationCacheKey`, `GraphKey`, `ShapelessKey`, `CompileOptionsKey`, `EncodeParamCacheKey`, `IncludeInCacheKey`, `IgnoredInCacheKey`, `Fingerprint`, `SetParam`, `Builder`, `GetCacheKey`, `BuildGraphKey`, `PromotedScalar`, `MaybePromotedScalar`
- **dtype-and-precision** - seeds: `ToElementType`, `ConvertTo`, `TorchEquivalentBitwidth`, `RealComponentOf`, `IsValidElemType`, `ValidateTensorByteSize`, `GetAllowExcessPrecision`, `SetAllowExcessPrecision`, `GetDefaultExecutableBuildOptions`, `PySetGlobalPrecision`, `PyGetGlobalPrecision`, `PyPushPrecision`, `PyPopPrecision`, `GetAndAddPrecisionTo`, `ApplyComputationDtype`, `PrecisionContextState`
- **error-handling** - seeds: `TtError`, `TranslateToC10ErrorAndThrow`, `StatusBuilder`, `StatusBuilderWithMessage`, `MaybeAddCppSourceLoc`, `SetRootOpNamePayload`, `GetRootOpNamePayload`, `GetMessageWithCppErrorTrace`, `TorchShowCppStacktraces`, `ReturnIfErrorAdaptor`, `ThrowIfErrorAdapter`, `GetStatus`, `AdaptXlaError`, `IsXlaOomError`, `AdaptExternalErrorMessage`, `GetEnableDebugChecks`, `CheckDeviceIsTpu`, `SafeWrapDim`
- **python-context-and-op-names** - seeds: `OpNameStack`, `ScopedOpName`, `ScopedPythonContextCapturer`, `ScopedPythonContextProvider`, `PythonContext`, `PythonTraceback`, `PythonStackFrame`, `GetTracebackMode`, `CaptureCurrentPythonTrackback`, `MaybeGetContext`, `GetRootOpName`, `RemoveOpSuffix`, `MakeMlirLocation`, `OpName`, `ToString`, `ToBaseName`, `IsMetadataOnly`
- **aten-kernel-authoring** - seeds: `UnaryOp`, `UnaryOpOut`, `UnaryOpInPlace`, `BinaryOp`, `BinaryOpOut`, `TernaryOpOut`, `DispatchBinaryOp`, `DispatchTernaryOp`, `FinalizeOpOutput`, `BinaryOpOptions`, `OutputOpMode`, `CheckTensorsUsed`, `CheckTtKernelArgList`, `CheckKernelArgTypes`, `KernelArgCheckerContext`, `LogKernelArgs`, `LogKernelName`, `ParseArgTypesOrEmpty`, `CollectPromotedScalarPointers`, `InvalidOpParamCacheKeys`, `AppendPromotedScalarPointers`
- **stablehlo-op-builder** - seeds: `GetTensorTypeOrDie`, `GetElementTypeOrSelf`, `GetElementType`, `GetElementTypeOrDie`, `GetMlirType`, `MakeConstant`, `MakeConstantLike`, `MakeScalarConstant`, `MlirOpResults`, `MlirOpResultsTraits`, `BroadcastIfNeeded`, `InferSize`, `GetDims`, `GetDimensions`, `SerializeBytecode`, `SerializePortableArtifact`, `GetVerifiedMinFiniteValue`, `BuildAddShlo`, `IsFloatType`
- **view-decomposition** - seeds: (subsystem: csrc/ops/view_decomposition)
- **scaled-dot-product-attention** - seeds: `flash_attention_backward_kernel#CreateBackwardDkvKernel`, `flash_attention_backward_kernel#CreateBackwardDqKernel`, `flash_attention_backward_kernel#buildBackwardDkvModule`, `flash_attention_backward_kernel#buildBackwardDqModule`, `flash_attention_config#FlashAttnConfig`, `flash_attention_config#FlashAttnConfig.batch_size`, `flash_attention_config#FlashAttnConfig.dropout_rate`, `flash_attention_config#FlashAttnConfig.num_heads`, `flash_attention_config#FlashAttnConfig.padded_kv_sequence_length`, `flash_attention_config#FlashAttnConfig.padded_q_sequence_length`, `flash_attention_config#FlashAttnConfig.vo_head_dim`, `flash_attention_config#Tiling`, `flash_attention_config#Tiling.kt`, `flash_attention_config#Tiling.qt`, `flash_attention_kernel#CreateKernel`, `flash_attention_kernel#buildModule`, `flash_attention_kernel_builder#CreateCustomCallOp`, `flash_attention_kernel_builder#CreateFlashAttentionBackwardKernel`, `flash_attention_kernel_builder#CreateFlashAttentionKernelImpl`, `flash_attention_kernel_builder#CreateFlashAttnConfig`, `flash_attention_kernel_builder#PadSequenceDim`, `flash_attention_kernel_builder#ValidateDtype`, `helpers#EmptyLikeStrides`, `helpers#flatten_batch_dims`, `helpers#get_batch_size`, `helpers#unflatten_batch_dims`, `scaled_dot_product_attention_aten_kernels#AtenFusedSdpChoice`, `scaled_dot_product_attention_aten_kernels#AtenScaledDotProductFlashAttention`, `scaled_dot_product_attention_aten_kernels#AtenScaledDotProductFlashAttentionBackward`, `scaled_dot_product_attention_aten_kernels#AtenScaledDotProductFusedAttentionOverrideable`, `scaled_dot_product_attention_aten_kernels#AtenScaledDotProductFusedAttentionOverrideableBackward`, `scaled_dot_product_attention_shlo#BuildScaledDotProductAttentionBackward`, `scaled_dot_product_attention_shlo#BuildScaledDotProductFusedAttentionShloOp`, `scaled_dot_product_attention_shlo#PrepareAttentionLogits`, `scaled_dot_product_attention_shlo#ScaledDotProductFusedAttentionShlo`, `scaled_dot_product_attention_shlo#ScaledDotProductFusedAttentionShloBackward`, `util#AppendScalarPrefetchTypes`, `util#ConvertElementType`, `util#CreateSymbolTransformIndicesAttr`, `util#GetOpString`, `util#LoadTile`, `util#LoadTileSlice`, `util#ScaleValue`, `util#StoreTile`, `util#ZeroTile`, `splash_attention#splash_sdpa`, `splash_attention#_make_splash_attention_fn`, `splash_attention#_SplashAttentionFn`
- **reductions** - seeds: (subsystem: csrc/ops/reductions)
- **scan-and-custom-kernels** - seeds: (subsystem: csrc/ops::custom_kernels)
- **fused-optimizers** - seeds: `AtenFusedAdam`, `AtenFusedAdamw`, `AtenFusedSgd`, `AtenFusedAdagrad`, `AtenFusedAdamTensorLr`, `AtenFusedAdamwTensorLr`, `AtenFusedSgdTensorLr`, `AtenFusedAdagradTensorLr`, `BuildAdagradShlo`, `BuildAdagradStepForOneParam`
- **host-device-transfer** - seeds: `AtenCopyFrom`, `AtenCopy_`, `CopyCpuToTpu`, `CopyCpuToTpuBuffer`, `CopyTpuToCpu`, `CopyTpuToTpu`, `GetCopyType`, `CopyTensor`, `MaterializeAndReturn`, `BatchTransferH2D`, `BatchTransferD2H`, `TransferFuture`, `Await`
- **torch-compile-backend** - seeds: `compiled_mode#ExtractMlirFromGraph`, `compiled_mode#MakeConstantTensor`, `compiled_mode#MakePlaceholder`, `compiled_mode#PrepareCompiledModeArgumentsImpl`, `compiled_mode#TraverseAndCompile`, `compiled_mode#AsyncCompileResult`, `dispatch_scan#BuildAssociativeScan`, `dispatch_scan#CollectAssociativeScanBuffers`, `dispatch_scan#EmitIdentityInitChloScan`, `dispatch_scan#EmitSliceInitScan`, `dispatch_scan#EvalAdd`, `dispatch_scan#EvalMax`, `dispatch_scan#EvalMin`, `dispatch_scan#EvalNeg`, `dispatch_scan#EvalSubtract`, `dispatch_scan#EvalSymbolicOp`, `dispatch_scan#GetInputBuffers`, `dispatch_scan#GetOutputShapes`, `dispatch_scan#IdentityKind`, `dispatch_scan#ImportStepFunctionCallees`, `dispatch_scan#PyCreateAssociativeScanOp`, `dispatch_scan#PyCreateScanOp`, `dispatch_scan#SymVal`, `dispatch_scan#SymVal.Const`, `dispatch_scan#SymVal.IsSameVar`, `dispatch_scan#SymVal.NegInf`, `dispatch_scan#SymVal.Tag`, `tpu_torch_compile#CompiledExecutableRunner.RunWithGenerators`, `tpu_torch_compile#ForceStrides`, `tpu_torch_compile#PyBuildMlir`, `tpu_torch_compile#PyCompileMlir`, `tpu_torch_compile#PyGetDeviceStateTensor`, `tpu_torch_compile#PyGetOrCompileSliceModule`, `tpu_torch_compile#PyMakeDynamicPlaceholder`, `tpu_torch_compile#PyMakePlaceholder`, `tpu_torch_compile#PyMakePlaceholderLike`, `tpu_torch_compile#PyParseMlirText`, `tpu_torch_compile#PyPrecompilePadModule`, `tpu_torch_compile#PySetDeviceStateTensor`, `tpu_torch_compile#TorchTpuPyBindInitModule_tpu_torch_compile`, `_backend#TpuBackend`, `_backend#make_backend_compiler`, `_backend#async_compile`, `compiler#Compiler`, `compiler#StaticCompiler`, `torch_tpu_compiled_executable#TorchTpuCompiledExecutable`, `torch_tpu_compiled_executable#CompiledArtifact`, `scan#_tpu_associative_scan`
- **dynamic-shapes-compiler** - seeds: (subsystem: src/torch_tpu/_internal/compile/dynamic)
- **eager-dynamic-shapes** - seeds: `dynamism#GetDynamismInfo`, `dynamism#MarkDynamic`, `dynamism_ops#GetDynamicPadModule`, `dynamism_ops#GetPadModule`, `dynamism_ops#GetSliceModule`, `dynamism_ops#MakeDynamicTensorType`, `dynamism_ops#Pad`, `dynamism_ops#Slice`, `dynamism_py#PyGetDynamismInfo`, `dynamism_py#PyMarkDynamic`, `dynamism_py#TorchTpuPyBindInitModule__tpu_torch_dynamism`
- **process-group-and-collectives** - seeds: (subsystem: csrc/distributed)
- **distributed-launchers-and-spmd** - seeds: (subsystem: src/torch_tpu/_internal/distributed)
- **pjrt-client** - seeds: (subsystem: csrc/pjrt)
- **device-backend-registration** - seeds: (subsystem: src/torch_tpu/_internal/device)
- **synchronization** - seeds: `sync#AwaitUniqueDeviceBufferLists`, `sync#BufferRefToVarMap`, `sync#DerefTensor`, `sync#GetComputationGraphviz`, `sync#GetComputationMlir`, `sync#GetTraversal`, `sync#IsMaterialized`, `sync#IsMaterializing`, `sync#NeedsExplicitSyncMaterialization`, `sync#SynchronizeTensors`, `sync#SynchronizeTensorsImpl`, `sync#WaitOnExecution`, `sync#WaitOnExecution.kNo`, `sync#WaitOnExecution.kYes`, `sync_py#AllTensorsAreDenseOnDevice`, `sync_py#GetComputationBuffers`, `sync_py#GetComputationGraph`, `sync_py#PyGetComputationGraphviz`, `sync_py#PyGetComputationMlir`, `sync_py#PyIsMaterialized`, `sync_py#PyIsMaterializing`, `sync_py#PySync`, `sync_py#PySyncSpan`, `sync_py#PySyncSpanNoWait`, `sync_py#PySynchronize`, `sync_py#PySynchronizeDevice`, `sync_py#PySynchronizeSequence`, `sync_py#TorchTpuPyBindInitModule__tpu_torch_sync`, `sync_py#TrySyncSingleTensorFast`, `sync_py#TrySyncTensorArrayFast`, `sync_py#UnwrapTensor`, `op_names#OpName`, `sync#synchronize`, `sync#computation_mlir`, `sync#computation_graphviz`, `sync#dump_computation_mlir`, `sync#dump_computation_graphviz`
- **pallas-custom-kernels** - seeds: (subsystem: src/torch_tpu/_internal/pallas)
- **profiler** - seeds: `profiler_api#PyStartProfilerServer`, `profiler_api#PyStartTrace`, `profiler_api#PyStopProfilerServer`, `profiler_api#PyStopTrace`, `profiler_api#TorchTpuPyBindInitModule__profiler_backend`, `profiler_api#TpuProfilerServer`, `profiler_api#TpuProfilerServer.Start`, `profiler_api#TpuProfilerServer.Stop`, `profiler_api#TpuProfilerServer.mutex_`, `profiler_api#TpuProfilerServer.operator`, `profiler_api#TpuProfilerServer.server_`, `profiler_api#TpuProfilerSession`, `profiler_api#TpuProfilerSession.Start`, `profiler_api#TpuProfilerSession.Stop`, `profiler_api#TpuProfilerSession.mutex_`, `profiler_api#TpuProfilerSession.operator`, `profiler_api#TpuProfilerSession.session_`, `profiler_api#pybind11_init__profiler_backend`, `tpu_profiler_plugin#AddActivityStringMetadata`, `tpu_profiler_plugin#GetProfilerBaseOutputDir`, `tpu_profiler_plugin#ParseAndSetUint`, `tpu_profiler_plugin#ProfilerOption`, `tpu_profiler_plugin#SetAdvancedConfigValue`, `tpu_profiler_plugin#SplitConfig`, `tpu_profiler_plugin#SplitConfigItem`, `tpu_profiler_plugin#TpuKinetoProfilerSession.start`, `tpu_profiler_plugin#TpuKinetoProfilerSession.stop`, `tpu_profiler_plugin#UnescapeString`, `tpu_profiler_plugin#UpdateProfileOption`, `tpu_profiler_plugin#UpdateProfileOptions`, `tpu_profiler_plugin#TpuKinetoProfilerSession`, `tpu_profiler_plugin#TpuKinetoProfilerSession.callback_handler_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.config_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.errors_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.mutex_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.run_dir_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.session_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.worker_rank_`, `tpu_profiler_plugin#TpuKinetoProfilerSession.xspace_`, `tpu_profiler_plugin#TpuProfiler.configure`, `xprof_callback_handler#IsProfilerEnabled`, `xprof_callback_handler#XProfCallbackHandler.Register`, `xprof_callback_handler#XProfCallbackHandler.XProfCallbackHandler`, `xprof_callback_handler#XProfCallbackHandler`, `_api#profile`, `_api#ProfilerActivity`, `_api#xprof_trace_handler`, `_api#register_kineto_backend`, `_impl#start_trace`, `_impl#stop_trace`, `_impl#ProfileOptions`, `profiler_config#TpuProfilerConfig`
- **hardware-introspection** - seeds: (subsystem: src/torch_tpu/_internal/utils)
- **mlir-optimization-passes** - seeds: (subsystem: csrc/mlir)
