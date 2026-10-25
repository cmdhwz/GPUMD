from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "src/measure/rpmd_ja_qnep_prepare.cu").read_text(encoding="utf-8")
cuda_test = (ROOT / "tests/rpmd_ja_qnep_prepare_cuda_test.cu").read_text(encoding="utf-8")
integration_test = (ROOT / "tests/rpmd_ja_fit_prepare_integration_cuda_test.cu").read_text(encoding="utf-8")
cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
header = (ROOT / "src/measure/rpmd_ja_qnep_prepare.cuh").read_text(encoding="utf-8")

# Every H tile is passed to gesvd with its actual shape. The workspace must
# therefore cover the full tile and each possible tail-shape combination.
assert "cusolverDnDgesvd_bufferSize(solver,m,n,&lwork)" in source
for required_shape in (
    "query_svd_workspace_elements(solver,full_extent,full_extent)",
    "query_svd_workspace_elements(solver,tile_size,tail)",
    "query_svd_workspace_elements(solver,tail,tail)",
    "if(m<n||n<=0)",
    "const int m = std::max(rows, cols), n = std::min(rows, cols)",
):
    assert required_shape in source, f"workspace query omits an actual H tile shape: {required_shape}"

# Failures must retain enough context to identify the failing H tile and the
# two distinct cuSOLVER/CUDA status layers, with devInfo only when readable.
for required in (
    "CUSOLVER_STATUS_EXECUTION_FAILED",
    "H direction",
    "tile row",
    "tile column",
    "ldvt",
    "lwork",
    "cudaPeekAtLastError",
    "devInfo",
):
    assert required in source, f"missing SVD failure diagnostic field: {required}"

api_failure = source.split("if(svd_status!=CUSOLVER_STATUS_SUCCESS){", 1)[1].split(
    "cuda_check(cudaDeviceSynchronize(), (\"synchronize SVD", 1)[0]
assert "svd_status_allows_dense_fallback(svd_status)" in api_failure
assert "cuda_status==cudaSuccess" in api_failure and "info_readable" in api_failure
assert "host_info==unwritten_info||host_info>=0" in api_failure
assert "disable_h_compression=true" in api_failure and "result=dense_tile" in api_failure
assert "throw std::runtime_error(diagnostic)" in api_failure
fallback_status = source.split("bool svd_status_allows_dense_fallback", 1)[1].split("int query_svd_workspace_elements", 1)[0]
for allowed_status in ("INTERNAL_ERROR", "EXECUTION_FAILED", "NOT_SUPPORTED", "ARCH_MISMATCH"):
    assert f"CUSOLVER_STATUS_{allowed_status}" in fallback_status
for rejected_status in ("INVALID_VALUE", "NOT_INITIALIZED", "ALLOC_FAILED"):
    assert f"CUSOLVER_STATUS_{rejected_status}" not in fallback_status
assert "if(host_info<0)" in source and "if(host_info>0)" in source
assert "qNEP H SVD input contains a non-finite value" in source
assert "remove_file(tmp_path)" in source and "created==output_path" in source

for required in (
    "constexpr int n=260",
    "shape_128x128",
    "shape_128x12",
    "shape_12x128",
    "shape_12x12",
    "run_ill_conditioned_h_tile_edges(base)",
    "run_positive(base,260,4)",
):
    assert required in cuda_test, f"missing native H SVD boundary regression: {required}"
assert "--prepare-svd-regression" in integration_test
assert "rpmd_ja_fit_prepare_integration_cuda_test --prepare-svd-regression" in cmake

for fault in (
    "ApiInvalidValueUnwrittenInfo",
    "ApiInvalidValueWrittenInfo",
    "ApiInvalidValueCudaStatusUnavailable",
    "ApiInternalErrorUnwrittenInfo",
    "ApiExecutionFailedZeroInfo",
    "ApiNotSupportedUnwrittenInfo",
    "ApiInternalErrorNegativeInfo",
    "ApiInternalErrorCudaStatusUnavailable",
    "NegativeDevInfo",
    "PositiveDevInfo",
    "NonFiniteFactor",
    "CorruptReconstruction",
    "NonFiniteReconstruction",
):
    assert fault in header and fault in source and fault in cuda_test, f"missing native SVD fault regression: {fault}"
assert "run_svd_fault_injection_regressions(base)" in integration_test
assert "raw_input_before==read_file_bytes(raw)" in cuda_test
assert "GPUMD_RPMD_JA_QNEP_PREPARE_FAULT_INJECTION_TEST" in cmake
