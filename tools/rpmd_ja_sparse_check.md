# Sparse RPMD-JA reference files

The v2 file uses layout `xyz_soa;csr_output_row_input_col;runtime_translation_projection;cheb_rowmajor_degree_rank;fixed_d0_edges_v1`. CSR rows are outputs and columns are inputs. The three site matrices store `B_alpha^T`; runtime projects translations with `D=P D0 P` and `B_alpha^T=B0_alpha^T P`. Chebyshev vectors are row-major `[degree+1, rank]`. The fixed edge policy is `fixed_d0_raw_nep_image_keys_v1`; metadata records its version, key count, and FNV64 summary of sorted `(center, neighbor, image[3], d0[3])` entries.

V2 wire order is: magic[8], version/u32 endian/u32/atom count/i32, temperature/fd step/fingerprint, null-terminated units and v2 layout, cell[9]/f64 and PBC[3]/i32, raw type[N]/i32, mass[N]/f64, position[3N]/f64, edge-policy version/i32 plus edge-count/fingerprint/u64, then four CSR matrices (`nnz/u64`, row offsets `[3N+1]/u64`, columns `[nnz]/i32`, values `[nnz]/f64`). The tail is spectral bound/U, kernel error[2], actual compressed S2[2], degree/P rank/Q rank, FD diagnostic D/B[4], P values, Q values, P vectors, Q vectors, then EOF. Fixed-size arrays are raw bytes without per-vector count prefixes. V1 retains its original layout and reader.

If the selected table does not cover `tau*sqrt(spectral_bound)`, generation saves the sparse matrices and edge metadata as a pending reference, reports the required U and path, and fails. It has no stability sidecar and cannot be enabled. Rebind to a wider complete table without repeating NEP finite differences, then run the CPU stability check:

```powershell
python tools/rpmd_ja_sparse_check.py pending.rpmdja --kernel-table tests/data/rpmd_ja_kernel_U32.txt --output rebound.rpmdja
python tools/rpmd_ja_sparse_check.py rebound.rpmdja
```

The checker anchors one atom's x/y/z coordinates, constructs the principal minor of `P D0 P`, and performs NumPy Cholesky. The sidecar binds the full reference-file FNV64, atom count, minimum pivot, and reconstruction residual. This is a numerical positive-definiteness check, not an interval certificate. For 5,896 atoms, the anchored minor is 17,685 square: one float64 matrix uses about 2.33 GiB. The checker estimates three such matrices plus CSR and a 256 MiB workspace reserve (roughly 7.24 GiB plus CSR) and performs O(N^3) Cholesky; it is offline CPU work and no system of this size has been certified by the small fixtures.

Generation uses `4*(3N)+1` NEP evaluations (70,753 at N=5,896). It does not allocate a dense `3N x 3N` matrix, but real generation cost depends on the reference graph and NEP model. Reported generation storage is a tracked-capacity estimate, not OS/GPU measurement; opaque NEP private buffers and allocator overhead are excluded.
