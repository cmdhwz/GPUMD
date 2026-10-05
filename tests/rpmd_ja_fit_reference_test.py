from pathlib import Path
import contextlib
import io
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import struct
import tempfile
import argparse
from unittest import mock
import numpy as np
import rpmd_ja_fit_reference as fit


def test_additive_package_roundtrip_and_rejects_trailing_bytes():
    data = _package()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.bin"
        old_tmp=path.with_name(path.name+".tmp"); old_tmp.write_bytes(b"keep")
        fit.write_additive(path, data)
        assert old_tmp.read_bytes()==b"keep"
        got = fit.read_additive(path)
        assert got["source_fingerprint"] == data["source_fingerprint"]
        np.testing.assert_array_equal(got["sites"][0]["B"], data["sites"][0]["B"])
        try:
            fit.write_additive(path,data)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing coefficient package was overwritten")
        path.write_bytes(path.read_bytes() + b"x")
        _raises(lambda: fit.read_additive(path), "trailing")


def test_triclinic_image_alignment_and_time_unwrap():
    h = np.array([[5., 1.2, 0.3], [0., 4., 0.6], [0., 0., 6.]])
    frac0 = np.array([[.95,.2,.2],[.2,.3,.3]])
    frac1 = np.array([[.04,.2,.2],[.21,.3,.3]])
    p0, p1 = frac0 @ h.T, frac1 @ h.T
    aligned, state = fit.align_bead_frame(p0, p1, h, None)
    np.testing.assert_allclose(aligned[0], p0[0] + np.array([.09,0,0]) @ h.T, atol=1e-12)
    continuous = fit.unwrap_centroid_frame(aligned, h, state)
    assert np.isfinite(continuous).all()
    frac=np.array([[0.,0,0],[.3,0,0],[.6,0,0],[.3,0,0]])
    ring=fit.align_bead_ring([f@h.T for f in frac],h)
    np.testing.assert_allclose(np.mean(ring,axis=0),np.array([.3,0,0])@h.T,atol=1e-12)


def test_collect_rejects_zero_winding_ring_that_disagrees_with_production_centroid():
    cell=np.eye(3)*10.; frac=[0.,.3,.6,.3]
    with tempfile.TemporaryDirectory() as tmp:
        td=Path(tmp); files=[]
        for b,s in enumerate(frac):
            path=td/f"b{b}.xyz"; files.append(path)
            path.write_text(f'1\nTime=0 pbc="T T T" Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:force:R:3\nH {10*s} 0 0 0 0 0\n')
        try: fit.collect(files,300.,td/"samples.npz",td/"mean.xyz",[1.])
        except ValueError as exc: assert "production centroid" in str(exc)
        else: raise AssertionError("collector accepted an unsupported zero-winding centroid branch")
        assert not (td/"samples.npz").exists() and not (td/"mean.xyz").exists()


def test_collect_uses_full_precision_tilted_cell_model_and_rejects_mismatch():
    cell=np.array([[8.123456789,1.2,.3],[0.,7.234567891,.6],[0.,0.,9.345678912]])
    rounded=np.round(cell,8)
    with tempfile.TemporaryDirectory() as tmp:
        td=Path(tmp); model=td/"model.xyz"; model.write_text(
            f'2\npbc="t t t" Lattice="{" ".join(map(str,cell.T.ravel()))}" Properties=species:S:1:pos:R:3\nH 1 1 1\nO 2 2 2\n')
        files=[]
        for b in range(2):
            path=td/f"b{b}.xyz"; files.append(path)
            with path.open("w") as f:
                for step in range(3):
                    shift=cell[:,0] if b else np.zeros(3)
                    f.write(f'2\nTime={step} pbc="T T T" Lattice="{" ".join(map(str,rounded.T.ravel()))}" Properties=species:S:1:pos:R:3:force:R:3\n')
                    f.write(f'H {1+shift[0]} {1+shift[1]} {1+shift[2]} 0 0 0\nO {2+shift[0]} {2+shift[1]} {2+shift[2]} 0 0 0\n')
        samples=td/"samples.npz"; mean=td/"mean.xyz"
        fit.collect(files,300.,samples,mean,[1.,16.],model)
        with np.load(samples) as z: np.testing.assert_array_equal(z["cell"],cell)
        assert f'"{" ".join(map(str,cell.T.ravel()))}"' in mean.read_text()
        bad=td/"bad.xyz"; bad.write_text(model.read_text().replace(str(cell[0,0]),"8.1234567"))
        try: fit.collect(files,300.,td/"bad.npz",td/"badmean.xyz",[1.,16.],bad)
        except ValueError as exc: assert "cell model" in str(exc)
        else: raise AssertionError("collector accepted incompatible full-precision model cell")
        assert not (td/"bad.npz").exists() and not (td/"badmean.xyz").exists()
        nonperiodic=td/"nonperiodic.xyz"; nonperiodic.write_text(model.read_text().replace('pbc="t t t"','pbc="t f t"'))
        try: fit._read_cell_model(nonperiodic)
        except ValueError as exc: assert "fully periodic" in str(exc)
        else: raise AssertionError("cell model accepted an explicit nonperiodic boundary")


def test_additive_symmetry_validation_is_scaled_for_extreme_finite_values():
    for magnitude in (1e308,1e-300):
        data=_package(); block=data["sites"][0]["B"]; block[0,1]=magnitude; block[1,0]=0.
        _raises(lambda:fit._validate_additive(data),"asymmetric")
    data=_package(); data["sites"][0]["B"][:]=np.eye(3)*1e308
    assert fit._validate_additive(data)
    data=_package(); data["sites"][0]["B"][:]=np.eye(3)*1e-300
    assert fit._validate_additive(data)


def test_fit_branch_gate_rejects_continuous_integer_cell_drift_with_fixed_com():
    cell=np.diag([10.,12.,14.]); reference=np.array([[2.,3.,4.],[4.,3.,4.]])
    shift=cell[:,0].copy()
    frame=reference+np.array([shift,-shift])
    np.testing.assert_allclose(np.sum(frame-reference,axis=0),0.)
    # A MIC-folded gate would map both +/- one-cell displacements back to zero.
    fractional=(frame-reference)@np.linalg.inv(cell).T
    np.testing.assert_allclose(fractional-np.rint(fractional),0.)
    _raises(lambda:fit._check_reference_branch(np.array([frame]),reference,cell),"0.45")


def test_cross_block_assembly_matches_directional_derivative_oracle():
    graph = _graph()
    sites = [_site(0, [1,2], [0,0]), _site(1, [0,2], [0,0]), _site(2, [0,1], [0,0])]
    rng=np.random.default_rng(81)
    for s in sites:
        raw=rng.normal(size=(6,6)); s["B"]=(raw+raw.T)/2 + (s["i"]+2)*np.eye(6)
        s["B"][0,4] += .17; s["B"][4,0] += .17
    pack = _package(sites=sites, hessian=[])
    a = fit.assemble_additive(pack, graph["positions"], graph["cell"])
    rng = np.random.default_rng(13)
    x, v = rng.normal(size=(2, 9))
    for alpha in range(3):
        lhs = x @ a["Hadd"][alpha] @ v
        # Independent contraction of the frozen local edge formula in SoA layout.
        rhs=0.
        for site in pack["sites"]:
            i=site["i"]; local=np.zeros(3*len(site["neighbors"]))
            for e,(j,img) in enumerate(site["neighbors"]):
                for mu in range(3): local[3*e+mu]=x[mu*3+j]-x[mu*3+i]
            forcegrad=site["B"]@local
            for e,(j,img) in enumerate(site["neighbors"]):
                edge=graph["positions"][j]+graph["cell"]@np.asarray(img)-graph["positions"][i]
                for mu in range(3): rhs -= edge[alpha]*forcegrad[3*e+mu]*v[mu*3+j]
        assert abs(lhs-rhs) < 2e-11
        assert np.linalg.norm(a["Hadd"][alpha]-a["Hadd"][alpha].T)>1e-6


def test_zero_correction_recovers_native_harmonic_target():
    d = 6
    kt = np.diag(np.r_[np.ones(3)*2., np.zeros(3)])
    ht = np.arange(3*d*d, dtype=float).reshape(3,d,d) / 7
    pack = _package(hessian=[])
    assembled = fit.assemble_additive(pack, np.zeros((2,3)), np.eye(3))
    np.testing.assert_allclose(assembled["Kadd"], 0.)
    np.testing.assert_allclose(assembled["Hadd"], 0.)
    np.testing.assert_allclose(kt + assembled["Kadd"], kt)
    np.testing.assert_allclose(ht + assembled["Hadd"], ht)


def test_graph_linear_center_term_cancels_raw_gradient():
    sites=[_site(0,[1,2],[0,0]),_site(1,[0,2],[0,0]),_site(2,[0,1],[0,0])]
    b=np.array([[.2,-.1,.3],[-.4,.2,-.1],[.2,-.1,-.2]])
    fit._set_linear_balance(sites,b)
    pack=_package(sites=sites)
    graph=_graph(); got=fit.assemble_additive(pack,graph["positions"],graph["cell"])
    np.testing.assert_allclose(got["linear_gradient"],-b,atol=1e-14)


def test_known_recoverable_quadratic_force_fit():
    x = np.array([[1.,0.,-1., 0.,1.,-1., 1.,-1.,0.],
                  [0.,1.,-1., -1.,0.,1., 2.,-1.,-1.],
                  [-1.,-1.,2., 1.,-1.,0., -1.,2.,-1.],
                  [2.,-1.,-1., -1.,2.,-1., -1.,-1.,2.]])
    mass = np.ones(3)
    k = np.diag(np.arange(1.,10.))
    f = -(x @ k)
    got = fit.fit_force_least_squares(x, f, np.zeros_like(k), [k], mass)
    np.testing.assert_allclose(got["theta"], [1.], atol=1e-12)


def test_harmonic_force_target_fits_zero_addition():
    q=np.array([[1.,0.,-1.,0.,1.,-1.,1.,-1.,0.],[-1.,0.,1.,0.,-1.,1.,-1.,1.,0.],
                [0.,1.,-1.,-1.,0.,1.,2.,-1.,-1.],[0.,-1.,1.,1.,0.,-1.,-2.,1.,1.]])
    k=np.diag(np.arange(1.,10.)); force=-(q@k)
    got=fit.fit_force_least_squares(q,force,k,[k],np.ones(3))
    np.testing.assert_allclose(got["theta"],[0.],atol=1e-12)


def test_spectral_cut_qp_closes_minimum_eigenvalue():
    h0 = np.diag([-1., 2.])
    modes = [np.diag([1.,0.])]
    q = fit.solve_psd_qp(np.array([[1.]]), np.array([0.]), h0, modes, 0.25)
    assert q["theta"][0] >= 1.25 - 1e-9
    assert q["min_eigenvalue"] >= .25 - 1e-9
    assert q["cuts"] >= 1


def test_response_rejects_single_bad_direction_across_large_scales():
    temp=300.; kbt=fit.KB_EV*temp
    stiffness=np.diag([1e-8,1.,1.])
    covariance=kbt*np.diag([1e8,1.,2.])
    report=fit.check_independent_response(stiffness,covariance,np.eye(3),.001,temp)
    assert report["passed"] is False
    assert report["max_relative_error"]>=.99


def test_response_rejects_one_bad_direction_in_384_mode_probe():
    temp=300.; n=384; stiffness=np.eye(n); covariance=fit.KB_EV*temp*np.eye(n); covariance[-1,-1]*=2
    report=fit.check_independent_response(stiffness,covariance,np.eye(n),.1,temp)
    assert report["passed"] is False
    assert report["max_relative_error"]>.99


def test_ibp_uses_largest_singular_error_not_dimension_average():
    n=384; temp=300.; q=np.eye(n); force=-fit.KB_EV*temp*n*np.eye(n); force[-1,-1]*=2
    report=fit.check_integration_by_parts(q,force,temp)
    assert report["relative_error"]>.99


def test_baseline_projection_has_independent_relative_gate():
    projector=fit._translation_projector(np.ones(2)); raw=projector+.1*(np.eye(6)-projector)
    got=fit.check_baseline_projection(raw,np.ones(2),.05)
    assert got["relative_change"]>.05 and not got["passed"]
    huge_addition=1e6*np.eye(6)
    assert np.linalg.norm(projector@raw@projector-raw)/np.linalg.norm(raw) == got["relative_change"]
    assert huge_addition.shape==raw.shape


def test_mass_weighted_baseline_asymmetry_gate():
    k=np.zeros((6,6)); k[1,1]=100.; k[0,2]=.1
    cart_error=np.linalg.norm(k-k.T)/max(np.linalg.norm(.5*(k+k.T)),1e-300)
    got=fit.check_baseline_projection(k,np.array([1.,100.]),.05)
    assert cart_error<.05
    assert got["asymmetry"]>.05 and not got["passed"]


def test_independent_response_failure_includes_cross_covariance():
    reference = np.eye(2)
    data = fit.KB_EV*300.*np.array([[1.,.8],[.8,1.]])
    report = fit.check_independent_response(reference,data,[np.array([1.,0.]),np.array([0.,1.])],.1,300.)
    assert report["max_relative_error"] > .1
    assert report["passed"] is False


def test_response_probe_vectors_are_columns_for_rectangular_probe_sets():
    temp=300.; probes=[np.array([1.,0.,0.]),np.array([0.,1.,0.])]
    report=fit.check_independent_response(np.eye(3),fit.KB_EV*temp*np.eye(3),probes,.01,temp)
    assert report["passed"] is True


def test_heldout_force_residual_uses_uncentered_reference_displacement():
    d=np.diag([2.,3.]); q=np.array([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]])+np.array([.25,-.1])
    force=-(q@d.T)
    assert fit._heldout_force_relative_residual(q,force,d)<1e-14
    centered=q-q.mean(axis=0)
    assert fit._heldout_force_relative_residual(centered,force,d)>.1


def test_raw_versions_and_collect_to_fit_synthetic_chain():
    n=2; masses=np.array([1.,3.]); temp=300.; kbt=fit.KB_EV*temp
    cell=np.eye(3)*10.; base=np.array([[2.,2.,2.],[3.,2.,2.]])
    eig=np.array([2.,3.,4.]); kt=np.zeros((6,6)); ktrue=np.zeros((6,6))
    for a,lam in enumerate(eig):
        kval=lam/(1/masses[0]+1/masses[1]); i=a*n
        ktrue[i:i+2,i:i+2]=kval*np.array([[1.,-1.],[-1.,1.]])
        kt[i:i+2,i:i+2]=.5*ktrue[i:i+2,i:i+2]
    z=np.array([[np.sqrt(masses[1]/masses.sum()),0,0],[-np.sqrt(masses[0]/masses.sum()),0,0],
                [0,np.sqrt(masses[1]/masses.sum()),0],[0,-np.sqrt(masses[0]/masses.sum()),0],
                [0,0,np.sqrt(masses[1]/masses.sum())],[0,0,-np.sqrt(masses[0]/masses.sum())]])
    trainvals=[]; validvals=[]
    for a,lam in enumerate(eig):
        amp=np.sqrt(3*kbt/lam); mode=z[:,a]
        trainvals += [amp*mode,-amp*mode]*4
        validvals += [amp*mode,-amp*mode]*2
    train=np.array(trainvals); valid=np.array(validvals); q=np.concatenate([train,valid])
    qcart=(q/np.tile(np.sqrt(masses),3)).reshape(-1,3,2).transpose(0,2,1)
    x=base[None,:,:]+qcart; force=np.empty_like(x)
    for t in range(len(x)): force[t]=-(ktrue@(q[t]/np.tile(np.sqrt(masses),3))).reshape(3,2).T
    with tempfile.TemporaryDirectory() as tmp:
        td=Path(tmp); beadpaths=[]
        for b in range(2):
            path=td/f"bead{b}.xyz"; beadpaths.append(path)
            with path.open("w") as f:
                for t in range(len(x)):
                    pos=x[t].copy()
                    if b: pos[0]+=cell[:,0]
                    f.write(f'2\nTime={t} pbc="T T T" Lattice="{' '.join(map(str,cell.T.ravel()))}" Properties=species:S:1:pos:R:3:vel:R:3:force:R:3\n')
                    for i in range(n): f.write(f"{'H' if i==0 else 'O'} {' '.join(map(str,pos[i]))} 0 0 0 {' '.join(map(str,force[t,i]))}\n")
        samples=td/"samples.npz"; mean=td/"mean.xyz"
        make_temp=fit._unique_temp_path; temp_calls=[0]
        def fail_second_temp(destination):
            temp_calls[0]+=1
            if temp_calls[0]==2: raise OSError("injected second-temp failure")
            return make_temp(destination)
        with mock.patch.object(fit,"_unique_temp_path",side_effect=fail_second_temp):
            try: fit.collect(beadpaths,temp,samples,mean,masses)
            except OSError as exc: assert "second-temp" in str(exc)
            else: raise AssertionError("second temporary-file creation failure was not injected")
        assert not list(td.glob(".*.tmp"))
        real_link=fit.os.link; calls=[0]
        def fail_second_link(src,dst):
            calls[0]+=1
            if calls[0]==2: raise OSError("injected second-install failure")
            return real_link(src,dst)
        with mock.patch.object(fit.os,"link",side_effect=fail_second_link):
            try: fit.collect(beadpaths,temp,samples,mean,masses)
            except OSError as exc: assert "injected" in str(exc)
            else: raise AssertionError("collect installation failure was not injected")
        assert not samples.exists() and not mean.exists()
        assert not list(td.glob(".*.tmp"))
        fit.collect(beadpaths,temp,samples,mean,masses)
        rawpaths=[]
        for version in (1,2,3):
            rp=td/f"raw{version}.qraw"; rawpaths.append(rp); _write_raw_fixture(rp,version,base,cell,masses,kt)
            parsed=fit.read_raw(rp); assert parsed["version"]==version
            for arrname in ("v","c","k","coarse_v","coarse_c"): parsed[arrname]._mmap.close()
            if version==2:
                with rp.open("r+b") as f:
                    f.seek(-18*8,2); stats=np.frombuffer(f.read(18*8),dtype="<f8").copy(); stats[0]=10.; stats[2]=.9; f.seek(-18*8,2); f.write(stats.astype("<f8").tobytes())
                parsed=fit.read_raw(rp); assert parsed["version"]==2
                for arrname in ("v","c","k","coarse_v","coarse_c"): parsed[arrname]._mmap.close()
            if version==1:
                with rp.open("r+b") as f:
                    f.seek(-18*8,2); stats=np.frombuffer(f.read(18*8),dtype="<f8").copy(); stats[2]=.051; f.seek(-18*8,2); f.write(stats.astype("<f8").tobytes())
                _raises(lambda:fit.read_raw(rp),"diagnostics")
                _write_raw_fixture(rp,version,base,cell,masses,kt)
        args=argparse.Namespace(raw=str(rawpaths[-1]),samples=str(samples),output=str(td/"corr.bin"),cutoff=2.5,epsilon=.01,response_tolerance=1e-8)
        fit.fit_cli(args); package=fit.read_additive(args.output)
        assembled=fit.assemble_additive(package,base,cell)
        np.testing.assert_allclose(assembled["Kadd"],.5*ktrue,atol=2e-8,rtol=2e-8)
        assert package["training_frames"]==24 and package["validation_frames"]==12

        shifted=base+np.array([[1e-5,0.,0.],[-1e-5/3,0.,0.]])
        shifted_raw=td/"raw_shifted.qraw"; _write_raw_fixture(shifted_raw,3,shifted,cell,masses,kt)
        shifted_output=td/"shifted.gpjaadd"
        shifted_args=argparse.Namespace(raw=str(shifted_raw),samples=str(samples),output=str(shifted_output),cutoff=2.5,epsilon=.01,response_tolerance=1e-7)
        log=io.StringIO()
        with contextlib.redirect_stdout(log): fit.fit_cli(shifted_args)
        report=log.getvalue()
        shifted_pack=fit.read_additive(shifted_output)
        raw_shifted=fit.read_raw(shifted_raw)
        try:
            prepared=fit.assemble_additive(shifted_pack,raw_shifted["positions"],raw_shifted["cell"])
            import importlib.util
            qprep_spec=importlib.util.spec_from_file_location("qprep_shifted_reference",Path(__file__).resolve().parents[1]/"tools"/"rpmd_ja_qnep_prepare.py")
            qprep=importlib.util.module_from_spec(qprep_spec); qprep_spec.loader.exec_module(qprep)
            prepared_by_prepare=qprep.prepare_additive(shifted_pack,raw_shifted["positions"],raw_shifted["cell"])
            np.testing.assert_allclose(prepared_by_prepare["Kadd"],prepared["Kadd"])
            np.testing.assert_allclose(prepared_by_prepare["Hadd"],prepared["Hadd"])
            train=len(x)*2//3; sqrtm=np.tile(np.sqrt(masses),3)
            trans=np.zeros((6,3))
            for axis in range(3): trans[axis*2:(axis+1)*2,axis]=np.sqrt(masses)/np.linalg.norm(np.sqrt(masses))
            z=np.linalg.svd(trans,full_matrices=True)[0][:,3:]
            baseline=fit.check_baseline_projection(np.asarray(raw_shifted["k"]),masses,.05)
            mass_add=prepared["Kadd"]/(sqrtm[:,None]*sqrtm[None,:])
            internal=fit._translation_projector(masses)
            d0=baseline["projected_D"]+internal@mass_add@internal
            dint=z.T@d0@z
            qval=((x[train:]-raw_shifted["positions"]).transpose(0,2,1).reshape(len(x)-train,-1)*sqrtm)@z
            fval=(force[train:].transpose(0,2,1).reshape(len(x)-train,-1)/sqrtm)@z
            expected=fit._heldout_force_relative_residual(qval,fval,dint)
            printed=float(report.split("heldout_force_relative_residual=")[1].split()[0])
            assert abs(printed-expected)<5e-7, (printed, expected)
            assert expected>1e-7
        finally:
            for arrname in ("v","c","k","coarse_v","coarse_c"): raw_shifted[arrname]._mmap.close()

        for cancelled in (False,True):
            gradient_raw=td/("gradient_cancel.qraw" if cancelled else "gradient_net.qraw")
            _write_raw_fixture(gradient_raw,3,base,cell,masses,kt)
            raw_meta=fit.read_raw(gradient_raw)
            v=np.zeros((6,2)); v[0,0]=1e200
            if cancelled: v[1,0]=-1e200
            with gradient_raw.open("r+b") as f:
                f.seek(raw_meta["start"]); f.write(v.astype("<f8").tobytes())
            for arrname in ("v","c","k","coarse_v","coarse_c"): raw_meta[arrname]._mmap.close()
            gradient_output=td/("gradient_cancel.gpjaadd" if cancelled else "gradient_net.gpjaadd")
            gradient_args=argparse.Namespace(raw=str(gradient_raw),samples=str(samples),output=str(gradient_output),cutoff=2.5,epsilon=.01,response_tolerance=1e-8)
            if cancelled:
                fit.fit_cli(gradient_args)
                assert gradient_output.exists()
            else:
                _raises(lambda:fit.fit_cli(gradient_args),"nonzero translation component")
                assert not gradient_output.exists()


def _write_raw_fixture(path,version,positions,cell,masses,k):
    n=len(masses); d=3*n; pos=positions.T.reshape(-1); zero_v=np.zeros((d,n)); zero_c=np.zeros((3,d,d)); stats=np.zeros(18); stats[16]=.01; stats[17]=version
    with path.open("wb") as f:
        f.write(fit.RAW_MAGIC); f.write(struct.pack("<IIii",version,fit.ENDIAN,n,d)); f.write(struct.pack("<ddQQ",300.,.01,11,22))
        f.write(struct.pack("<iid",1,1,1.)); f.write(fit.RAW_LAYOUT); h=np.zeros(18); h[:9]=cell.ravel(); h[9:]=np.linalg.inv(cell).ravel(); f.write(h.astype("<f8").tobytes())
        f.write(np.ones(3,dtype="<i4").tobytes()); f.write(np.arange(n,dtype="<i4").tobytes()); f.write(masses.astype("<f8").tobytes())
        f.write(pos.astype("<f8").tobytes()); f.write(struct.pack("<d",0.)); f.write(np.zeros(n,dtype="<f8").tobytes()); f.write(np.zeros(d,dtype="<f8").tobytes()); f.write(np.zeros(9*n,dtype="<f8").tobytes())
        f.write(zero_v.astype("<f8").tobytes()); f.write(zero_c.astype("<f8").tobytes()); f.write(k.astype("<f8").tobytes()); f.write(zero_v.astype("<f8").tobytes()); f.write(zero_c.astype("<f8").tobytes()); f.write(stats.astype("<f8").tobytes())


def test_fit_rejects_oversized_raw_before_reading_samples():
    with tempfile.TemporaryDirectory() as td:
        raw=Path(td)/"large.qraw"
        raw.write_bytes(fit.RAW_MAGIC+struct.pack("<IIii",3,fit.ENDIAN,130,390))
        args=argparse.Namespace(raw=str(raw),samples=str(Path(td)/"missing.npz"),output=str(Path(td)/"out.bin"),cutoff=2.,epsilon=.01,response_tolerance=.1)
        _raises(lambda:fit.fit_cli(args),"before qraw scan")


def _raises(fn, word):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        assert word in str(exc).lower()
    else:
        raise AssertionError("expected rejection")


def _site(i, js, ims):
    return {"i": i, "neighbors": [(j, (ims[k],0,0)) for k,j in enumerate(js)],
            "ell": np.zeros((len(js),3)), "B": np.zeros((3*len(js),3*len(js)))}


def _graph():
    return {"positions": np.array([[0.,0,0],[1.,0,0],[0.,1.,.3]]),
            "cell": np.array([[4.,.6,0],[0.,4.,.3],[0.,0,4.]])}


def _package(sites=None, hessian=None):
    if sites is None:
        sites = [{"i":i,"neighbors":[],"ell":np.zeros((0,3)),"B":np.zeros((0,0))} for i in range(2)]
        sites[0]["neighbors"]=[(1,(0,0,0))]; sites[0]["ell"]=np.zeros((1,3)); sites[0]["B"]=np.zeros((3,3))
        sites[1]["neighbors"]=[(0,(0,0,0))]; sites[1]["ell"]=np.zeros((1,3)); sites[1]["B"]=np.zeros((3,3))
    return {"n":len(sites), "beads":4, "source_fingerprint":123,
            "temperature":300., "epsilon":.01,"response_max":0.,
            "response_tolerance":.1,"training_frames":3,"validation_frames":1,
            "sites":sites}


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"): value()
    print("additive fit tests passed")










