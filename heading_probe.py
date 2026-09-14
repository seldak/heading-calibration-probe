"""A model-explicit heading calibration experiment, not a general calibrator.

Model: z(t) = theta(t + tau) + phi + iid N(0, sigma**2).
The unwrapped reference theta(t), including its absolute offset, is KNOWN.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def theta(t, acceleration, omega=0.8, theta0=0.2):
    return theta0 + omega * t + 0.5 * acceleration * t**2


def sensitivity(t, tau, acceleration, omega=0.8):
    return np.column_stack((omega + acceleration * (t + tau), np.ones_like(t)))


def identifiability(jacobian, sigma, parameter_scales, rank_rtol=1e-10, weak_rtol=1e-2):
    """Local SVD of a whitened, dimensionless Jacobian.

    Scales mean p = diag(scales) u. Directions returned are in u coordinates.
    Noise is known iid with scalar std sigma. Numerical rank is not a global
    identifiability or practical accuracy guarantee.
    """
    J = np.asarray(jacobian, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if sigma <= 0 or np.any(scales <= 0):
        raise ValueError("Noise standard deviation and parameter scales must be positive")
    A = J * scales[None, :] / sigma
    _, s, vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > rank_rtol * s[0])) if s.size and s[0] else 0
    weak = [i for i in range(rank) if s[i] < weak_rtol * s[0]]
    return {
        "rank": rank,
        "parameter_count": J.shape[1],
        "singular_values": s.tolist(),
        "rank_relative_threshold": rank_rtol,
        "weak_relative_threshold": weak_rtol,
        "parameter_scales": scales.tolist(),
        "constrained_scaled_directions": vt[:rank].tolist(),
        "null_scaled_directions": vt[rank:].tolist(),
        "weak_constrained_scaled_directions": vt[weak].tolist(),
        "worst_constrained_scaled_std": float(1 / s[rank-1]) if rank else None,
        "individual_parameters_supported": rank == J.shape[1],
    }


def svd_covariance(jacobian, sigma, parameter_scales, rank_rtol=1e-10):
    """Full-rank local covariance and log information determinant via SVD.

    Work with the whitened, scaled Jacobian directly: normal equations square
    its condition number. A deficient Jacobian does not support this covariance.
    Returned covariance is in physical parameter units; the determinant uses
    the declared scaled coordinates.
    """
    scales = np.asarray(parameter_scales, dtype=float)
    if sigma <= 0 or np.any(scales <= 0):
        raise ValueError("Noise standard deviation and parameter scales must be positive")
    A = np.asarray(jacobian, dtype=float) * scales[None, :] / sigma
    _, singular_values, vt = np.linalg.svd(A, full_matrices=False)
    if (len(singular_values) < A.shape[1] or not singular_values.size
            or singular_values[-1] <= rank_rtol * singular_values[0]):
        raise ValueError("Individual-parameter covariance requires full numerical rank")
    B = scales[:, None] * vt.T / singular_values[None, :]
    covariance = B @ B.T
    logdet = 2 * np.log(singular_values).sum()
    return covariance, float(logdet)


def fit_heading(t, z, acceleration, sigma, scales=(0.1, 0.1), omega=0.8):
    """Return only an estimable combination if separate tau,phi are unsupported."""
    y = z - theta(t, acceleration, omega)
    # Structural rank is independent of tau for this quadratic reference model.
    report = identifiability(sensitivity(t, 0., acceleration, omega), sigma, scales)
    if report["rank"] < 2:
        combination = {"expression": "omega * tau + phi",
            "omega_rad_per_s": omega, "estimate_rad": float(y.mean()),
            "std_rad": float(sigma / np.sqrt(len(t)))} if acceleration == 0 else None
        return {"status": "individual_parameters_not_identifiable" if acceleration == 0 else "numerically_rank_deficient",
                "acceleration_rad_per_s2": acceleration, "tau_s": None,
                "phi_rad": None, "individual_parameter_covariance": None,
                "estimable_combination": combination,
                "local_report": report}
    slope, intercept = np.linalg.lstsq(np.column_stack((t, np.ones_like(t))), y, rcond=None)[0]
    tau = slope / acceleration
    phi = intercept - omega * tau - 0.5 * acceleration * tau**2
    J = sensitivity(t, tau, acceleration, omega)
    report = identifiability(J, sigma, scales)
    cov, _ = svd_covariance(J, sigma, scales)
    return {"status": "locally_identifiable", "acceleration_rad_per_s2": acceleration,
            "tau_s": float(tau), "phi_rad": float(phi),
            "individual_parameter_covariance": cov.tolist(),
            "local_std_tau_s": float(np.sqrt(cov[0, 0])),
            "local_std_phi_rad": float(np.sqrt(cov[1, 1])),
            "local_report": report}


def selection_experiment(acceleration=0.2, sigma=0.03, tau=0.12):
    # All segments cost 6 observations and 0.5 seconds; select 4 of 12.
    segments = np.arange(12)[:, None] + np.linspace(0., 0.5, 6)[None, :]
    choices = list(combinations(range(len(segments)), 4))
    def evaluate(choice):
        t = segments[list(choice)].ravel()
        J = sensitivity(t, tau, acceleration)
        covariance, logdet = svd_covariance(J, sigma, [0.1, 0.1])
        expected_std = sigma / (abs(acceleration) * np.sqrt(np.sum((t-t.mean())**2)))
        np.testing.assert_allclose(np.sqrt(covariance[0, 0]), expected_std, rtol=1e-12)
        return float(logdet), float(np.sqrt(covariance[0, 0]))
    metrics = np.array([evaluate(c) for c in choices])
    best = int(np.argmax(metrics[:, 0]))
    even = (0, 4, 7, 11)
    even_metrics = evaluate(even)
    # Random baseline is the EXACT distribution over uniformly sampled subsets.
    result = {
        "candidate_segments": 12, "selected_segments": 4,
        "acceleration_rad_per_s2": acceleration,
        "observations_per_segment": 6, "duration_per_segment_s": 0.5,
        "number_of_subsets_enumerated": len(choices),
        "selection_score": "2 * sum(log(singular_values(J_scaled_whitened)))",
        "even_subset_zero_based": list(even), "even_tau_std_s": even_metrics[1],
        "information_subset_zero_based": list(choices[best]),
        "information_tau_std_s": float(metrics[best, 1]),
        "random_tau_std_s_quantiles_10_50_90": np.quantile(metrics[:, 1], [.1, .5, .9]).tolist(),
        "information_gain_determinant_over_even": float(np.exp(metrics[best, 0] - even_metrics[0])),
        "constant_rate_all_subset_ranks": sorted(set(identifiability(
            sensitivity(segments[list(c)].ravel(), tau, 0.), sigma, [.1, .1])["rank"] for c in choices)),
        "interpretation": "For this quadratic reference, D-optimal selection is maximum time spread. This is a correctness check, not a new selection method.",
    }
    # Paired Monte Carlo: each strategy sees subsets of the same simulated
    # full recording. Random chooses one uniformly random subset per trial.
    rng = np.random.default_rng(42)
    trials = 5000
    errors = rng.normal(0., sigma, (trials, segments.size))
    def slope_error(choice, errors):
        indices = (np.array(choice)[:, None]*6 + np.arange(6)).ravel()
        times = segments.ravel()[indices]
        centered = times - times.mean()
        return errors[..., indices] @ centered / (acceleration * (centered @ centered))
    selected_errors = slope_error(choices[best], errors)
    even_errors = slope_error(even, errors)
    random_choices = rng.integers(len(choices), size=trials)
    random_errors = np.array([slope_error(choices[j], errors[i]) for i,j in enumerate(random_choices)])
    result["monte_carlo"] = {"trials": trials, "seed": 42, "paired_full_recordings": True,
        "information_tau_rmse_s": float(np.sqrt(np.mean(selected_errors**2))),
        "even_tau_rmse_s": float(np.sqrt(np.mean(even_errors**2))),
        "random_tau_rmse_s": float(np.sqrt(np.mean(random_errors**2))),
        "random_predicted_root_mean_variance_s": float(np.sqrt(np.mean(metrics[:,1]**2)))}
    np.testing.assert_allclose(result["monte_carlo"]["information_tau_rmse_s"], result["information_tau_std_s"], rtol=.05)
    np.testing.assert_allclose(result["monte_carlo"]["even_tau_rmse_s"], result["even_tau_std_s"], rtol=.05)
    np.testing.assert_allclose(result["monte_carlo"]["random_tau_rmse_s"], result["monte_carlo"]["random_predicted_root_mean_variance_s"], rtol=.05)
    return result, segments, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0., 8., 81)
    truth = np.array([.12, -.08])
    sigma = .03
    rng = np.random.default_rng(20260914)
    noise = rng.normal(0., sigma, t.size)
    results, observations, checks = {}, {}, []
    for name, acceleration in [("constant_rate", 0.), ("accelerating", .2)]:
        z = theta(t + truth[0], acceleration) + truth[1] + noise
        observations[name] = z
        fit = fit_heading(t, z, acceleration, sigma)
        results[name] = fit
        # Change tau units from seconds to milliseconds and scale consistently.
        J = sensitivity(t, truth[0], acceleration)
        base = identifiability(J, sigma, [.1, .1])
        converted = identifiability(J @ np.diag([.001, 1.]), sigma, [100., .1])
        np.testing.assert_allclose(base["singular_values"], converted["singular_values"], atol=1e-12)
        checks.append(name + ": singular values invariant to seconds/milliseconds with physical scales preserved")
        if acceleration == 0:
            assert fit["tau_s"] is None and fit["individual_parameter_covariance"] is None
            assert fit["local_report"]["rank"] == 1
            for shift in [-1., -.2, .5, 2.]:
                prediction = theta(t + truth[0] + shift, 0.) + truth[1] - .8 * shift
                np.testing.assert_allclose(prediction, theta(t + truth[0], 0.) + truth[1], atol=2e-15)
            checks.append("constant rate: exact likelihood ridge and refusal of individual estimates")
        else:
            p = np.array([fit["tau_s"], fit["phi_rad"]])
            residual = lambda x: (theta(t + x[0], acceleration) + x[1] - z) / sigma
            nonlinear = least_squares(residual, [0., 0.], jac=lambda x: sensitivity(t, x[0], acceleration)/sigma,
                                      gtol=1e-12, xtol=1e-12, ftol=1e-12)
            np.testing.assert_allclose(p, nonlinear.x, atol=1e-10)
            noiseless = fit_heading(t, theta(t + truth[0], acceleration)+truth[1], acceleration, sigma)
            np.testing.assert_allclose([noiseless["tau_s"], noiseless["phi_rad"]], truth, atol=1e-12)
            checks.append("accelerating: exact regression oracle matches nonlinear least squares; noiseless truth recovered")
    selection, segments, metrics = selection_experiment()
    assert selection["constant_rate_all_subset_ranks"] == [1]
    assert selection["information_tau_std_s"] <= selection["even_tau_std_s"]
    checks.append("selection: all 495 equal-budget subsets checked; none rescues constant-rate degeneracy")
    checks.append("selection: shared SVD covariance matches analytic timing variance for every subset")
    checks.append("selection: 5000 paired Monte Carlo recordings agree with predicted timing variance within 5 percent")
    assert identifiability(sensitivity(np.ones(10), truth[0], .2), sigma, [.1,.1])["rank"] == 1
    checks.append("acceleration with repeated identical sample times remains rank deficient")
    weak_a = 1e-4
    weak_fit = fit_heading(t, theta(t+truth[0],weak_a)+truth[1], weak_a, sigma)
    assert weak_fit["local_report"]["rank"] == 2
    assert weak_fit["local_std_tau_s"] > 1.
    assert len(weak_fit["local_report"]["weak_constrained_scaled_directions"]) == 1
    results["weak_acceleration"] = weak_fit
    checks.append("weak acceleration: full numerical rank does not imply useful precision; near-null direction flagged")
    very_weak_a = 1e-8
    near_singular = fit_heading(t, theta(t+truth[0],very_weak_a)+truth[1], very_weak_a, sigma)
    expected_tau_std = sigma / (very_weak_a * np.sqrt(np.sum((t-t.mean())**2)))
    assert np.all(np.diag(near_singular["individual_parameter_covariance"]) > 0)
    np.testing.assert_allclose(near_singular["local_std_tau_s"], expected_tau_std, rtol=1e-7)
    same_time_fit = fit_heading(np.ones(10),np.zeros(10),.2,sigma)
    assert same_time_fit["tau_s"] is None
    checks.append("near-singular a=1e-8 covariance agrees with exact timing variance; no squared-condition-number failure")
    physical_covariance, _ = svd_covariance(sensitivity(t, truth[0], very_weak_a), sigma, [.1, .1])
    ms_covariance, _ = svd_covariance(sensitivity(t, truth[0], very_weak_a) @ np.diag([.001, 1.]), sigma, [100., .1])
    to_seconds = np.diag([.001, 1.])
    np.testing.assert_allclose(to_seconds @ ms_covariance @ to_seconds.T, physical_covariance, rtol=1e-7)
    try:
        svd_covariance(sensitivity(t, truth[0], 0.), sigma, [.1, .1])
    except ValueError:
        pass
    else:
        raise AssertionError("Rank-one covariance must be refused")
    checks.append("shared SVD covariance: physical covariance invariant to timing units; rank-one input refused")
    results["selection"] = selection
    results["checks"] = checks
    results["assumptions"] = {"reference": "known, unwrapped, theta0=0.2 rad, omega0=0.8 rad/s",
        "parameters": ["tau [s]", "phi [rad]"], "truth": truth.tolist(), "noise_std_rad": sigma,
        "samples": len(t), "noise_seed": 20260914,
        "scope": "Local scaled Jacobian report; structural degeneracy also proven analytically for the stated model. Reference uncertainty, wrapping, drift and nuisance calibration parameters excluded."}
    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "device": "CPU"}
    (out / "heading-checks.log").write_text("\n".join("PASS: " + check for check in checks) + "\n")
    (out / "heading-results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    np.savetxt(out / "heading-observations.csv", np.column_stack((t, observations["constant_rate"], observations["accelerating"])), delimiter=",", header="t_s,constant_rate_heading_rad,accelerating_heading_rad", comments="")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
    ts = np.linspace(truth[0]-.12, truth[0]+.12, 350)
    ps = np.linspace(truth[1]-.23, truth[1]+.23, 400)
    T, P = np.meshgrid(ts, ps)
    for ax, name, a, title in zip(axes, results, [0., .2], ["Constant angular velocity: rank 1", "Accelerating rotation: rank 2"]):
        prediction = theta(t[None,None,:]+T[:,:,None], a)+P[:,:,None]
        cost = .5*np.sum(((prediction-observations[name])/sigma)**2, axis=-1)
        fit = results[name]
        if a == 0:
            optimum = fit["estimable_combination"]["estimate_rad"]
            best_cost = .5*np.sum(((theta(t,0.)+optimum-observations[name])/sigma)**2)
            ax.plot(ts, optimum-.8*ts, color="#2563eb", linewidth=2, label="Equal-likelihood ridge")
            note = "Separate delay and mounting yaw refused"
        else:
            best_cost = .5*np.sum(((theta(t+fit["tau_s"],a)+fit["phi_rad"]-observations[name])/sigma)**2)
            ax.scatter(fit["tau_s"],fit["phi_rad"],s=55,color="#2563eb",label="Estimated parameters",zorder=5)
            note = f"Delay {fit['tau_s']*1000:.1f} ms; yaw {fit['phi_rad']:.3f} rad"
        ax.contourf(T,P,np.minimum(cost-best_cost,15),levels=[0.,1.,3.,6.,15.], colors=["#b9d6f2","#d9e7f5","#edf3fa","#fafbfc"])
        ax.contour(T,P,cost-best_cost,levels=[1.,3.,6.],colors="#7c9abc",linewidths=.8)
        ax.scatter(*truth,marker="x",s=70,color="#b45309",linewidths=2,label="Injected truth",zorder=6)
        ax.set(xlabel="Timing offset τ [s]",ylabel="Mounting yaw φ [rad]",title=title)
        ax.text(.5,1.01,note,transform=ax.transAxes,ha="center",fontsize=10)
        ax.set_title(title,pad=28,weight="bold")
        ax.legend(loc="lower left",fontsize=9)
    fig.suptitle("Can this recording distinguish timing offset from mounting yaw?",weight="bold",fontsize=14)
    fig.savefig(out / "heading-identifiability.png",dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8,4.2),layout="constrained")
    ax.hist(metrics[:,1]*1000,bins=30,color="#dbe6f1",edgecolor="white",label="Uniform random subset (all 495)")
    ax.axvline(selection["even_tau_std_s"]*1000,color="#b45309",lw=2,label="Evenly spaced segments")
    ax.axvline(selection["information_tau_std_s"]*1000,color="#2563eb",lw=2,label="Maximum information subset")
    ax.set(xlabel="Predicted local delay standard deviation [ms] — smaller is better",ylabel="Number of subsets",title="Same budget: 4 segments × 6 heading observations")
    ax.legend(fontsize=9)
    fig.savefig(out / "heading-selection.png",dpi=170)
    plt.close(fig)
    print(json.dumps({k: v for k,v in results.items() if k != "local_report"},indent=2))


if __name__ == "__main__":
    main()
