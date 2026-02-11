from math import floor, ceil
from efficient_fpt.models import DDModel, piecewise_const_func
from efficient_fpt.utils import get_alternating_addm_mu_array
import numpy as np
import pandas as pd

class aDDModel(DDModel):
    """
    One trial of an attentional drift diffusion model with alternating drift mu1 and mu2.
    """
    def __init__(self, mu1, mu2, sacc_array, flag, sigma, a, b, x0):
        super().__init__(x0)
        # drift parameters
        self.mu1 = mu1
        self.mu2 = mu2
        self.sacc_array = sacc_array
        self.flag = flag # indicates whether the process starts with mu1 (flag=0) or mu2 (flag=1)
        self.d = len(sacc_array) # number of stages
        self.mu_array = get_alternating_addm_mu_array(mu1, mu2, self.d, flag)
        # diffusion parameter
        self.sigma = sigma
        # symmetric linear boundary parameters
        self.a = a
        self.b = b

    def drift_coeff(self, X: float, t: float) -> float:
        return piecewise_const_func(t, self.mu_array, self.sacc_array)

    def diffusion_coeff(self, X: float, t: float) -> float:
        return self.sigma

    @property
    def is_update_vectorizable(self) -> bool:
        return True

    def upper_bdy(self, t: float) -> float:
        return self.a - self.b * t

    def lower_bdy(self, t: float) -> float:
        return -self.a + self.b * t

def expand_addm_fixations(sacc_data, flag_data, rt_data, dt):
    """
    Parameters
    ----------
    sacc_data : list of 1D np.ndarray
        saccade times per trial (seconds)
    flag_data : 1D np.ndarray
        initial fixation per trial (0/1 or similar)
    rt_data : 1D np.ndarray
        reaction time per trial (seconds)
    dt : float
        timestep size (seconds)

    Encoding
    --------
      0 -> transition
      1 -> fixation with fix_start = 0
      2 -> fixation with fix_start = 1

    Returns
    -------
    array
        Each element is a tuple of fixation locations for one trial
    """
    all_trials = []

    for saccs, start_fix, rt in zip(sacc_data, flag_data, rt_data):

        fix_len = int(floor(rt / dt)) + 1

        # start in transition
        fix = np.zeros(fix_len, dtype=int)

        if len(saccs) > 0:
            switch_idxs = [int(ceil(s / dt)) for s in saccs]

            # fixation identity determined by start_fix
            current_fix = 1 if start_fix == 0 else 2
            state = "fixation"  # first switch enters fixation

            for idx in switch_idxs[1:]:
                if idx <= 0 or idx >= fix_len:
                    continue

                if state == "fixation":
                    fix[idx:] = current_fix
                    state = "transition"
                else:
                    fix[idx:] = 0
                    current_fix = 1 if current_fix == 2 else 2
                    state = "fixation"

        all_trials.append(tuple(fix.tolist()))

    return all_trials

def rasterize_data(
    df: pd.DataFrame,
    subject_col: str,
    trial_col: str,
    seq_col: str = "fixation",
    fill_codes: set = {0, 4},
    start_col: str = "fix_start",
    end_col: str = "fix_end",
    loc_col: str = "fix_location",
    fixnum_col: str | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Expand per-(subject, trial) fixation sequences into fixation-level rows.
    Zero-valued segments are treated as transitions and excluded.
    """

    df = df.copy()

    if keep_cols is None:
        keep_cols = [
            c for c in df.columns
            if c not in {subject_col, trial_col, seq_col}
        ]

    rows = []

    for _, row in df.iterrows():
        seq = np.asarray(row[seq_col])

        changes = np.diff(seq, prepend=seq[0])
        starts = np.where(changes != 0)[0]

        fix_num = 0

        for i, start_idx in enumerate(starts):
            loc = seq[start_idx]

            end_idx = (
                starts[i + 1]
                if i + 1 < len(starts)
                else len(seq)
            )

            # Skip transitions
            if loc in fill_codes:
                continue

            data = {
                subject_col: row[subject_col],
                trial_col: row[trial_col],
                start_col: start_idx,
                end_col: end_idx,
                loc_col: loc,
            }

            if fixnum_col is not None:
                data[fixnum_col] = fix_num
                fix_num += 1

            for col in keep_cols:
                data[col] = row[col]

            rows.append(data)

    return pd.DataFrame(rows)

def simulate_empirical_trial(
    n,
    r1_data,
    r2_data,
    empirical_distributions,
    *,
    eta, kappa, sigma, a, b, T, x0, dt, seed
):
    rng = np.random.default_rng(seed)

    r1 = r1_data[n]
    r2 = r2_data[n]

    # --- Generate fixations ---
    # flag = rng.binomial(1, empirical_distributions['probFixLeftFirst']) # 1 for left first, 0 for right first
    flag = rng.binomial(1, 0.5) 

    fixations = []
    total_dur = 0.0

    latency = rng.choice(empirical_distributions['latencies'])
    fixations.append(latency)
    total_dur += latency

    diff = r1 - r2 # left - right
    if not flag: # right first
        diff = -diff
    fix_1 = rng.choice(empirical_distributions['fixations'][1][diff])
    fixations.append(fix_1)
    total_dur += fix_1

    while total_dur <= T:
        transition = rng.choice(empirical_distributions['transitions'])
        fixations.append(transition)
        total_dur += transition

        diff = -diff
        fix_dur = rng.choice(empirical_distributions['fixations'][2][diff])
        fixations.append(fix_dur)
        total_dur += fix_dur

    sacc_array = np.insert(np.cumsum(fixations), 0, 0.0)
    sacc_array = sacc_array[sacc_array < T] * dt

    # --- First fixation side ---
    mu1 = mu2 = None
    if flag: # left first
        mu1 = kappa * (r1 - eta * r2)
        mu2 = kappa * (eta * r1 - r2)
    else: # right first
        mu1 = kappa * (eta * r1 - r2)
        mu2 = kappa * (r1 - eta * r2)

    addm = aDDModel(
        mu1=mu1, mu2=mu2,
        sacc_array=sacc_array,
        flag=flag,
        sigma=sigma, a=a, b=b, x0=x0
    )

    decision = addm.simulate_fpt_datum(dt=dt)

    sacc_array = sacc_array[sacc_array < decision[0]]
    d = len(sacc_array)
    mu_array = get_alternating_addm_mu_array(mu1, mu2, d, flag)

    return decision, mu_array, sacc_array, r1, r2, flag