"""Fermionic bitstring algebra on fixed-N determinants."""

from __future__ import annotations

import numpy as np

def popcount(x: int) -> int:
    return int(x.bit_count())
def mode_to_bitpos(mode: int, n_qubits: int) -> int:
    """
    OpenFermion-consistent mapping inferred from your identity check:
    fermionic mode 0 is the LEFTMOST bit in the printed binary string.
    """
    if not (0 <= mode < n_qubits):
        raise ValueError(f"mode {mode} out of range for n_qubits={n_qubits}")
    return n_qubits - 1 - mode
def mode_is_occupied(bitstring: int, mode: int, n_qubits: int) -> int:
    pos = mode_to_bitpos(mode, n_qubits)
    return (bitstring >> pos) & 1
def parity_sign(bitstring: int, mode: int, n_qubits: int) -> int:
    """
    Fermionic JW sign for acting on 'mode':
    (-1)^(number of occupied modes with label < mode).
    IMPORTANT: this is NOT the same as counting lower integer bit positions
    once mode 0 is mapped to the MSB.
    """
    occ_before = 0
    for k in range(mode):
        occ_before += mode_is_occupied(bitstring, k, n_qubits)
    return -1 if (occ_before % 2 == 1) else 1
def apply_annihilate(bitstring: int, mode: int, n_qubits: int):
    pos = mode_to_bitpos(mode, n_qubits)
    if ((bitstring >> pos) & 1) == 0:
        return None, 0
    sign = parity_sign(bitstring, mode, n_qubits)
    return bitstring & ~(1 << pos), sign
def apply_create(bitstring: int, mode: int, n_qubits: int):
    pos = mode_to_bitpos(mode, n_qubits)
    if ((bitstring >> pos) & 1) == 1:
        return None, 0
    sign = parity_sign(bitstring, mode, n_qubits)
    return bitstring | (1 << pos), sign
def closed_shell_hf_bitstring(n_electrons, n_spatial):
    if n_electrons % 2 != 0:
        raise ValueError("This helper assumes closed-shell (even electron count).")

    n_qubits = 2 * n_spatial
    occ = n_electrons // 2
    b = 0

    for i in range(occ):
        a_mode = 2 * i
        b_mode = 2 * i + 1
        b |= (1 << mode_to_bitpos(a_mode, n_qubits))
        b |= (1 << mode_to_bitpos(b_mode, n_qubits))

    return b
def omega_mask_from_bitstring(bitstring: int, n_spatial: int) -> int:
    """
    Returns an n_spatial-bit mask in ORBITAL LABEL ORDER.
    Bit i of the returned mask corresponds to spatial orbital i.
    """
    n_qubits = 2 * n_spatial
    mask = 0
    for i in range(n_spatial):
        oa = mode_is_occupied(bitstring, 2 * i,     n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        if oa ^ ob:
            mask |= (1 << i)
    return mask
def occ_lists_alpha_beta(bitstring: int, n_spatial: int):
    n_qubits = 2 * n_spatial
    occ_a = [i for i in range(n_spatial) if mode_is_occupied(bitstring, 2 * i,     n_qubits)]
    occ_b = [i for i in range(n_spatial) if mode_is_occupied(bitstring, 2 * i + 1, n_qubits)]
    return occ_a, occ_b
def format_omega_mask(mask: int, n_spatial: int) -> str: #ai code to make it easier to look at
    """Human-friendly Ω-pattern string, i=0 on the left."""
    return "".join(str((mask >> i) & 1) for i in range(n_spatial))
def build_hf_occ_modes(bitstring: int, n_qubits: int):
    return [m for m in range(n_qubits) if mode_is_occupied(bitstring, m, n_qubits)]
def build_hf_virt_modes(bitstring: int, n_qubits: int):
    return [m for m in range(n_qubits) if not mode_is_occupied(bitstring, m, n_qubits)]
def excite(bitstring: int, annihilators, creators, n_qubits: int):
    """
    Apply a_p^† ... a_q^† a_j ... a_i to a determinant bitstring, returning (new_bitstring, phase)
    using your JW sign convention via apply_annihilate/apply_create.
    """
    b = int(bitstring)
    phase = 1

    # annihilate in a chosen order (common: descending mode index can reduce sign bugs)
    for m in annihilators:
        b2, s = apply_annihilate(b, m, n_qubits)
        if b2 is None:
            return None, 0
        b, phase = b2, phase * s

    # create
    for m in creators:
        b2, s = apply_create(b, m, n_qubits)
        if b2 is None:
            return None, 0
        b, phase = b2, phase * s

    return b, phase
def build_cisd_basis_bitstrings(hf_b: int, n_qubits: int):
    occ = build_hf_occ_modes(hf_b, n_qubits)
    virt = build_hf_virt_modes(hf_b, n_qubits)

    cisd_set = {int(hf_b)}

    # Singles: i -> a
    for i in occ:
        for a in virt:
            b_new, _ = excite(hf_b, [i], [a], n_qubits)
            if b_new is not None:
                cisd_set.add(int(b_new))

    # Doubles: i,j -> a,b
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            i, j = occ[ii], occ[jj]
            for aa in range(len(virt)):
                for bb in range(aa + 1, len(virt)):
                    a, b = virt[aa], virt[bb]
                    b_new, _ = excite(hf_b, [j, i], [a, b], n_qubits)  # note ordering choice
                    if b_new is not None:
                        cisd_set.add(int(b_new))

    return sorted(cisd_set)
def print_fci_state(
    v_sub,
    basis_bitstrings,
    n_spatial,
    topk=None,
    amp_tol=1e-10,
    sort_by_weight=True,
):

    coeffs = np.asarray(v_sub, dtype=np.complex128)
    coeffs = coeffs / np.linalg.norm(coeffs)

    rows = []
    for idx, (b, c) in enumerate(zip(basis_bitstrings, coeffs)):
        amp = abs(c)
        if amp < amp_tol:
            continue

        wt = amp * amp
        occ_a, occ_b = occ_lists_alpha_beta(int(b), n_spatial)
        omega_mask = omega_mask_from_bitstring(int(b), n_spatial)
        omega_pat = format_omega_mask(omega_mask, n_spatial)
        sen = popcount(omega_mask)
        bitstr = format(int(b), f"0{2*n_spatial}b")

        rows.append({
            "idx": idx,
            "bitstring": bitstr,
            "occ_a": occ_a,
            "occ_b": occ_b,
            "omega": omega_pat,
            "sen": sen,
            "coeff": c,
            "weight": wt,
        })

    if sort_by_weight:
        rows.sort(key=lambda r: r["weight"], reverse=True)

    if topk is not None:
        rows = rows[:topk]

    print("\n=== FCI state in N-electron determinant basis ===")
    print(f"Components shown: {len(rows)}")
    print("rank   idx        bitstring            occ_a        occ_b      Ω-mask  sen        coeff                         weight      cumulative")

    cumulative = 0.0
    for rank, r in enumerate(rows, start=1):
        cumulative += r["weight"]
        c = r["coeff"]
        coeff_str = f"{c.real:+.10f}{c.imag:+.10f}j"
        print(
            f"{rank:>4d} {r['idx']:>6d}   {r['bitstring']}   "
            f"{str(r['occ_a']):>10s}  {str(r['occ_b']):>10s}   "
            f"{r['omega']:>5s}   {r['sen']:>3d}   "
            f"{coeff_str:>28s}   {r['weight']:>10.8f}   {cumulative:>10.8f}"
        )

    shown_weight = float(sum(r["weight"] for r in rows))
    total_weight = float(np.sum(np.abs(coeffs) ** 2))
    print(f"\nShown weight = {shown_weight:.10f}")
    print(f"Total weight = {total_weight:.10f}")
