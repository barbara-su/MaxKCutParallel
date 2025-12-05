import numpy as np


# =====================================================================================
# Parallel Rank-r Algorithm Implementation
# =====================================================================================

def process_combination_batch(args):
    """
    Process a batch of combinations for the rank-r algorithm.

    Parameters
    ----------
    args : tuple
        Contains (V, K, r, Q, combinations_batch, batch_id)

    Returns
    -------
    best_score : float
        Best score found in this batch
    best_candidate : ndarray
        Best candidate solution found in this batch
    batch_id : int
        ID of this batch for tracking
    """
    V, K, r, Q, combinations_batch, batch_id = args

    n = V.shape[0]
    best_score = float('-inf')
    best_candidate = None

    # Process each combination in the batch
    for combo in combinations_batch:
        try:
            # Extract the corresponding rows from V_tilde
            V_tilde = compute_vtilde(V)
            I = np.array(combo)
            VI = V_tilde[I]

            # Find intersection of hyperplanes
            c_tilde = find_intersection(VI)
            phi, sign_c = determine_phi_sign_c(c_tilde)

            # Check if the last angle is within the decision region
            if -np.pi/K < phi[2*r-2] <= np.pi/K:
                # Create a new candidate solution
                candidate = np.zeros(n, dtype=complex)

                # Adjust c_tilde sign
                c_tilde = c_tilde * sign_c

                # Convert to complex form
                c = convert_ctilde_to_complex(c_tilde, r)

                # Get row mapping
                row_mapping, inverse_mapping = get_row_mapping(n, K)

                # Get the original V indices that correspond to the selected V_tilde rows
                v_indices_used = set()
                for idx in I:
                    v_row, _ = row_mapping[idx]
                    v_indices_used.add(v_row)

                # K-th roots of unity
                s = np.exp(1j * 2 * np.pi * np.arange(K) / K)

                # For vertices not in the selected hyperplanes
                for k in range(n):
                    if k not in v_indices_used:
                        v_c = V[k] @ c
                        metric = np.real(np.conj(s) * v_c)
                        s_idx = np.argmax(metric)
                        candidate[k] = s[s_idx]

                # For vertices in the selected hyperplanes
                for v_idx in v_indices_used:
                    v_tilde_indices_for_v = [idx for idx in inverse_mapping[v_idx] if idx in I]

                    for vtilde_idx in v_tilde_indices_for_v:
                        pos = np.where(I == vtilde_idx)[0][0]
                        VI_minus = np.delete(VI, pos, axis=0)

                        try:
                            new_c_tilde = find_intersection_fixed_angle(VI_minus, r, K)
                            new_c = convert_ctilde_to_complex(new_c_tilde, r)

                            v_c = V[v_idx] @ new_c
                            metric = np.real(np.conj(s) * v_c)
                            s_idx = np.argmax(metric)
                            candidate[v_idx] = s[s_idx]
                            break
                        except ValueError:
                            continue

                    # Fallback if needed
                    if np.abs(candidate[v_idx]) < 1e-10:
                        v_c = V[v_idx] @ c
                        metric = np.real(np.conj(s) * v_c)
                        s_idx = np.argmax(metric)
                        candidate[v_idx] = s[s_idx]

                # Evaluate candidate
                score = np.real(candidate.conj() @ Q @ candidate)

                if score > best_score:
                    best_score = score
                    best_candidate = candidate.copy()

        except (ValueError, np.linalg.LinAlgError):
            continue

    return best_score, best_candidate, batch_id
