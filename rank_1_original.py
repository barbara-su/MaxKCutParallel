def process_rank1_parallel(V, Q, K, n_workers):
    """
    Parallel processing for rank-1 case.
    """
    n = V.shape[0]

    # Extract real and imaginary parts
    real_q1 = np.real(V).flatten()
    im_q1 = np.imag(V).flatten()

    # Initial partition assignment
    phis = np.zeros(n)
    ks = np.zeros(n, dtype=int)

    # Vectorized angle calculation
    thetas = np.arctan2(im_q1, real_q1)
    thetas = np.where(thetas < 0, thetas + 2*np.pi, thetas)

    # Map angles to partitions
    b = K * thetas / (2 * np.pi)
    b_floor = np.floor(b).astype(int)
    ks = b_floor % K

    # Calculate angular deviations
    phi_hat = 0.5 - b + b_floor
    phis = 2 * np.pi * phi_hat / K

    # Initial solution
    best_candidate = np.exp(2 * np.pi * 1j * ks / K)
    best_score = np.real(best_candidate.conj() @ Q @ best_candidate)

    # Try reassignments in parallel
    indices_sorted = np.argsort(phis)

    # Split indices for parallel processing
    chunk_size = max(1, len(indices_sorted) // n_workers)
    chunks = [indices_sorted[i:i+chunk_size] for i in range(0, len(indices_sorted), chunk_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        for chunk in chunks:
            future = executor.submit(
                process_rank1_chunk,
                ks.copy(), Q, K, chunk
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                chunk_score, chunk_ks = future.result()
                if chunk_score > best_score:
                    best_score = chunk_score
                    best_candidate = np.exp(2 * np.pi * 1j * chunk_ks / K)
            except Exception as e:
                logger.error(f"Error in rank-1 chunk: {e}")

    return best_score, best_candidate


def process_rank1_chunk(ks, Q, K, indices):
    """
    Process a chunk of vertices for rank-1 reassignment.
    """
    best_ks = ks.copy()
    best_score = float('-inf')

    for i in indices:
        # Try reassigning vertex i
        ks_test = ks.copy()
        ks_test[i] = (ks_test[i] + 1) % K

        # Evaluate
        candidate = np.exp(2 * np.pi * 1j * ks_test / K)
        score = np.real(candidate.conj() @ Q @ candidate)

        if score > best_score:
            best_score = score
            best_ks = ks_test.copy()

    return best_score, best_ks