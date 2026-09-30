import numpy as np
import pandas as pd  # used to save CSV files


class Individual:
    """An individual holding a (multi-dimensional) objective vector."""

    def __init__(self, obj_list):
        self.obj = np.array(obj_list)
        self.n_obj = len(self.obj)

    def dominate(self, other) -> bool:
        """Return True if this individual dominates ``other`` (minimization)."""
        if not isinstance(other, Individual):
            raise TypeError("comparison target must be an Individual instance")
        return (np.all(self.obj <= other.obj) and np.any(self.obj < other.obj))


class NonDominatedSort:
    """Fast non-dominated sorting."""

    @staticmethod
    def sort(population):
        """Parameters
        ----------
        population : list of Individual

        Returns
        -------
        fronts : list of lists of Individual, ordered from lowest to highest
            domination rank.
        """
        popsize = len(population)
        if popsize == 0:
            return []
        domination_count = np.zeros(popsize, dtype=int)
        dominated_solutions = [[] for _ in range(popsize)]
        fronts = []

        for i in range(popsize):
            for j in range(popsize):
                if i == j:
                    continue
                if population[i].dominate(population[j]):
                    dominated_solutions[i].append(j)
                elif population[j].dominate(population[i]):
                    domination_count[i] += 1

        current_front = [i for i in range(popsize) if domination_count[i] == 0]
        fronts.append([population[i] for i in current_front])

        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append([population[i] for i in next_front])
            current_front = next_front

        return fronts


class HyperVolume:
    """Hypervolume computation, supporting 2D and 3D.

    The input point set is expected to be a Pareto front (non-dominated
    solutions); otherwise the result is inaccurate. A class method is provided
    to compute directly from an arbitrary point set (the Pareto front is
    extracted internally).
    """

    def __init__(self, points, ref_point):
        self.points = np.array(points)
        self.ref = np.array(ref_point)
        self.m = self.points.shape[1] if len(self.points.shape) > 1 else 0
        if self.m not in [2, 3]:
            raise ValueError("only 2D or 3D objective spaces are supported")

    def _extract_pareto_front(self):
        """Extract the first front (non-dominated solutions) from all points."""
        pop = [Individual(p) for p in self.points]
        fronts = NonDominatedSort.sort(pop)
        if not fronts:
            return np.empty((0, self.m))
        pareto = np.array([ind.obj for ind in fronts[0]])
        return pareto

    def _hv_2d(self, points_2d):
        """2D hypervolume (points_2d must be a Pareto front)."""
        if len(points_2d) == 0:
            return 0.0
        sorted_idx = np.argsort(points_2d[:, 0])
        sorted_pts = points_2d[sorted_idx]
        prev = self.ref[0]
        area = 0.0
        for pt in sorted_pts:
            x, y = pt[0], pt[1]
            width = prev - x
            height = self.ref[1] - y
            if width > 0 and height > 0:
                area += width * height
            prev = x
        return area

    def _hv_3d(self, points_3d):
        """3D hypervolume (recursive slicing)."""
        if len(points_3d) == 0:
            return 0.0
        sorted_idx = np.argsort(points_3d[:, 0])
        sorted_pts = points_3d[sorted_idx]
        n = len(sorted_pts)
        prev = self.ref[0]
        total_vol = 0.0

        for i in range(n):
            x = sorted_pts[i, 0]
            sub_2d = sorted_pts[i:, 1:3]
            if len(sub_2d) == 0:
                break
            pop_2d = [Individual(p) for p in sub_2d]
            fronts_2d = NonDominatedSort.sort(pop_2d)
            if not fronts_2d:
                sub_pareto = np.empty((0, 2))
            else:
                sub_pareto = np.array([ind.obj for ind in fronts_2d[0]])
            hv_2d = self._hv_2d_with_ref(sub_pareto, (self.ref[1], self.ref[2]))
            width = prev - x
            if width > 0 and hv_2d > 0:
                total_vol += width * hv_2d
            prev = x
        return total_vol

    def _hv_2d_with_ref(self, points_2d, ref_2d):
        """2D hypervolume using a specified reference point."""
        if len(points_2d) == 0:
            return 0.0
        sorted_idx = np.argsort(points_2d[:, 0])
        sorted_pts = points_2d[sorted_idx]
        prev = ref_2d[0]
        area = 0.0
        for pt in sorted_pts:
            x, y = pt[0], pt[1]
            width = prev - x
            height = ref_2d[1] - y
            if width > 0 and height > 0:
                area += width * height
            prev = x
        return area

    def compute(self, use_pareto_front=True):
        """Compute the hypervolume.

        use_pareto_front : if True, extract the Pareto front internally;
            if False, use the passed point set directly.
        """
        if use_pareto_front:
            points = self._extract_pareto_front()
        else:
            points = self.points

        if len(points) == 0:
            return 0.0

        if self.m == 2:
            return self._hv_2d(points)
        elif self.m == 3:
            return self._hv_3d(points)
        else:
            raise NotImplementedError("only 2D and 3D are supported")


# ========== Vectorized layered non-dominated sorting (fronts 1..N) ==========
def nondominated_sort_layers(points, max_layers=10):
    """Layered non-dominated sorting (NSGA-II style).

    Layer 1 = the Pareto front; after removing it the procedure is repeated on
    the remaining points to obtain layer 2, layer 3, and so on.

    Parameters
    ----------
    points : numpy array (n_samples, m), minimization directions, must not
        contain NaN (drop NaNs before calling).
    max_layers : maximum number of layers to return; stops early if the points
        are exhausted.

    Returns
    -------
    layers : list of np.ndarray, each holding the row indices of the original
        array for that layer (ordered by layer).
    """
    points = np.asarray(points, dtype=float)
    if np.isnan(points).any():
        raise ValueError("points contain NaN; drop them before non-dominated sorting")
    remaining = np.arange(len(points))
    layers = []
    while len(remaining) > 0 and len(layers) < max_layers:
        sub = points[remaining]
        m = len(sub)
        # Vectorized dominance relation: dominates[i, j] is True if sub[i] dominates sub[j].
        leq = np.all(sub[:, None, :] <= sub[None, :, :], axis=2)
        strict = np.all(sub[:, None, :] < sub[None, :, :], axis=2)
        dominates = leq & (strict | ~np.eye(m, dtype=bool))
        dominated = dominates.any(axis=0)
        front_local = np.where(~dominated)[0]
        layers.append(remaining[front_local])
        remaining = remaining[np.where(dominated)[0]]
    return layers


# ========== Extract and save the first N fronts ==========
def save_fronts_to_csv(points, n_fronts=10, base_filename='front', include_rank_col=True):
    """Non-dominated sort the input points, extract the first n_fronts fronts
    and save them as CSV.

    Parameters
    ----------
    points : numpy array (n_samples, m) of objective values (already
        normalized, all in minimization direction).
    n_fronts : number of fronts to save.
    base_filename : base name for the CSV file(s). If include_rank_col=True a
        single file is written; otherwise each front is saved separately as
        'front_1.csv', 'front_2.csv', ...
    include_rank_col : if True, merge all fronts into one CSV with a
        'front_rank' column; if False, save each front separately.
    """
    pop = [Individual(p) for p in points]
    fronts = NonDominatedSort.sort(pop)
    n_available = len(fronts)
    n_save = min(n_fronts, n_available)
    print(f"{n_available} fronts in total; saving the first {n_save}.")

    if include_rank_col:
        # Merge all fronts and add a rank column.
        all_points = []
        ranks = []
        for rank, front in enumerate(fronts[:n_save], start=1):
            for indiv in front:
                all_points.append(indiv.obj.tolist())
                ranks.append(rank)
        df = pd.DataFrame(all_points, columns=[f'obj_{i+1}' for i in range(points.shape[1])])
        df['front_rank'] = ranks
        filename = f"{base_filename}_top{n_save}.csv"
        df.to_csv(filename, index=False)
        print(f"Merged fronts saved to {filename}")
    else:
        # Save each front separately.
        for rank, front in enumerate(fronts[:n_save], start=1):
            arr = np.array([indiv.obj for indiv in front])
            df = pd.DataFrame(arr, columns=[f'obj_{i+1}' for i in range(points.shape[1])])
            filename = f"{base_filename}_{rank}.csv"
            df.to_csv(filename, index=False)
            print(f"Front {rank} (size {len(front)}) saved to {filename}")

    return fronts[:n_save]  # list of Individual for the first n fronts


# ================== Example usage ==================
if __name__ == "__main__":
    # Example 1: 100 random 3D points in [0, 1] (lower is better).
    np.random.seed(42)
    n_points = 100
    points = np.random.rand(n_points, 3)

    # Compute the hypervolume.
    ref_point = [1.0, 1.0, 1.0]
    hv_calculator = HyperVolume(points, ref_point)
    hv_value = hv_calculator.compute(use_pareto_front=True)
    print(f"3D hypervolume (HV) = {hv_value:.6f}")

    # Save the first 10 fronts (merged).
    save_fronts_to_csv(points, n_fronts=10, base_filename='random_front', include_rank_col=True)

    # Example 2: load from an existing normalized CSV (columns 'MIC_norm',
    # 'TOXIN_norm', 'AIP_norm'). Uncomment to use a real file.
    # df = pd.read_csv('normalized_with_seq.csv')
    # points_from_file = df[['MIC_norm', 'TOXIN_norm', 'AIP_norm']].values
    # save_fronts_to_csv(points_from_file, n_fronts=10, base_filename='my_front', include_rank_col=True)
