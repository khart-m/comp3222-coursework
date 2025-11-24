import numpy as np

class TestEnsembleHelpers:
    def __init__(self, n_estimators=3, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def random_sample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        sampled_X = []
        sampled_y = []

        for _ in range(self.n_estimators):
            idx = rng.integers(0, n_samples, size=n_samples)
            sampled_X.append(X[idx])
            sampled_y.append(y[idx])

        return np.array(sampled_X), np.array(sampled_y)

    def random_features(self, sampled_X):
        rng = np.random.default_rng(self.random_state)

        self.feature_subsets_ = []
        reduced_X_list = []

        for X_i in sampled_X:
            n_features = X_i.shape[1]

            # Random Forest rule: m = sqrt(p)
            m = max(1, int(np.sqrt(n_features)))

            feat_idx = rng.choice(n_features, size=m, replace=False)
            self.feature_subsets_.append(feat_idx)

            reduced_X_list.append(X_i[:, feat_idx])

        return np.array(reduced_X_list, dtype=object)

    def random_quality(self):
        rng = np.random.default_rng(self.random_state)
        possible = np.array(["ig", "gain_ratio", "chi2", "chi2_yates"])
        self.quality_measures_ = rng.choice(possible, size=self.n_estimators)
        return self.quality_measures_

# -----------------------
# Run Test
# -----------------------

if __name__ == "__main__":
    # Simple fake dataset
    X = np.array([
        ["red",   0, "true"],
        ["green", 1, "false"],
        ["blue",  0, "true"],
        ["red",   1, "false"]
    ])

    y = np.array(["1", "2", "3", "4"])

    tester = TestEnsembleHelpers(n_estimators=3, random_state=1)

    # --- Test random_sample ---
    print("\n=== random_sample ===")
    sampled_X, sampled_y = tester.random_sample(X, y)
    print("sampled_X:\n", sampled_X)
    print("sampled_y:\n", sampled_y)

    # --- Test random_features ---
    print("\n=== random_features ===")
    reduced_X = tester.random_features(sampled_X)
    print("feature subsets:", tester.feature_subsets_)
    print("reduced_X:\n", reduced_X)

    # --- Test random_quality ---
    print("\n=== random_quality ===")
    qm = tester.random_quality()
    print("quality_measures:\n", qm)
