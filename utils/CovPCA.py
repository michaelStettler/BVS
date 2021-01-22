import numpy as np

"""
Important note, if you get memory allocation error. You may want to change the settings of your computer.
First open a terminal and set it as root (sudo -i).
Override the overcommit memory: echo 1 > /proc/sys/vm/overcommit_memory
"""

class CovPCA:
    """
    Compute the Principal Component Analysis (PCA) using the covariance matrix
    The method is used to simply compare the indexes retained for dimensionality reduction

    :param data:
    :param dim:
    :param return_index:
    :return:
    """

    def __init__(self, n_components=2, var_threshold=0.01, max_dim=5000):
        self.n_components = n_components
        self.eig_vectors = None
        self.eig_values = None
        self.max_index = None
        self.var_threshold = var_threshold
        self.max_dim = max_dim
        self.thresh_index = None
        self.used_thresh_var = False
        self.explained_variance_ratio_ = None

    def _update_params(self):
        sum_variance = np.sum(self.eig_values)
        self.explained_variance_ratio_ = np.real(self.eig_values / sum_variance)

    def _remove_variance(self, data):
        # remove all the entry with small variance
        var = np.std(data, axis=0)
        # filter out all variance smaller than var_threshold
        thresh_index = np.arange(data.shape[-1])
        self.thresh_index = thresh_index[var > self.var_threshold]
        print("[CovPCA] thresh index[:{}]".format(self.n_components), self.thresh_index[:self.n_components])
        data = data[:, self.thresh_index]

        # control that the dimensionality reduction was sufficient
        if data.shape[-1] > self.max_dim:
            raise ValueError("Dimensionality reduction did not work (max_dim: {}, current: {})."
                             "You will likely have a memory error. Try setting a higher threshold "
                             "(current: {}), or a higher dimension. (Max variance: {})".format(self.max_dim,
                                                                                               data.shape[-1],
                                                                                               self.var_threshold,
                                                                                               np.amax(var)))
        print("[CovPCA] Variance threshold, data dimension reduced to: {}".format(data.shape[-1]))
        return data


    def fit(self, data, verbose=False):
        print("[CovPCA] shape data", np.shape(data))
        var_data = np.std(data, axis=0)
        max_var_data = np.flip(np.argsort(var_data))
        print("[CovPCA] max_var_data index", max_var_data[:self.n_components])

        # check for dimensionality
        if data.shape[-1] > self.max_dim:
            self.used_thresh_var = True
            data = self._remove_variance(data)

        # compute cov matrix on column
        data_cov = np.cov(data, rowvar=False)

        # compute the eigenvalues and eigenvectors
        eig_values, eig_vectors = np.linalg.eig(data_cov)
        self.eig_values = np.real(eig_values)
        self.eig_vectors = np.real(eig_vectors)

        if verbose:
            print("eig_values:")
            print(eig_values)

        # pick eigenvectors whose eigenvalues are highest
        print("[CovPCA] shape eig_values", np.shape(eig_values))
        max_index = np.flip(np.argsort(eig_values))
        self.max_index = max_index[:self.n_components]
        self.eig_vectors = eig_vectors[:, self.max_index]
        self.eig_values = eig_values[self.max_index]

        # map index back to original space if the var_threshold has been used
        if self.used_thresh_var:
            self.max_index = self.thresh_index[self.max_index]

        self._update_params()

    def fit_transform(self, data, verbose=False):
        # check for dimensionality
        if data.shape[-1] > self.max_dim:
            self.used_thresh_var = True
            data = self._remove_variance(data)

        # fit PCA
        self.fit(data, verbose=verbose)

        # map index back to original space if the var_threshold has been used
        if self.used_thresh_var:
            self.max_index = self.thresh_index[self.max_index]

        # transform feature
        return np.real(np.transpose(np.matmul(np.transpose(self.eig_vectors), np.transpose(data))))

    def transform(self, data):
        # transform feature
        return np.real(np.transpose(np.matmul(np.transpose(self.eig_vectors), np.transpose(data))))


if __name__ == '__main__':
    """
    These test compares the PCA implemented by sklearn, the one using the covariance to get the indexes, and a
    self-built one using SVD decomposition as the sklearn implementation. 
    
    The all give the same results besides some sign inverse
    
    run: python -m utils.CovPCA
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    # set a random seed
    np.random.seed(6)  # 3 -> doesn't have the highest variance in index 0 as 1

    # ==================================================================================================================
    # Create a toy example dataset and declare some variables
    data = np.random.rand(4, 16)
    data = StandardScaler().fit_transform(data)
    data[:, 2] = 0
    data[:, 4] = 0
    data[:, 12] = 0
    print("data", np.shape(data))
    print(data)

    final_dim = 2

    # ==================================================================================================================
    # Compare highest variance
    var = np.std(data, axis=0)
    print("variance", np.shape(var))
    print(var)
    print()

    # ==================================================================================================================
    # apply normal PCA from sklearn
    pca = PCA(n_components=final_dim)
    x_new = pca.fit_transform(data)
    print("x_new", np.shape(x_new))
    print(x_new)
    print()

    # ==================================================================================================================
    # apply personal PCA
    # get PCA
    pca = CovPCA(n_components=final_dim, max_dim=14)
    new_data = pca.fit_transform(data)
    index = pca.max_index
    print("covPCA new_data", np.shape(new_data))
    print(new_data)
    print("index", index)
    print()

    # plt.figure()
    # plt.plot(new_data)
    # plt.title("PCA")
    # plt.savefig("PCA")

    # ==================================================================================================================
    # apply PCA using SVD
    u, s, v = np.linalg.svd(data, full_matrices=False)  # It's not necessary to compute the full matrix of U or V

    # transform feature
    new_data = np.dot(u[:, :final_dim], np.diag(s[:final_dim]))
    print("shape new data", new_data.shape)
    print(new_data)

    # plt.figure()
    # plt.plot(new_data)
    # plt.title("SVD")
    # plt.savefig("SVD")
