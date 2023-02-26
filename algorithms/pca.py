import numpy as np
import matplotlib.pyplot as plt

def PCA(data, labels, datapath, savepath):
    target_prop = 0.9
    eigenvecs = [i for i in range(dim + 1)]
    pca_data, prop_var_sum, eigenvals, pcs = PCA(data, target_prop, dim)
    print(pca_data.shape)

    def PCA(X, target_prop, dim):
     
        X_mean = X - np.mean(X, axis=0)
        
        cov_mat = np.cov(X_mean, rowvar=False)
        
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        # numpy sorts vals and vecs ascending
        # we could just [::-1], but just for safety
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        print('Largest 2 eigenvalues:')
        print(sorted_eigenvalue[0])
        print(sorted_eigenvalue[1])

        sum = 0
        num_components = 0
        prop_var_sum = [0]
        normalized_eigenvalue = [i for i in sorted_eigenvalue]
        for i in sorted_eigenvalue:
            sum += i
        for i in range(dim):
            normalized_eigenvalue[i] /= sum
            prop_var_sum.append(round(prop_var_sum[-1] + normalized_eigenvalue[i], 4))
            if prop_var_sum[-1] >= target_prop and num_components == 0:
                num_components = i + 1
                print('Reached minimum proportion of variance = {} at {} eigenvectors.'.format(target_prop, num_components))
                print('Prop. var: {}'.format(prop_var_sum[-1]))
                
        W = np.dot(sorted_eigenvectors.transpose(), X_mean.transpose()).transpose()
        print('Projection matrix W (shape {}):'.format(W.shape))
        print(W)
        return W, prop_var_sum, sorted_eigenvalue, num_components
