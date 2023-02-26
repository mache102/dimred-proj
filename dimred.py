import numpy as np
import matplotlib.pyplot as plt

class DimRed:

    def __init__(self, alg, datapath, savepath, verbose):
        self.alg = alg
        self.datapath = datapath 
        self.savepath = savepath
        self.verbose = verbose

    def data_proc(self):
        '''
        input data (csv) preprocessing for the default format:
        row1
        row2
        ...
        rowN

        where each row has K digits for K dimensions, followed by a numeric label
        -> last col contains all the labels
        e.g. 

        Label ----------- V

        1, 0, 0, 9, 5, 7, 3
        0, 0, 1, 8, 6, 7, 3
        5, 4, 6, 2, 7, 7, 2
        9, 9, 8, 1, 1, 8, 4
        '''

        labels = []
        data = []
        with open(self.datapath, 'r') as f:
            for line in f:
                temp = line.strip().split(',')
                labels.append(int(temp[-1]))
                data.append(temp[:-1])

        data = np.array(data).astype('int16')

        self.samples = data.shape[0]
        self.dim = data.shape[1]
        if self.verbose:
            print(self.datapath)
            print('{} rows'.format(self.samples))
            print('{} dimensions'.format(self.dim))

        self.data = data
        self.labels = labels

    def scatter_plt(self, X, class_idxs, ms=3, ax=None, alpha=0.1, 
                           legend=True, figsize=None, title=None, xlabel=None, ylabel=None):
        ## Plot a 2D matrix with corresponding class labels: each class diff colour
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        classes = list(np.unique(class_idxs))
        markers = 'os' * len(classes)
        colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

        for i, cls in enumerate(classes):
            mark = markers[i]
            ax.plot(X[class_idxs==cls, 0], X[class_idxs==cls, 1], marker=mark, 
                linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
                markeredgecolor='black', markeredgewidth=0.4)
        if legend:
            ax.legend()
        
        if title:
            plt.title(title)
        
        if xlabel:
            plt.xlabel(xlabel)
        
        if ylabel:
            plt.ylabel(ylabel)

        if self.savepath:
            plt.tight_layout()
            plt.savefig('{}/pca_scatter.png'.format(self.savepath))
        
        plt.clf()
        return ax
        

    def dimred(self):
        algorithms = ['pca', 'mds', 'lda', 'le', 'tsne']

        run = 'self.{}()'.format(self.alg.lower())
        try:
            exec(run)
        except:
            raise AttributeError('"{}" is not a valid algorithm.'.format(self.alg))

    def pca(self):
        ###################################################################################
        target_prop = 0.9

        def PCA(X, target_prop, dim):
            X_mean = X - np.mean(X, axis=0)
            
            cov_mat = np.cov(X_mean, rowvar=False)
            
            eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

            # numpy sorts vals and vecs ascending
            # we could just [::-1], but just for safety
            sorted_index = np.argsort(eigen_values)[::-1]
            sorted_eigenvalue = eigen_values[sorted_index]
            sorted_eigenvectors = eigen_vectors[:,sorted_index]
            if self.verbose == 2:
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
                    if self.verbose:
                        print('Reached minimum proportion of variance = {} at {} eigenvectors.'.format(target_prop, num_components))
                        print('Prop. var: {}'.format(prop_var_sum[-1]))
                    
            W = np.dot(sorted_eigenvectors.transpose(), X_mean.transpose()).transpose()
            if self.verbose == 2:
                print('Projection matrix W (shape {}):'.format(W.shape))
                print(W)
            return W, prop_var_sum, sorted_eigenvalue, num_components
    
        eigenvecs = [i for i in range(self.dim + 1)]
        pca_data, prop_var_sum, eigenvals, pcs = PCA(self.data, target_prop, self.dim)

        self.scatter_plt(pca_data, self.labels, alpha=1.0, ms=6, figsize=(9, 6), 
        title='After PCA', xlabel='First Eigenvector (PC1)', ylabel='Second Eigenvector (PC2)')
        
        if self.savepath:
            plt.plot(eigenvecs[1:], eigenvals, color='k', marker='+', linewidth=1)
            plt.grid(True, linestyle='--')

            plt.title('Scree Plot')
            plt.xlabel('Eigenvectors')
            plt.ylabel('Eigenvalues')

            plt.savefig('{}/pca_scree.png'.format(self.savepath))
            plt.clf()


            opt_prop = prop_var_sum[pcs]

            plt.plot(eigenvecs, prop_var_sum, color='k', marker='+', linewidth=1)
            plt.plot(pcs, opt_prop, markersize=6, marker='o', color='r')
            plt.text(pcs + 2, prop_var_sum[pcs] - 0.06, '({}, {})'.format(pcs, opt_prop))
            plt.grid(True, linestyle='--')

            plt.title("Proportion of Variance Summed")
            plt.xlabel("Eigenvectors")
            plt.ylabel("Prop of Var")

            plt.savefig('{}/pca_propvar.png'.format(self.savepath))
            plt.clf()