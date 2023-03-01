import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# laplacian eigenmaps
from scipy.sparse.csgraph import laplacian as csg_laplacian

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

    def scatter_plt(self, X, class_idxs, savename, ms=3, ax=None, alpha=0.1, 
                           legend=True, figsize=None, title=None, xlabel=None, ylabel=None):
        if self.savepath:
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

            
            plt.tight_layout()
            plt.savefig('{}/{}.png'.format(self.savepath, savename))
            plt.clf()
            return ax
        

    def dimred(self):
        algorithms = ['pca', 'mds', 'lda', 'lem', 'tsne']

        run = 'self.{}()'.format(self.alg.lower())
        try:
            exec(run)
        except:
            raise AttributeError('"{}" is not a valid algorithm.'.format(self.alg))

    def pca(self):

        def PCA(X, target_prop):
            dim = X.shape[1]
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
    
        ###################################################################################
        target_prop = 0.9
        savename = 'pca_scatter'

        eigenvecs = [i for i in range(self.dim + 1)]
        pca_data, prop_var_sum, eigenvals, pcs = PCA(self.data, target_prop, self.dim)

        self.scatter_plt(pca_data, self.labels, savename, alpha=1.0, ms=6, figsize=(9, 6), 
            title='Post PCA', 
            xlabel='First Eigenvector (PC1)', 
            ylabel='Second Eigenvector (PC2)')

        if self.savepath:
            
            # PCA Scree Plot
            plt.plot(eigenvecs[1:], eigenvals, color='k', marker='+', linewidth=1)
            plt.grid(True, linestyle='--')

            plt.title('Scree Plot')
            plt.xlabel('Eigenvectors')
            plt.ylabel('Eigenvalues')

            plt.savefig('{}/pca_scree.png'.format(self.savepath))
            plt.clf()

            # PCA Prop of Var
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

    def mds(self):

        def MDS(data, n_components=2):

            samples = data.shape[0]
            D = np.zeros([samples, samples])
            for i in tqdm(range(samples)):
                for j in range(samples):
                    # d_rs = ||xr - xs||^2
                    D[i][j] = np.sum(np.square(np.subtract(data[i], data[j])))

            H = np.identity(samples) - (np.ones(samples) / samples)
            B = -0.5 * np.matrix(H) * np.matrix(D) * np.matrix(H)

            eigenvalues, eigenvectors = np.linalg.eig(B)
            eigenvalues = np.array(eigenvalues).real
            eigenvectors = np.array(eigenvectors).real

            # Y = C*sqrt(lambda)
            # C = eigenvecs
            # lambda = diag matrix, diag values are eigenvals (n_components, n_components)

            lmbda = np.identity(n_components)
            for i in range(n_components):
                lmbda[i, i] = np.sqrt(eigenvalues[i])
                
            if self.verbose == 2:
                print('sqrt(lambda), largest 2 eigenvalues:')
                print(lmbda)
            Y = np.dot(eigenvectors[:, :n_components], lmbda)
            return Y

        ###################################################################################
        target_comp = 2
        savename = 'pca_scatter'

        data_mds = MDS(self.data, target_comp)

        self.scatter_plt(data_mds, self.labels, savename, alpha=1.0, ms=6, show=True, figsize=(9, 6), 
            title='Post MDS', 
            xlabel='First Eigenvector (Y1)', 
            ylabel='Second Eigenvector (Y2)')

    def lda(self):

        def LDA(X, y):

            dim = X.shape[1]
            class_labels = np.unique(y)

            # Within class scatter:
            # SW = sum(Si)

            # Between class scatter:
            # SB = sum(class_occurences * (mean_c - mean_overall).dot((mean_c - mean_overall).T))

            mean_overall = np.mean(X, axis=0)
            SW = np.zeros((dim, dim))
            SB = np.zeros((dim, dim))
            for c in class_labels:

                # class mean
                X_c = X[y == c]
                mean_c = np.mean(X_c, axis=0)
                SW += (X_c - mean_c).T.dot((X_c - mean_c))

                # how many times this class occurs
                n_c = X_c.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(dim, 1)
                SB += n_c * (mean_diff).dot(mean_diff.T)

            # Calc SW^-1 * SB
            A = np.linalg.pinv(SW).dot(SB)
            # Get eigenvalues and eigenvectors of SW^-1 * SB
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # no imag
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            
            # sort eigenvals desc.
            idxs = np.argsort(abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[:, idxs]

            if self.verbose == 2:
                print('Largest 2 eigenvalues:')
                print(eigenvalues[0])
                print(eigenvalues[1])

            return np.dot(X, eigenvectors)

        ###################################################################################
        savename = 'lda_scatter'

        W = LDA(self.data, np.array(self.labels))

        if self.verbose == 2:
            print('Projection matrix W (shape {}):'.format(W.shape))
            print(W)

        self.scatter_plt(W, self.labels, savename, alpha=1.0, ms=6, show=True, figsize=(9, 6), 
            title='Post LDA', 
            xlabel='First Eigenvector (LD1)', 
            ylabel='Second Eigenvector (LD2)')
        
    def lem(self):

        def LEM(B, samples, n_components=2):

            n_components += 1

            #laplacian
            laplacian, diag = csg_laplacian(B, normed=True, return_diag=True)
            laplacian *= -1

            # calc eigenval/vecs
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

            #ignore smallest eigenval, since it is ~= 0
            if self.verbose == 2:
                print('Smallest eigenvalue:')
                print(eigenvalues[-2])
                print('2nd smallest eigenvalue:')
                print(eigenvalues[-3])

            embedding = eigenvectors.T[-n_components:][::-1] / diag
            # correct sign
            for i in range(n_components):
                embedding[i] *= np.sign(embedding[i, np.argmax(np.abs(embedding[i]))])

            W = embedding[1:n_components].T
            if self.verbose == 2:
                print('Projection Matrix W:')
                print(W)
            return W
        
        ###################################################################################
        savename = 'lem_scatter'
        target_comp = 2
        sigma = 10

        B = np.zeros([self.samples, self.samples])
        for i in tqdm(range(self.samples)):
            for j in range(self.samples):
                B[i][j] = np.exp(-np.sum(np.square(np.subtract(self.data[i], self.data[j]))) / (2 * (sigma ** 2)))

        LEM = LEM(B, self.samples, n_components=target_comp)

        self.scatter_plt(LEM, self.labels, savename, alpha=1.0, ms=6, show=True, figsize=(9, 6), 
            title='Post LEM', 
            xlabel='First Eigenvector (LE1)', 
            ylabel='Second Eigenvector (LE2)')
        
    def tsne(self):

        ###################################################################################
        '''
        Default Configs 
        PERPLEXITY = 20
        SEED = 1                   
        MOMENTUM = 0.9
        LEARNING_RATE = 10.
        NUM_ITERS = 100                        
        NUM_PLOTS = 10         
        '''
        PERPLEXITY = 20
        SEED = 1                   
        MOMENTUM = 0.9
        LEARNING_RATE = 10.
        NUM_ITERS = 100                        
        NUM_PLOTS = 10   
        obj = TSNE(self.data, self.labels, self.savepath)      

        rng = np.random.RandomState(SEED)
        P = obj.p_joint(PERPLEXITY)

        Z = obj.estimate_sne(P, rng,
            num_iters =     NUM_ITERS,
            q_fn =          obj.q_tsne,
            grad_fn =       obj.tsne_grad,
            learning_rate = LEARNING_RATE,
            momentum =      MOMENTUM,
            plot =          NUM_PLOTS)

class TSNE:
    def __init__(self, data, labels, savepath):
        self.X = data
        self.labels = labels
        self.savepath = savepath

        # tsne_args = ['perplexity', 'momentum', 'learning_rate', 'num_iters', 'num_plots']
        # for count, arg in enumerate(tsne_args):
        #     run = 'self.{} = {}'.format(arg, config[count])
        #     exec(run)

    def scatter_plt(self, X, class_idxs, savename, ms=3, ax=None, alpha=0.1, 
                           legend=True, figsize=None, title=None, xlabel=None, ylabel=None):
        if self.savepath:
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

            
            plt.tight_layout()
            plt.savefig('{}/{}.png'.format(self.savepath, savename))
            plt.clf()
            return ax
        
    '''
    P-Joint Fn Set
    '''
    def neg_squared_euc_dists(self, X):
        # NxN matrix D, with entry D_ij = negative squared
        # euclidean distance between rows X_i and X_j 
        # sum of every row squared
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D

    def softmax(self, X, diag_zero=True):
        # softmax every row in X
        # Subtract max for numerical stability
        # diag probs = 0 if true
        # numerical stability 

        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        if diag_zero:
            np.fill_diagonal(e_x, 0.)
        e_x += 1e-8

        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    def calc_prob_matrix(self, distances, sigmas=None):
        # dist -> prob matrix
        if sigmas is not None:
            two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
            return self.softmax(distances / two_sig_sq)
        else:
            return self.softmax(distances)

    def binary_search(self, eval_fn, target, tol=1e-10, max_iter=10000, 
                    lower=1e-20, upper=1000.):
        # optimize eval_fn, until tolerant
        
        for _ in range(max_iter):
            # is float
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break

        return guess

    def calc_perplexity(self, distances, sigmas):
        #Perp(Pi)
        cond_prob = self.calc_prob_matrix(distances, sigmas)
        return 2 **-np.sum(cond_prob * np.log2(cond_prob), 1)

    def find_optimal_sigmas(self, distances, target_perplexity):
        # sigma for tgt perplexity for each dist row

        sigmas = [] 
        # For each row of the matrix (each point in our dataset)
        for i in range(distances.shape[0]):
            # Make fn that returns perplexity of this row given sigma
            eval_fn = lambda sigma: self.calc_perplexity(distances[i:i+1, :], np.array(sigma))
            # Binary search over sigmas to achieve target perplexity
            correct_sigma = self.binary_search(eval_fn, target_perplexity)
            # Append the resulting sigma to our output array
            sigmas.append(correct_sigma)
        return np.array(sigmas)

    def p_joint(self, perplexity):

        # negative squared euclidean dist
        # Find optimal sigma for each row of this distances matrix
        # pi|j
        #pij = (pj|i + pi|j) / 2N

        distances = self.neg_squared_euc_dists(self.X)
        sigmas = self.find_optimal_sigmas(distances, perplexity)
        P = self.calc_prob_matrix(distances, sigmas)
        return (P + P.T) / (2 * P.shape[0])        

    '''
    TSNE
    '''

    def q_tsne(self, Z):
    
        # compute pairwise affinities pij
        distances = self.neg_squared_euc_dists(Z)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances
    def tsne_grad(self, P, Q, Z, inv_distances):
        
        # gradient w.r.t Z
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        z_diffs = np.expand_dims(Z, 1) - np.expand_dims(Z, 0)

        # expand inverse dist to multiply more
        distances_expanded = np.expand_dims(inv_distances, 2)

        # Multiply this by inverse distances matrix
        z_diffs_wt = z_diffs * distances_expanded

        # Multiply then sum over j's
        grad = 4. * (pq_expanded * z_diffs_wt).sum(1)
        return grad

    def estimate_sne(self, P, rng, num_iters, q_fn, grad_fn, learning_rate,
                    momentum, plot):

        X = self.X
        y = self.labels
        # Set initial solution Z(theta) N(0, 10^-4)
        Z = rng.normal(0., 0.0001, [X.shape[0], 2])

        # Initialise past values (used for momentum)
        if momentum:
            Z_t1 = Z.copy()
            Z_t2 = Z.copy()

        # Start gradient descent loop
        for i in tqdm(range(num_iters)):
            # Get Q and distances

            # low-dim affinity
            Q, distances = q_fn(Z)
            # gradient (partialC / partialZ)
            grads = grad_fn(P, Q, Z, distances)

            # Update Z
            Z -= learning_rate * grads
            if momentum:  # Add momentum
                Z += momentum * (Z_t1 - Z_t2)
                # Update previous Z's for momentum
                Z_t2 = Z_t1.copy()
                Z_t1 = Z.copy()

            # plot 
            if plot and (i + 1) % (num_iters / plot) == 0:
                savename = 'tsne_scatter_{}'.format((i + 1) % (num_iters / plot))

                self.scatter_plt(Z, y, savename, alpha=1.0, ms=6, figsize=(9, 6), 
                title='Post TSNE', xlabel='First Eigenvector (Z1)', ylabel='Second Eigenvector (Z2)')

        return Z