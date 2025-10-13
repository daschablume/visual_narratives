import numpy as np
import pandas as pd
from relatio.embeddings import Embeddings
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import umap


# TODO: get rid of Relatio
EMBEDDING_MODEL = Embeddings(
    embeddings_type="SentenceTransformer",
    embeddings_model='all-MiniLM-L6-v2',
)


def print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector):
    v1, v2 = pca_phrase2vector[phrase1].reshape(1, -1), pca_phrase2vector[phrase2].reshape(1, -1)
    sim = cosine_similarity(v1, v2)[0, 0]
    print(f'PCA: {phrase1} <-> {phrase2}: {sim:.4f}')
    v1, v2 = umap_phrase2vector[phrase1].reshape(1, -1), umap_phrase2vector[phrase2].reshape(1, -1)
    sim = cosine_similarity(v1, v2)[0, 0]
    print(f'UMAP: {phrase1} <-> {phrase2}: {sim:.4f}')
    print('')


def inspect_similarity(phrase1, phrase2, phrase2vector):
    v1, v2 = phrase2vector[phrase1].reshape(1, -1), phrase2vector[phrase2].reshape(1, -1)
    sim = cosine_similarity(v1, v2)[0, 0]
    print(f'{phrase1} <-> {phrase2}: {sim:.4f}')


input_path = '../experiments5/np.csv'
batch_size = 10000

df = pd.read_csv(input_path)
df['count'] = df.groupby('word')['word'].transform('count')
phrases = list(set(df['word'].tolist()))
batch = phrases[0:batch_size]  # first batch
batch_vectors = EMBEDDING_MODEL.get_vectors(batch, progress_bar=True)
# reduce dimensionality here with PCA; number of components is 3 according to sihouette score
pca_args = {'n_components': 3, 'svd_solver': 'full'}
pca_model = PCA(**pca_args).fit(batch_vectors)
pca_training_vectors = pca_model.transform(batch_vectors)
pca_phrase2vector = dict(zip(batch, pca_training_vectors))

umap_args = {"n_neighbors": 15, "n_components": 2, "random_state": 0}
umap_model = umap.umap_.UMAP(**umap_args).fit(pca_training_vectors)
umap_training_vectors = umap_model.transform(pca_training_vectors)
umap_phrase2vector = dict(zip(batch, umap_training_vectors))


co2 = [
    'CO2 levels',
    'the atmospheric CO₂ level',
]

clean = ['100 % CLEAN ENERGY', 'clean renewable energy', 'a carbon neutral goal']

print('printing cosine similarity (SHOULD BE clustered together)')
for i in range(len(co2) - 1):
    phrase1, phrase2 = co2[i], co2[i+1]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)
for i in range(len(clean) - 1):
    phrase1, phrase2 = co2[i], co2[i+1]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)

print('\n')
print('printing cosine similarity (SHOULD NOT BE clustered together)')
for i in range(len(clean)):
    phrase1, phrase2 = co2[i], clean[i]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)


'''
Results from things up: UMAP overclusters, but PCA underclusters (both do it severely)
'''

# ----------
# TRY#2: UMAP with 10 dimensions (more than 2 like before)
# ----------

umap_args = {
    "n_neighbors": 15,        # Lower = preserve local structure better
    "n_components": 30,        # More components = less compression
    "min_dist": 0.1,           # Allow some spacing between points
    "metric": "cosine",
    "random_state": 0,
    "densmap": False,          # Standard UMAP
    "output_metric": "cosine"  # Ensure output respects cosine distances
}
umap_model = umap.umap_.UMAP(**umap_args).fit(pca_training_vectors)
umap_training_vectors = umap_model.transform(pca_training_vectors)
umap_phrase2vector = dict(zip(batch, umap_training_vectors))

print('printing cosine similarity (SHOULD BE clustered together)')
for i in range(len(co2) - 1):
    phrase1, phrase2 = co2[i], co2[i+1]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)
for i in range(len(clean) - 1):
    phrase1, phrase2 = co2[i], co2[i+1]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)

print('\n')
print('printing cosine similarity (SHOULD NOT BE clustered together)')
for i in range(len(clean)):
    phrase1, phrase2 = co2[i], clean[i]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)

'''
Results: again, UMAP overclusters
'''

# ----------
# Tuning of PCA
# ----------

import numpy as np
pca_full = PCA(svd_solver='full').fit(batch_vectors)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1

pca_model = PCA(n_components=n_components, svd_solver='full')
pca_training_vectors = pca_model.fit_transform(batch_vectors)
pca_phrase2vector = dict(zip(batch, pca_training_vectors))

print('printing cosine similarity (SHOULD BE clustered together)')
for i in range(len(co2) - 1):
    phrase1, phrase2 = co2[i], co2[i+1]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)
for i in range(len(clean) - 1):
    phrase1, phrase2 = co2[i], co2[i+1]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)



print('\n')
print('printing cosine similarity (SHOULD NOT BE clustered together)')
for i in range(len(clean)):
    phrase1, phrase2 = co2[i], clean[i]
    print_pca_umap_comparison(phrase1, phrase2, pca_phrase2vector, umap_phrase2vector)


# ----------
# Exploration of umap trustworthiness
# ----------

umap_args = {
    "n_neighbors": 15,        
    "n_components": 30,        
    "min_dist": 0.1,           
    "metric": "cosine",
    "random_state": 0,
    "densmap": False,          
    "output_metric": "cosine"  
}
umap_model = umap.umap_.UMAP(**umap_args).fit(pca_training_vectors)
umap_training_vectors = umap_model.transform(pca_training_vectors)
umap_phrase2vector = dict(zip(batch, umap_training_vectors))
    
trust_score = trustworthiness(batch_vectors, umap_training_vectors, n_neighbors=15)
print(f"Trustworthiness: {trust_score:.3f}")  # Closer to 1.0 is better

# Sample subset for computation efficiency
indices = np.random.choice(len(batch_vectors), size=1000, replace=False)

# Compute pairwise distances
original_distances = pdist(batch_vectors[indices], metric='cosine')
umap_distances = pdist(umap_training_vectors[indices], metric='cosine')

# Correlation between original and UMAP distances
correlation, p_value = spearmanr(original_distances, umap_distances)
print(f"Distance correlation: {correlation:.3f}")  # Higher is better

'''
Results: 
    Trustworthiness: 0.502
    Distance correlation: -0.002
Extremely poor results!
'''

# ----------
# Last attempt to tune UMAP
# ----------

umap_args = {
    "n_neighbors": 50,          # INCREASE: Larger neighborhoods = better global structure
    "n_components": 100,         # INCREASE: More dimensions = less compression/distortion
    "min_dist": 0.3,             # INCREASE: Allow more spacing between points
    "metric": "cosine",
    "random_state": 0,
    "densmap": False,
    "output_metric": "cosine",
    "negative_sample_rate": 5,   # Default, but can try 10-20 for better global structure
    "spread": 1.5                # INCREASE from default 1.0 = allows points to spread out more
}
umap_model = umap.umap_.UMAP(**umap_args).fit(pca_training_vectors)
umap_training_vectors = umap_model.transform(pca_training_vectors)
umap_phrase2vector = dict(zip(batch, umap_training_vectors))
    
trust_score = trustworthiness(batch_vectors, umap_training_vectors, n_neighbors=50)
print(f"Trustworthiness: {trust_score:.3f}")  # Closer to 1.0 is better

# Sample subset for computation efficiency
indices = np.random.choice(len(batch_vectors), size=1000, replace=False)

# Compute pairwise distances
original_distances = pdist(batch_vectors[indices], metric='cosine')
umap_distances = pdist(umap_training_vectors[indices], metric='cosine')

# Correlation between original and UMAP distances
correlation, p_value = spearmanr(original_distances, umap_distances)
print(f"Distance correlation: {correlation:.3f}")  # Higher is better

'''
Results:
Trustworthiness: 0.503
Distance correlation: -0.007
'''

# ----------
# Let's also measure it for the initial tuning just for fun
# ----------
pca_args = {'n_components': 50, 'svd_solver': 'full'}
pca_model = PCA(**pca_args).fit(batch_vectors)
pca_training_vectors = pca_model.transform(batch_vectors)
pca_phrase2vector = dict(zip(batch, pca_training_vectors))

umap_args = {"n_neighbors": 15, "n_components": 2, "random_state": 0}
umap_model = umap.umap_.UMAP(**umap_args).fit(pca_training_vectors)
umap_training_vectors = umap_model.transform(pca_training_vectors)
umap_phrase2vector = dict(zip(batch, umap_training_vectors))

trust_score = trustworthiness(batch_vectors, umap_training_vectors, n_neighbors=15)
print(f"Trustworthiness: {trust_score:.3f}")  # Closer to 1.0 is better

# Sample subset for computation efficiency
indices = np.random.choice(len(batch_vectors), size=1000, replace=False)

# Compute pairwise distances
original_distances = pdist(batch_vectors[indices], metric='cosine')
umap_distances = pdist(umap_training_vectors[indices], metric='cosine')

# Correlation between original and UMAP distances
correlation, p_value = spearmanr(original_distances, umap_distances)
print(f"Distance correlation: {correlation:.3f}")  # Higher is better

'''
Results:
Trustworthiness: 0.925
Distance correlation: 0.064
Ok, I ditch UMAP
'''

# ----------
# Let's find optimal PCA dimensions
# ----------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Fit PCA with all components first
pca_full = PCA(svd_solver='full').fit(batch_vectors)

# Plot cumulative explained variance
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
# Find n_components for 90% variance
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
print(f"Components for 90% variance: {n_components_90}")
print(f"Components for 95% variance: {np.argmax(cumsum_variance >= 0.95) + 1}")

pca = PCA(n_components=n_components_90, svd_solver='full')
pca_model = PCA(**pca_args).fit(batch_vectors)
pca_training_vectors = pca_model.transform(batch_vectors)
pca_phrase2vector = dict(zip(batch, pca_training_vectors))


# --------
# Ok, Claude suggests me to pick the best number of components using silhouette score
# ---------


def evaluate_pca_for_clustering(vectors, n_components_list, n_clusters=10):
    """
    Test different PCA dimensions and evaluate clustering quality
    """
    results = []
    
    for n_comp in n_components_list:
        # Apply PCA
        pca = PCA(n_components=n_comp, svd_solver='full')
        reduced_vectors = pca.fit_transform(vectors)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced_vectors)
        
        # Evaluate
        silhouette = silhouette_score(reduced_vectors, labels, metric='cosine')
        davies_bouldin = davies_bouldin_score(reduced_vectors, labels)
        variance_explained = pca.explained_variance_ratio_.sum()
        
        results.append({
            'n_components': n_comp,
            'silhouette': silhouette,  # Higher is better
            'davies_bouldin': davies_bouldin,  # Lower is better
            'variance_explained': variance_explained
        })
        
        print(f"n_components={n_comp}: Silhouette={silhouette:.3f}, "
              f"DB={davies_bouldin:.3f}, Variance={variance_explained:.3f}")
    
    return results


'''
The best result is actually 3. Ok, let's try clustering
'''

#  --------
# Inspect similarity
# --------

pca_args = {'n_components':3, 'svd_solver': 'full'}  # ATTENTION: n_components == 3
pca_model = PCA(**pca_args).fit(batch_vectors)
training_vectors = pca_model.transform(batch_vectors)
phrase2vector = dict(zip(batch, training_vectors))
id2phrase = dict(zip(range(len(batch)), batch))

tests = [
    "the environment and people 's health",
    'the homes and the environment',
    'new fossil fuel projects',
    'environmental injustice',
    'fossil fuel executives',
    'low - carbon investments',
    'a clean environment',
    'better environmental practices',
    'environmental issues and social justice',
    'environmental justice and voting',
    'human activity and urbanization',
    'IPCC and UN scientific reports',
    'environmental awareness and protection',
    'the factory and the environment',
    'environmental sustainability'
]

for i in range(len(tests) - 1):
    phrase1, phrase2 = tests[i], tests[i+1]
    inspect_similarity(phrase1, phrase2, phrase2vector)


# ----------
# Let's try it out with higher n_components
# ----------
pca_args = {'n_components':50, 'svd_solver': 'full'}  # ATTENTION: n_components == 3
pca_model = PCA(**pca_args).fit(batch_vectors)
training_vectors = pca_model.transform(batch_vectors)
phrase2vector = dict(zip(batch, training_vectors))
id2phrase = dict(zip(range(len(batch)), batch))

for i in range(len(tests) - 1):
    phrase1, phrase2 = tests[i], tests[i+1]
    inspect_similarity(phrase1, phrase2, phrase2vector)


# ----------
# Clustering
# ----------
clust = AgglomerativeClustering(
    metric="cosine",
    linkage="complete",
    distance_threshold=1 - 0.7,
    n_clusters=None
)
labels = clust.fit_predict(training_vectors)

from collections import defaultdict

clusters = defaultdict(list)
for i, lbl in enumerate(labels):
    clusters[lbl].append(i)

clusters = list(clusters.values())

tests2 = ['frustration with climate change and a call for action or change',
'climate change awareness or action',
'the end of the day and the urgency of climate change',
'the climate change awareness campaign',
'the urgency of climate change and the need for collective effort',
'climate change awareness or efforts',
'climate change policies and actions',
'climate change and the urgency of action',
'climate change and the need to act',
'the urgency of addressing climate change and the need for action to prevent extinction',
'climate change and the need for action',
'global unity and the importance of diverse voices in addressing climate change',
'climate change and the importance of encouraging action',
'the urgency of climate change and the need for immediate action',
'climate change action now',
'the urgency of addressing climate change and the importance of protecting the planet',
"action and recognition of climate change 's urgency",
'The climate change awareness campaign',
'the urgency of climate change and the collective effort to raise awareness',
'change and action on climate change',
'hope and the need for action on climate change',]


for i in range(len(tests2) - 1):
    phrase1, phrase2 = tests2[i], tests2[i+1]
    inspect_similarity(phrase1, phrase2, phrase2vector)