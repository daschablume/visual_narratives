"""
Based on Embeddings class from relatio package: https://github.com/relatio-nlp/relatio/blob/master/relatio/embeddings.py
Original copyright (c) 2023-2024 ETH Zurich, Andrei V. Plamada, et al.
Modified by Dariia Puhach, Info-Lab, Uppsala University, 2025.
Licensed under MIT License.
"""
import warnings
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


class Embeddings():
    """
    A wrapper class for generating text embeddings using SentenceTransformer.
    
    Attributes:
        normalize (bool): Whether to normalize the embedding vectors.
        size_vectors (int): Dimensionality of the embedding vectors.
    """
    
    def __init__(self, normalize: bool = True) -> None:
        """        
        Args:
            normalize (bool): If True, normalize embedding vectors to unit length.
                Defaults to True.
        """
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
        self._normalize = normalize
        self._size_vectors = self._model.get_sentence_embedding_dimension()

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def size_vectors(self) -> int:
        return self._size_vectors

    def get_vector(self, phrase: str) -> np.ndarray:
        """
        Generate an embedding vector for a single phrase.
                    
        Returns:
            np.ndarray: The embedding vector. Returns NaN array if encoding fails.
        """
        res = self._get_default_vector(phrase)

        # In case the result is fishy it will return a vector of np.nans and raise a warning
        if np.isnan(res).any() or np.count_nonzero(res) == 0:
            warnings.warn(
                f"Unable to compute an embedding for phrase: {phrase}.", RuntimeWarning
            )
            a = np.empty((self.size_vectors,))
            a[:] = np.nan
            return a

        if self.normalize:
            return res / norm(res)
        else:
            return res

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        """
        Encode a phrase using the underlying SentenceTransformer model.
            
        Returns:
            np.ndarray: The raw embedding vector from the model.
        """
        return self._model.encode(phrase)

    def get_vectors(self, phrases, progress_bar: bool = True) -> np.ndarray:
        """
        Generate embedding vectors for multiple phrases.
            
        Returns:
            np.ndarray: Array of embedding vectors with shape (n_phrases, embedding_dim).
        """
        phrases_list = list(phrases)
        
        if progress_bar:
            print("Computing phrase embeddings...")
            phrases_list = tqdm(phrases_list)

        vectors_list = []
        for phrase in phrases_list:
            vector = self.get_vector(phrase)
            vectors_list.append(vector)
        
        vectors = np.array(vectors_list)
        return vectors
