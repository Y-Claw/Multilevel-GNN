import os
import pickle
import numpy as np

cache_path = 'data_cache/'
book_path = os.path.join('data_cache', 'idx_book.pkl')
if os.path.exists(book_path):
    idx_book = pickle.load(open(book_path, 'rb'))
else:
    idx_book = None

def have_cached_data(patient_ids, cancer_type):
    global idx_book
    if idx_book is not None:
        return tuple(patient_ids+[cancer_type]) in idx_book
    else:
        return False

def cache_data(patient_ids, cancer_type, mutual_info_mask, mutual_info, pca_components, edges, edge_attrs, gene_pca_match):
    global idx_book
    if idx_book is None:
        idx_book = {}
        save_idx = 0
    else:
        save_idx = max(idx_book.values()) + 1
    idx_book[tuple(patient_ids+[cancer_type])] = save_idx
    pickle.dump(idx_book, open(book_path, 'wb+'))
    data_cache_path = os.path.join(cache_path, str(save_idx))
    os.mkdir(data_cache_path)
    pickle.dump(mutual_info_mask, open(os.path.join(data_cache_path, 'mutual_info_mask.pkl'), 'wb+'))
    pickle.dump(mutual_info, open(os.path.join(data_cache_path, 'mutual_info.pkl'), 'wb+'))
    pickle.dump(pca_components, open(os.path.join(data_cache_path, 'pca_components.pkl'), 'wb+'))
    pickle.dump(edges, open(os.path.join(data_cache_path, 'edges.pkl'), 'wb+'))
    pickle.dump(edge_attrs, open(os.path.join(data_cache_path, 'edge_attrs.pkl'), 'wb+'))
    pickle.dump(gene_pca_match, open(os.path.join(data_cache_path, 'gene_pca_match.pkl'), 'wb+'))


def get_cached_data(patient_ids, cancer_type):
    global idx_book
    save_idx = idx_book[tuple(patient_ids+[cancer_type])]
    data_cache_path = os.path.join(cache_path, str(save_idx))
    mutual_info_mask = pickle.load(open(os.path.join(data_cache_path, 'mutual_info_mask.pkl'), 'rb+'))
    mutual_info = pickle.load(open(os.path.join(data_cache_path, 'mutual_info.pkl'), 'rb+'))
    pca_components = pickle.load(open(os.path.join(data_cache_path, 'pca_components.pkl'), 'rb+'))
    edges = pickle.load(open(os.path.join(data_cache_path, 'edges.pkl'), 'rb+'))
    edge_attrs = pickle.load(open(os.path.join(data_cache_path, 'edge_attrs.pkl'), 'rb+'))
    gene_pca_match = pickle.load(open(os.path.join(data_cache_path, 'gene_pca_match.pkl'), 'rb+'))
    return mutual_info_mask, mutual_info, pca_components, edges, edge_attrs, gene_pca_match
