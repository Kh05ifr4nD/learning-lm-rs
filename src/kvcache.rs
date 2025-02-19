use crate::tensor::Tensor;
pub struct KVCache<T> {
  cache_k: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
  cache_v: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
  #[allow(unused)]
  len_max_seq: usize,
  d_qkv: usize,
  len_seq: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
  pub fn new(n_lyr: usize, len_max_seq: usize, d_qkv: usize, len_init: usize) -> Self {
    KVCache {
      cache_k: (0..n_lyr).map(|_| Tensor::default(&[len_max_seq, d_qkv])).collect(),
      cache_v: (0..n_lyr).map(|_| Tensor::default(&[len_max_seq, d_qkv])).collect(),
      len_max_seq,
      d_qkv,
      len_seq: len_init,
    }
  }

  pub fn cache_k(&mut self, lyr: usize, start: usize) -> Tensor<T> {
    self.cache_k[lyr].slice(start * self.d_qkv, &[self.len_seq - start, self.d_qkv])
  }

  pub fn cache_v(&mut self, lyr: usize, start: usize) -> Tensor<T> {
    self.cache_v[lyr].slice(start * self.d_qkv, &[self.len_seq - start, self.d_qkv])
  }

  pub fn inc(&mut self, seq_len: usize) {
    self.len_seq += seq_len;
  }

  pub fn len(&self) -> usize {
    self.len_seq
  }
}
