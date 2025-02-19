use std::fs::File;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{gather, masked_softmax, matmul_transb, rand_samp, rms_norm, rope, swiglu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
  len_vocab: usize,       // vocab size
  n_lyr: usize,           // number of layers
  n_q_head: usize,        // number of heads for q
  n_kv_head: usize,       // number of heads for k and v
  d_hidden: usize,        // dimension of hidden states
  d_qkv: usize,           // length of a single q, k, or v vector
  d_int: usize,           // dimension of intermediate states
  eps: f32,               // epsilon for RMS normalization
  rope_theta: f32,        // rope theta for rope initialization
  len_max_seq: usize,     // maximum sequence length
  params: LLamaParams<T>, // trained weights of this model
  bos_token_id: u32,      // start token id
  eos_token_id: u32,      // end token id
}

impl Llama<f32> {
  pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
    let config: LlamaConfigJson =
      serde_json::from_reader(File::open(model_dir.as_ref().join("config.json")).unwrap()).unwrap();
    let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
    let safetensor = SafeTensors::deserialize(&model_file).unwrap();
    let params = LLamaParams::from_safetensors(&safetensor, &config);

    Self {
      len_vocab: config.vocab_size,
      n_lyr: config.num_hidden_layers,
      n_q_head: config.num_attention_heads,
      n_kv_head: config.num_key_value_heads,
      d_hidden: config.hidden_size,
      d_qkv: config.hidden_size / config.num_attention_heads,
      d_int: config.intermediate_size,
      eps: config.rms_norm_eps,
      rope_theta: config.rope_theta,
      len_max_seq: config.max_position_embeddings,
      params,
      bos_token_id: config.bos_token_id,
      eos_token_id: config.eos_token_id,
    }
  }

  pub fn new_cache(&self) -> KVCache<f32> {
    KVCache::new(self.n_lyr, self.len_max_seq, self.n_kv_head * self.d_qkv, 0)
  }

  pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
    let len_seq = input.len();
    let len_past_seq = cache.len();
    cache.inc(len_seq);
    let len_ttl_seq = len_past_seq + len_seq;
    let n_grp = self.n_q_head / self.n_kv_head;

    // Some pre-allocated buffers that will be reused
    let mut res = Tensor::<f32>::default(&[len_seq, self.d_hidden]);
    let mut hidden_stat = Tensor::<f32>::default(&[len_seq, self.d_hidden]);
    let mut buf_q = Tensor::<f32>::default(&[len_seq, self.n_q_head * self.d_qkv]);
    let mut attn_sc = Tensor::<f32>::default(&[self.n_kv_head, n_grp, len_seq, len_ttl_seq]);
    let mut buf_gate = Tensor::<f32>::default(&[len_seq, self.d_int]);
    let mut buf_up = Tensor::<f32>::default(&[len_seq, self.d_int]);

    // Computation Starts Here
    // Embedding lookup
    gather(&mut res, input, &self.params.embed_tbl);

    for lyr in 0..self.n_lyr {
      rms_norm(&mut hidden_stat, &res, &self.params.w_rms_attn[lyr], self.eps);

      let q = buf_q.reshp(&[len_seq, self.n_q_head * self.d_qkv]); // (seq, n_h * dqkv)
      let k = &mut cache.cache_k(lyr, len_past_seq); // (seq, n_kv_h * dqkv)
      let v = &mut cache.cache_v(lyr, len_past_seq); // (seq, n_kv_h * dqkv)
      matmul_transb(q, 0., &hidden_stat, &self.params.w_q[lyr], 1.0);
      matmul_transb(k, 0., &hidden_stat, &self.params.w_k[lyr], 1.0);
      matmul_transb(v, 0., &hidden_stat, &self.params.w_v[lyr], 1.0);
      rope(q.reshp(&[len_seq, self.n_q_head, self.d_qkv]), len_past_seq, self.rope_theta);
      rope(k.reshp(&[len_seq, self.n_kv_head, self.d_qkv]), len_past_seq, self.rope_theta);

      let full_k = &mut cache.cache_k(lyr, 0); // (total_seq, n_kv_h * dqkv)
      let full_v = &mut cache.cache_v(lyr, 0); // (total_seq, n_kv_h * dqkv)

      self_attention(
        &mut hidden_stat,
        &mut attn_sc,
        q,
        full_k,
        full_v,
        self.n_kv_head,
        n_grp,
        len_seq,
        len_ttl_seq,
        self.d_qkv,
      );
      matmul_transb(&mut res, 1., &hidden_stat, &self.params.w_o[lyr], 1.);

      mlp(
        &mut res,
        &mut hidden_stat,
        &mut buf_gate,
        &mut buf_up,
        &self.params.w_up[lyr],
        &self.params.w_down[lyr],
        &self.params.w_gate[lyr],
        &self.params.w_rms_ffn[lyr],
        self.eps,
      );
    }

    // No matter what seq_len, the output is always a 1D vector of length vocab,
    // which contains the probabilities for the next token.
    let mut logits = Tensor::<f32>::default(&[1, self.len_vocab]);
    let mut hidden_stat = hidden_stat.slice((len_seq - 1) * self.d_hidden, &[1, self.d_hidden]);
    let res = res.slice((len_seq - 1) * self.d_hidden, &[self.d_hidden]);

    rms_norm(&mut hidden_stat, &res, &self.params.w_rms_out, self.eps);
    matmul_transb(&mut logits, 0., &hidden_stat, &self.params.lm_head, 1.0);

    logits
  }

  pub fn generate(
    &self,
    token_ids: &[u32],
    len_max: usize,
    top_p: f32,
    top_k: u32,
    temp: f32,
  ) -> Vec<u32> {
    let mut rslt = vec![];
    let mut kvcache = self.new_cache();
    rslt.push(rand_samp(
      &self.forward(&Tensor::new(token_ids.to_vec(), &[token_ids.len()]), &mut kvcache),
      top_p,
      top_k,
      temp,
    ));
    while rslt.len() < len_max {
      let new_id = rand_samp(
        &self.forward(&Tensor::new(rslt[rslt.len() - 1..].to_vec(), &[1usize]), &mut kvcache),
        top_p,
        top_k,
        temp,
      );
      rslt.push(new_id);
      if new_id == self.eos_token_id {
        break;
      }
    }
    rslt
  }
}

fn self_attention(
  hidden_stat: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
  attn_sc: &mut Tensor<f32>,     // (n_kv_h, n_groups, seq, total_seq)
  q: &Tensor<f32>,               // (seq, n_kv_h * n_groups * dqkv)
  k: &Tensor<f32>,               // (total_seq, n_kv_h * dqkv)
  v: &Tensor<f32>,               // (total_seq, n_kv_h * dqkv)
  n_kv_head: usize,
  n_grp: usize,
  len_seq: usize,
  len_ttl_seq: usize,
  d_qkv: usize,
) {
  let q = q.data();
  let k = k.data();
  let v = v.data();
  let attn = unsafe { attn_sc.data_mut() };
  #[allow(clippy::cast_precision_loss)]
  for kv_head in 0..n_kv_head {
    for grp in 0..n_grp {
      for seq in 0..len_seq {
        for ttl_seq in 0..len_ttl_seq {
          attn[kv_head * n_grp * len_seq * len_ttl_seq
            + grp * len_seq * len_ttl_seq
            + seq * len_ttl_seq
            + ttl_seq] = q[(seq * n_kv_head * d_qkv * n_grp + kv_head * n_grp * d_qkv + grp * d_qkv)
            ..(seq * n_kv_head * d_qkv * n_grp + kv_head * n_grp * d_qkv + (grp + 1) * d_qkv)]
            .iter()
            .zip(
              &k[(ttl_seq * n_kv_head * d_qkv + kv_head * d_qkv)
                ..(ttl_seq * n_kv_head * d_qkv + (kv_head + 1) * d_qkv)],
            )
            .map(|(&q, &k)| q * k)
            .sum::<f32>() / (d_qkv as f32).sqrt();
        }
      }
    }
  }
  masked_softmax(attn_sc);
  let attn_sc = attn_sc.data();
  let hidden_stat = unsafe { hidden_stat.data_mut() };
  for kv_head in 0..n_kv_head {
    for grp in 0..n_grp {
      for seq in 0..len_seq {
        let hidden_base = seq * n_kv_head * n_grp * d_qkv + kv_head * n_grp * d_qkv + grp * d_qkv;
        attn_sc[kv_head * n_grp * len_seq * len_ttl_seq
          + grp * len_seq * len_ttl_seq
          + seq * len_ttl_seq
          ..kv_head * n_grp * len_seq * len_ttl_seq
            + grp * len_seq * len_ttl_seq
            + (seq + 1) * len_ttl_seq]
          .iter()
          .zip(v.chunks_exact(n_kv_head * d_qkv))
          .for_each(|(&attn_val, v_chunk)| {
            let v_values = &v_chunk[kv_head * d_qkv..(kv_head + 1) * d_qkv];

            hidden_stat[hidden_base..hidden_base + d_qkv]
              .iter_mut()
              .zip(v_values)
              .for_each(|(h, &v)| *h += attn_val * v);
          });
      }
    }
  }
}

fn mlp(
  res: &mut Tensor<f32>,
  hidden_stat: &mut Tensor<f32>,
  gate: &mut Tensor<f32>,
  up: &mut Tensor<f32>,
  w_up: &Tensor<f32>,
  w_down: &Tensor<f32>,
  w_gate: &Tensor<f32>,
  w_rms: &Tensor<f32>,
  eps: f32,
) {
  rms_norm(hidden_stat, res, w_rms, eps);
  matmul_transb(gate, 0., hidden_stat, w_gate, 1.);
  matmul_transb(up, 0., hidden_stat, w_up, 1.);
  swiglu(up, gate);
  matmul_transb(res, 1., up, w_down, 1.);
}

#[test]
pub fn test_mlp() {
  let seq_len = 4;
  let d = 2;
  let di = 3;
  let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &[seq_len, d]);
  let mut hidden_states = Tensor::<f32>::default(&[seq_len, d]);
  let mut gate_buf = Tensor::<f32>::default(&[seq_len, di]);
  let mut up_buf = Tensor::<f32>::default(&[seq_len, di]);
  let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[di, d]);
  let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[d, di]);
  let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[di, d]);
  let rms_w = Tensor::<f32>::new(vec![1., 1.], &[d]);
  let eps = 1e-6;
  mlp(
    &mut residual,
    &mut hidden_states,
    &mut gate_buf,
    &mut up_buf,
    &w_up,
    &w_down,
    &w_gate,
    &rms_w,
    eps,
  );

  assert!(residual.cl_to(
    &Tensor::<f32>::new(
      vec![
        1.342_996_4,
        1.729_073_9,
        1.342_996_4,
        1.729_073_9,
        1.342_996_4,
        1.729_073_9,
        1.342_996_4,
        1.729_073_9
      ],
      &[seq_len, d]
    ),
    1e-3
  ));
}

#[test]
pub fn test_load_safetensors() {
  use crate::tensor::f32_eq;
  use std::path::PathBuf;
  let project_dir = env!("CARGO_MANIFEST_DIR");
  let model_dir = PathBuf::from(project_dir).join("model").join("story");
  let model = Llama::from_safetensors(model_dir);
  assert_eq!(model.len_vocab, 2048);
  assert_eq!(model.n_lyr, 2);
  assert_eq!(model.n_q_head, 8);
  assert_eq!(model.n_kv_head, 4);
  assert_eq!(model.d_hidden, 128);
  assert_eq!(model.d_qkv, 16);
  assert_eq!(model.d_int, 384);

  assert!(f32_eq(model.params.embed_tbl.data()[50], 0.144_531_25, 1e-6));
  #[allow(clippy::float_cmp)]
  {
    assert_eq!(model.params.lm_head.data()[10], model.params.embed_tbl.data()[10]);
  }
  assert!(f32_eq(model.params.w_rms_attn[0].data()[10], 0.186_523_44, 1e-6));
  assert!(f32_eq(model.params.w_rms_ffn[1].data()[10], 0.324_218_75, 1e-6));
  assert!(f32_eq(model.params.w_rms_out.data()[100], 0.730_468_75, 1e-6));
  assert!(f32_eq(model.params.w_down[0].data()[100], -0.0625, 1e-6));
  assert!(f32_eq(model.params.w_up[0].data()[100], 1.46875, 1e-6));
  assert!(f32_eq(model.params.w_gate[1].data()[100], 0.296_875, 1e-6));
  assert!(f32_eq(model.params.w_q[1].data()[100], 0.032_226_563, 1e-6));
  assert!(f32_eq(model.params.w_k[1].data()[100], -0.213_867_19, 1e-6));
  assert!(f32_eq(model.params.w_v[0].data()[100], 0.041_015_625, 1e-6));
  assert!(f32_eq(model.params.w_o[0].data()[100], 0.019_653_32, 1e-6));
}
