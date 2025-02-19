#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
  pub bos_token_id: u32,
  pub eos_token_id: u32,
  pub hidden_size: usize,
  pub intermediate_size: usize,
  pub max_position_embeddings: usize,
  pub num_attention_heads: usize,
  pub num_hidden_layers: usize,
  pub num_key_value_heads: usize,
  pub vocab_size: usize,
  #[serde(default = "dflt_rms_norm_eps")]
  pub rms_norm_eps: f32,
  #[serde(default = "dflt_rope_theta")]
  pub rope_theta: f32,
  pub torch_dtype: String,
  #[serde(default = "dflt_tie_word_embeddings")]
  pub tie_word_embeddings: bool,
}

const fn dflt_rms_norm_eps() -> f32 {
  1e-5
}

const fn dflt_rope_theta() -> f32 {
  1e4
}

const fn dflt_tie_word_embeddings() -> bool {
  false
}
