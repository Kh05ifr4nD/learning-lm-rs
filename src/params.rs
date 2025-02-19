use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
  // token_id to embedding lookup table
  pub embed_tbl: Tensor<T>, // (vocab_size, dim)
  // decoder layer
  pub w_rms_attn: Vec<Tensor<T>>, // (hidden_size, ) x layers
  pub w_k: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
  pub w_o: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
  pub w_q: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
  pub w_v: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
  // FFN layer
  pub w_rms_ffn: Vec<Tensor<T>>, // (hidden_size, ) x layers
  pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
  pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
  pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
  // output
  pub lm_head: Tensor<T>,   // (vocab_size, dim)
  pub w_rms_out: Tensor<T>, // (hidden_size, )
}

impl LLamaParams<f32> {
  pub fn from_safetensors(safetensor: &SafeTensors<'_>, cfg: &LlamaConfigJson) -> Self {
    let get_tensor = |name: &str| {
      let view = safetensor.tensor(name).unwrap();
      Tensor::new(
        unsafe {
          core::ptr::slice_from_raw_parts(
            #[allow(clippy::cast_ptr_alignment)]
            view.data().as_ptr().cast::<f32>(),
            view.data().len() / size_of::<f32>(),
          )
          .as_ref()
          .unwrap()
        }
        .to_vec(),
        view.shape(),
      )
    };

    let mut w_rms_attn = Vec::<Tensor<f32>>::new();
    let mut w_rms_ffn = Vec::<Tensor<f32>>::new();
    let mut w_down = Vec::<Tensor<f32>>::new();
    let mut w_gate = Vec::<Tensor<f32>>::new();
    let mut w_up = Vec::<Tensor<f32>>::new();
    let mut w_k = Vec::<Tensor<f32>>::new();
    let mut w_o = Vec::<Tensor<f32>>::new();
    let mut w_q = Vec::<Tensor<f32>>::new();
    let mut w_v = Vec::<Tensor<f32>>::new();

    for i in 0..cfg.num_hidden_layers {
      let prefix = format!("model.layers.{i}.");
      w_rms_attn.push(get_tensor(&format!("{prefix}input_layernorm.weight")));
      w_rms_ffn.push(get_tensor(&format!("{prefix}post_attention_layernorm.weight")));
      w_down.push(get_tensor(&format!("{prefix}mlp.down_proj.weight")));
      w_gate.push(get_tensor(&format!("{prefix}mlp.gate_proj.weight")));
      w_up.push(get_tensor(&format!("{prefix}mlp.up_proj.weight")));
      w_k.push(get_tensor(&format!("{prefix}self_attn.k_proj.weight")));
      w_o.push(get_tensor(&format!("{prefix}self_attn.o_proj.weight")));
      w_q.push(get_tensor(&format!("{prefix}self_attn.q_proj.weight")));
      w_v.push(get_tensor(&format!("{prefix}self_attn.v_proj.weight")));
    }

    LLamaParams {
      embed_tbl: if cfg.tie_word_embeddings {
        get_tensor("lm_head.weight")
      } else {
        get_tensor("model.embed_tokens.weight")
      },
      w_rms_attn,
      w_q,
      w_k,
      w_v,
      w_o,
      w_rms_ffn,
      w_up,
      w_gate,
      w_down,
      w_rms_out: get_tensor("model.norm.weight"),
      lm_head: get_tensor("lm_head.weight"),
    }
  }
}
