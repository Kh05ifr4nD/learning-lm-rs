mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
  let prj_dir = env!("CARGO_MANIFEST_DIR");
  let mdl_dir = PathBuf::from(prj_dir).join("model").join("story");
  let llama = model::Llama::<f32>::from_safetensors(&mdl_dir);
  let tknizer = Tokenizer::from_file(mdl_dir.join("tokenizer.json")).unwrap();
  let inp = "Once upon a time";
  let encing = tknizer.encode(inp, true).unwrap();
  let ls_inp_id = encing.get_ids();
  print!("\n{inp}");
  let ls_out_id = llama.generate(ls_inp_id, 500, 0.8, 30, 1.);
  println!("{}", tknizer.decode(&ls_out_id, true).unwrap());
}
