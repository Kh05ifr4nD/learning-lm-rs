use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, ls_idx: &Tensor<u32>, table: &Tensor<f32>) {
  let shp_tbl = table.shp();
  assert!(shp_tbl.len() == 2);
  let d = shp_tbl[1];
  let len = ls_idx.len();
  assert_eq!(y.len(), len * d);

  let ls_idx = ls_idx.data();
  let tbl = table.data();
  let y = unsafe { y.data_mut() };

  ls_idx.iter().enumerate().for_each(|(i, &index)| {
    let (src, dest) = (index as usize * d, i * d);
    y[dest..dest + d].copy_from_slice(&tbl[src..src + d]);
  });
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
  let shp = y.shp();
  assert!(shp.len() == 3);
  let (len_seq, n_head, d) = (shp[0], shp[1], shp[2]);

  let data = unsafe { y.data_mut() };
  for tk in 0..len_seq {
    let pos = start_pos + tk;
    for head in 0..n_head {
      for i in 0..d / 2 {
        let a = data[tk * n_head * d + head * d + i];
        let b = data[tk * n_head * d + head * d + i + d / 2];
        #[allow(clippy::cast_precision_loss)]
        let (sin, cos) = (pos as f32 / theta.powf((i * 2) as f32 / d as f32)).sin_cos();
        data[tk * n_head * d + head * d + i] = a * cos - b * sin;
        data[tk * n_head * d + head * d + i + d / 2] = b * cos + a * sin;
      }
    }
  }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
  let n_d = y.shp().len();
  assert!(n_d >= 2);
  let (len_seq, len_ttl_seq) = (y.shp()[n_d - 2], y.shp()[n_d - 1]);
  let batch = y.len() / (len_seq * len_ttl_seq);
  let data = unsafe { y.data_mut() };
  for b in 0..batch {
    let base = b * len_seq * len_ttl_seq;
    for i in 0..len_seq {
      let ofst = base + i * len_ttl_seq;
      let bdy = len_ttl_seq - len_seq + i + 1;
      let max = data[ofst..ofst + bdy].iter().fold(data[ofst], |a, &b| a.max(b));
      let sum = (0..bdy)
        .map(|j| {
          let e = (data[ofst + j] - max).exp();
          data[ofst + j] = e;
          e
        })
        .sum::<f32>();

      (0..bdy).for_each(|j| data[ofst + j] /= sum);
      data[ofst + bdy..ofst + len_ttl_seq].fill(0.0);
    }
  }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
  let shp_y = y.shp();
  let shp_x = x.shp();
  let shp_w = w.shp();
  let len_x = shp_x.len();
  let len_y = shp_y.len();
  assert!(len_y == len_x || (len_y == len_x + 1 && shp_y[0] == 1));
  let d = shp_y[len_y - 1];
  assert_eq!(d, shp_x[len_x - 1]);
  assert_eq!(d, w.len());

  let batch = shp_w[0];
  let n_batch = x.len() / batch;

  let (x, w, y) = (x.data(), w.data(), unsafe { y.data_mut() });

  for i in 0..n_batch {
    let ofst = i * batch;
    let (x, y) = (&x[ofst..ofst + batch], &mut y[ofst..ofst + batch]);
    #[allow(clippy::cast_precision_loss)]
    let rms = (x.iter().map(|x| x * x).sum::<f32>() / batch as f32 + epsilon).sqrt();
    w.iter().zip(x).zip(y).for_each(|((w, x), y)| {
      *y = w * x / rms;
    });
  }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
  assert_eq!(x.len(), y.len());
  unsafe { y.data_mut() }.iter_mut().zip(x.data()).for_each(|(y, x)| {
    *y *= x / (1. + (-x).exp());
  });
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
#[allow(non_snake_case)]
pub fn matmul_transb(
  mat_c: &mut Tensor<f32>,
  beta: f32,
  mat_a: &Tensor<f32>,
  mat_b: &Tensor<f32>,
  alpha: f32,
) {
  let (shp_a, shp_b, shp_c) = (mat_a.shp(), mat_b.shp(), mat_c.shp());
  assert!(shp_a.len() == 2 && shp_b.len() == 2 && shp_c.len() == 2, "维度不匹配");

  let (d_k, d_l, d_m) = (shp_a[0], shp_a[1], shp_b[0]);
  assert!(d_k == shp_c[0] && d_l == shp_b[1] && d_m == shp_c[1], "维度不匹配");

  let (mat_a, mat_b, mat_c) = (mat_a.data(), mat_b.data(), unsafe { mat_c.data_mut() });

  mat_c.iter_mut().enumerate().for_each(|(idx, elem_c)| {
    let (i, j) = (idx / d_m, idx % d_m);
    let row_a = &mat_a[i * d_l..(i + 1) * d_l];
    let row_b = &mat_b[j * d_l..(j + 1) * d_l];

    *elem_c = beta * *elem_c + alpha * row_a.iter().zip(row_b).fold(0.0, |acc, (a, b)| acc + a * b);
  });
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
  assert_eq!(x.len(), y.len());
  x.data().iter().zip(y.data()).map(|(&a, &b)| a * b).sum()
}

// Sample a index from a tensor (treated as a probability vector)
pub fn rand_samp(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
  #[derive(Clone, Copy, PartialEq, Debug)]
  struct Probability {
    val: f32,
    tok: u32,
  }
  impl Eq for Probability {}
  impl PartialOrd for Probability {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
      Some(self.cmp(other))
    }
  }
  impl Ord for Probability {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
      match self.val.total_cmp(&other.val) {
        std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
        ord => ord.reverse(),
      }
    }
  }
  impl From<(usize, &f32)> for Probability {
    #[inline]
    fn from((i, p): (usize, &f32)) -> Self {
      #[allow(clippy::cast_possible_truncation)]
      Self { val: *p, tok: i as _ }
    }
  }

  assert!(x.shp()[x.shp().len() - 1] == x.len());

  if temperature <= 0. || top_k < 2 || top_p <= 0. {
    #[allow(clippy::cast_possible_truncation)]
    return x.data().iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
      as _;
  }

  // sort
  let mut logits = x.data().iter().enumerate().map(Probability::from).collect::<Vec<_>>();
  logits.sort_unstable();
  let max = core::mem::replace(&mut logits[0].val, 1.);
  // softmax & sum
  for i in 1..logits.len() {
    logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
  }
  // topk & topp & random
  let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
  let pp = logits[logits.len() - 1].val * top_p;
  let plimit = rand::random::<f32>() * f32::min(pk, pp);
  // sample
  logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
  let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &[1, 3]);
  let x = Tensor::<f32>::new(vec![1., 2., 3.], &[1, 3]);
  swiglu(&mut y, &x);
  assert!(y.cl_to(&Tensor::<f32>::new(vec![1.462_117_2, 5.284_782_4, 11.43089], &[1, 3]), 1e-3));
}

#[test]
fn test_rms_norm() {
  let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &[2, 2]);
  let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &[2, 2]);
  let w = Tensor::<f32>::new(vec![1., 2.], &[2]);
  rms_norm(&mut y, &x, &w, 1e-6);
  assert!(y.cl_to(
    &Tensor::<f32>::new(vec![0.632_455_4, 2.529_821_6, 0.848_528_1, 2.262_741_6], &[2, 2]),
    1e-3
  ));
}

#[test]
fn test_matmul_transb() {
  let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &[2, 2]);
  let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
  let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
  matmul_transb(&mut c, 1., &a, &b, 1.);
  assert!(c.cl_to(&Tensor::<f32>::new(vec![15., 34., 35., 81.], &[2, 2]), 1e-3));
}
