use std::rc::Rc;
pub struct Tensor<T> {
  data: Rc<Box<[T]>>,
  shp: Vec<usize>,
  ofst: usize,
  len: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
  pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
    Tensor {
      len: data.len(),
      data: Rc::new(data.into_boxed_slice()),
      shp: shape.to_owned(),
      ofst: 0,
    }
  }

  pub fn default(shape: &[usize]) -> Self {
    Self::new(vec![T::default(); shape.iter().product()], shape)
  }

  pub fn data(&self) -> &[T] {
    &self.data[self.ofst..self.ofst + self.len]
  }

  pub unsafe fn data_mut(&mut self) -> &mut [T] {
    core::slice::from_raw_parts_mut(self.data.as_ptr().add(self.ofst).cast_mut(), self.len)
  }

  pub fn shp(&self) -> &Vec<usize> {
    &self.shp
  }

  pub fn len(&self) -> usize {
    self.len
  }

  // Reinterpret the tensor as a new shape while preserving total size.
  pub fn reshp(&mut self, shp_new: &[usize]) -> &mut Self {
    assert_eq!(
      shp_new.iter().product::<usize>(),
      self.len,
      "New shape {shp_new:?} does not match tensor of {:?}",
      self.shp
    );
    self.shp.clone_from(&shp_new.to_vec());
    self
  }

  pub fn slice(&self, start: usize, shp: &[usize]) -> Self {
    let len_slice: usize = shp.iter().product();
    assert!(self.ofst + start + len_slice <= self.len);
    Tensor { data: self.data.clone(), shp: shp.to_vec(), ofst: self.ofst + start, len: len_slice }
  }
}

// Some helper functions for testing and debugging
impl Tensor<f32> {
  #[allow(unused)]
  pub fn cl_to(&self, oth: &Self, rel: f32) -> bool {
    if self.shp() == oth.shp() {
      self.data().iter().zip(oth.data()).all(|(&x, &y)| f32_eq(x, y, rel))
    } else {
      false
    }
  }
  #[allow(unused)]
  pub fn print(&self) {
    println!("shpae: {:?}, offset: {}, length: {}", self.shp, self.ofst, self.len);
    let dim = self.shp()[self.shp().len() - 1];
    let batch = self.len / dim;
    for i in 0..batch {
      let start = i * dim;
      println!("{:?}", &self.data()[start..][..dim]);
    }
  }
}

#[allow(clippy::inline_always)]
#[inline(always)]
pub fn f32_eq(x: f32, y: f32, rel: f32) -> bool {
  (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.
}
