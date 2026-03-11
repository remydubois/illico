use ndarray::{Array1, ArrayView1};
use pyo3::prelude::*;

// TODO: assert shape and indptr match and indptr sorted
// impl CSCMatrix {
//     pub fn
// }
use num_traits::{Float, PrimInt, ToPrimitive, Zero};

pub trait SparseFloat: Float + Zero + Send + Sync + numpy::Element + 'static {
    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;
}

// impl SparseFloat for f16 {
//     fn to_f64(self) -> f64 { self as f64 }
//     fn to_f32(self) -> f32 { self as f32 }
// }
impl SparseFloat for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl SparseFloat for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}

pub trait SparseIndex: PrimInt + ToPrimitive + Send + Sync + numpy::Element + 'static {
    fn to_usize(self) -> usize;
}

impl SparseIndex for i32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl SparseIndex for i64 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

pub struct CSCMatrix<'a, D: SparseFloat = f32, I: SparseIndex = i32> {
    pub data: ArrayView1<'a, D>,
    pub indices: ArrayView1<'a, I>,
    pub indptr: ArrayView1<'a, I>,
    pub shape: (usize, usize),
}

// impl<'a, D: SparseFloat, I: SparseIndex> CSCMatrix<'a, D, I> {
//     pub fn new(data: ArrayView1<'a, D>, indices: ArrayView1<'a, I>, indptr: ArrayView1<'a, I>, shape: (usize, usize)) -> CSCMatrix<'a, D, I>{
//         let n_cols = shape.1;
//         let n_parcels = indptr.len() - 1;
//         if n_parcels != n_cols {
//             let error_str = format!("Ill-defined CSC matrix: {} columns but indptr contains {} parcels.", n_cols, n_parcels);
//             panic!("{error_str}")
//         } else {
//             CSCMatrix { data: data, indices: indices, indptr: indptr, shape }
//         }
//     }
// }

// macro_rules! new_sparse_matrix {
//     ($mat_type:ident, $axis:expr) => {
//         impl<'a, D: SparseFloat, I: SparseIndex> $mat_type<'a, D, I> {
//             pub fn new(data: ArrayView1<'a, D>, indices: ArrayView1<'a, I>, indptr: ArrayView1<'a, I>, shape: (usize, usize)) -> $mat_type<'a, D, I>{
//                 // let axis: usize = $axis;
//                 // let dim = if $axis == 0 {shape.0} else {shape.1};
//                 // let dim = if $axis ==  {shape.0} else {shape.1};
//                 let dim = match ($mat_type) {
//                     CSRMatrix => 0,
//                     OwnedCSRMatrix => 0,
//                     CSCMatrix => 0,
//                     OwnedCSCMatrix => 0,
//                     _ => panic!("Wrong matrix type.")
//                 };
//                 let n_parcels = indptr.len() - 1;
//                 if n_parcels != dim {
//                     let error_str = format!("Ill-defined matrix: {} {} but indptr contains {} parcels.", dim, if $axis == 0 {"rows"} else {"cols"}, n_parcels);
//                     panic!("{error_str}")
//                 } else {
//                     $mat_type { data: data, indices: indices, indptr: indptr, shape }
//                 }
//             }
//         }

//     };
// }
// new_sparse_matrix!(CSRMatrix, 0);

pub struct CSRMatrix<'py, D: SparseFloat = f32, I: SparseIndex = i32> {
    pub data: ArrayView1<'py, D>,
    pub indices: ArrayView1<'py, I>,
    pub indptr: ArrayView1<'py, I>,
    pub shape: (usize, usize),
}

pub struct OwnedCSCMatrix<D: SparseFloat = f32, I: SparseIndex = i32> {
    pub data: Array1<D>,
    pub indices: Array1<I>,
    pub indptr: Array1<I>,
    pub shape: (usize, usize),
}

pub struct OwnedCSRMatrix<D: SparseFloat = f32, I: SparseIndex = i32> {
    pub data: Array1<D>,
    pub indices: Array1<I>,
    pub indptr: Array1<I>,
    pub shape: (usize, usize),
}

pub struct PyCSRMatrix<'py> {
    pub data: Bound<'py, PyAny>,
    pub indices: Bound<'py, PyAny>,
    pub indptr: Bound<'py, PyAny>,
    pub shape: (usize, usize),
}

pub struct PyCSCMatrix<'py> {
    pub data: Bound<'py, PyAny>,
    pub indices: Bound<'py, PyAny>,
    pub indptr: Bound<'py, PyAny>,
    pub shape: (usize, usize),
}

impl<'py> FromPyObject<'py, 'py> for PyCSRMatrix<'py> {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'py, 'py, PyAny>) -> Result<PyCSRMatrix<'py>, Self::Error> {
        let data = obj.getattr("data")?.extract()?;
        let indices = obj.getattr("indices")?;
        let indptr = obj.getattr("indptr")?;
        let shape: (usize, usize) = obj.getattr("shape")?.extract()?;
        Ok(Self {
            data: data,
            indices: indices,
            indptr: indptr,
            shape: shape,
        })
    }
}

impl<'py> FromPyObject<'py, 'py> for PyCSCMatrix<'py> {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'py, 'py, PyAny>) -> Result<PyCSCMatrix<'py>, Self::Error> {
        let data = obj.getattr("data")?.extract()?;
        let indices = obj.getattr("indices")?;
        let indptr = obj.getattr("indptr")?;
        let shape: (usize, usize) = obj.getattr("shape")?.extract()?;
        Ok(Self {
            data: data,
            indices: indices,
            indptr: indptr,
            shape: shape,
        })
    }
}

// Those are never used as well in the final lib, only used for debugging
// impl OwnedCSCMatrix {
//     pub fn as_view(&self) -> CSCMatrix<'_> {
//         CSCMatrix {
//             data: self.data.view(),
//             indices: self.indices.view(),
//             indptr: self.indptr.view(),
//             shape: self.shape,
//         }
//     }
// }

// impl OwnedCSRMatrix {
//     pub fn as_view(&self) -> CSRMatrix<'_> {
//         CSRMatrix {
//             data: self.data.view(),
//             indices: self.indices.view(),
//             indptr: self.indptr.view(),
//             shape: self.shape,
//         }
//     }
// }

// fn extract_py_matrix<'py>(
//     obj: pyo3::Borrowed<'py, 'py, PyAny>,
// ) -> Result<
//     (
//         PyReadonlyArray1<'py, f32>,
//         PyReadonlyArray1<'py, i32>,
//         PyReadonlyArray1<'py, i32>,
//         (usize, usize),
//     ),
//     PyErr,
// > {
//     let data: PyReadonlyArray1<'py, f32> = obj.getattr("data")?.extract().map_err(|e| {
//         PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Could not extract data: {}", e))
//     })?;
//     let indices: PyReadonlyArray1<'py, i32> = obj.getattr("indices")?.extract().map_err(|e| {
//         PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Could not extract indices: {}", e))
//     })?;
//     let indptr: PyReadonlyArray1<'py, i32> = obj.getattr("indptr")?.extract().map_err(|e| {
//         PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Could not extract indptr: {}", e))
//     })?;
//     let shape: (usize, usize) = obj.getattr("shape")?.extract().map_err(|e| {
//         PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Could not extract shape: {}", e))
//     })?;
//     Ok((data, indices, indptr, shape))
// }

// Those are actually never used in the final library
// impl<'py> IntoPyObject<'py> for CSCMatrix<'py> {
//     type Target = PyTuple;
//     type Error = PyErr;
//     type Output = Bound<'py, PyTuple>;
//     fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
//         Ok((
//             PyArray1::from_array(py, &self.data),
//             PyArray1::from_array(py, &self.indices),
//             PyArray1::from_array(py, &self.indptr),
//             &self.shape,
//         )
//             .into_pyobject(py)?)
//     }
// }

// impl<'py> IntoPyObject<'py> for CSRMatrix<'py> {
//     type Target = PyTuple;
//     type Error = PyErr;
//     type Output = Bound<'py, PyTuple>;
//     fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
//         Ok((
//             PyArray1::from_array(py, &self.data),
//             PyArray1::from_array(py, &self.indices),
//             PyArray1::from_array(py, &self.indptr),
//             &self.shape,
//         )
//             .into_pyobject(py)?)
//     }
// }

// Those ones can no longer be used in the dtype agnostic setup, unless I do macros for all the dtypes combinations but its useless
// considering how simple the function is
// This one turns it into a proper CSCMatrix, with proper ndarray attributes
// Note: if my class had only native types, would I need the PyCSCMatrix at all ?
// impl<'py> PyCSCMatrix<'py> {
//     pub fn as_csc_matrix(&'py self) -> CSCMatrix<'py> {
//         CSCMatrix {
//             data: self.data.as_array(),
//             indices: self.indices.as_array(),
//             indptr: self.indptr.as_array(),
//             shape: self.shape,
//         }
//     }
// }

// impl<'py> PyCSRMatrix<'py> {
//     pub fn as_csr_matrix(&'py self) -> CSRMatrix<'py> {
//         CSRMatrix {
//             data: self.data.as_array(),
//             indices: self.indices.as_array(),
//             indptr: self.indptr.as_array(),
//             shape: self.shape,
//         }
//     }
// pub fn as_owned_csr_matrix(&'py self) -> OwnedCSRMatrix {
//     OwnedCSRMatrix {
//         data: self.data.as_array().to_owned(),
//         indices: self.indices.as_array().to_owned(),
//         indptr: self.indptr.as_array().to_owned(),
//         shape: self.shape,
//     }
// }
// }
