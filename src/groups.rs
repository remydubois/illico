use ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// This is the basic Rust struct that will play the same role as a NamedTuple
pub struct GroupContainer<'py> {
    pub encoded_groups: ArrayView1<'py, usize>,
    pub counts: ArrayView1<'py, usize>,
    pub indices: ArrayView1<'py, usize>,
    pub indptr: ArrayView1<'py, usize>,
    pub encoded_ref_group: isize,
}

// This struct is necessary as it holds PyArrays (Python arrays), and not Arrays (Struct arrays). This is the input to our #[pyfunction]
pub struct GroupContainerNamedTuple<'py> {
    encoded_groups: PyReadonlyArray1<'py, usize>,
    counts: PyReadonlyArray1<'py, usize>,
    indices: PyReadonlyArray1<'py, usize>,
    indptr: PyReadonlyArray1<'py, usize>,
    encoded_ref_group: isize,
}

// This method is called automatically by Rust when converting arguments given in to Python interpreter into Rust types
impl<'py> FromPyObject<'py, 'py> for GroupContainerNamedTuple<'py> {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'py, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self {
            encoded_groups: obj.getattr("encoded_groups")?.extract()?,
            counts: obj.getattr("counts")?.extract()?,
            indices: obj.getattr("indices")?.extract()?,
            indptr: obj.getattr("indptr")?.extract()?,
            encoded_ref_group: obj.getattr("encoded_ref_group")?.extract()?,
        })
    }
}

// Convert from GroupContainerNamedTuple (Rust type) to GroupContainer, this one just converts the PyArrays to Rust's Arrays
impl<'py> GroupContainerNamedTuple<'py> {
    pub fn as_group_container(&'py self) -> GroupContainer<'py> {
        GroupContainer {
            encoded_groups: self.encoded_groups.as_array(),
            counts: self.counts.as_array(),
            indices: self.indices.as_array(),
            indptr: self.indptr.as_array(),
            encoded_ref_group: self.encoded_ref_group,
        }
    }
}
