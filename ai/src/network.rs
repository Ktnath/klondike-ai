use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::path::Path;

#[pyclass]
#[derive(Clone)]
pub struct NeuralNet {
    model: PyObject,
    optimizer: PyObject,
}

impl NeuralNet {
    pub fn new(input_shape: usize, action_size: usize) -> PyResult<Self> {
        Python::with_gil(|py| {
            let _torch = py.import("torch")?;
            let nn = py.import("torch.nn")?;
            let optim = py.import("torch.optim")?;

            // Définir l'architecture du réseau
            // Simple linear model as placeholder
            let model_any = nn.getattr("Linear")?.call1((input_shape, action_size))?;
            let model: PyObject = model_any.into_py(py);
            let params = model.as_ref(py).getattr("parameters")?.call0()?;
            let optimizer_any = optim.call_method1("Adam", (params,))?;
            let optimizer: PyObject = optimizer_any.into_py(py);

            Ok(Self { model, optimizer })
        })
    }

    pub fn train(&mut self, examples: Vec<(Vec<f32>, Vec<f32>, f32)>) -> PyResult<()> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;

            for (board, _pi, _v) in examples {
                let board_tensor = torch.call_method1("tensor", (board,))?;

                // Placeholder training step - to be implemented
                let _ = self.model.call_method1(py, "forward", (board_tensor,))?;
                self.optimizer.call_method0(py, "zero_grad")?;
                // TODO: compute loss and call backward
                self.optimizer.call_method0(py, "step")?;
            }

            Ok(())
        })
    }

    pub fn predict(&self, board: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;

            let board_tensor = torch.call_method1("tensor", (board,))?;
            let (pi, v) = self
                .model
                .call_method1(py, "forward", (board_tensor,))?
                .extract::<(PyObject, PyObject)>(py)?;

            let pi: Vec<f32> = pi.extract(py)?;
            let v: f32 = v.extract(py)?;

            Ok((pi, v))
        })
    }

    pub fn save_checkpoint(&self, folder: &str, filename: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            let path = Path::new(folder).join(filename);

            let state = py.eval("{ 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() }", Some([("model", self.model.as_ref(py)), ("optimizer", self.optimizer.as_ref(py))].into_py_dict(py)), None)?;

            torch.call_method1("save", (state, path.to_str().unwrap()))?;

            Ok(())
        })
    }

    pub fn load_checkpoint(&mut self, folder: &str, filename: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            let path = Path::new(folder).join(filename);

            let checkpoint = torch.call_method1("load", (path.to_str().unwrap(),))?;
            self.model.call_method1(py, "load_state_dict", (checkpoint.get_item("state_dict")?,))?;
            self.optimizer.call_method1(py, "load_state_dict", (checkpoint.get_item("optimizer")?,))?;

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_net_creation() {
        let result = NeuralNet::new(156, 96); // Dimensions pour Klondike
        assert!(result.is_ok());
    }

    #[test]
    fn test_prediction() {
        let net = NeuralNet::new(156, 96).unwrap();
        let board = vec![0.0; 156];
        let result = net.predict(board);
        assert!(result.is_ok());

        let (pi, v) = result.unwrap();
        assert_eq!(pi.len(), 96);
        assert!(v >= -1.0 && v <= 1.0);
    }
}