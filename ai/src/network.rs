use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use std::path::Path;

#[pyclass]
pub struct NeuralNet {
    model: PyObject,
    optimizer: PyObject,
}

impl NeuralNet {
    pub fn new(input_shape: usize, action_size: usize) -> PyResult<Self> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            let nn = py.import("torch.nn")?;
            let optim = py.import("torch.optim")?;

            // Définir l'architecture du réseau
            let model = py.eval("""
                class KlondikeNet(torch.nn.Module):
                    def __init__(self, input_shape, action_size):
                        super().__init__()
                        self.conv1 = torch.nn.Conv2d(input_shape[0], 128, 3, padding=1)
                        self.conv2 = torch.nn.Conv2d(128, 256, 3, padding=1)
                        self.conv3 = torch.nn.Conv2d(256, 256, 3)
                        
                        self.bn1 = torch.nn.BatchNorm2d(128)
                        self.bn2 = torch.nn.BatchNorm2d(256)
                        self.bn3 = torch.nn.BatchNorm2d(256)

                        self.fc1 = torch.nn.Linear(256 * (input_shape[1]-2) * (input_shape[2]-2), 1024)
                        self.fc_bn1 = torch.nn.BatchNorm1d(1024)

                        self.fc2 = torch.nn.Linear(1024, 512)
                        self.fc_bn2 = torch.nn.BatchNorm1d(512)

                        self.policy_head = torch.nn.Linear(512, action_size)
                        self.value_head = torch.nn.Linear(512, 1)

                    def forward(self, x):
                        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
                        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
                        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))

                        x = x.view(x.size(0), -1)

                        x = torch.nn.functional.relu(self.fc_bn1(self.fc1(x)))
                        x = torch.nn.functional.relu(self.fc_bn2(self.fc2(x)))

                        pi = torch.nn.functional.log_softmax(self.policy_head(x), dim=1)
                        v = torch.tanh(self.value_head(x))

                        return pi, v
            """, None, None)?;

            let model = model.call_method1("KlondikeNet", (input_shape, action_size))?;
            let optimizer = optim.call_method1("Adam", (model.getattr("parameters")()?,))?;

            Ok(Self { model, optimizer })
        })
    }

    pub fn train(&mut self, examples: Vec<(Vec<f32>, Vec<f32>, f32)>) -> PyResult<()> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;

            for (board, pi, v) in examples {
                let board_tensor = torch.call_method1("tensor", (board,))?;
                let pi_tensor = torch.call_method1("tensor", (pi,))?;
                let v_tensor = torch.call_method1("tensor", (v,))?;

                self.optimizer.call_method0("zero_grad")?;

                let (pi_pred, v_pred) = self.model.call_method1("forward", (board_tensor,))?.extract::<(PyObject, PyObject)>()?;

                let loss_pi = torch.call_method1("mean", (torch.call_method1("sum", (pi_tensor * pi_pred,))?,))?;
                let loss_v = torch.call_method1("mean", (torch.call_method1("pow", (v_tensor - v_pred, 2))?,))?;
                let total_loss = loss_pi + loss_v;

                total_loss.call_method0("backward")?;
                self.optimizer.call_method0("step")?;
            }

            Ok(())
        })
    }

    pub fn predict(&self, board: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;

            let board_tensor = torch.call_method1("tensor", (board,))?;
            let (pi, v) = self.model.call_method1("forward", (board_tensor,))?.extract::<(PyObject, PyObject)>()?;

            let pi: Vec<f32> = pi.extract()?;
            let v: f32 = v.extract()?;

            Ok((pi, v))
        })
    }

    pub fn save_checkpoint(&self, folder: &str, filename: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            let path = Path::new(folder).join(filename);

            let state = py.eval("{
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }", Some([(
                "model", &self.model),
                ("optimizer", &self.optimizer)
            ].into_py_dict(py)), None)?;

            torch.call_method1("save", (state, path.to_str().unwrap()))?;

            Ok(())
        })
    }

    pub fn load_checkpoint(&mut self, folder: &str, filename: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            let path = Path::new(folder).join(filename);

            let checkpoint = torch.call_method1("load", (path.to_str().unwrap(),))?;
            self.model.call_method1("load_state_dict", (checkpoint.get_item("state_dict")?,))?;
            self.optimizer.call_method1("load_state_dict", (checkpoint.get_item("optimizer")?,))?;

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