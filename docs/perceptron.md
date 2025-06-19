# 🧠 Perceptron – Vredict Library

The **Perceptron** is one of the earliest supervised learning algorithms, introduced by Frank Rosenblatt in 1958. It’s a **binary classifier** that updates its weights based on errors in predictions.

---
$$x^2$$
## 📌 Algorithm Summary

- **Goal**: Learn a linear decision boundary that separates two classes (binary labels 0 and 1).
- **Type**: Linear, online learning algorithm
- **Learning rule**: Update weights only when a misclassification occurs.

---

## 🔢 Mathematical Formulation

### Net Input Function

The **net input** \( z \) is calculated as:

\[
z = \mathbf{w}^\top \mathbf{x} + b
\]

Where:
- \( \mathbf{w} \) is the weight vector
- \( \mathbf{x} \) is the input sample
- \( b \) is the bias term

### Prediction Rule (Unit Step Function)

\[
\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}
\]

### Weight Update Rule

For each training sample:

\[
w_j := w_j + \eta \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}
\]
\[
b := b + \eta \cdot (y^{(i)} - \hat{y}^{(i)})
\]

---

## 🧪 Implementation Notes

- Weight vector `self.w_` is initialized from a normal distribution (mean = 0, std = 0.01).
- `self.b_` is the bias term.
- `self.errors_` tracks the number of misclassifications per epoch.
- The model supports reproducibility through a `random_state` seed.
- Training occurs over `n_iter` epochs using online (one-sample-at-a-time) updates.

---

## 🛠 Usage

```python
from vredict.perceptron import Perceptron

model = Perceptron(eta=0.01, n_iter=10, random_state=1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

