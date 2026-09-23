"""Regression checks for the Value class taught in the scratch notebook."""

import ast
import json
import math
from pathlib import Path
import unittest


def notebook_value():
    notebook = Path(__file__).resolve().parents[1] / "micrograd_from_scratch_yay.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells
                  if "class Value:" in "".join(c["source"]))
    definition = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    namespace = {"math": math}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace["Value"]


class NotebookGradientsTest(unittest.TestCase):
    def setUp(self):
        self.Value = notebook_value()

    def test_product_and_sum_chain(self):
        a, b, c, f = [self.Value(x) for x in (2, -3, 10, -2)]
        loss = (a * b + c) * f
        loss.backward()
        self.assertEqual(loss.data, -8)
        self.assertEqual([v.grad for v in (a, b, c, f)], [6, -4, -2, 4])

    def test_reused_operands_accumulate(self):
        for operation, expected in [(lambda x: x + x, 2),
                                    (lambda x: x * x, 6)]:
            with self.subTest(expected=expected):
                x = self.Value(3)
                operation(x).backward()
                self.assertEqual(x.grad, expected)

    def test_neuron_matches_finite_differences(self):
        inputs = [2.0, 0.0, -3.0, 1.0, 6.7]
        values = [self.Value(x) for x in inputs]
        x1, x2, w1, w2, b = values
        output = (x1 * w1 + x2 * w2 + b).tanh()
        output.backward()
        def forward(xs):
            return math.tanh(xs[0] * xs[2] + xs[1] * xs[3] + xs[4])
        self.assertAlmostEqual(output.data, forward(inputs))
        for index, value in enumerate(values):
            plus, minus = inputs.copy(), inputs.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            expected = (forward(plus) - forward(minus)) / 2e-6
            self.assertAlmostEqual(value.grad, expected, places=7)

    def test_tanh_accumulates_across_branches(self):
        x = self.Value(0)
        (x.tanh() + x.tanh()).backward()
        self.assertEqual(x.grad, 2)

    def test_tanh_handles_large_input(self):
        x = self.Value(1000)
        y = x.tanh()
        y.backward()
        self.assertEqual((y.data, x.grad), (1, 0))


if __name__ == "__main__":
    unittest.main()
