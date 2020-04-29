# cifar10-fast: a simplified reimplementation

This is simplified and refactored reimplementation of the [original fast CIFAR10 training code](https://github.com/davidcpage/cifar10-fast).
The code in this repository is currently based on the `demo.ipynb`.

This code was tested with PyTorch 1.4.0 and tries to follow PyTorch best practices.

```bash
./main.py
```

```
  1. epoch: train accuracy = 0.4099 (1.62e+00) test accuracy = 0.5293 (1.42e+00)
  2. epoch: train accuracy = 0.6660 (9.44e-01) test accuracy = 0.6067 (1.33e+00)
  3. epoch: train accuracy = 0.7457 (7.24e-01) test accuracy = 0.7249 (7.94e-01)
  4. epoch: train accuracy = 0.7828 (6.23e-01) test accuracy = 0.6958 (9.32e-01)
  5. epoch: train accuracy = 0.8075 (5.57e-01) test accuracy = 0.7876 (6.14e-01)
  6. epoch: train accuracy = 0.8286 (4.97e-01) test accuracy = 0.8264 (5.08e-01)
  7. epoch: train accuracy = 0.8486 (4.43e-01) test accuracy = 0.8287 (5.12e-01)
  8. epoch: train accuracy = 0.8602 (4.08e-01) test accuracy = 0.8145 (5.20e-01)
  9. epoch: train accuracy = 0.8671 (3.88e-01) test accuracy = 0.8042 (5.72e-01)
 10. epoch: train accuracy = 0.8762 (3.62e-01) test accuracy = 0.8469 (4.64e-01)
 11. epoch: train accuracy = 0.8826 (3.46e-01) test accuracy = 0.8411 (4.76e-01)
 12. epoch: train accuracy = 0.8909 (3.23e-01) test accuracy = 0.8332 (4.89e-01)
 13. epoch: train accuracy = 0.8958 (3.07e-01) test accuracy = 0.8714 (3.74e-01)
 14. epoch: train accuracy = 0.9041 (2.85e-01) test accuracy = 0.8425 (4.67e-01)
 15. epoch: train accuracy = 0.9103 (2.69e-01) test accuracy = 0.8801 (3.55e-01)
 16. epoch: train accuracy = 0.9144 (2.52e-01) test accuracy = 0.8811 (3.52e-01)
 17. epoch: train accuracy = 0.9219 (2.29e-01) test accuracy = 0.8900 (3.28e-01)
 18. epoch: train accuracy = 0.9277 (2.15e-01) test accuracy = 0.8957 (3.15e-01)
 19. epoch: train accuracy = 0.9392 (1.85e-01) test accuracy = 0.9041 (2.78e-01)
 20. epoch: train accuracy = 0.9465 (1.63e-01) test accuracy = 0.9041 (2.82e-01)
 21. epoch: train accuracy = 0.9533 (1.41e-01) test accuracy = 0.9272 (2.17e-01)
 22. epoch: train accuracy = 0.9638 (1.14e-01) test accuracy = 0.9211 (2.29e-01)
 23. epoch: train accuracy = 0.9731 (8.99e-02) test accuracy = 0.9387 (1.88e-01)
 24. epoch: train accuracy = 0.9785 (7.52e-02) test accuracy = 0.9406 (1.75e-01)
```
