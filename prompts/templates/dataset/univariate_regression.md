**Family:** univariate regression on `[0, 1]`.

- Input and target are 1-D scalars.
- Training and test points are sampled uniformly on the domain.
- The target is a symbolic expression of `x`. Whether training labels carry
  noise is stated in the synthesis code and the protocol below; the test split
  always uses the exact target function.
