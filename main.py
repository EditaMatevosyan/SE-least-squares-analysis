# # # main.py

# # import numpy as np
# # from least_squares import least_squares_normal, residual_analysis


# # def main():
# #     # Simple example: true model y = 2 + 3 t
# #     t = np.array([0., 1., 2., 3., 4.])
# #     y = 2 + 3 * t  # exact values, no noise
# #     # y = 2 + 3 * t + np.random.randn(len(t)) * 0.5  # noisy version

# #     # Build matrix A: [1, t_i]
# #     A = np.column_stack([
# #         np.ones_like(t),  # column of ones for intercept
# #         t                 # time feature
# #     ])

# #     # Call least squares solver
# #     x = least_squares_normal(A, y)

# #     # residual, residual_norm = residual_analysis(A, y, x)

# #     beta0, beta1 = x
# #     print("Estimated parameters:")
# #     print(f"  beta0 = {beta0:.4f}")
# #     print(f"  beta1 = {beta1:.4f}")

# #     # print("\nEstimated coefficients:", x)
# #     # print("\nResidual vector:", residual)
# #     # print("\nResidual norm ||r||_2 =", residual_norm)


# # if __name__ == "__main__":
# #     main()

# import numpy as np
# from least_squares import IncrementalLeastSquares

# def main():
#     # True model: y = 2 + 3 t
#     rng = np.random.default_rng(0)

#     # Simulate a larger dataset in two batches
#     t_all = np.linspace(0, 10, 1000)
#     y_all = 2 + 3 * t_all + rng.normal(0, 0.5, size=t_all.shape)  # noisy

#     # Build full A (just to split into batches)
#     A_all = np.column_stack([np.ones_like(t_all), t_all])

#     # Split into two batches
#     A_batch1, b_batch1 = A_all[:500], y_all[:500]
#     A_batch2, b_batch2 = A_all[500:], y_all[500:]

#     # Create incremental LS with 2 features (intercept + t)
#     inc_ls = IncrementalLeastSquares(n_features=2)

#     # Add first batch
#     inc_ls.add_batch(A_batch1, b_batch1)
#     x1 = inc_ls.solve()
#     print("After first batch:")
#     print("  beta0 ≈", x1[0])
#     print("  beta1 ≈", x1[1])

#     # Add second batch
#     inc_ls.add_batch(A_batch2, b_batch2)
#     x2 = inc_ls.solve()
#     print("\nAfter second batch (all data):")
#     print("  beta0 ≈", x2[0])
#     print("  beta1 ≈", x2[1])

# if __name__ == "__main__":
#     main()
