#include <h_signature/h_signature.h>
#include <eigen3/unsupported/Eigen/Splines>
#include <iostream>

HSignature get_h_signature(Loop const &loop, Skeleton const &skeleton)
{
    // dscrretize loop so we can integrate the field along it
    int n_dscr = 1000;
    Loop const &loop_dscr = discretize_loop(loop, n_dscr);

    // compute the deltas between each dscrretized point on the loop
    Eigen::Matrix3Xd loop_deltas = loop_dscr.rightCols(n_dscr - 1) - loop_dscr.leftCols(n_dscr - 1);

    // iterate over the skeleton and compute the field at each point
    HSignature h;
    for (auto const &[name, obstacle_loop_i] : skeleton)
    {
        // compute the field direction
        Eigen::Matrix3Xd const &bs = skeleton_field_dirs(obstacle_loop_i, loop_dscr.leftCols(n_dscr - 1));

        // integrate by summing the field directions and the deltas
        double h_d = bs.cwiseProduct(loop_deltas).sum();

        // round to the nearest integer and take the absolute value
        auto const h_i = static_cast<int>(std::abs(std::round(h_d)));

        // add to the H-signature
        h.push_back(h_i);
    }

    return h;
}

Loop discretize_loop(Loop const &loop, int n_dscr)
{
    // ensure the loop is not empty
    if (loop.cols() == 0)
    {
        throw std::invalid_argument("The loop must not be empty.");
    }

    // also ensure the loop is closed
    if ((loop.col(0) - loop.col(loop.cols() - 1)).norm() > 1e-6)
    {
        throw std::invalid_argument("The loop must be closed.");
    }

    // linearly interpolate between the points on the loop
    Eigen::Spline3d spline = Eigen::SplineFitting<Eigen::Spline3d>::Interpolate(loop, 1);

    Eigen::Matrix3Xd loop_dscr(3, n_dscr);
    for (int i = 0; i < n_dscr; i++)
    {
        auto const t = double(i) / n_dscr;
        loop_dscr.col(i) = spline(t);
    }

    return loop_dscr;
}

Eigen::Matrix3Xd skeleton_field_dirs(Eigen::Matrix3Xd const &obstacle_loop_i, Loop const &loop)
{
    // Check that the loop is closed
    if ((obstacle_loop_i.col(0) - obstacle_loop_i.col(obstacle_loop_i.cols() - 1)).norm() > 1e-6)
    {
        throw std::invalid_argument("The loop must be closed.");
    }

    Eigen::Matrix3Xd const s_prev = obstacle_loop_i.leftCols(obstacle_loop_i.cols() - 1);  // [3, n]
    Eigen::Matrix3Xd const s_next = obstacle_loop_i.rightCols(obstacle_loop_i.cols() - 1); // [3, n]

    // Extend the dimensions of to [3, n, 1] and [3, 1, m] respectively
    Eigen::TensorMap<Eigen::Tensor<const double, 3>> s_prev_tensor(s_prev.data(), 3, s_prev.cols(), 1);
    Eigen::TensorMap<Eigen::Tensor<const double, 3>> s_next_tensor(s_next.data(), 3, s_next.cols(), 1);
    Eigen::TensorMap<Eigen::Tensor<const double, 3>> loop_tensor(loop.data(), 3, 1, loop.cols());
    Tensor3d s_prev_tensor_b = s_prev_tensor.broadcast(Eigen::array<long int, 3>({1, 1, loop.cols()}));
    Tensor3d s_next_tensor_b = s_next_tensor.broadcast(Eigen::array<long int, 3>({1, 1, loop.cols()}));
    Tensor3d loop_tensor_b = loop_tensor.broadcast(Eigen::array<long int, 3>({1, s_prev.cols(), 1}));

    // Compute the 3D tensor of differences p between s_prev and r [3, n, m]
    Tensor3d p_prev = s_prev_tensor_b - loop_tensor_b; // [3, n, m]
    Tensor3d p_next = s_next_tensor_b - loop_tensor_b; // [3, n, m]

    // squared_segment_lens = squared_norm(s_next - s_prev, keepdims=True)
    // Compute the norm of each difference vector
    Eigen::VectorXd squared_segment_lens = (s_next - s_prev).colwise().norm(); // [4]

    // d = np.cross((s_next - s_prev), np.cross(p_next, p_prev)) / squared_segment_lens  # [b, n, 3]
    // Compute the cross product of p_next and p_prev
    // Tensor's don't have a cross function, so we write it out
    Tensor3d p_next_cross_p_prev = batch_cross(p_next, p_prev);
    Eigen::Matrix3Xd s_next_minus_s_pref = s_next - s_prev;
    // Expand the dimensions of s_next_minus_s_pref to [3, n, m]
    Tensor3d s_next_minus_s_pref_tensor = Eigen::TensorMap<Eigen::Tensor<const double, 3>>(s_next_minus_s_pref.data(), 3, s_next_minus_s_pref.cols(), 1);
    Tensor3d s_next_minus_s_pref_b = s_next_minus_s_pref_tensor.broadcast(Eigen::array<long int, 3>({1, 1, loop.cols()}));

    Tensor3d d_unnormalized = batch_cross(s_next_minus_s_pref_b, p_next_cross_p_prev); // [3, n, m]
    // broadcast squared_segment_lens to [3, n, m]
    Tensor3d squared_segment_lens_tensor = Eigen::TensorMap<Eigen::Tensor<const double, 3>>(squared_segment_lens.data(), 1, s_next_minus_s_pref.cols(), 1);
    Tensor3d squared_segment_lens_b = squared_segment_lens_tensor.broadcast(Eigen::array<long int, 3>({3, 1, loop.cols()}));

    Tensor3d d = d_unnormalized / squared_segment_lens_b; // [3, n, m]

    // # bs is a matrix [b, n,3] where each bs[i, j] corresponds to a line segment in the skeleton
    // squared_d_lens = squared_norm(d, keepdims=True)
    Tensor2d squared_d_lens = d.square().sum(Eigen::array<long int, 1>{0}); // [n, m]
    double eps = 1e-6;
    Tensor2d p_next_lens = p_next.square().sum(Eigen::array<long int, 1>{0}).sqrt() + eps;
    Tensor2d p_prev_lens = p_prev.square().sum(Eigen::array<long int, 1>{0}).sqrt() + eps;
    // # Epsilon is added to the denominator to avoid dividing by zero, which would happen for points _on_ the skeleton.
    // d_scale = np.where(squared_d_lens > ε, 1 / (squared_d_lens + ε), 0)
    Tensor2d squared_d_lens_inv = 1 / (squared_d_lens + eps);
    Eigen::Tensor<bool, 2> d_valid = (squared_d_lens > eps);
    Tensor2d zeros(Eigen::array<long int, 2>({squared_d_lens.dimension(0), squared_d_lens.dimension(1)}));
    zeros.setZero();
    Tensor2d d_scale = d_valid.select(squared_d_lens_inv, zeros); // [n, m]

    Tensor3d d_cross_p_next = batch_cross(d, p_next);
    Tensor3d d_cross_p_prev = batch_cross(d, p_prev);
    // p_next_lens is only [n,m] so first broadcast to [3, n, m]
    Tensor3d p_next_lens_tensor = Eigen::TensorMap<Eigen::Tensor<const double, 3>>(p_next_lens.data(), 1, p_next_lens.dimension(0), p_next_lens.dimension(1));
    Tensor3d p_prev_lens_tensor = Eigen::TensorMap<Eigen::Tensor<const double, 3>>(p_prev_lens.data(), 1, p_prev_lens.dimension(0), p_prev_lens.dimension(1));
    Tensor3d d_scale_3d = Eigen::TensorMap<Eigen::Tensor<const double, 3>>(d_scale.data(), 1, d_scale.dimension(0), d_scale.dimension(1));
    Tensor3d d_scale_b = d_scale_3d.broadcast(Eigen::array<long int, 3>({3, 1, 1}));
    Tensor3d p_next_lens_b = p_next_lens_tensor.broadcast(Eigen::array<long int, 3>({3, 1, 1}));
    Tensor3d p_prev_lens_b = p_prev_lens_tensor.broadcast(Eigen::array<long int, 3>({3, 1, 1}));
    Tensor3d bs = d_scale_b * (d_cross_p_next / p_next_lens_b - d_cross_p_prev / p_prev_lens_b); // [3, n, m]
    Tensor2d b = bs.sum(Eigen::array<long int, 1>{1}) / (4 * M_PI);                              // [3, m]

    // convert to Matrix
    Eigen::Matrix3Xd b_mat = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(b.data(), 3, b.dimension(1));
    std::cout << b_mat << std::endl;
    return b_mat;
}

Tensor3d batch_cross(Tensor3d a, Tensor3d b)
{
    Tensor2d cross_i = a.chip(1, 0) * b.chip(2, 0) - a.chip(2, 0) * b.chip(1, 0);
    Tensor2d cross_j = a.chip(2, 0) * b.chip(0, 0) - a.chip(0, 0) * b.chip(2, 0);
    Tensor2d cross_k = a.chip(0, 0) * b.chip(1, 0) - a.chip(1, 0) * b.chip(0, 0);
    Tensor3d cross = Tensor3d(Eigen::array<long int, 3>({3, a.dimension(1), a.dimension(2)}));
    cross.chip(0, 0) = cross_i;
    cross.chip(1, 0) = cross_j;
    cross.chip(2, 0) = cross_k;
    return cross;
}

// define the << operator for HSignature
std::ostream &operator<<(std::ostream &os, const HSignature &h_signature)
{
    os << "[";
    for (auto const &h : h_signature)
    {
        os << h << ", ";
    }
    os << "], ";
    return os;
}