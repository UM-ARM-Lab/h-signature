#include <h_signature/h_signature.h>
#include <eigen3/unsupported/Eigen/Splines>

HSignature get_h_signature(Loop const &loop, Skeleton const & skeleton) {
    // discretize loop so we can integrate the field along it
    int n_disc = 1000;
    Loop const &loop_disc = discretize_loop(loop, n_disc);

    // compute the deltas between each discretized point on the loop
    Eigen::Matrix3Xd loop_deltas = loop_disc.rightCols(n_disc - 1) - loop_disc.leftCols(n_disc - 1);

    // iterate over the skeleton and compute the field at each point
    HSignature h;
    for (auto const & [name, obstacle_loop_i] : skeleton) {
        // compute the field direction
        Eigen::Matrix3Xd const &bs = skeleton_field_dirs(obstacle_loop_i, loop_disc, n_disc);

        // // integrate by summing the field directions and the deltas
        // double hd = (bs * loop_deltas).sum(1).sum(0);

        // // round to the nearest integer and take the absolute value
        // auto const h_i = static_cast<int>(std::abs(std::round(hd)));

        // // add to the H-signature
        // h.push_back(h_i);
    }

    return h;
}

Loop discretize_loop(Loop const & loop, int n_disc) {
    // ensure the loop is not empty
    if (loop.cols() == 0) {
        throw std::invalid_argument("The loop must not be empty.");
    }

    // also ensure the loop is closed
    if ((loop.col(0) - loop.col(loop.cols() - 1)).norm() > 1e-6) {
        throw std::invalid_argument("The loop must be closed.");
    }

    // linearly interpolate between the points on the loop
    Eigen::Spline3d spline = Eigen::SplineFitting<Eigen::Spline3d>::Interpolate(loop, 1);

    Eigen::Matrix3Xd loop_disc(3, n_disc);
    for (int i = 0; i < n_disc; i++) {
        auto const t = double(i) / n_disc;
        loop_disc.col(i) = spline(t);
    }

    return loop_disc;
}

Eigen::Matrix3Xd skeleton_field_dirs(Eigen::Matrix3Xd const & obstacle_loop_i, Loop const & loop_disc, int n_disc) {
    // Check that the loop is closed
    if ((obstacle_loop_i.col(0) - obstacle_loop_i.col(obstacle_loop_i.cols() - 1)).norm() > 1e-6) {
        throw std::invalid_argument("The loop must be closed.");
    }

    // convert the following python code to C++:
    // s_prev = skeleton[:-1][None]  # [1, n, 3]
    // s_next = skeleton[1:][None]  # [1, n, 3]

    // p_prev = s_prev - r[:, None]  # [b, n, 3]
    // p_next = s_next - r[:, None]  # [b, n, 3]
    // squared_segment_lens = squared_norm(s_next - s_prev, keepdims=True)
    // d = np.cross((s_next - s_prev), np.cross(p_next, p_prev)) / squared_segment_lens  # [b, n, 3]

    // # bs is a matrix [b, n,3] where each bs[i, j] corresponds to a line segment in the skeleton
    // squared_d_lens = squared_norm(d, keepdims=True)
    // p_next_lens = norm(p_next, axis=-1, keepdims=True) + 1e-6
    // p_prev_lens = norm(p_prev, axis=-1, keepdims=True) + 1e-6

    // # Epsilon is added to the denominator to avoid dividing by zero, which would happen for points _on_ the skeleton.
    // ε = 1e-6
    // d_scale = np.where(squared_d_lens > ε, 1 / (squared_d_lens + ε), 0)

    // bs = d_scale * (np.cross(d, p_next) / p_next_lens - np.cross(d, p_prev) / p_prev_lens)

    // b = bs.sum(axis=1) / (4 * np.pi)
    // return b

    Eigen::Matrix3Xd const s_prev = obstacle_loop_i.leftCols(obstacle_loop_i.cols() - 1);
    Eigen::Matrix3Xd const s_next = obstacle_loop_i.rightCols(obstacle_loop_i.cols() - 1);

}


// define the << operator for HSignature
std::ostream& operator<<(std::ostream& os, const HSignature& h_signature) {
    os << "[";
    for (auto const & h : h_signature) {
        os << h << ", ";
    }
    os << "], ";
    return os;
}