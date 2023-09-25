
#include <vector>
#include <map>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

// type alias for HSignature
typedef Eigen::Matrix3Xd Loop;
typedef std::map<std::string, Eigen::Matrix3Xd> Skeleton;
typedef std::vector<int> HSignature;
typedef Eigen::Tensor<double, 3> Tensor3d;
typedef Eigen::Tensor<double, 2> Tensor2d;

HSignature get_h_signature(Loop const &loop, Skeleton const & skeleton);

Loop discretize_loop(Loop const & loop, int n_disc);

/// @brief Computes the field direction at the input points, where the conductor is the skeleton of an obstacle.
// A skeleton is defined by a set of points in 3D, like a line-strip, and can represent only a genus-1 obstacle (donut)
// Assumes μ and I are 1.
// Based on this paper: https://www.roboticsproceedings.org/rss07/p02.pdf
// Variables in my code <--> math in the paper:

//     s_prev = s_i^j
//     s_next = s_i^j'
//     p_prev = p
//     p_next = p'

// Args:
//     skeleton: [3, 3] the points that define the skeleton
//     r: [3, b] the points at which to compute the field.
Eigen::Matrix3Xd skeleton_field_dirs(Loop const & obstacle_loop_i, Loop const & loop_disc);

/// @brief Computes the cross product between each pair of vectors in a and b.
/// @param a [3, n, m]
/// @param b [3, n, m]
/// @return [3, n, m]
Tensor3d batch_cross(Tensor3d a, Tensor3d b);

// define the << operator for HSignature
std::ostream& operator<<(std::ostream& os, const HSignature& h_signature);
