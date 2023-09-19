
#include <vector>
#include <map>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

// type alias for HSignature
typedef Eigen::Matrix3Xd Loop;
typedef std::map<std::string, Eigen::Matrix3Xd> Skeleton;
typedef std::vector<int> HSignature;

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
Eigen::Matrix3Xd skeleton_field_dirs(Eigen::Matrix3Xd const & obstacle_loop_i, Loop const & loop_disc, int n_disc);

// define the << operator for HSignature
std::ostream& operator<<(std::ostream& os, const HSignature& h_signature);
