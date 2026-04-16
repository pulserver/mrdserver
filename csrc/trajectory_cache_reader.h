/**
 * @file trajectory_cache_reader.h
 * @brief Standalone reader for pulseqlib binary cache trajectory data.
 *
 * No dependency on pulserverlib — reads the binary cache format directly.
 * Reads section 2 (GENINSTRUCTIONS) for rotation matrices and
 * section 5 (TRAJECTORY) for kshot library, encoding spaces, and table.
 */

#ifndef TRAJECTORY_CACHE_READER_H
#define TRAJECTORY_CACHE_READER_H

#include <vector>
#include <array>
#include <string>
#include <map>
#include <cstdint>

namespace mrdserver {

struct Kshot {
    std::vector<float> k; /**< k-space values [num_samples], Hz·s/m */
};

struct EncodingSpace {
    float fov[3];
    float matrix[3];
    float nav_fov[3];
    float nav_matrix[3];
    int   subseq_idx;
    int   nav_subseq_offset;
};

struct TrajTableEntry {
    int   kx_shot_id, ky_shot_id, kz_shot_id;
    float gx_amplitude, gy_amplitude, gz_amplitude;
    int   rotation_id;
    int   slc, seg, rep, avg, set, eco, phs, lin, par, acq;
    uint64_t flags;            /**< ISMRMRD-compatible flag bitmask    */
    int   center_sample;       /**< k-zero sample index                */
    float sample_time_us;      /**< ADC dwell time (us)                */
    int   encoding_space_ref;  /**< encoding space index                */
};

struct LabelLimit { int min, max; };

struct TrajectoryCache {
    std::vector<Kshot>                      kshots;
    std::vector<std::array<float, 9>>       rotations;
    std::vector<EncodingSpace>              encoding_spaces;
    std::vector<TrajTableEntry>             table;
    std::map<std::string, std::vector<std::string>> definitions;
    struct {
        LabelLimit slc, phs, rep, avg, seg, set, eco, par, lin, acq;
    } label_limits;
};

/**
 * Pre-computed trajectory data for a single encoding space.
 * ndim == 0 means Cartesian (no trajectory attached to acquisitions).
 * Layout: [ndim × num_samples] interleaved, repeated for each readout.
 * i.e. data[readout * ndim * num_samples + sample * ndim + dim]
 */
struct PrecomputedTrajectory {
    int ndim        = 0; /**< 0=Cartesian, 2 or 3 */
    int num_samples = 0; /**< ADC samples per readout */
    int num_readouts = 0;
    std::vector<float> data; /**< [num_readouts × num_samples × ndim] */
};

/**
 * @brief Read trajectory data from a pulseqlib binary cache file.
 *
 * @param cache_path  Path to the .bin cache file.
 * @return Populated TrajectoryCache; empty if file not found or
 *         trajectory section is missing.
 * @throws std::runtime_error on I/O or format errors.
 */
TrajectoryCache read_trajectory_cache(const std::string& cache_path);

/**
 * @brief Pre-compute per-encoding-space trajectory arrays from a loaded cache.
 *
 * Applies kshot scaling (amplitude), rotation matrices, and axis pruning.
 * Returns one PrecomputedTrajectory per encoding space; ndim==0 entries are
 * Cartesian and need no trajectory attached to ISMRMRD acquisitions.
 *
 * @param cache  Populated TrajectoryCache (from read_trajectory_cache).
 * @return Vector of PrecomputedTrajectory, indexed by encoding_space_ref.
 */
std::vector<PrecomputedTrajectory> pre_compute_trajectories(const TrajectoryCache& cache);

} // namespace mrdserver

#endif // TRAJECTORY_CACHE_READER_H
