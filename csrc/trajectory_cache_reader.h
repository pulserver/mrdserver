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
};

struct TrajectoryCache {
    std::vector<Kshot>                      kshots;
    std::vector<std::array<float, 9>>       rotations;
    std::vector<EncodingSpace>              encoding_spaces;
    std::vector<TrajTableEntry>             table;
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

} // namespace mrdserver

#endif // TRAJECTORY_CACHE_READER_H
