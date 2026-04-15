/**
 * @file trajectory_cache_reader.cpp
 * @brief Standalone reader for pulseqlib binary cache.
 *
 * Reads two sections from the cache:
 *   - Section 2 (GENINSTRUCTIONS): rotation matrices
 *   - Section 5 (TRAJECTORY): kshot library, encoding spaces, table
 *
 * No dependency on pulserverlib. All integer and float fields are 4 bytes.
 */

#include "trajectory_cache_reader.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace mrdserver {

namespace {

constexpr int32_t CACHE_ENDIAN_MARKER          = 0x01020304;
constexpr int     SECTION_GENINSTRUCTIONS      = 2;
constexpr int     SECTION_TRAJECTORY           = 5;
constexpr int     MAX_GRAD_SHOTS               = 16;

// ---------- byte-swap helpers ----------

void swap4(void* p) {
    auto* b = static_cast<uint8_t*>(p);
    std::swap(b[0], b[3]);
    std::swap(b[1], b[2]);
}

void swap4_array(void* p, int count) {
    for (int i = 0; i < count; ++i)
        swap4(static_cast<uint8_t*>(p) + static_cast<size_t>(i) * 4);
}

// ---------- typed I/O ----------

bool read4(std::ifstream& f, void* p, int count) {
    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(count) * 4);
    return f.good();
}

int read_int(std::ifstream& f, bool do_swap) {
    int32_t v;
    if (!read4(f, &v, 1)) throw std::runtime_error("unexpected EOF in cache");
    if (do_swap) swap4(&v);
    return v;
}

float read_float(std::ifstream& f, bool do_swap) {
    float v;
    if (!read4(f, &v, 1)) throw std::runtime_error("unexpected EOF in cache");
    if (do_swap) swap4(&v);
    return v;
}

void skip_ints(std::ifstream& f, int count) {
    f.seekg(static_cast<std::streamoff>(count) * 4, std::ios::cur);
    if (!f.good()) throw std::runtime_error("unexpected EOF skipping ints");
}

// ---------- read rotation matrices from GENINSTRUCTIONS (section 2) ----------

std::vector<std::array<float, 9>> read_rotations_from_geninstructions(
    std::ifstream& f, long section_offset, int section_size, bool do_swap)
{
    std::vector<std::array<float, 9>> rotations;

    f.seekg(section_offset, std::ios::beg);
    if (!f.good()) throw std::runtime_error("cannot seek to GENINSTRUCTIONS");

    // First, skip the collection-level header (6 ints):
    //   num_subsequences, num_repetitions, total_unique_segments,
    //   total_unique_adcs, total_blocks, total_duration_us
    int num_subseq = read_int(f, do_swap);
    skip_ints(f, 5); // remaining collection header fields

    // For each subsequence, skip its info (4 ints) then read the descriptor
    // We only need the first subsequence's rotations for now
    for (int s = 0; s < num_subseq; ++s) {
        // subsequence_info: 4 ints
        skip_ints(f, 4);

        // --- Descriptor scalars: 11 ints + 12 floats ---
        skip_ints(f, 11);  // num_prep_blocks..vendor
        skip_ints(f, 12);  // fov[3], matrix[3], nav_fov[3], nav_matrix[3]

        // --- Block definitions ---
        int num_unique_blocks = read_int(f, do_swap);
        skip_ints(f, num_unique_blocks * 7);

        // --- Block table ---
        int num_blocks = read_int(f, do_swap);
        skip_ints(f, num_blocks * 16);

        // --- RF definitions ---
        int num_unique_rfs = read_int(f, do_swap);
        // each RF def: 6 ints + 13 floats (stats) = 19 fields (but stats has 11 float + 2 int...)
        // Actually: id, mag_shape_id, phase_shape_id, time_shape_id, delay, num_channels (6 ints)
        // then rf_stats: 8 float + 1 int + 2 float + 1 int + 1 int = 13 x 4 bytes
        // Total: 6 + 13 = 19 fields per rf_def (but let's check)
        // rf_stats: flip_angle_deg(f), act_amplitude_hz(f), area(f), abs_width(f),
        //   eff_width(f), duty_cycle(f), max_pulse_width(f), duration_us(f),
        //   isodelay_us(i), bandwidth_hz(f), base_amplitude_hz(f),
        //   num_samples(i), num_instances(i) = 13 fields
        // rf_def: 6 + 13 = 19 fields
        skip_ints(f, num_unique_rfs * 19);

        // --- RF table ---
        int rf_table_size = read_int(f, do_swap);
        // per row: id(i), amplitude(f), freq_offset(f), phase_offset(f), rf_use(i) = 5 fields
        skip_ints(f, rf_table_size * 5);

        // --- Grad definitions ---
        int num_unique_grads = read_int(f, do_swap);
        // per grad_def: 8 ints + 7 * MAX_GRAD_SHOTS floats
        //   id, type, rise_time_or_unused, flat_time_or_unused,
        //   fall_time_or_num_uncompressed_samples, unused_or_time_shape_id,
        //   delay, num_shots (8 ints)
        //   shot_shape_ids[16] (16 ints), max_amplitude[16], min_amplitude[16],
        //   slew_rate[16], energy[16], first_value[16], last_value[16]
        //   = 8 + 16 + 6*16 = 8 + 16 + 96 = 120 fields
        // Wait, shot_shape_ids is int[16], that's 16 ints.
        // max_amplitude..last_value are 6 * float[16] = 96 floats
        // Total: 8 + 16 + 96 = 120 fields per grad_def
        skip_ints(f, num_unique_grads * 120);

        // --- Grad table ---
        int grad_table_size = read_int(f, do_swap);
        skip_ints(f, grad_table_size * 3);  // id, shot_index, amplitude

        // --- ADC definitions ---
        int num_unique_adcs = read_int(f, do_swap);
        skip_ints(f, num_unique_adcs * 4);

        // --- ADC table ---
        int adc_table_size = read_int(f, do_swap);
        skip_ints(f, adc_table_size * 3);

        // --- Freq-mod definitions (legacy, count=0 expected) ---
        int num_fmod = read_int(f, do_swap);
        // each: id, num_samples, raster_us, duration_us, then waveform arrays + ref_integral + ref_time
        // For the cache, freq_mod_defs are written as count=0, so skip nothing
        if (num_fmod > 0) {
            throw std::runtime_error("non-zero freq_mod_definitions in cache not supported");
        }

        // --- RF shim definitions ---
        int num_shims = read_int(f, do_swap);
        for (int sh = 0; sh < num_shims; ++sh) {
            skip_ints(f, 1);  // id
            int nch = read_int(f, do_swap);
            skip_ints(f, nch * 2);  // magnitudes[nch] + phases[nch]
        }

        // --- Rotations (THIS IS WHAT WE WANT) ---
        int num_rotations = read_int(f, do_swap);
        rotations.resize(static_cast<size_t>(num_rotations));
        for (int r = 0; r < num_rotations; ++r) {
            if (!read4(f, rotations[r].data(), 9))
                throw std::runtime_error("EOF reading rotations");
            if (do_swap) swap4_array(rotations[r].data(), 9);
        }

        // Only read first subsequence's rotations
        break;
    }

    return rotations;
}

} // anonymous namespace

// ---------- Public API ----------

TrajectoryCache read_trajectory_cache(const std::string& cache_path)
{
    TrajectoryCache cache;

    std::ifstream f(cache_path, std::ios::binary);
    if (!f.is_open()) return cache;  // empty if file not found

    // Read header
    int32_t marker;
    if (!read4(f, &marker, 1)) return cache;

    bool do_swap = false;
    if (marker != CACHE_ENDIAN_MARKER) {
        swap4(&marker);
        if (marker != CACHE_ENDIAN_MARKER)
            throw std::runtime_error("invalid cache file: bad endian marker");
        do_swap = true;
    }

    int version_major = read_int(f, do_swap);
    int version_minor = read_int(f, do_swap);
    int vendor        = read_int(f, do_swap);
    int stored_size   = read_int(f, do_swap);
    int num_sections  = read_int(f, do_swap);

    (void)version_minor; (void)vendor; (void)stored_size;

    if (num_sections <= 0 || num_sections > 16)
        throw std::runtime_error("invalid cache: bad num_sections");

    // Read section index
    struct SectionEntry { int id, offset, size; };
    std::vector<SectionEntry> sections(static_cast<size_t>(num_sections));
    for (int i = 0; i < num_sections; ++i) {
        sections[i].id     = read_int(f, do_swap);
        sections[i].offset = read_int(f, do_swap);
        sections[i].size   = read_int(f, do_swap);
    }

    // Find sections 2 (GENINSTRUCTIONS) and 5 (TRAJECTORY)
    const SectionEntry* geninst_section = nullptr;
    const SectionEntry* traj_section    = nullptr;
    for (auto& s : sections) {
        if (s.id == SECTION_GENINSTRUCTIONS) geninst_section = &s;
        if (s.id == SECTION_TRAJECTORY)      traj_section    = &s;
    }

    if (!traj_section) return cache;  // no trajectory data

    // Read rotation matrices from GENINSTRUCTIONS
    if (geninst_section) {
        cache.rotations = read_rotations_from_geninstructions(
            f, geninst_section->offset, geninst_section->size, do_swap);
    }

    // Read trajectory section
    f.seekg(traj_section->offset, std::ios::beg);
    if (!f.good()) throw std::runtime_error("cannot seek to TRAJECTORY section");

    // Kshot library
    int num_shots = read_int(f, do_swap);
    cache.kshots.resize(static_cast<size_t>(num_shots));
    for (int i = 0; i < num_shots; ++i) {
        int ns = read_int(f, do_swap);
        cache.kshots[i].k.resize(static_cast<size_t>(ns));
        if (ns > 0) {
            if (!read4(f, cache.kshots[i].k.data(), ns))
                throw std::runtime_error("EOF reading kshot data");
            if (do_swap) swap4_array(cache.kshots[i].k.data(), ns);
        }
    }

    // Encoding spaces
    int num_es = read_int(f, do_swap);
    cache.encoding_spaces.resize(static_cast<size_t>(num_es));
    for (int i = 0; i < num_es; ++i) {
        auto& es = cache.encoding_spaces[i];
        if (!read4(f, es.fov, 3))        throw std::runtime_error("EOF reading encoding space");
        if (!read4(f, es.matrix, 3))     throw std::runtime_error("EOF reading encoding space");
        if (!read4(f, es.nav_fov, 3))    throw std::runtime_error("EOF reading encoding space");
        if (!read4(f, es.nav_matrix, 3)) throw std::runtime_error("EOF reading encoding space");
        es.subseq_idx       = read_int(f, do_swap);
        es.nav_subseq_offset = read_int(f, do_swap);
        if (do_swap) {
            swap4_array(es.fov, 3);
            swap4_array(es.matrix, 3);
            swap4_array(es.nav_fov, 3);
            swap4_array(es.nav_matrix, 3);
        }
    }

    // Trajectory table
    int num_entries = read_int(f, do_swap);
    cache.table.resize(static_cast<size_t>(num_entries));
    for (int i = 0; i < num_entries; ++i) {
        auto& e = cache.table[i];
        e.kx_shot_id    = read_int(f, do_swap);
        e.ky_shot_id    = read_int(f, do_swap);
        e.kz_shot_id    = read_int(f, do_swap);
        e.gx_amplitude  = read_float(f, do_swap);
        e.gy_amplitude  = read_float(f, do_swap);
        e.gz_amplitude  = read_float(f, do_swap);
        e.rotation_id   = read_int(f, do_swap);
        e.slc = read_int(f, do_swap);
        e.seg = read_int(f, do_swap);
        e.rep = read_int(f, do_swap);
        e.avg = read_int(f, do_swap);
        e.set = read_int(f, do_swap);
        e.eco = read_int(f, do_swap);
        e.phs = read_int(f, do_swap);
        e.lin = read_int(f, do_swap);
        e.par = read_int(f, do_swap);
        e.acq = read_int(f, do_swap);
    }

    return cache;
}

} // namespace mrdserver
