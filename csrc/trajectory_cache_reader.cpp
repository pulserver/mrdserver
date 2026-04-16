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

// ---------- read descriptor data from GENINSTRUCTIONS (section 2) ----------

std::vector<std::array<float, 9>> read_rotations_from_geninstructions(
    std::ifstream& f, long section_offset, int section_size, bool do_swap,
    TrajectoryCache& cache)
{
    std::vector<std::array<float, 9>> rotations;

    f.seekg(section_offset, std::ios::beg);
    if (!f.good()) throw std::runtime_error("cannot seek to GENINSTRUCTIONS");
    (void)section_size;

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
        // per rf_def: 6 scalars + 11 from rf_stats = 17 fields
        skip_ints(f, num_unique_rfs * 17);

        // --- RF table ---
        int rf_table_size = read_int(f, do_swap);
        // per row: id(i), amplitude(f), freq_offset(f), phase_offset(f), rf_use(i) = 5 fields
        skip_ints(f, rf_table_size * 5);

        // --- Grad definitions ---
        int num_unique_grads = read_int(f, do_swap);
        // per grad_def: 8 + MAX_GRAD_SHOTS + 6*MAX_GRAD_SHOTS = 120 fields
        skip_ints(f, num_unique_grads * (8 + 7 * MAX_GRAD_SHOTS));

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

        // --- Rotations ---
        int num_rotations = read_int(f, do_swap);
        rotations.resize(static_cast<size_t>(num_rotations));
        for (int r = 0; r < num_rotations; ++r) {
            if (!read4(f, rotations[r].data(), 9))
                throw std::runtime_error("EOF reading rotations");
            if (do_swap) swap4_array(rotations[r].data(), 9);
        }

        // --- Triggers ---
        int num_triggers = read_int(f, do_swap);
        skip_ints(f, num_triggers * 5);

        // --- Shapes ---
        int num_shapes = read_int(f, do_swap);
        for (int sh = 0; sh < num_shapes; ++sh) {
            skip_ints(f, 1);  // num_uncompressed_samples
            int ns = read_int(f, do_swap);
            if (ns > 0) skip_ints(f, ns);
        }

        // --- TR descriptor (10 fields) ---
        skip_ints(f, 10);

        // --- Segment definitions ---
        {
            int num_segs_def = read_int(f, do_swap);
            for (int sg = 0; sg < num_segs_def; ++sg) {
                skip_ints(f, 1);  // start_block
                int seg_nblocks = read_int(f, do_swap);
                skip_ints(f, 1);  // max_energy_start_block
                if (seg_nblocks > 0)
                    skip_ints(f, seg_nblocks * 5);  // 5 arrays × num_blocks
                skip_ints(f, 2);  // trigger_id, is_nav
            }
        }

        // --- Segment table ---
        {
            skip_ints(f, 1);  // num_unique_segments (redundant)
            int n_prep = read_int(f, do_swap);
            if (n_prep > 0) skip_ints(f, n_prep);
            int n_main = read_int(f, do_swap);
            if (n_main > 0) skip_ints(f, n_main);
            int n_cool = read_int(f, do_swap);
            if (n_cool > 0) skip_ints(f, n_cool);
        }

        // --- Label table (skip: written with fwrite, same 4-byte fields) ---
        {
            int label_cols = read_int(f, do_swap);
            int label_rows = read_int(f, do_swap);
            if (label_rows > 0 && label_cols > 0)
                skip_ints(f, label_rows * label_cols);
        }

        // --- Label limits (10 × {min, max} = 20 ints) ---
        {
            LabelLimit ll[10];
            if (!read4(f, ll, 20))
                throw std::runtime_error("EOF reading label_limits");
            if (do_swap) swap4_array(ll, 20);
            cache.label_limits.slc = ll[0];
            cache.label_limits.phs = ll[1];
            cache.label_limits.rep = ll[2];
            cache.label_limits.avg = ll[3];
            cache.label_limits.seg = ll[4];
            cache.label_limits.set = ll[5];
            cache.label_limits.eco = ll[6];
            cache.label_limits.par = ll[7];
            cache.label_limits.lin = ll[8];
            cache.label_limits.acq = ll[9];
        }

        // --- Generic definitions ---
        {
            int num_defs = read_int(f, do_swap);
            for (int d = 0; d < num_defs; ++d) {
                // name: length-prefixed string (not null-terminated in file)
                int name_len = read_int(f, do_swap);
                std::string name(static_cast<size_t>(name_len), '\0');
                f.read(&name[0], name_len);
                if (!f.good()) throw std::runtime_error("EOF reading definition name");

                int value_size = read_int(f, do_swap);
                std::vector<std::string> values(static_cast<size_t>(value_size));
                for (int v = 0; v < value_size; ++v) {
                    int vlen = read_int(f, do_swap);
                    values[v].resize(static_cast<size_t>(vlen));
                    f.read(&values[v][0], vlen);
                    if (!f.good()) throw std::runtime_error("EOF reading definition value");
                }
                cache.definitions[std::move(name)] = std::move(values);
            }
        }

        // --- Scan table (skip) ---
        {
            int scan_len = read_int(f, do_swap);
            if (scan_len > 0)
                skip_ints(f, scan_len * 3);  // block_idx, tr_id, seg_id
        }

        // Only read first subsequence
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

    (void)version_major; (void)version_minor; (void)vendor; (void)stored_size;

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

    // Read rotation matrices, label_limits, definitions from GENINSTRUCTIONS
    if (geninst_section) {
        cache.rotations = read_rotations_from_geninstructions(
            f, geninst_section->offset, geninst_section->size, do_swap,
            cache);
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
        // New fields: flags (2 ints), center_sample, sample_time_us, encoding_space_ref
        {
            uint32_t flags_lo = static_cast<uint32_t>(read_int(f, do_swap));
            uint32_t flags_hi = static_cast<uint32_t>(read_int(f, do_swap));
            e.flags = (static_cast<uint64_t>(flags_hi) << 32) | static_cast<uint64_t>(flags_lo);
        }
        e.center_sample      = read_int(f, do_swap);
        e.sample_time_us     = read_float(f, do_swap);
        e.encoding_space_ref = read_int(f, do_swap);
    }

    return cache;
}

std::vector<PrecomputedTrajectory> pre_compute_trajectories(const TrajectoryCache& cache)
{
    const int num_es = static_cast<int>(cache.encoding_spaces.size());
    std::vector<PrecomputedTrajectory> result(static_cast<size_t>(num_es));

    for (int es = 0; es < num_es; ++es) {
        // Collect ADC indices for this encoding space
        std::vector<int> adc_indices;
        for (int t = 0; t < static_cast<int>(cache.table.size()); ++t) {
            if (cache.table[t].encoding_space_ref == es)
                adc_indices.push_back(t);
        }
        if (adc_indices.empty()) continue;

        // Determine num_samples from the first ADC's kshot
        const auto& first = cache.table[adc_indices[0]];
        int nsamples = 0;
        if (first.kx_shot_id >= 0 && first.kx_shot_id < static_cast<int>(cache.kshots.size()))
            nsamples = static_cast<int>(cache.kshots[first.kx_shot_id].k.size());
        else if (first.ky_shot_id >= 0 && first.ky_shot_id < static_cast<int>(cache.kshots.size()))
            nsamples = static_cast<int>(cache.kshots[first.ky_shot_id].k.size());
        else if (first.kz_shot_id >= 0 && first.kz_shot_id < static_cast<int>(cache.kshots.size()))
            nsamples = static_cast<int>(cache.kshots[first.kz_shot_id].k.size());
        if (nsamples == 0) continue;

        const int num_ro = static_cast<int>(adc_indices.size());
        std::vector<float> kx_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);
        std::vector<float> ky_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);
        std::vector<float> kz_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);

        for (int r = 0; r < num_ro; ++r) {
            const auto& entry = cache.table[adc_indices[r]];
            float* px = &kx_all[static_cast<size_t>(r) * nsamples];
            float* py = &ky_all[static_cast<size_t>(r) * nsamples];
            float* pz = &kz_all[static_cast<size_t>(r) * nsamples];

            auto compose = [&](int shot_id, float amp, float* dst) {
                if (shot_id >= 0 && shot_id < static_cast<int>(cache.kshots.size())) {
                    const auto& sk = cache.kshots[shot_id].k;
                    for (int i = 0; i < std::min(nsamples, static_cast<int>(sk.size())); ++i)
                        dst[i] = sk[i] * amp;
                }
            };
            compose(entry.kx_shot_id, entry.gx_amplitude, px);
            compose(entry.ky_shot_id, entry.gy_amplitude, py);
            compose(entry.kz_shot_id, entry.gz_amplitude, pz);

            if (entry.rotation_id >= 0 &&
                entry.rotation_id < static_cast<int>(cache.rotations.size()))
            {
                const auto& R = cache.rotations[entry.rotation_id];
                for (int i = 0; i < nsamples; ++i) {
                    float rx = R[0]*px[i] + R[1]*py[i] + R[2]*pz[i];
                    float ry = R[3]*px[i] + R[4]*py[i] + R[5]*pz[i];
                    float rz = R[6]*px[i] + R[7]*py[i] + R[8]*pz[i];
                    px[i] = rx; py[i] = ry; pz[i] = rz;
                }
            }
        }

        auto is_zero = [](const std::vector<float>& v) {
            for (float f : v) if (f != 0.0f) return false;
            return true;
        };
        bool has_x = !is_zero(kx_all);
        bool has_y = !is_zero(ky_all);
        bool has_z = !is_zero(kz_all);
        int ndim = (has_x ? 1 : 0) + (has_y ? 1 : 0) + (has_z ? 1 : 0);
        if (ndim == 0) continue; // Cartesian

        auto& pt = result[es];
        pt.ndim        = ndim;
        pt.num_samples = nsamples;
        pt.num_readouts = num_ro;
        pt.data.resize(static_cast<size_t>(ndim) * nsamples * num_ro);

        // Pack active axes: interleaved [ax0_s0, ax1_s0, ..., ax0_s1, ...]
        for (int r = 0; r < num_ro; ++r) {
            const float* axes[3] = {
                has_x ? &kx_all[static_cast<size_t>(r) * nsamples] : nullptr,
                has_y ? &ky_all[static_cast<size_t>(r) * nsamples] : nullptr,
                has_z ? &kz_all[static_cast<size_t>(r) * nsamples] : nullptr
            };
            float* dst = &pt.data[static_cast<size_t>(r) * ndim * nsamples];
            for (int s = 0; s < nsamples; ++s) {
                int d = 0;
                if (has_x) dst[s * ndim + d++] = axes[0][s];
                if (has_y) dst[s * ndim + d++] = axes[1][s];
                if (has_z) dst[s * ndim + d++] = axes[2][s];
            }
        }
    }

    return result;
}

} // namespace mrdserver
