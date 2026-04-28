/**
 * @file trajectory_cache_reader.cpp
 * @brief Standalone reader for pulseqlib binary cache.
 *
 * Reads sections from the cache:
 *   - Section 2 (GENINSTRUCTIONS): rotation matrices
 *   - Section 5 (TRAJECTORY): kshot library, encoding spaces, table
 *   - Section 6 (SEQUENCEDESCRIPTION): event lists, RF shapes, shims (optional)
 *
 * No dependency on pulserverlib. All integer and float fields are 4 bytes.
 */

#include "trajectory_cache_reader.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <algorithm>

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/version.h"
#include "ismrmrd/waveform.h"

namespace mrdserver {

namespace {

constexpr int32_t CACHE_ENDIAN_MARKER          = 0x01020304;
constexpr int     SECTION_GENINSTRUCTIONS      = 2;
constexpr int     SECTION_TRAJECTORY           = 4;
constexpr int     SECTION_SEQUENCEDESCRIPTION  = 5;
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
    SequenceCache& cache)
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

        // --- Label limits (global, kept for backward compatibility but superseded by per-ES limits) ---
        {
            LabelLimit ll[10];
            if (!read4(f, ll, 20))
                throw std::runtime_error("EOF reading label_limits");
            // Consumed but not stored — per-encoding-space limits in TRAJECTORY section take precedence.
            (void)ll;
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

SequenceCache read_sequence_cache(const std::string& cache_path)
{
    SequenceCache cache;

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

    // Find sections 2 (GENINSTRUCTIONS), 4 (TRAJECTORY), and 5 (SEQUENCEDESCRIPTION)
    const SectionEntry* geninst_section = nullptr;
    const SectionEntry* traj_section    = nullptr;
    const SectionEntry* seqdesc_section = nullptr;
    for (auto& s : sections) {
        if (s.id == SECTION_GENINSTRUCTIONS)     geninst_section = &s;
        if (s.id == SECTION_TRAJECTORY)          traj_section    = &s;
        if (s.id == SECTION_SEQUENCEDESCRIPTION) seqdesc_section = &s;
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
        // Per-encoding-space label limits (10 × {min,max} = 20 ints)
        {
            LabelLimit ll[10];
            if (!read4(f, ll, 20))
                throw std::runtime_error("EOF reading per-ES label_limits");
            if (do_swap) swap4_array(ll, 20);
            es.label_limits.slc = ll[0];
            es.label_limits.phs = ll[1];
            es.label_limits.rep = ll[2];
            es.label_limits.avg = ll[3];
            es.label_limits.seg = ll[4];
            es.label_limits.set = ll[5];
            es.label_limits.eco = ll[6];
            es.label_limits.par = ll[7];
            es.label_limits.lin = ll[8];
            es.label_limits.acq = ll[9];
        }
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
        e.off                = read_int(f, do_swap);
    }

    // Read Section 5 — sequence description (optional; graceful skip if absent)
    if (seqdesc_section) {
        f.seekg(seqdesc_section->offset, std::ios::beg);
        if (f.good()) {
            try {
                // Sequence parameters
                auto& sp = cache.seq_params;
                sp.min_te_us           = read_float(f, do_swap);
                sp.min_tr_us           = read_float(f, do_swap);
                sp.max_tr_us           = read_float(f, do_swap);
                sp.max_flip_angle_deg  = read_float(f, do_swap);
                sp.total_scan_time_us  = read_float(f, do_swap);
                sp.num_subseqs         = read_int(f, do_swap);
                read_int(f, do_swap); read_int(f, do_swap); read_int(f, do_swap); // reserved

                cache.seq_descs.resize(static_cast<size_t>(sp.num_subseqs));

                for (int ss = 0; ss < sp.num_subseqs; ++ss) {
                    auto& sd = cache.seq_descs[static_cast<size_t>(ss)];
                    sd.subseq_idx     = read_int(f, do_swap);
                    sd.tr_duration_us = read_float(f, do_swap);

                    // RF shape tuples
                    int num_tuples = read_int(f, do_swap);
                    sd.rf_shape_tuples.resize(static_cast<size_t>(num_tuples));
                    for (int t = 0; t < num_tuples; ++t) {
                        auto& tup = sd.rf_shape_tuples[static_cast<size_t>(t)];
                        tup.tuple_id     = read_int(f, do_swap);
                        tup.N_tx         = read_int(f, do_swap);
                        tup.N_samples    = read_int(f, do_swap);
                        tup.rf_raster_us = read_float(f, do_swap);
                        tup.num_bands    = read_int(f, do_swap);
                        for (int b = 0; b < SEQDESC_MAX_BANDS; ++b)
                            tup.band_freq_offsets_hz[b] = read_float(f, do_swap);
                        tup.band_bandwidth_hz = read_float(f, do_swap);
                        tup.total_b1sq_power  = read_float(f, do_swap);
                        int tot = tup.N_tx * tup.N_samples;
                        if (tot > 0) {
                            tup.mag.resize(static_cast<size_t>(tot));
                            if (!read4(f, tup.mag.data(), tot))
                                throw std::runtime_error("EOF reading RF mag");
                            if (do_swap) swap4_array(tup.mag.data(), tot);
                        }
                        int has_phase = read_int(f, do_swap);
                        if (has_phase && tot > 0) {
                            tup.phase.resize(static_cast<size_t>(tot));
                            if (!read4(f, tup.phase.data(), tot))
                                throw std::runtime_error("EOF reading RF phase");
                            if (do_swap) swap4_array(tup.phase.data(), tot);
                        }
                        int has_time = read_int(f, do_swap);
                        if (has_time && tup.N_samples > 0) {
                            tup.time.resize(static_cast<size_t>(tup.N_samples));
                            if (!read4(f, tup.time.data(), tup.N_samples))
                                throw std::runtime_error("EOF reading RF time");
                            if (do_swap) swap4_array(tup.time.data(), tup.N_samples);
                        }
                    }

                    // Shim definitions
                    int num_shims = read_int(f, do_swap);
                    sd.shim_defs.resize(static_cast<size_t>(num_shims));
                    for (int s2 = 0; s2 < num_shims; ++s2) {
                        auto& sh = sd.shim_defs[static_cast<size_t>(s2)];
                        sh.shim_id_local = read_int(f, do_swap);
                        sh.N_ch          = read_int(f, do_swap);
                        sh.magnitudes.resize(static_cast<size_t>(sh.N_ch));
                        for (int c = 0; c < sh.N_ch; ++c)
                            sh.magnitudes[static_cast<size_t>(c)] = read_float(f, do_swap);
                        sh.phases.resize(static_cast<size_t>(sh.N_ch));
                        for (int c = 0; c < sh.N_ch; ++c)
                            sh.phases[static_cast<size_t>(c)] = read_float(f, do_swap);
                    }

                    // Events
                    int num_events = read_int(f, do_swap);
                    sd.events.resize(static_cast<size_t>(num_events));
                    for (int e2 = 0; e2 < num_events; ++e2) {
                        auto& ev = sd.events[static_cast<size_t>(e2)];
                        ev.type = static_cast<SeqEventType>(read_int(f, do_swap));
                        for (int p = 0; p < 7; ++p)
                            ev.params[p] = read_float(f, do_swap);
                    }

                    // Composite RF groups
                    int num_groups = read_int(f, do_swap);
                    sd.composite_rf_groups.resize(static_cast<size_t>(num_groups));
                    for (int g = 0; g < num_groups; ++g) {
                        auto& cg = sd.composite_rf_groups[static_cast<size_t>(g)];
                        cg.group_id        = read_int(f, do_swap);
                        cg.first_event_idx = read_int(f, do_swap);
                        cg.last_event_idx  = read_int(f, do_swap);
                        cg.num_pulses      = read_int(f, do_swap);
                        cg.eff_te_us       = read_float(f, do_swap);
                    }
                }

                cache.has_seq_desc = true;
            } catch (...) {
                // Sequence description is optional — degrade gracefully
                cache.seq_descs.clear();
                cache.has_seq_desc = false;
            }
        }
    }

    return cache;
}

std::vector<PrecomputedTrajectory> pre_compute_trajectories(const SequenceCache& cache)
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

        // If no kshot data but rotation is present, synthesise a unit radial line along kx.
        // The rotation matrix for each readout will orient the spoke to its actual direction.
        bool synthetic_radial = false;
        if (nsamples == 0) {
            if (es < static_cast<int>(cache.encoding_spaces.size()))
                nsamples = static_cast<int>(cache.encoding_spaces[es].matrix[0]);
            if (nsamples == 0 || first.rotation_id < 0) continue;
            synthetic_radial = true;
        }

        const int num_ro = static_cast<int>(adc_indices.size());
        std::vector<float> kx_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);
        std::vector<float> ky_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);
        std::vector<float> kz_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);

        for (int r = 0; r < num_ro; ++r) {
            const auto& entry = cache.table[adc_indices[r]];
            float* px = &kx_all[static_cast<size_t>(r) * nsamples];
            float* py = &ky_all[static_cast<size_t>(r) * nsamples];
            float* pz = &kz_all[static_cast<size_t>(r) * nsamples];

            if (synthetic_radial) {
                // Unit line: [-0.5, 0.5] centred on center_sample, normalised by nsamples
                const float center = static_cast<float>(entry.center_sample);
                const float norm   = static_cast<float>(nsamples);
                for (int i = 0; i < nsamples; ++i)
                    px[i] = (static_cast<float>(i) - center) / norm;
                // ky, kz remain zero — rotation will spread kx into the spoke direction
            } else {
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
            }

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

namespace {

/* Map cfgradcoil identifier to the GE coef base name.
 * Reference values from EPIC pulserver.allcv.h:
 *   1=CRD, 2=Roemer, 101=HGC, 102=Vectra, 103=Permanent.
 * Add new entries as more gradient subsystems become relevant. */
const char* cfgradcoil_to_coef_name(int id)
{
    switch (id) {
        case 1:   return "crd";
        case 2:   return "roemer";
        case 101: return "hgc";
        case 102: return "vectra";
        case 103: return "permanent";
        default:  return nullptr;
    }
}

std::string sequence_resource_base_dir()
{
    const char* env = std::getenv("GADGETRON_RESOURCE_DIR");
    if (env && env[0] != '\0') return std::string(env);
    return std::string("/usr/g/bin");
}

} // anonymous namespace

void set_user_parameter_string(ISMRMRD::IsmrmrdHeader& hdr,
                               const std::string& name,
                               const std::string& value)
{
    if (!hdr.userParameters) hdr.userParameters = ISMRMRD::UserParameters();
    auto& strs = hdr.userParameters->userParameterString;
    for (auto& p : strs) {
        if (p.name == name) { p.value = value; return; }
    }
    ISMRMRD::UserParameterString p;
    p.name  = name;
    p.value = value;
    strs.push_back(p);
}

void add_sequence_resource_paths(ISMRMRD::IsmrmrdHeader& hdr,
                                 int tensor_index,
                                 int grad_coil_id)
{
    const std::string base = sequence_resource_base_dir();

    if (tensor_index > 0) {
        std::ostringstream oss;
        oss << base << "/tensor" << tensor_index << ".dat";
        set_user_parameter_string(hdr, "tensor_dat_path", oss.str());
    }

    if (const char* coef = cfgradcoil_to_coef_name(grad_coil_id)) {
        std::ostringstream oss;
        oss << base << "/" << coef << ".coef";
        set_user_parameter_string(hdr, "grad_coef_path", oss.str());
    }
}

void enrich_ismrmrd_header(ISMRMRD::IsmrmrdHeader& hdr, const SequenceCache& cache)
{
    // --- Sequence parameters from definitions ---
    {
        ISMRMRD::SequenceParameters seqp;
        auto get_floats = [&](const char* key) -> std::vector<float> {
            std::vector<float> out;
            auto it = cache.definitions.find(key);
            if (it != cache.definitions.end()) {
                for (const auto& sv : it->second) {
                    try { out.push_back(std::stof(sv)); } catch (...) {}
                }
            }
            return out;
        };
        auto tr = get_floats("TR");
        auto te = get_floats("TE");
        auto ti = get_floats("TI");
        auto fa = get_floats("FlipAngle");
        if (!tr.empty()) seqp.TR = tr;
        if (!te.empty()) seqp.TE = te;
        if (!ti.empty()) seqp.TI = ti;
        if (!fa.empty()) seqp.flipAngle_deg = fa;
        // Only override if we actually have something to set
        if (seqp.TR || seqp.TE || seqp.TI || seqp.flipAngle_deg)
            hdr.sequenceParameters = seqp;
    }

    // --- Encoding limits, encodedSpace and reconSpace ---
    // cache.encoding_spaces is the authoritative flat list: normal and
    // navigator encoding spaces are already interleaved at the correct
    // indices (encoding_space_ref values in the table use these directly).
    // We map 1:1: hdr.encoding[i] ↔ cache.encoding_spaces[i].
    if (!cache.encoding_spaces.empty()) {
        const int num_es = static_cast<int>(cache.encoding_spaces.size());
        if (static_cast<int>(hdr.encoding.size()) < num_es)
            hdr.encoding.resize(static_cast<size_t>(num_es));

        auto make_limit = [](const LabelLimit& ll) {
            ISMRMRD::Limit lim;
            lim.minimum = 0;
            lim.maximum = static_cast<uint16_t>(ll.max);
            lim.center  = static_cast<uint16_t>(ll.max / 2);
            return lim;
        };

        // encodedSpace, reconSpace and encodingLimits: one entry per encoding space
        for (int es = 0; es < num_es; ++es) {
            const auto& ces = cache.encoding_spaces[es];
            ISMRMRD::EncodingSpace space;
            space.matrixSize.x     = static_cast<uint16_t>(ces.matrix[0]);
            space.matrixSize.y     = static_cast<uint16_t>(ces.matrix[1]);
            space.matrixSize.z     = static_cast<uint16_t>(ces.matrix[2]);
            space.fieldOfView_mm.x = ces.fov[0];
            space.fieldOfView_mm.y = ces.fov[1];
            space.fieldOfView_mm.z = ces.fov[2];
            hdr.encoding[es].encodedSpace = space;
            hdr.encoding[es].reconSpace   = space;

            const auto& ll = ces.label_limits;
            auto& enc = hdr.encoding[es].encodingLimits;
            enc.kspace_encoding_step_1 = make_limit(ll.lin);
            enc.kspace_encoding_step_2 = make_limit(ll.par);
            enc.slice                  = make_limit(ll.slc);
            enc.average                = make_limit(ll.avg);
            enc.contrast               = make_limit(ll.eco);
            enc.phase                  = make_limit(ll.phs);
            enc.repetition             = make_limit(ll.rep);
            enc.set                    = make_limit(ll.set);
            enc.segment                = make_limit(ll.seg);
        }
    }
}

void enrich_ismrmrd_acquisition(
    ISMRMRD::Acquisition& acq,
    int acquisition_index,
    uint32_t measurement_uid,
    float table_position_z,
    const SequenceCache& cache,
    const std::vector<PrecomputedTrajectory>& trajectories,
    const std::vector<int>& readout_index_in_es,
    const uint32_t* physio_stamps)
{
    // Scan-invariant fields not correctly populated by the GE converter
    acq.measurement_uid() = measurement_uid;
    acq.patient_table_position()[0] = 0.0f;
    acq.patient_table_position()[1] = 0.0f;
    acq.patient_table_position()[2] = table_position_z;

    // acquisition_time_stamp: ISMRMRD convention is ms since midnight;
    // the GE converter sets time(NULL) (seconds since epoch)
    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    acq.acquisition_time_stamp() =
        static_cast<uint32_t>((t->tm_hour * 3600 + t->tm_min * 60 + t->tm_sec) * 1000);

    // Physiological trigger timestamps: ms since midnight for each trigger type
    // (0=ECG, 1=PPG, 2=Respiratory).  Always zero-initialise so scans without
    // physio recording leave a well-defined default value.
    using namespace ISMRMRD;  // bring enum values into scope
    for (int i = 0; i < ISMRMRD_PHYS_STAMPS; ++i)
        acq.physiology_time_stamp()[i] = (physio_stamps != nullptr) ? physio_stamps[i] : 0u;

    // Trajectory cache metadata and pre-computed trajectory
    if (cache.table.empty() ||
        acquisition_index < 0 ||
        acquisition_index >= static_cast<int>(cache.table.size()))
        return;

    const auto& entry = cache.table[acquisition_index];

    auto& idx = acq.idx();
    idx.kspace_encode_step_1 = static_cast<uint16_t>(entry.lin);
    idx.kspace_encode_step_2 = static_cast<uint16_t>(entry.par);
    idx.slice                = static_cast<uint16_t>(entry.slc);
    idx.average              = static_cast<uint16_t>(entry.avg);
    idx.contrast             = static_cast<uint16_t>(entry.eco);
    idx.phase                = static_cast<uint16_t>(entry.phs);
    idx.repetition           = static_cast<uint16_t>(entry.rep);
    idx.set                  = static_cast<uint16_t>(entry.set);
    idx.segment              = static_cast<uint16_t>(entry.seg);

    const_cast<ISMRMRD::AcquisitionHeader&>(acq.getHead()).flags = entry.flags;
    acq.center_sample()      = static_cast<uint16_t>(entry.center_sample);
    acq.sample_time_us()     = entry.sample_time_us;
    acq.encoding_space_ref() = static_cast<uint16_t>(entry.encoding_space_ref);

    const int es = entry.encoding_space_ref;

    // FIRST_IN / LAST_IN flags: compare each idx field against the actual
    // per-encoding-space min/max from label_limits.  min is the real observed
    // minimum (not zero-filled), so flags fire at the correct boundary values.
    if (es >= 0 && es < static_cast<int>(cache.encoding_spaces.size())) {
        const auto& ll = cache.encoding_spaces[es].label_limits;
        // helper: OR in ISMRMRD first/last flag when idx field hits the limit
#define SETFL(idxf, llf, fst, lst) do { \
    if (static_cast<int>(idx.idxf) == ll.llf.min) acq.setFlag(fst); \
    if (static_cast<int>(idx.idxf) == ll.llf.max) acq.setFlag(lst); \
} while(0)
        SETFL(kspace_encode_step_1, lin, ISMRMRD_ACQ_FIRST_IN_ENCODE_STEP1, ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);
        SETFL(kspace_encode_step_2, par, ISMRMRD_ACQ_FIRST_IN_ENCODE_STEP2, ISMRMRD_ACQ_LAST_IN_ENCODE_STEP2);
        SETFL(average,              acq, ISMRMRD_ACQ_FIRST_IN_AVERAGE,       ISMRMRD_ACQ_LAST_IN_AVERAGE);
        SETFL(slice,                slc, ISMRMRD_ACQ_FIRST_IN_SLICE,         ISMRMRD_ACQ_LAST_IN_SLICE);
        SETFL(contrast,             eco, ISMRMRD_ACQ_FIRST_IN_CONTRAST,      ISMRMRD_ACQ_LAST_IN_CONTRAST);
        SETFL(phase,                phs, ISMRMRD_ACQ_FIRST_IN_PHASE,         ISMRMRD_ACQ_LAST_IN_PHASE);
        SETFL(repetition,           rep, ISMRMRD_ACQ_FIRST_IN_REPETITION,    ISMRMRD_ACQ_LAST_IN_REPETITION);
        SETFL(set,                  set, ISMRMRD_ACQ_FIRST_IN_SET,           ISMRMRD_ACQ_LAST_IN_SET);
        SETFL(segment,              seg, ISMRMRD_ACQ_FIRST_IN_SEGMENT,       ISMRMRD_ACQ_LAST_IN_SEGMENT);
#undef SETFL
    }

    if (es < 0 || es >= static_cast<int>(trajectories.size()))
        return;

    const auto& pt = trajectories[es];
    const int ro_idx = (acquisition_index < static_cast<int>(readout_index_in_es.size()))
                       ? readout_index_in_es[acquisition_index] : -1;
    if (pt.ndim > 0 && ro_idx >= 0 && ro_idx < pt.num_readouts) {
        acq.resize(acq.number_of_samples(), acq.active_channels(), pt.ndim);
        const float* src = &pt.data[static_cast<size_t>(ro_idx) * pt.ndim * pt.num_samples];
        std::memcpy(acq.getTrajPtr(), src, static_cast<size_t>(pt.ndim) * pt.num_samples * sizeof(float));
    }
}

void add_waveform_information(ISMRMRD::IsmrmrdHeader& hdr,
                              bool has_ecg, bool has_ppg, bool has_resp)
{
    struct Desc { bool enabled; ISMRMRD::WaveformType type; const char* name; };
    const Desc descs[] = {
        { has_ecg,  ISMRMRD::WaveformType::ECG,         "ECG"         },
        { has_ppg,  ISMRMRD::WaveformType::PULSE,        "PPG"         },
        { has_resp, ISMRMRD::WaveformType::RESPIRATORY,  "Respiratory" },
    };
    for (const auto& d : descs) {
        if (!d.enabled) continue;
        ISMRMRD::WaveformInformation info;
        info.waveformName = d.name;
        info.waveformType = d.type;
        hdr.waveformInformation.push_back(info);
    }
}

ISMRMRD::Waveform make_physio_waveform(uint16_t waveform_id,
                                       uint32_t measurement_uid,
                                       uint32_t scan_counter,
                                       uint32_t time_stamp_ms,
                                       float sample_time_us,
                                       const std::vector<const int16_t*>& channels,
                                       uint16_t num_samples)
{
    const uint16_t num_channels = static_cast<uint16_t>(channels.size());
    ISMRMRD::Waveform wav(num_samples, num_channels);
    wav.head.version          = ISMRMRD_VERSION_MAJOR;
    wav.head.measurement_uid  = measurement_uid;
    wav.head.scan_counter     = scan_counter;
    wav.head.time_stamp       = time_stamp_ms;
    wav.head.number_of_samples = num_samples;
    wav.head.channels         = num_channels;
    wav.head.sample_time_us   = sample_time_us;
    wav.head.waveform_id      = waveform_id;

    // Channel-major layout: all samples of ch0, then ch1, etc.
    // int16_t sign-extended to uint32_t
    uint32_t* dst = wav.begin_data();
    for (uint16_t ch = 0; ch < num_channels; ++ch) {
        const int16_t* src = channels[ch];
        for (uint16_t s = 0; s < num_samples; ++s) {
            dst[static_cast<size_t>(ch) * num_samples + s] =
                static_cast<uint32_t>(static_cast<int32_t>(src[s]));
        }
    }
    return wav;
}

// ---------- Helper: build a float-payload waveform ----------
// All values are stored as bit-casts of float32 into uint32 channels.
// num_channels = 1 (all data serialised as a flat uint32 stream).
static ISMRMRD::Waveform make_float_payload_waveform(
    uint16_t         waveform_id,
    uint32_t         measurement_uid,
    uint32_t         scan_counter,
    const std::vector<uint32_t>& payload)
{
    const auto n = static_cast<uint16_t>(
        payload.size() > 65535u ? 65535u : payload.size());
    ISMRMRD::Waveform wav(n, 1);
    wav.head.version           = ISMRMRD_VERSION_MAJOR;
    wav.head.measurement_uid   = measurement_uid;
    wav.head.scan_counter      = scan_counter;
    wav.head.time_stamp        = 0;
    wav.head.number_of_samples = n;
    wav.head.channels          = 1;
    wav.head.sample_time_us    = 1.0f;
    wav.head.waveform_id       = waveform_id;
    std::memcpy(wav.begin_data(), payload.data(), n * sizeof(uint32_t));
    return wav;
}

static uint32_t f2u(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// ================================================================
// Sequence-description waveform factory functions
// ================================================================

ISMRMRD::Waveform make_seqdesc_header_waveform(
    const SequenceCache& cache,
    uint32_t measurement_uid,
    uint32_t scan_counter)
{
    const auto& sp = cache.seq_params;
    std::vector<uint32_t> p;
    p.reserve(8);
    p.push_back(static_cast<uint32_t>(sp.num_subseqs));
    p.push_back(f2u(sp.min_te_us));
    p.push_back(f2u(sp.min_tr_us));
    p.push_back(f2u(sp.max_tr_us));
    p.push_back(f2u(sp.max_flip_angle_deg));
    p.push_back(f2u(sp.total_scan_time_us));
    return make_float_payload_waveform(WAVEFORM_ID_SEQDESC_HEADER,
                                       measurement_uid, scan_counter, p);
}

ISMRMRD::Waveform make_seqdesc_events_waveform(
    const SequenceDescription& desc,
    uint32_t measurement_uid,
    uint32_t scan_counter)
{
    // Header: subseq_idx, tr_duration_us, num_events
    // Per event: type (int), params[7] (float x7)
    std::vector<uint32_t> p;
    p.reserve(3 + desc.events.size() * 8);
    p.push_back(static_cast<uint32_t>(desc.subseq_idx));
    p.push_back(f2u(desc.tr_duration_us));
    p.push_back(static_cast<uint32_t>(desc.events.size()));
    for (const auto& ev : desc.events) {
        p.push_back(static_cast<uint32_t>(ev.type));
        for (int i = 0; i < 7; ++i)
            p.push_back(f2u(ev.params[i]));
    }
    return make_float_payload_waveform(WAVEFORM_ID_SEQDESC_EVENTS,
                                       measurement_uid, scan_counter, p);
}

ISMRMRD::Waveform make_seqdesc_rf_shapes_waveform(
    const SequenceDescription& desc,
    uint32_t measurement_uid,
    uint32_t scan_counter)
{
    // Header: subseq_idx, num_tuples
    // Per tuple: tuple_id, N_tx, N_samples, rf_raster_us, num_bands,
    //            band_freq_offsets_hz[8], band_bandwidth_hz, total_b1sq_power,
    //            has_phase, has_time,
    //            mag[N_tx*N_samples], [phase[N_tx*N_samples]], [time[N_samples]]
    std::vector<uint32_t> p;
    p.push_back(static_cast<uint32_t>(desc.subseq_idx));
    p.push_back(static_cast<uint32_t>(desc.rf_shape_tuples.size()));
    for (const auto& t : desc.rf_shape_tuples) {
        p.push_back(static_cast<uint32_t>(t.tuple_id));
        p.push_back(static_cast<uint32_t>(t.N_tx));
        p.push_back(static_cast<uint32_t>(t.N_samples));
        p.push_back(f2u(t.rf_raster_us));
        p.push_back(static_cast<uint32_t>(t.num_bands));
        for (int b = 0; b < SEQDESC_MAX_BANDS; ++b)
            p.push_back(f2u(t.band_freq_offsets_hz[b]));
        p.push_back(f2u(t.band_bandwidth_hz));
        p.push_back(f2u(t.total_b1sq_power));
        // Magnitude
        for (float v : t.mag)   p.push_back(f2u(v));
        // Phase
        p.push_back(t.phase.empty() ? 0u : 1u);
        if (!t.phase.empty())
            for (float v : t.phase) p.push_back(f2u(v));
        // Time
        p.push_back(t.time.empty() ? 0u : 1u);
        if (!t.time.empty())
            for (float v : t.time) p.push_back(f2u(v));
    }
    return make_float_payload_waveform(WAVEFORM_ID_SEQDESC_RF_SHAPES,
                                       measurement_uid, scan_counter, p);
}

ISMRMRD::Waveform make_seqdesc_shims_waveform(
    const SequenceDescription& desc,
    uint32_t measurement_uid,
    uint32_t scan_counter)
{
    std::vector<uint32_t> p;
    p.push_back(static_cast<uint32_t>(desc.subseq_idx));
    p.push_back(static_cast<uint32_t>(desc.shim_defs.size()));
    for (const auto& s : desc.shim_defs) {
        p.push_back(static_cast<uint32_t>(s.shim_id_local));
        p.push_back(static_cast<uint32_t>(s.N_ch));
        for (float v : s.magnitudes) p.push_back(f2u(v));
        for (float v : s.phases)     p.push_back(f2u(v));
    }
    return make_float_payload_waveform(WAVEFORM_ID_SEQDESC_SHIMS,
                                       measurement_uid, scan_counter, p);
}

} // namespace mrdserver
