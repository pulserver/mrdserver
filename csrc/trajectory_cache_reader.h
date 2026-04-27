/**
 * @file trajectory_cache_reader.h
 * @brief Standalone reader for pulseqlib binary cache trajectory data.
 *
 * No dependency on pulserverlib — reads the binary cache format directly.
 * Reads section 2 (GENINSTRUCTIONS) for rotation matrices and
 * section 5 (TRAJECTORY) for kshot library, encoding spaces, and table.
 * Reads section 6 (SEQUENCEDESCRIPTION) when present.
 */

#ifndef TRAJECTORY_CACHE_READER_H
#define TRAJECTORY_CACHE_READER_H

#include <vector>
#include <array>
#include <string>
#include <map>
#include <cstdint>

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/waveform.h"
#include "ismrmrd/xml.h"

namespace mrdserver {

struct Kshot {
    std::vector<float> k; /**< k-space values [num_samples], Hz·s/m */
};

struct LabelLimit { int min, max; };

struct EncodingSpace {
    float fov[3];
    float matrix[3];
    float nav_fov[3];
    float nav_matrix[3];
    int   subseq_idx;
    int   nav_subseq_offset;
    struct {
        LabelLimit slc, phs, rep, avg, seg, set, eco, par, lin, acq;
    } label_limits;
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

/* ------------------------------------------------------------------ */
/*  Sequence-description structs (Section 6)                          */
/* ------------------------------------------------------------------ */

static constexpr int SEQDESC_MAX_BANDS = 8;

/** ADC role constants */
enum AdcRole {
    ADC_ROLE_NON_ACQUIRED = 0,
    ADC_ROLE_SINGLE       = 1,
    ADC_ROLE_ECHO_CENTER  = 2,
    ADC_ROLE_NON_CENTER   = 3,
};

/** Sequence event type */
enum SeqEventType {
    SEQ_EVENT_WAIT = 0,
    SEQ_EVENT_RF   = 1,
    SEQ_EVENT_ADC  = 2,
};

/**
 * @brief Single TR event (WAIT / RF / ADC).
 * See pulseqlib_types.h pulseqlib_seq_event for param layout.
 */
struct SeqEvent {
    SeqEventType type;
    float params[7];

    /* Accessors for RF events */
    float rf_center_us()       const { return params[0]; }
    float rf_duration_us()     const { return params[1]; }
    float rf_flip_angle_deg()  const { return params[2]; }
    int   rf_tuple_id()        const { return (int)params[3]; }
    int   rf_shim_id_local()   const { return (int)params[4]; }
    bool  rf_slice_selective() const { return params[5] > 0.5f; }

    /* Accessors for ADC events */
    float adc_center_us()  const { return params[0]; }
    float adc_duration_us()const { return params[1]; }
    int   adc_num_samples()const { return (int)params[2]; }
    float adc_dwell_us()   const { return params[3]; }
    AdcRole adc_role()     const { return static_cast<AdcRole>((int)params[4]); }

    /* Accessor for WAIT */
    float wait_duration_us() const { return params[0]; }
};

/** @brief Normalised RF shape tuple. */
struct RfShapeTuple {
    int   tuple_id;
    int   N_tx;         /**< transmit channels */
    int   N_samples;    /**< per-channel samples */
    float rf_raster_us;
    /* multiband annotation */
    int   num_bands;
    float band_freq_offsets_hz[SEQDESC_MAX_BANDS];
    float band_bandwidth_hz;
    float total_b1sq_power;
    /* waveform arrays — all per-channel interleaved [N_tx * N_samples] */
    std::vector<float> mag;    /**< magnitude (normalised, peak=1)  */
    std::vector<float> phase;  /**< phase (rad); empty if all zero  */
    std::vector<float> time;   /**< sample times (us); empty if uniform */
};

/** @brief Local shim definition (pTx weights) for one subsequence. */
struct ShimDefLocal {
    int   shim_id_local;
    int   N_ch;
    std::vector<float> magnitudes;
    std::vector<float> phases;
};

/** @brief Composite RF group (contiguous RF run, num_pulses >= 2). */
struct CompositeRfGroup {
    int   group_id;
    int   first_event_idx;
    int   last_event_idx;
    int   num_pulses;
    float eff_te_us;
};

/** @brief Per-subsequence sequence description (event list + libraries). */
struct SequenceDescription {
    int   subseq_idx     = 0;
    float tr_duration_us = 0.0f;
    std::vector<RfShapeTuple>     rf_shape_tuples;
    std::vector<ShimDefLocal>     shim_defs;
    std::vector<SeqEvent>         events;
    std::vector<CompositeRfGroup> composite_rf_groups;
};

/** @brief Scan-global sequence parameters. */
struct SequenceParameters {
    float min_te_us           = 0.0f;
    float min_tr_us           = 0.0f;
    float max_tr_us           = 0.0f;
    float max_flip_angle_deg  = 0.0f;
    float total_scan_time_us  = 0.0f;
    int   num_subseqs         = 0;
};

/* ------------------------------------------------------------------ */

struct SequenceCache {
    std::vector<Kshot>                      kshots;
    std::vector<std::array<float, 9>>       rotations;
    std::vector<EncodingSpace>              encoding_spaces;
    std::vector<TrajTableEntry>             table;
    std::map<std::string, std::vector<std::string>> definitions;
    /* Section 6 — sequence description (populated when present) */
    std::vector<SequenceDescription>        seq_descs;
    SequenceParameters                      seq_params;
    bool                                    has_seq_desc = false;
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
 * @return Populated SequenceCache; empty if file not found or
 *         trajectory section is missing.
 * @throws std::runtime_error on I/O or format errors.
 */
SequenceCache read_sequence_cache(const std::string& cache_path);

/**
 * @brief Pre-compute per-encoding-space trajectory arrays from a loaded cache.
 *
 * Applies kshot scaling (amplitude), rotation matrices, and axis pruning.
 * Returns one PrecomputedTrajectory per encoding space; ndim==0 entries are
 * Cartesian and need no trajectory attached to ISMRMRD acquisitions.
 *
 * @param cache  Populated SequenceCache (from read_sequence_cache).
 * @return Vector of PrecomputedTrajectory, indexed by encoding_space_ref.
 */
std::vector<PrecomputedTrajectory> pre_compute_trajectories(const SequenceCache& cache);

/**
 * @brief Enrich an ISMRMRD header with Pulseq sequence parameters and
 *        encoding geometry derived from a loaded trajectory cache.
 *
 * Overrides (not appends):
 *  - sequenceParameters: TR, TE, TI, flipAngle_deg from definitions
 *  - encoding[i].encodingLimits (i==0): kspace limits from label_limits
 *  - encoding[i].encodedSpace / reconSpace: FOV and matrix from each
 *    encoding space entry (1:1 mapping — navigator encoding spaces are
 *    already present as separate entries in cache.encoding_spaces at the
 *    indices determined at cache-write time).
 *
 * @param hdr    Header to modify in-place (already deserialised).
 * @param cache  Populated SequenceCache.
 */
void enrich_ismrmrd_header(ISMRMRD::IsmrmrdHeader& hdr, const SequenceCache& cache);

/**
 * @brief Enrich an ISMRMRD Acquisition with trajectory cache metadata,
 *        pre-computed trajectory data, and scan-invariant header fields.
 *
 * All parameters that would require Orchestra SDK calls must be resolved
 * by the caller before invoking this function.
 *
 * Fills:
 *  - measurement_uid, patient_table_position (scan-invariant, from caller)
 *  - acquisition_time_stamp (ms since midnight, computed from system clock)
 *  - idx fields (lin, par, slc, avg, eco, phs, rep, set, seg)
 *  - flags, center_sample, sample_time_us, encoding_space_ref
 *  - trajectory data (memcpy from pre-computed array for the encoding space)
 *
 * @param acq                ISMRMRD Acquisition to modify in-place.
 * @param acquisition_index  Global readout counter (0-based).
 * @param measurement_uid    Stable per-scan identifier (e.g. exam<<16 ^ series).
 * @param table_position_z   Patient table S-axis isocenter (mm).
 * @param cache              Populated SequenceCache.
 * @param trajectories       Pre-computed trajectories (from pre_compute_trajectories).
 * @param readout_index_in_es Per-table-entry readout index within its encoding space.
 * @param physio_stamps      Optional array of 3 ms-since-midnight timestamps for the most
 *                           recent physiological trigger of each type
 *                           (index 0 = ECG, 1 = PPG, 2 = Respiratory).
 *                           Pass nullptr when physio is not enabled.
 */
void enrich_ismrmrd_acquisition(
    ISMRMRD::Acquisition& acq,
    int acquisition_index,
    uint32_t measurement_uid,
    float table_position_z,
    const SequenceCache& cache,
    const std::vector<PrecomputedTrajectory>& trajectories,
    const std::vector<int>& readout_index_in_es,
    const uint32_t* physio_stamps = nullptr);

/**
 * @brief Add WaveformInformation entries to an ISMRMRD header for the
 *        physiological signal types that will be sent.
 *
 * Call this in OnPrep after deserialization, before serializing and sending
 * the ISMRMRD header to Gadgetron.
 *
 * Waveform IDs follow the MRD standard:
 *   0 = ECG, 1 = Pulse Oximetry (PPG), 2 = Respiratory
 *
 * @param hdr         Header to modify in-place.
 * @param has_ecg     Include ECG waveform information.
 * @param has_ppg     Include pulse oximetry (PPG) waveform information.
 * @param has_resp    Include respiratory waveform information.
 */
void add_waveform_information(ISMRMRD::IsmrmrdHeader& hdr,
                              bool has_ecg, bool has_ppg, bool has_resp);

/**
 * @brief Create a single ISMRMRD Waveform from one or more int16_t channel arrays.
 *
 * Each element of @p channels is a pointer to an array of @p num_samples int16_t
 * samples. Channels may differ in the waveform but must all have the same length.
 * int16_t samples are sign-extended to uint32_t in the output data array.
 *
 * ISMRMRD data layout: channel-major — all samples of channel 0, then channel 1, etc.
 *
 * @param waveform_id      MRD waveform type ID (0=ECG, 1=PPG, 2=Resp, 3/4=ext).
 * @param measurement_uid  Stable per-scan identifier.
 * @param scan_counter     Index of the next acquisition after this waveform.
 * @param time_stamp_ms    Start timestamp of the waveform (ms since midnight).
 * @param sample_time_us   Time between samples in microseconds.
 * @param channels         Vector of pointers to int16_t sample arrays.
 * @param num_samples      Number of samples per channel.
 * @return Populated ISMRMRD::Waveform ready to send.
 */
ISMRMRD::Waveform make_physio_waveform(uint16_t waveform_id,
                                       uint32_t measurement_uid,
                                       uint32_t scan_counter,
                                       uint32_t time_stamp_ms,
                                       float sample_time_us,
                                       const std::vector<const int16_t*>& channels,
                                       uint16_t num_samples);

/* ------------------------------------------------------------------ */
/*  Section 6 waveform factory functions                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Waveform IDs for the sequence-description custom waveforms.
 *
 * 999  — sequence-description header (one waveform per scan)
 * 1000 — per-TR event list (one per subsequence)
 * 1002 — RF shape tuples (one per subsequence)
 * 1005 — shim definitions (one per subsequence, omitted if empty)
 */
constexpr uint16_t WAVEFORM_ID_SEQDESC_HEADER = 999;
constexpr uint16_t WAVEFORM_ID_SEQDESC_EVENTS = 1000;
constexpr uint16_t WAVEFORM_ID_SEQDESC_RF_SHAPES = 1002;
constexpr uint16_t WAVEFORM_ID_SEQDESC_SHIMS = 1005;

/**
 * @brief Create the sequence-description header waveform (ID 999).
 *
 * Contains: num_subseqs, min_te_us, min_tr_us, max_tr_us,
 *           max_flip_angle_deg, total_scan_time_us  (as uint32 bit-casts).
 */
ISMRMRD::Waveform make_seqdesc_header_waveform(
    const SequenceCache& cache,
    uint32_t measurement_uid,
    uint32_t scan_counter);

/**
 * @brief Create the event-list waveform (ID 1000) for one subsequence.
 *
 * Encodes all SeqEvent entries as float32 words packed into uint32 channels:
 *   channel 0: event type (0=WAIT, 1=RF, 2=ADC)
 *   channels 1-7: params[0..6]
 */
ISMRMRD::Waveform make_seqdesc_events_waveform(
    const SequenceDescription& desc,
    uint32_t measurement_uid,
    uint32_t scan_counter);

/**
 * @brief Create the RF-shape-tuple waveform (ID 1002) for one subsequence.
 *
 * Packs the flattened mag/phase/time arrays for all tuples.
 * Header per tuple (as uint32 words):
 *   tuple_id, N_tx, N_samples, rf_raster_us (float), num_bands,
 *   band_freq_offsets_hz[8], band_bandwidth_hz, total_b1sq_power,
 *   has_phase, has_time
 * Followed by the waveform samples.
 */
ISMRMRD::Waveform make_seqdesc_rf_shapes_waveform(
    const SequenceDescription& desc,
    uint32_t measurement_uid,
    uint32_t scan_counter);

/**
 * @brief Create the shim-definitions waveform (ID 1005) for one subsequence.
 *
 * Per shim: shim_id_local, N_ch, magnitudes[N_ch], phases[N_ch].
 * Returns an empty (zero-sample) waveform if no shims are defined.
 */
ISMRMRD::Waveform make_seqdesc_shims_waveform(
    const SequenceDescription& desc,
    uint32_t measurement_uid,
    uint32_t scan_counter);

} // namespace mrdserver

#endif // TRAJECTORY_CACHE_READER_H
