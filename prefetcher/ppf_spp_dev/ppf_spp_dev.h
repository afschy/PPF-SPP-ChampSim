#ifndef SPP_H
#define SPP_H

#include <cstdint>
#include <vector>

namespace spp
{
// SPP functional knobs
constexpr bool LOOKAHEAD_ON = true;
constexpr bool FILTER_ON = true;
constexpr bool GHR_ON = true;
constexpr bool SPP_SANITY_CHECK = true;
constexpr bool SPP_DEBUG_PRINT = false;

// Signature table parameters
constexpr std::size_t ST_SET = 1;
constexpr std::size_t ST_WAY = 256;
constexpr unsigned ST_TAG_BIT = 16;
constexpr uint32_t ST_TAG_MASK = ((1 << ST_TAG_BIT) - 1);
constexpr unsigned SIG_SHIFT = 3;
constexpr unsigned SIG_BIT = 12;
constexpr uint32_t SIG_MASK = ((1 << SIG_BIT) - 1);
constexpr unsigned SIG_DELTA_BIT = 7;

// Pattern table parameters
constexpr std::size_t PT_SET = 512;
constexpr std::size_t PT_WAY = 4;
constexpr unsigned C_SIG_BIT = 4;
constexpr unsigned C_DELTA_BIT = 4;
constexpr uint32_t C_SIG_MAX = ((1 << C_SIG_BIT) - 1);
constexpr uint32_t C_DELTA_MAX = ((1 << C_DELTA_BIT) - 1);

// Prefetch filter parameters
constexpr unsigned QUOTIENT_BIT = 10;
constexpr unsigned REMAINDER_BIT = 6;
constexpr unsigned HASH_BIT = (QUOTIENT_BIT + REMAINDER_BIT + 1);
constexpr std::size_t FILTER_SET = (1 << QUOTIENT_BIT);
constexpr uint32_t FILL_THRESHOLD = 90;
constexpr uint32_t PF_THRESHOLD = 25;

// Global register parameters
constexpr unsigned GLOBAL_COUNTER_BIT = 10;
constexpr uint32_t GLOBAL_COUNTER_MAX = ((1 << GLOBAL_COUNTER_BIT) - 1);
constexpr std::size_t MAX_GHR_ENTRY = 8;

// PPF parameters
constexpr int32_t PPF_T_HI;
constexpr int32_t PPF_T_LO;
constexpr int32_t PPF_THETA_P;
constexpr int32_t PPF_THETA_N;
constexpr uint8_t MAX_SPECULATION_DEPTH = 10;

constexpr unsigned PERCEPTRON_WEIGHT_BITS = 5;
constexpr int32_t PERCEPTRON_MIN_WEIGHT = -( 1 << (PERCEPTRON_WEIGHT_BITS - 1) );
constexpr int32_t PERCEPTRON_MAX_WEIGHT = ( 1 << (PERCEPTRON_WEIGHT_BITS - 1) ) - 1;
constexpr uint8_t PPF_TRAINING_DELTA = 1;

enum FILTER_REQUEST { SPP_L2C_PREFETCH, SPP_LLC_PREFETCH, L2C_DEMAND, L2C_EVICT }; // Request type for prefetch filter
uint64_t get_hash(uint64_t key);

class SIGNATURE_TABLE
{
public:
	bool valid[ST_SET][ST_WAY];
	uint32_t tag[ST_SET][ST_WAY], last_offset[ST_SET][ST_WAY], sig[ST_SET][ST_WAY], lru[ST_SET][ST_WAY];

	SIGNATURE_TABLE()
	{
		for (uint32_t set = 0; set < ST_SET; set++)
			for (uint32_t way = 0; way < ST_WAY; way++) {
				valid[set][way] = 0;
				tag[set][way] = 0;
				last_offset[set][way] = 0;
				sig[set][way] = 0;
				lru[set][way] = way;
			}
	};

	void read_and_update_sig(uint64_t page, uint32_t page_offset, uint32_t& last_sig, uint32_t& curr_sig, int32_t& delta);
};

class PATTERN_TABLE
{
public:
	int delta[PT_SET][PT_WAY];
	uint32_t c_delta[PT_SET][PT_WAY], c_sig[PT_SET];

	PATTERN_TABLE()
	{
		for (uint32_t set = 0; set < PT_SET; set++) {
			for (uint32_t way = 0; way < PT_WAY; way++) {
				delta[set][way] = 0;
				c_delta[set][way] = 0;
			}
			c_sig[set] = 0;
		}
	}

	void update_pattern(uint32_t last_sig, int curr_delta);
    void read_pattern(uint32_t curr_sig, std::vector<int>&prefetch_delta, std::vector<uint32_t>&confidence_q,
					    uint32_t&lookahead_way, uint32_t&lookahead_conf, uint32_t&pf_q_tail, uint32_t&depth);
};

class PREFETCH_FILTER
{
public:
	uint64_t remainder_tag[FILTER_SET];
	bool valid[FILTER_SET]; // Consider this as "prefetched"
	bool useful[FILTER_SET]; // Consider this as "used"

	PREFETCH_FILTER()
	{
		for (uint32_t set = 0; set < FILTER_SET; set++) {
			remainder_tag[set] = 0;
			valid[set] = 0;
			useful[set] = 0;
		}
	}

	bool check(uint64_t pf_addr, FILTER_REQUEST filter_request);
};

class GLOBAL_REGISTER
{
public:
	// Global counters to calculate global prefetching accuracy
	uint32_t pf_useful, pf_issued;
	uint32_t global_accuracy; // Alpha value in Section III. Equation 3

	// Global History Register (GHR) entries
	uint8_t valid[MAX_GHR_ENTRY];
	uint32_t sig[MAX_GHR_ENTRY], confidence[MAX_GHR_ENTRY], offset[MAX_GHR_ENTRY];
	int delta[MAX_GHR_ENTRY];

	GLOBAL_REGISTER()
	{
		pf_useful = 0;
		pf_issued = 0;
		global_accuracy = 0;

		for (uint32_t i = 0; i < MAX_GHR_ENTRY; i++) {
			valid[i] = 0;
			sig[i] = 0;
			confidence[i] = 0;
			offset[i] = 0;
			delta[i] = 0;
		}
	}

	void update_entry(uint32_t pf_sig, uint32_t pf_confidence, uint32_t pf_offset, int pf_delta);
	uint32_t check_entry(uint32_t page_offset);
};

// Data stored in the prefetch table and the reject table
struct METADATA_ENTRY
{
    bool valid;
    uint64_t tag;
    bool useful;

    // uint16_t pc;
    // uint64_t physical_address;
    // uint16_t curr_sig;
    // uint16_t pc_hash;
    // uint8_t delta;
    // uint16_t confidence;
    // uint8_t depth;

	uint16_t ind_physical_address;
	uint16_t ind_cache_line;
	uint16_t ind_page_address;
	uint16_t ind_pc_xor_depth;
	uint16_t ind_pc_hash;
	uint16_t ind_pc_xor_delta;
	uint16_t ind_confidence;
	uint16_t ind_page_address_xor_confidence;
	uint16_t ind_curr_sig_xor_delta;

    METADATA_ENTRY() {
        valid = false;
        tag = 0;
        useful = false;
    }
};

class PPF_MODULE
{
public:
    // Weight tables for inference
    // weights are from -16 to 15, requiring 5 bits
    std::vector<int8_t> physical_address;
    std::vector<int8_t> cache_line;
    std::vector<int8_t> page_address;
    std::vector<int8_t> pc_xor_depth;
    std::vector<int8_t> pc_hash;
    std::vector<int8_t> pc_xor_delta;
    std::vector<int8_t> confidence;
    std::vector<int8_t> page_address_xor_confidence;
    std::vector<int8_t> curr_sig_xor_delta;

	// Prefetch table and reject table
	std::vector<METADATA_ENTRY> prefetch_table;
	std::vector<METADATA_ENTRY> reject_table;

	PPF_MODULE() {
		physical_address.resize(4096); std::fill(physical_address.begin(), physical_address.end(), 0);
		cache_line.resize(4096); std::fill(cache_line.begin(), cache_line.end(), 0);
		page_address.resize(4096); std::fill(page_address.begin(), page_address.end(), 0);
		pc_xor_depth.resize(4096); std::fill(pc_xor_depth.begin(), pc_xor_depth.end(), 0);
		pc_hash.resize(4096); std::fill(pc_hash.begin(), pc_hash.end(), 0);
		pc_xor_delta.resize(4096); std::fill(pc_xor_delta.begin(), pc_xor_delta.end(), 0);
		confidence.resize(4096); std::fill(confidence.begin(), confidence.end(), 0);
		page_address_xor_confidence.resize(4096); std::fill(page_address_xor_confidence.begin(), page_address_xor_confidence.end(), 0);
		curr_sig_xor_delta.resize(4096); std::fill(curr_sig_xor_delta.begin(), curr_sig_xor_delta.end(), 0);

		prefetch_table.resize(FILTER_SET);
		reject_table.resize(FILTER_SET);
	}

	inline void bounded_increment(std::vector<int8_t>& vec, const uint16_t ind) {
		vec[ind] = std::min(vec[ind] + PPF_TRAINING_DELTA, PERCEPTRON_MAX_WEIGHT);
	}

	inline void bounded_decrement(std::vector<int8_t>& vec, const uint16_t ind) {
		vec[ind] = std::max(vec[ind] - PPF_TRAINING_DELTA, PERCEPTRON_MIN_WEIGHT);
	}

	int get_sum(const METADATA_ENTRY& metadata) {
		int sum = 0;
		sum += physical_address[metadata.ind_physical_address];
		sum += cache_line[metadata.ind_cache_line];
		sum += page_address[metadata.ind_page_address];
		sum += pc_xor_depth[metadata.ind_pc_xor_depth];
		sum += pc_hash[metadata.ind_pc_hash];
		sum += pc_xor_delta[metadata.ind_pc_xor_delta];
		sum += confidence[metadata.ind_confidence];
		sum += page_address_xor_confidence[metadata.ind_page_address_xor_confidence];
		sum += curr_sig_xor_delta[metadata.ind_curr_sig_xor_delta];

		return sum;
	}

	void retrain(const bool success, const METADATA_ENTRY& metadata) {
		int sum = get_sum(metadata);

		if(success && (sum < PPF_THETA_P)) {
			bounded_increment(physical_address, metadata.ind_physical_address);
			bounded_increment(cache_line, metadata.ind_cache_line);
			bounded_increment(page_address, metadata.ind_page_address);
			bounded_increment(pc_xor_depth, metadata.ind_pc_xor_depth);
			bounded_increment(pc_hash, metadata.ind_pc_hash);
			bounded_increment(pc_xor_delta, metadata.ind_pc_xor_delta);
			bounded_increment(confidence, metadata.ind_confidence);
			bounded_increment(page_address_xor_confidence, metadata.ind_page_address_xor_confidence);
			bounded_increment(curr_sig_xor_delta, metadata.ind_curr_sig_xor_delta);
		}
		else if(!success && (sum > PPF_THETA_N)) {
			bounded_decrement(physical_address, metadata.ind_physical_address);
			bounded_decrement(cache_line, metadata.ind_cache_line);
			bounded_decrement(page_address, metadata.ind_page_address);
			bounded_decrement(pc_xor_depth, metadata.ind_pc_xor_depth);
			bounded_decrement(pc_hash, metadata.ind_pc_hash);
			bounded_decrement(pc_xor_delta, metadata.ind_pc_xor_delta);
			bounded_decrement(confidence, metadata.ind_confidence);
			bounded_decrement(page_address_xor_confidence, metadata.ind_page_address_xor_confidence);
			bounded_decrement(curr_sig_xor_delta, metadata.ind_curr_sig_xor_delta);
		}
	}

	METADATA_ENTRY create_metadata(uint64_t v_pc, uint64_t v_physical_address, uint64_t v_curr_sig, uint64_t v_pc_hash,
						 			uint64_t v_delta, uint64_t v_confidence, uint64_t v_depth) {

		uint64_t v_page = physical_address >> LOG2_PAGE_SIZE;
		uint64_t v_cache_line = physical_address >> LOG2_BLOCK_SIZE;

		METADATA_ENTRY metadata;
		metadata.ind_physical_address = get_hash(v_physical_address) % physical_address.size();
		metadata.ind_cache_line = get_hash(v_cache_line) % cache_line.size();
		metadata.ind_page_address = get_hash(v_page) % page_address.size();
		metadata.ind_pc_xor_depth = get_hash(v_pc ^ v_depth) % pc_xor_depth.size();
		metadata.ind_pc_hash = get_hash(v_pc_hash) % pc_hash.size();
		metadata.ind_pc_xor_delta = get_hash(v_pc ^ v_delta) % pc_xor_delta.size();
		metadata.ind_confidence = get_hash(v_confidence) % confidence.size();
		metadata.ind_page_address_xor_confidence = get_hash(v_page ^ v_confidence) % page_address_xor_confidence.size();
		metadata.ind_curr_sig_xor_delta = get_hash(v_curr_sig ^ v_delta) % curr_sig_xor_delta.size();

		return metadata;
	}

	// Called every time something is demanded, or something is evicted
	// hit_or_evict is true for hit, false for evict
	void feedback_update(uint64_t addr, bool hit_or_evict) {
		uint64_t cache_line = addr >> LOG2_BLOCK_SIZE;
		uint64_t hash = get_hash(cache_line);
		uint64_t quotient = (hash >> REMAINDER_BIT) & ((1 << QUOTIENT_BIT) - 1);
		uint64_t remainder = hash % (1 << REMAINDER_BIT);

		METADATA_ENTRY& prefetch_entry = prefetch_table[quotient];
		METADATA_ENTRY& reject_entry = reject_table[quotient];

		if(hit_or_evict) {
			if(prefetch_entry.valid && prefetch_entry.tag == remainder)
				retrain(true, prefetch_entry);
			if(reject_entry.valid && reject_entry.tag == remainder)
				retrain(true, reject_entry);
		}
		else {
			if(prefetch_entry.valid && prefetch_entry.tag == remainder) {
				retrain(false, prefetch_entry);
				prefetch_entry.valid = false;
				prefetch_entry.tag = 0;
			}
			if(reject_entry.valid && reject_entry.tag == remainder) {
				retrain(false, reject_entry);
				reject_entry.valid = false;
				reject_entry.tag = 0;
			}
		}
	}
};

} // namespace spp

#endif
