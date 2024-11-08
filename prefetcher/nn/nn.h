#ifndef SPP_H
#define SPP_H

#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
using std::vector;

// SPP functional knobs
#define LOOKAHEAD_ON
#define FILTER_ON
#define GHR_ON
#define SPP_SANITY_CHECK

//#define SPP_DEBUG_PRINT
#ifdef SPP_DEBUG_PRINT
#define SPP_DP(x) x
#else
#define SPP_DP(x)
#endif

//#define SPP_PERC_WGHT
#ifdef SPP_PERC_WGHT
#define SPP_PW(x) x
#else 
#define SPP_PW(x)
#endif

// Signature table parameters
#define ST_SET 1
#define ST_WAY 256
#define ST_TAG_BIT 16
#define ST_TAG_MASK ((1 << ST_TAG_BIT) - 1)
#define SIG_SHIFT 3
#define SIG_BIT 12
#define SIG_MASK ((1 << SIG_BIT) - 1)
#define SIG_DELTA_BIT 7

// Pattern table parameters
#define PT_SET 512
#define PT_WAY 4
#define C_SIG_BIT 4
#define C_DELTA_BIT 4
#define C_SIG_MAX ((1 << C_SIG_BIT) - 1)
#define C_DELTA_MAX ((1 << C_DELTA_BIT) - 1)

// Prefetch filter parameters
#define QUOTIENT_BIT    10
#define REMAINDER_BIT 6
#define HASH_BIT (QUOTIENT_BIT + REMAINDER_BIT + 1)
#define FILTER_SET (1 << QUOTIENT_BIT)

#define QUOTIENT_BIT_REJ    10
#define REMAINDER_BIT_REJ 8
#define HASH_BIT_REJ (QUOTIENT_BIT_REJ + REMAINDER_BIT_REJ + 1)
#define FILTER_SET_REJ (1 << QUOTIENT_BIT_REJ)

// Global register parameters
#define GLOBAL_COUNTER_BIT 10
#define GLOBAL_COUNTER_MAX ((1 << GLOBAL_COUNTER_BIT) - 1) 
#define MAX_GHR_ENTRY 8
#define PAGES_TRACKED 6

// Perceptron paramaters
#define PERC_ENTRIES 4096 //Upto 12-bit addressing in hashed perceptron
#define PERC_FEATURES 9 //Keep increasing based on new features
#define PERC_COUNTER_MAX 15 //-16 to +15: 5 bits counter 
#define PERC_THRESHOLD_HI    -5
#define PERC_THRESHOLD_LO    -15
#define POS_UPDT_THRESHOLD    90
#define NEG_UPDT_THRESHOLD   -80

// NN parameters
#define FIRST_LAYER_SIZE 20
#define LAST_LAYER_SIZE 2
#define LEARN_RATE 0.2
#define LLC_DELTA -0.5
#define L2C_DELTA 0.0

enum FILTER_REQUEST { SPP_L2C_PREFETCH, SPP_LLC_PREFETCH, L2C_DEMAND, L2C_EVICT, SPP_PERC_REJECT}; // Request type for prefetch filter
uint64_t get_hash(uint64_t key);
void get_perc_index(uint64_t base_addr, uint64_t ip, uint64_t ip_1, uint64_t ip_2, uint64_t ip_3, int32_t cur_delta, uint32_t last_sig, uint32_t curr_sig, uint32_t confidence, uint32_t depth, uint64_t perc_set[PERC_FEATURES]);

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

    void update_pattern(uint32_t last_sig, int curr_delta),
                read_pattern(uint32_t curr_sig, vector<int> &delta_q, vector<uint32_t> &confidence_q, vector<int32_t> &perc_sum_q, uint32_t &lookahead_way, uint32_t &lookahead_conf, uint32_t &pf_q_tail, uint32_t &depth, uint64_t addr, uint64_t base_addr, uint64_t train_addr, uint64_t curr_ip, int32_t train_delta, uint32_t last_sig, uint32_t pq_occupancy, uint32_t pq_SIZE, uint32_t mshr_occupancy, uint32_t mshr_SIZE);
};

class PREFETCH_FILTER
{
public:
    uint64_t remainder_tag[FILTER_SET],
                        pc[FILTER_SET],
                        pc_1[FILTER_SET],
                        pc_2[FILTER_SET],
                        pc_3[FILTER_SET],
                        address[FILTER_SET];
    bool         valid[FILTER_SET],    // Consider this as "prefetched"
                        useful[FILTER_SET]; // Consider this as "used"
    int32_t	 delta[FILTER_SET],
                        perc_sum[FILTER_SET];
    uint32_t last_signature[FILTER_SET],
                        confidence[FILTER_SET],
                        cur_signature[FILTER_SET],
                        la_depth[FILTER_SET];

    uint64_t remainder_tag_reject[FILTER_SET_REJ],
                        pc_reject[FILTER_SET_REJ],
                        pc_1_reject[FILTER_SET_REJ],
                        pc_2_reject[FILTER_SET_REJ],
                        pc_3_reject[FILTER_SET_REJ],
                        address_reject[FILTER_SET_REJ];
    bool 	 valid_reject[FILTER_SET_REJ]; // Entries which the perceptron rejected
    int32_t	 delta_reject[FILTER_SET_REJ],
                        perc_sum_reject[FILTER_SET_REJ];
    uint32_t last_signature_reject[FILTER_SET_REJ],
                        confidence_reject[FILTER_SET_REJ],
                        cur_signature_reject[FILTER_SET_REJ],
                        la_depth_reject[FILTER_SET_REJ];

    // Tried the set-dueling idea which din't work out
    uint32_t PSEL_1;
    uint32_t PSEL_2;

    // To enable / disable negative training using reject filter
    // Set to 1 in the prefetcher file
    bool train_neg;

    float hist_hits[55];
    float hist_tots[55];

    PREFETCH_FILTER()
    {
        for (int i = 0; i < 55; i++) {
            hist_hits[i] = 0;
            hist_tots[i] = 0;
        }
        for (uint32_t set = 0; set < FILTER_SET; set++) {
            remainder_tag[set] = 0;
            valid[set] = 0;
            useful[set] = 0;
        }
        for (uint32_t set = 0; set < FILTER_SET_REJ; set++) {
            valid_reject[set] = 0;
            remainder_tag_reject[set] = 0;
        }
        train_neg = 0;
    }

    bool check(uint64_t pf_addr, uint64_t base_addr, uint64_t ip, FILTER_REQUEST filter_request, int32_t cur_delta, uint32_t last_sign, uint32_t cur_sign, uint32_t confidence, int32_t sum, uint32_t depth);
    bool add_to_filter(uint64_t check_addr, uint64_t base_addr, uint64_t ip, FILTER_REQUEST filter_request, int cur_delta, uint32_t last_sig, uint32_t curr_sig, uint32_t conf, int32_t sum, uint32_t depth);
};

class PERCEPTRON
{
public:
    // Perc Weights
    int32_t perc_weights[PERC_ENTRIES][PERC_FEATURES];

    // Only for dumping csv
    bool perc_touched[PERC_ENTRIES][PERC_FEATURES];

    // CONST depths for different features
    int32_t PERC_DEPTH[PERC_FEATURES];

    PERCEPTRON() {
        PERC_DEPTH[0] = 2048;     //base_addr;
        PERC_DEPTH[1] = 4096;     //cache_line;
        PERC_DEPTH[2] = 4096;    	//page_addr;
        PERC_DEPTH[3] = 4096;     //confidence ^ page_addr;
        PERC_DEPTH[4] = 1024;	//curr_sig ^ sig_delta;
        PERC_DEPTH[5] = 4096; 	//ip_1 ^ ip_2 ^ ip_3;		
        PERC_DEPTH[6] = 1024; 	//ip ^ depth;
        PERC_DEPTH[7] = 2048;     //ip ^ sig_delta;
        PERC_DEPTH[8] = 128;     	//confidence;

        for (int i = 0; i < PERC_ENTRIES; i++) {
            for (int j = 0;j < PERC_FEATURES; j++) {
                perc_weights[i][j] = 0;
                perc_touched[i][j] = 0;
            }
        }
    }

    void perc_update(uint64_t base_addr, uint64_t ip, uint64_t ip_1, uint64_t ip_2, uint64_t ip_3, int32_t cur_delta, uint32_t last_sig, uint32_t curr_sig, uint32_t confidence, uint32_t depth, bool direction, int32_t perc_sum)
    {
        uint64_t perc_set[PERC_FEATURES];
        // Get the perceptron indexes
        get_perc_index(base_addr, ip, ip_1, ip_2, ip_3, cur_delta, last_sig, curr_sig, confidence, depth, perc_set);

        int32_t sum = 0;
        for (int i = 0; i < PERC_FEATURES; i++) {
            // Marking the weights as touched for final dumping in the csv
            perc_touched[perc_set[i]][i] = 1;
        }
        // Restore the sum that led to the prediction
        sum = perc_sum;

        if (!direction) { // direction = 1 means the sum was in the correct direction, 0 means it was in the wrong direction
            // Prediction wrong
            for (int i = 0; i < PERC_FEATURES; i++) {
                if (sum >= PERC_THRESHOLD_HI) {
                    // Prediction was to prefetch -- so decrement counters
                    if (perc_weights[perc_set[i]][i] > -1*(PERC_COUNTER_MAX+1) )
                    perc_weights[perc_set[i]][i]--;
                }
                if (sum < PERC_THRESHOLD_HI) {
                    // Prediction was to not prefetch -- so increment counters
                    if (perc_weights[perc_set[i]][i] < PERC_COUNTER_MAX)
                    perc_weights[perc_set[i]][i]++;
                }
            }
        }
        if (direction && sum > NEG_UPDT_THRESHOLD && sum < POS_UPDT_THRESHOLD) {
            // Prediction correct but sum not 'saturated' enough
            for (int i = 0; i < PERC_FEATURES; i++) {
                if (sum >= PERC_THRESHOLD_HI) {
                    // Prediction was to prefetch -- so increment counters
                    if (perc_weights[perc_set[i]][i] < PERC_COUNTER_MAX)
                    perc_weights[perc_set[i]][i]++;
                }
                if (sum < PERC_THRESHOLD_HI) {
                    // Prediction was to not prefetch -- so decrement counters
                    if (perc_weights[perc_set[i]][i] > -1*(PERC_COUNTER_MAX+1) )
                    perc_weights[perc_set[i]][i]--;
                }
            }
        }
    }
    int32_t	perc_predict(uint64_t base_addr, uint64_t ip, uint64_t ip_1, uint64_t ip_2, uint64_t ip_3, int32_t cur_delta, uint32_t last_sig, uint32_t curr_sig, uint32_t confidence, uint32_t depth)
    {
        uint64_t perc_set[PERC_FEATURES];
        // Get the indexes in perc_set[]
        get_perc_index(base_addr, ip, ip_1, ip_2, ip_3, cur_delta, last_sig, curr_sig, confidence, depth, perc_set);

        int32_t sum = 0;
        for (int i = 0; i < PERC_FEATURES; i++) {
            sum += perc_weights[perc_set[i]][i];	
            // Calculate Sum
        }
        // Return the sum
        return sum;
    }
};


class GLOBAL_REGISTER
{
public:
    // Global counters to calculate global prefetching accuracy
    uint64_t pf_useful,
                        pf_issued,
                        global_accuracy; // Alpha value in Section III. Equation 3

    // Global History Register (GHR) entries
    uint8_t valid[MAX_GHR_ENTRY];
    uint32_t sig[MAX_GHR_ENTRY], confidence[MAX_GHR_ENTRY], offset[MAX_GHR_ENTRY];
    int delta[MAX_GHR_ENTRY];

    uint64_t ip_0,
                        ip_1,
                        ip_2,
                        ip_3;

    uint64_t page_tracker[PAGES_TRACKED];

    // Stats Collection
    double 	    depth_val,
                        depth_sum,
                        depth_num;
    double 	    pf_total,
                        pf_l2c,
                        pf_llc,
                        pf_l2c_good;
    long 	    perc_pass,
                    perc_reject,
                    reject_update;
    // Stats

    GLOBAL_REGISTER()
    {
        pf_useful = 0;
        pf_issued = 0;
        global_accuracy = 0;
        ip_0 = 0;
        ip_1 = 0;
        ip_2 = 0;
        ip_3 = 0;

        // These are just for stats printing
        depth_val = 0;
        depth_sum = 0;
        depth_num = 0;
        pf_total = 0;
        pf_l2c = 0;
        pf_llc = 0;
        pf_l2c_good = 0;
        perc_pass = 0;
        perc_reject = 0;
        reject_update = 0;

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

double relu(double x) { return std::max(0.0, x); }
double d_relu(double x) { return (x > 0.0)? 1 : 0; }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double d_sigmoid(double x) { return x * (1 - x); }

vector<double> softmax(const vector<double>& vec) {
    vector<double> result(vec.size());
    double sum = 0;
    for(int i=0; i<vec.size(); i++) sum += exp(vec[i]);
    for(int i=0; i<vec.size(); i++) result[i] = exp(vec[i]) / sum;
    return result;
}

class NN_LAYER
{
public:
    int input_count, output_count;
    vector<vector<double>> w, dw;
    vector<double> b, db;

    vector<double> a, x_in;

    NN_LAYER(int ic, int oc)
    {
        input_count = ic;
        output_count = oc;
        
        w.resize(output_count);
        dw.resize(output_count);

        for(int i=0; i<w.size(); i++){
            w[i].resize(input_count);
            dw[i].resize(input_count);
        }

        b.resize(output_count);
        db.resize(output_count);

        a.resize(output_count);
        x_in.resize(input_count);
    }

    vector<double> forward(const vector<double>& x_in)
    {
        assert(x_in.size() == input_count && "input size mismatch at NN_LAYER.forward\n");
        this->x_in = x_in;
        vector<double> activations(output_count);

        for(int node_out=0; node_out < output_count; node_out++) {
            double weighted_input = b[node_out];
            for(int node_in=0; node_in < input_count; node_in++)
                weighted_input += x_in[node_in] * w[node_out][node_in];
            activations[node_out] = sigmoid(weighted_input);
        }

        this->a = activations;
        return activations;
    }

    vector<double> backward(const vector<double>& nodeval_next)
    {
        // nodeval_next.size() == nodeval_curr.size() == output_count
        assert(nodeval_next.size() == output_count && "nodeval_next size mismatch at NN_LAYER.backward\n");
        vector<double> nodeval_curr = nodeval_next;
        for(int i=0; i<nodeval_curr.size(); i++)
            nodeval_curr[i] *= d_sigmoid(this->a[i]);
        
        vector<double> nodeval_prev(input_count);

        for(int out=0; out<input_count; out++) {
            nodeval_prev[out] = 0;
            for(int in=0; in<output_count; in++)
                nodeval_prev[out] += w[in][out] * nodeval_curr[in];
        }

        for(int out=0; out<output_count; out++)
            for(int in=0; in<input_count; in++)
                dw[out][in] = nodeval_curr[out] * x_in[in];
        db = nodeval_curr;
        
        for(int out=0; out<output_count; out++) {
            b[out] -= LEARN_RATE * db[out];
            for(int in=0; in<input_count; in++)
                w[out][in] -= LEARN_RATE * dw[out][in];
        }

        return nodeval_prev;
    }
};

class NN
{
public:
    // Perc Weights
    double perc_weights[FIRST_LAYER_SIZE][PERC_ENTRIES][PERC_FEATURES];
    vector<double> perc_biases;
    vector<double> first_layer_outputs;

    // Only for dumping csv
    bool perc_touched[FIRST_LAYER_SIZE][PERC_ENTRIES][PERC_FEATURES];

    // CONST depths for different features
    int32_t PERC_DEPTH[PERC_FEATURES];

    vector<int> layer_size_list;
    vector<NN_LAYER> layers;

    NN(const vector<int>& layer_size_list)
    {
        first_layer_outputs.resize(FIRST_LAYER_SIZE);
        perc_biases.resize(FIRST_LAYER_SIZE);

        PERC_DEPTH[0] = 2048; //base_addr;
        PERC_DEPTH[1] = 4096; //cache_line;
        PERC_DEPTH[2] = 4096; //page_addr;
        PERC_DEPTH[3] = 4096; //confidence ^ page_addr;
        PERC_DEPTH[4] = 1024; //curr_sig ^ sig_delta;
        PERC_DEPTH[5] = 4096; //ip_1 ^ ip_2 ^ ip_3;		
        PERC_DEPTH[6] = 1024; //ip ^ depth;
        PERC_DEPTH[7] = 2048; //ip ^ sig_delta;
        PERC_DEPTH[8] = 128;  //confidence;
        
        for(int k=0; k < FIRST_LAYER_SIZE; k++) {
            for (int i = 0; i < PERC_ENTRIES; i++) {
                for (int j = 0;j < PERC_FEATURES; j++) {
                    perc_weights[k][i][j] = 1.00 * rand() / RAND_MAX;
                    perc_weights[k][i][j] *= sqrt(2.0 / PERC_FEATURES);
                    perc_touched[k][i][j] = 0;
                }
            }
        }
        
        this->layer_size_list = layer_size_list;
        for(int i=1; i<layer_size_list.size(); i++)
            layers.push_back( NN_LAYER(layer_size_list[i-1], layer_size_list[i]) );
    }

    vector<double> forward(const vector<double>& x_in)
    {
        assert(x_in.size() == layers[0].input_count && "input size mismatch at NN.forward\n");
        vector<double> curr = x_in;
        for(int i=0; i<layers.size(); i++)
            curr = layers[i].forward(curr);
        return curr;
    }

    vector<double> backward(const vector<double>& nodeval_out)
    {
        assert(nodeval_out.size() == layers[layers.size()-1].output_count && "output error size mismatch at NN.backward\n");
        vector<double> curr = nodeval_out;
        for(int i=layers.size()-1; i>=0; i--)
            curr = layers[i].backward(curr);
        return curr;
    }

    void nn_update(uint64_t base_addr, uint64_t ip, uint64_t ip_1, uint64_t ip_2, uint64_t ip_3, int32_t cur_delta, uint32_t last_sig, uint32_t curr_sig, uint32_t confidence, uint32_t depth, bool should_have_prefetched)
    {
        uint64_t perc_set[PERC_FEATURES];
        // Get the perceptron indexes
        get_perc_index(base_addr, ip, ip_1, ip_2, ip_3, cur_delta, last_sig, curr_sig, confidence, depth, perc_set);

        vector<double> expected_result({0.0, 0.0});
        if(should_have_prefetched) expected_result[1] = 1;
        else expected_result[0] = 1;

        vector<double> predicted_result = this->nn_predict(base_addr, ip, ip_1, ip_2, ip_3, cur_delta, last_sig, curr_sig, confidence, depth);
        vector<double> output_error(2);
        output_error[0] = predicted_result[0] - expected_result[0];
        output_error[1] = predicted_result[1] - expected_result[1];

        vector<double> nodevals = this->backward(output_error);
        for(int i=0; i<FIRST_LAYER_SIZE; i++)
            nodevals[i] *= d_sigmoid(this->first_layer_outputs[i]);
        
        for(int out=0; out<FIRST_LAYER_SIZE; out++) {
            perc_biases[out] -= LEARN_RATE * nodevals[out];
            for(int in=0; in<PERC_FEATURES; in++)
                perc_weights[out][perc_set[in]][in] -= LEARN_RATE * nodevals[out];
        }
    }
    
    // Returns a vector of two values
    // result[0]: likelihood of rejecting
    // result[1]: likelihood of accepting
    vector<double> nn_predict(uint64_t base_addr, uint64_t ip, uint64_t ip_1, uint64_t ip_2, uint64_t ip_3, int32_t cur_delta, uint32_t last_sig, uint32_t curr_sig, uint32_t confidence, uint32_t depth)
    {
        uint64_t perc_set[PERC_FEATURES];
        // Get the indexes in perc_set[]
        get_perc_index(base_addr, ip, ip_1, ip_2, ip_3, cur_delta, last_sig, curr_sig, confidence, depth, perc_set);

        for(int out=0; out < FIRST_LAYER_SIZE; out++) {
            first_layer_outputs[out] = perc_biases[out];
            for (int in = 0; in < PERC_FEATURES; in++)
                first_layer_outputs[out] += perc_weights[out][perc_set[in]][in];
        }

        for(int i=0; i < FIRST_LAYER_SIZE; i++)
            first_layer_outputs[i] = sigmoid(first_layer_outputs[i]);

        return softmax(this->forward(first_layer_outputs));
    }
};

#endif
