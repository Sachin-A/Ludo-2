//
#include "common.h"
//#include "cstlib"
#include "BloomFilter/bloom_filter.h"
// include libraries for other DS
#include "MinimalPerfectCuckoo/minimal_perfect_cuckoo.h"

#include <tuple>

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

using namespace mlpack;
using namespace mlpack::tree;


typedef uint32_t Key;
template <class Val>
class BloomFilterPerPort {
  BloomFilter<Key> bloomfilter;
  public:
    int port;
    Val next_hop;
    BloomFilterPerPort(int _port=0){
      port = port;
//      next_hop = _next_hop;
    }

    void insert(const Key &key) {
      bloomfilter.insert(key);
    }

    bool lookup(const Key &key) const {
      return bloomfilter.isMember(key);
    }

    void set_port(int _port){
      port = _port;
    }

    void set_next_hop(Val _next_hop){
      next_hop = _next_hop;
    }

};

bool is_hot_key(uint32_t key){
  return (key % 2) == 0;
}

template <int VL, class Val>
class Fib {
  int num_bloom_filters;
  std::unordered_map<Val, double> hot_vals_bf_fpr;
  std::unordered_map<Val, uint64_t> hot_vals_size;
  BloomFilter<Key>* heavyhitter;
  int flag = 0;
  int mem_flag = 0;
  bool hh_bf_exists = true;
  std::unordered_map<Val, BloomFilter<Key>* > hh_bloomfilters;
  ControlPlaneMinimalPerfectCuckoo<Key, Val, VL>* cp_hashtable;
  DataPlaneMinimalPerfectCuckoo<Key, Val, VL>* dp_hashtable;
  DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, double> decisiontree;
    int wrong_ht = 0;
public:
  // FOR DEBUGGING purposes only:	
  std::unordered_map<Key, Val> actual_keys_vals; 
  uint64_t total_memory = 0;
  std::vector<double> ht_lookup_time; // HH-BF lookup + HT
  std::vector<double> bf_lookup1_time;
  std::vector<int> bf_match_cts;
  std::vector<std::vector<double>> bf_ilookup_times;
  Fib(int flag_, int mem_flag_){
   flag = flag_;
   mem_flag = mem_flag_;
	  /*
	  num_bloom_filters = num_bf;
    bloomfilters = new BloomFilterPerPort<Val>*[num_bf];
    for(int i=0; i<num_bf; ++i){
      bloomfilters[i] = new BloomFilterPerPort<Val>();
    } */
  }

  void get_heavy_hitter_vals(vector<Key> &keys, vector<Val> &values, uint64_t nn)
  {
	  std::map<Val, uint64_t> per_val_key_ct;
	  for (uint64_t i = 0; i < nn; i++)
		  per_val_key_ct[ values[i] ] += 1;
	  for (auto &i : per_val_key_ct)
		  if (i.second > 0.01 * nn )
		  {
			  cout << "Val " << i.first << " with items:" << i.second << " is HAWT!! \n";
		  	  hot_vals_bf_fpr[i.first] = -1.0; // to be optimized later.
		  }
  }

  inline uint64_t memory_heuristic(uint64_t nn, uint64_t hh_keys)
  {
	  // use upto 5KB for all BFs: giving extra buffer of 1KB to cover for ML model.
	  if (mem_flag == 0)
	  	return ludo_hash_size<VL>(nn-hh_keys) + ludo_hash_size<VL>(hh_keys); // same as ludo
	//  else if (mem_flag == 1)
	  //	return ludo_hash_size<VL>(nn-hh_keys) + 1024*8 + ((flag > 0) ? 575*8 : 0); // 1KB for all BFs.
	  else
	  	return ludo_hash_size<VL>(nn-hh_keys) + 1024*8*mem_flag + ((flag > 0) ? 575*8 : 0); // 4KB for all BFs.
	  // return ludo_hash_size<VL>(nn-hh_keys) + std::min( (uint64_t) 1024*5*8, ludo_hash_size<VL>(hh_keys)/2 );
  }

  std::tuple<std::unordered_map<Val, double>, uint64_t, double> get_opt_bf_sizes(vector<Key> &keys, vector<Val> &values, uint64_t nn)
  {
	  // find #keys for each value, find HOT keys
        printf("IN FIB.H : Finding HH ports. nn: %i, key size: %i \n", nn, keys.size());
	  std::map<Val, uint64_t> per_val_key_ct;
        for (uint64_t i = 0; i < nn; i++)
                per_val_key_ct[ values[i] ] += 1;
        // HAWT keys:
        std::unordered_set<Val> hawt_vals;
        uint64_t total_hh_keys = 0;
        for (auto &i : per_val_key_ct)
        {
		if (i.second>9)
                	cout << "Value(nh): " << i.first << ", Key ct: " << i.second << endl;
                if (hot_vals_bf_fpr.find(i.first) != hot_vals_bf_fpr.end() )
		{
                        hawt_vals.insert(i.first);
                        total_hh_keys += i.second;
                        cout << "HAWT!! \n";
                }
        }
	printf("Total #unique vals: %i , total HH keys: %i \n", per_val_key_ct.size(), total_hh_keys);
	// Si = per_val_key_ct[hot_val]
        uint64_t memory_left = memory_heuristic(nn, total_hh_keys) - (ludo_hash_size<VL>(nn - total_hh_keys)) - ( (flag > 0) ? 575*8 : 0 );
        total_memory = memory_heuristic(nn, total_hh_keys);
	std::cout << "MEMORY heuristic for total memory to be used:" << total_memory << ", memory_left after removing Ludo[nonHH], ML:" << memory_left << std::endl;

	double nln_sum = 0.0;
        double n_sum = 0.0;
        
	if  (hawt_vals.size() == 0)
	{
		std::cout << "NO HH KEY and NO BF at the top" << std::endl;
		return std::make_tuple(std::unordered_map<Val, double> (), 0, -1.0);
	}
	
	for (auto &hv: hawt_vals)
        {
                nln_sum += per_val_key_ct[hv] * log(per_val_key_ct[hv]) ;
                n_sum += per_val_key_ct[hv];
                std::cout << "val:" << hv << " sz:" << per_val_key_ct[hv] << "nln sum: " << nln_sum << ", nsum: " << n_sum << std::endl;
        }
        // if there's BFh at the top (flag=0) or using backup for LM (flag=2)
        // Give less wt to fh, since Sh is bigger than Si's.
        double w_h = 1.0/hawt_vals.size();
        uint64_t hh_bf_keyset_sz = total_hh_keys;
	if (flag == 2)
	{
		// find all HH keys for which LM gives P(HH) < 0.5
		for (int i = 0; i < nn; i++)
			hh_bf_keyset_sz -= ( is_hot_val(values[i]) && (dt_classify_prob(keys[i]) > 0.5) );
		std::cout << "ADDING BACKUP BF for HH. Only need to store " << hh_bf_keyset_sz << " keys!!! \n";
	}

	if (flag == 0 || flag == 2)
	{
		nln_sum += hh_bf_keyset_sz * log(hh_bf_keyset_sz / w_h);
		n_sum += hh_bf_keyset_sz;
		std::cout << "After adding HH-BF: nln sum: " << nln_sum << ", nsum: " << n_sum << std::endl;
	}

	double muu = exp( -1.0 * (memory_left + nln_sum) / n_sum );
        std::cout << "muu: " << muu << std::endl;

        int total_iters = 0; // for logging
        bool some_fpr_invalid = true; // loop exit condn
        std::unordered_map<Val, double> fpr_iter; // current set of fprs for val
        std::unordered_set<Val> remaining_vals = hawt_vals; // which vals have a BF
        for (auto &hv: hawt_vals)
                fpr_iter[hv] = (per_val_key_ct[hv] * muu);
        bool remaining_hh_bf = (flag != 1);
        double hh_fpr_iter = remaining_hh_bf ? (hh_bf_keyset_sz * muu / w_h) : -1.0;

	while (some_fpr_invalid)
        {
                // are some fi's > 1? collect nln,n terms.
                int curr_sz = remaining_vals.size() + remaining_hh_bf;
                for (auto &hv: remaining_vals)
                        if (fpr_iter[hv] > 1.0)
                                remaining_vals.erase(hv);

                if (remaining_hh_bf && hh_fpr_iter > 1.0 )
                        remaining_hh_bf = false;

                // re-solve:
                if ( (remaining_vals.size() + remaining_hh_bf) == curr_sz)
                        some_fpr_invalid = false;
                else
		{
                        std::cout << "NOW " << remaining_vals.size() << " remaining keys! HH-BF?" << remaining_hh_bf << std::endl;
                        double nln_sum_i = (remaining_hh_bf) ? hh_bf_keyset_sz*log(hh_bf_keyset_sz/w_h) : 0.0;
                        double n_sum_i = (remaining_hh_bf) ? hh_bf_keyset_sz : 0.0;
                        for (auto &hv: remaining_vals)
                        {
                                nln_sum_i += per_val_key_ct[hv] * log(per_val_key_ct[hv]);
                                n_sum_i += per_val_key_ct[hv];
                        }

                        muu = exp( -1.0 * (memory_left + nln_sum_i) / (n_sum_i) );
                        for (auto &hv: remaining_vals)
                                fpr_iter[hv] = muu * per_val_key_ct[hv];
                        hh_fpr_iter = remaining_hh_bf ? (muu * hh_bf_keyset_sz)/w_h : -1.0;
                }

        }

        std::unordered_map<Val, double> per_bf_fpr;

	if (remaining_vals.size() != hot_vals_bf_fpr.size())
	{
		std::cerr << "DAMNNNN. One of the HOT NH's dont have a BF!!! Exiting! \n";
		abort();
	}

	hot_vals_bf_fpr.clear();
	hot_vals_size.clear();
	for (auto &hv: remaining_vals)
        {
                hot_vals_bf_fpr[hv] = fpr_iter[hv];
                hot_vals_size[hv] = per_val_key_ct[hv];
		std::cout << "For val=" << hv << " FINAL BF fpr: " << hot_vals_bf_fpr[hv] << std::endl;
        }
        std::cout << hh_fpr_iter << " : FINAL HH FPR" << std::endl;
	return std::make_tuple(per_bf_fpr, hh_bf_keyset_sz, hh_fpr_iter);

  }

  inline bool is_hot_val(Val x)
  {
	  return (hot_vals_bf_fpr.find(x) != hot_vals_bf_fpr.end());
  }

  void populate(vector<Key> &keys, vector<Val> &values, uint64_t nn){
	  // get HH values.
	  get_heavy_hitter_vals(keys, values, nn);

	  int count_ht_vals = 0; // #entries that'll go to Ludo
          std::unordered_map<Key, bool> key_heavyhitter_map;
          for (int i=0; i<nn; i++)
	  {
		  actual_keys_vals[(keys[i])] = values[i];
		  key_heavyhitter_map[ keys[i] ] = (is_hot_val(values[i]));
	  	  count_ht_vals += (!is_hot_val(values[i])); 
	  }

	  if (flag > 0)
	  {
		arma::mat dataset(4, key_heavyhitter_map.size());
		arma::Row<size_t> labels(key_heavyhitter_map.size());
		std::cout << "TRAINING! SIZE OF x MAT:" << size(dataset);
		std::cout << ", SIZE OF y MAT:" << size(labels);
		int i = 0;
		for (auto &kv: key_heavyhitter_map)
		{
			for (int j=0; j<4; j++)
				dataset(j,i) = kv.first & (255 << (8*j));
			labels(i) = kv.second;
			i++;
		}
		decisiontree.Train(dataset, labels, 2, 10, 1e-7, 3); // training on full data.
    	  }

	  std::tuple<std::unordered_map<Val, double>, uint64_t, double > per_bf_fpr_hh_fpr = get_opt_bf_sizes(keys, values, nn);
	  
	  hh_bf_exists = (std::get<2>(per_bf_fpr_hh_fpr) > 0);
	  int non_hot_keys = nn;
	  // initialize per-NH BFs: 
	  for(auto &hv : hot_vals_bf_fpr)
	  {
		non_hot_keys -= hot_vals_size[hv.first];
	  	hh_bloomfilters[hv.first] = new BloomFilter<Key>(hot_vals_size[hv.first], hv.second, log(1.0/hv.second) * log(2));
	  }
	  std::cout << "IS HH BF?" << hh_bf_exists << ", non hot keys:" << non_hot_keys << std::endl;
	  if ( flag != 1 && hh_bf_exists)
	  {
		  heavyhitter = new BloomFilter<Key>(std::get<1>(per_bf_fpr_hh_fpr), std::get<2>(per_bf_fpr_hh_fpr), log(1.0/std::get<2>(per_bf_fpr_hh_fpr) ) * log(2));
	  }
	  
	  // initialize LudoHashTable for non-HH keys:
	  cp_hashtable = new ControlPlaneMinimalPerfectCuckoo<Key, Val, VL>(non_hot_keys);

	  // actually populate the BFs, HT:
	  for (int i = 0; i < nn; i++)
	  {
		float dt_prob = (flag > 0) ? dt_classify_prob(keys[i]) : 0.0;
		if (is_hot_val(values[i]))
		{
			hh_bloomfilters[values[i]]->insert(keys[i]);
			if (hh_bf_exists && (flag == 0 || (flag == 2 && (dt_prob < 0.5) ) ) )
				heavyhitter->insert(keys[i]);
		}
		else
			cp_hashtable -> insert(keys[i], values[i]);	
	  }
	/*	------ OLD version:
	  // assign port and next_hop to bloomfilters
	std::pair<std::unordered_map<Val, double>, , double > per_bf_fpr_hh_fpr = get_opt_bf_sizes(keys, values, nn, memory); 
	// initialize BFs for each hot Val:
	int non_hot_keys = nn;
    	for(auto &hv : hot_vals_bf_fpr)
	{
		non_hot_keys -= hot_vals_size[hv.first];
		hh_bloomfilters[hv.first] = new BloomFilter<Key>(hot_vals_size[hv.first], hv.second, log(1.0/hv.second) * log(2));
    	}
	hh_bf_exists = (per_bf_fpr_hh_fpr.second > 0);
	std::cout << "IS THERE A BF AT THE TOP [FOR HH classificn] ? " << hh_bf_exists << std::endl;
	if (flag == 0 && (hh_bf_exists) ){
		heavyhitter = new BloomFilter<Key>(nn-non_hot_keys, per_bf_fpr_hh_fpr.second, log(1.0/per_bf_fpr_hh_fpr.second) * log(2) );
	} 
    	cp_hashtable = new ControlPlaneMinimalPerfectCuckoo<Key, Val, VL>(non_hot_keys); // VL: 32 for nextHop and ? for next port.
   
    // std::unordered_set<Val> unique_vals;
   	int count_ht_vals = 0;
   	std::unordered_map<Key, bool> key_heavyhitter_map;
    	for (int i=0; i<nn; i++){
		actual_keys_vals[(keys[i])] = values[i];
      		if(is_hot_val(values[i])){
			if (!flag && hh_bf_exists)
				heavyhitter->insert(keys[i]);
			key_heavyhitter_map[keys[i]] = 1;
			hh_bloomfilters[values[i]]->insert(keys[i]);  
        	} 
		else {
			key_heavyhitter_map[keys[i]] = 0;
			cp_hashtable -> insert(keys[i], values[i]);
			count_ht_vals++;
        	}
    }
    if (flag == 1){
	arma::mat dataset(4, key_heavyhitter_map.size());
	arma::Row<size_t> labels(key_heavyhitter_map.size());
	std::cout << "SIZE OF MAT:" << size(dataset);
	std::cout << "SIZE OF MAT:" << size(labels);
	int i = 0;
	for (auto &kv: key_heavyhitter_map){
		for (int j=0; j<4; j++){
			dataset(j,i) = kv.first & (255 << (8*j));
		}
		labels(i) = kv.second;
		i++;
	}
	decisiontree.Train(dataset, labels, 2, 10, 1e-7, 3); // training on full data.
    } */
	
	  // CHECKING the accuracy of the model at the top (flag0 : BF, flag1: LM, flag2: LM+BF)
	  // NOTE that this is the same as the forward logic.
	  int wrong_ct = 0;
    	  for (auto &kv: actual_keys_vals)
	  {
	  	// bool predicted = (flag == 0 && hh_bf_exists) ? (heavyhitter->isMember(kv.first)) : (flag == 1) ? (dt_classify(kv.first) ) : true;
	    	wrong_ct += ( (is_hot_key(kv.first) ) != (is_hot_val(kv.second)) );
	  }
    	printf("TOTAL keys: %i, ACCURACY OF HH BF/LM %f \n", actual_keys_vals.size(), (1.0 - ((float)wrong_ct/actual_keys_vals.size()) ) );
     	cout<<"Total values inserted in ht "<<count_ht_vals<<endl;
        
	dp_hashtable = new DataPlaneMinimalPerfectCuckoo<Key, Val, VL>(*cp_hashtable);
    	printf("UNIQUE KEYS: %i Hot: %d, Non-hot:%d", actual_keys_vals.size(), nn-non_hot_keys, non_hot_keys);
  }

  inline float dt_classify_prob(Key key)
  {
	arma::mat ip_prefixes(4, 1);
	arma::Row<size_t> output;
	arma::mat probability(1, 1);
	for (int i=0; i<4; i++){
		ip_prefixes(i, 0) = key & (255 << (8*i));
	}
	decisiontree.Classify(ip_prefixes, output, probability);
	float heavyHitterProb = probability(1);
  	return heavyHitterProb;
  }

  inline bool is_hot_key(Key key)
  {
	  bool predicted = true;
	  if (flag == 0 && hh_bf_exists)
          	predicted = (heavyhitter->isMember(key)); // ans = true if flag=0 & !hh_bf_exists.
          else if (flag == 1)
          	predicted = (dt_classify(key) );
          else if (flag == 2)
          	predicted = (dt_classify_prob(key) >= 0.5 ? 1 : ( (hh_bf_exists) ? (heavyhitter->isMember(key)) : 0) ); // call LM then backup-BF if needed.
  	  return predicted;
  }

  inline size_t dt_classify(Key key)
  {
		arma::mat ip_prefixes(4, 1);
		arma::Row<size_t> output;
		arma::mat probability(1, 1);
		for (int i=0; i<4; i++){
			ip_prefixes(i, 0) = key & (255 << (8*i));
		}
		decisiontree.Classify(ip_prefixes, output, probability);
		float heavyHitterProb = probability(1);
		/*if (!output(0)) {
			probability.raw_print("Probability: ");
			output.raw_print("Output: ");
		}*/
		return output(0);
  }

  inline Val forward(const Key &key){
     timeval start, end;
    // gettimeofday(&start, nullptr);
    struct timespec start_RT, end_RT;
	clock_gettime(CLOCK_REALTIME, &start_RT);

    Val next_hop;
    if (!is_hot_key(key))
    {
	    // lookup in HT
	    cp_hashtable -> lookUp(key, next_hop);
	    clock_gettime(CLOCK_REALTIME, &end_RT);
	    ht_lookup_time.push_back(diff_ns_RT(end_RT, start_RT));
	    return next_hop;
    }
    
    clock_gettime(CLOCK_REALTIME, &end_RT);
    bf_lookup1_time.push_back(diff_ns_RT(end_RT, start_RT));
    
    // need to lookup in per NH BFs now:
    
    std::vector<double> ilookup_times;
    int match_Ct = 0;
    for (auto &hv: hh_bloomfilters)
    {
	struct timespec start_RTi, end_RTi;
	clock_gettime(CLOCK_REALTIME, &start_RTi);
        if (hh_bloomfilters[hv.first]->isMember(key)){
        	next_hop = hv.first; // hh_bloomfilters[i]->next_hop;
      		match_Ct += 1;
        }
        clock_gettime(CLOCK_REALTIME, &end_RTi);
        ilookup_times.push_back(diff_ns_RT(end_RTi, start_RTi));
   }
   bf_ilookup_times.push_back(ilookup_times);
   bf_match_cts.push_back(match_Ct);
   struct timespec start_RTh, end_RTh;
   clock_gettime(CLOCK_REALTIME, &start_RTh);

   // doing sequential lookup in HT: will help improve accuracy.
   if (match_Ct == 0){
        cp_hashtable->lookUp(key, next_hop);
        clock_gettime(CLOCK_REALTIME, &end_RTh);
      
        bf_lookup1_time[bf_lookup1_time.size()-1] += diff_ns_RT(end_RTh, start_RTh);
      
	double tt_ht_lookup = diff_ns_RT(end_RTh, start_RTh);
	// cout<<"Not found in BFs, looking up in HT: "<<key<<" lookup_times "<<ilookup_times[i]<<" new lookup time "<<tt_ht_lookup<<" new sum ";  
   }
   /*
    if (match_Ct > 1)
	    std::cout << "MULTIPLE BF matches: " << match_Ct << " for key" << key << std::endl;
    else if (match_Ct == 0)
	    std::cout << "DAMN. 0 matches for " << key << ", is HH?" << is_hot_key(actual_keys_vals[key]) << std::endl;
    else if (next_hop != actual_keys_vals[key] )
	    std::cout << "DAMN. 1match AND WRONGG" << next_hop << ", actual:" << actual_keys_vals[key] << std::endl;
    */
    return next_hop;
  }
};
