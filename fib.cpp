//
#include "common.h"
#include "cstlib"
#include "BloomFilter/bloom_filter.h"
// include libraries for other DS
#include "MinimalPerfectCuckoo/minimal_perfect_cuckoo.h"

typedef uint32_t Key;
class BloomFilterPerPort {
  BloomFilter<Key> bloomfilter;
  public:
    int port;
    uint32_t next_hop;
    BloomFilterPerPort(int _port=0, uint32_t _next_hop=0){
      port = port;
      next_hop = _next_hop;
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

    void set_next_hop(string _next_hop){
      next_hop = _next_hop;
    }

}

class FIB {
  int num_bloom_filters;
  BloomFilter<Key> heavyhitter;
  BloomFilterPerPort** bloomfilters;
  ControlPlaneMinimalPerfectCuckoo<Key, uint32_t, 8>* cp_hashtable;
  DataPlaneMinimalPerfectCuckoo<Key, uint32_t, 8>* dp_hashtable;
public:
  FIB(int num_bf){
    num_bloom_filters = num_bf;
    bloomfilters = new (BloomFilterPerPort*)[num_bf];
    for(int i=0; i<num_bf; ++i){
      bloomfilters[i] = new BloomFilterPerPort();
    }
  }

  void populate(vector<Key> &keys, vector<Val> &values, uint64_t nn){
    // assign port and next_hop to bloomfilters
    cp_hastable = new ControlPlaneMinimalPerfectCuckoo<Key, uint32_t, 8>(nn);
    // need to initialize BF with opt fpr, #elements.
    for (int i=0; i<keys.size(); i++){
      if(is_hot_key(keys[i])){
        heavyhitter.insert(keys[i]);
        for (int j=0; j<num_bloom_filter; j++){
          if (bloomfilters[j].next_hop == values[i]){
            bloomfilters[j].insert(keys[i]);
          }
        }
      } else {
        cp_hashtable -> insert(keys[i], values[i]);
      }
    }
    dp_hastable = new DataPlaneMinimalPerfectCuckoo<Key, uint32_t, 8>(*cp_hashtable);
  }

  uint32_t forward(const Key &key){
    uint32_t next_hop;
    int out_port;
    if (!heavyhitter.isMember(key)){
      // lookup in hashtable
      dp_hashtable -> lookUp(key, next_hop);
      return next_hop;
    }
    for (int i=0; i<num_bloom_filters; i++){
      if (bloomfilters[i]->lookup(key)){
        next_hop = bloomfilters[i]->next_hop;
        out_port = bloomfilters[i]->port;
      }
    }
    return next_hop;
  }
}
