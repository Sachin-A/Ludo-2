#include "common.h"
#include "cstdlib"
#include "SetSep/setsep.h"
#include "BloomFilter/bloom_flitable.h"
#include "CuckooPresized/cuckoo_map.h"
#include "CuckooPresized/cuckoo_ht.h"
#include "CuckooPresized/cuckoo_filter_control_plane.h"
#include "CuckooPresized/cuckoo_filtable.h"
#include "Othello/data_plane_othello.h"
#include "MinimalPerfectCuckoo/minimal_perfect_cuckoo.h"
#include "DPH/dph.h"
#include "fib.h"
#include <utility> 
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/json.h>
#include <math.h>


std::string fname_global = "";
int version = 12;
int cores = min(20U, std::thread::hardware_concurrency());

template<class K>
struct OthelloChange {
  int8_t type;
  vector<uint32_t> cc;
  uint64_t xorTemplate;
  int marks[2];
};

typedef uint32_t Key;
const uint32_t lookupCnt = 1<<14; // 1 << 25;

template<int VL, class Val>
void testSetSep(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  try {
    Clocker construction("SetSep construction");
    SetSep<Key, Val, VL> cp(nn, true, keys, values);
    Counter::count("overflow", cp.overflow.size());
    cout << "overflow: " << cp.overflow.size() << endl;
    construction.stop();
    
    if (VL == 4 || VL == 20) {
      for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
        thread threads[threadCnt];
        uint32_t start[threadCnt];

        for (int i = 0; i < threadCnt; ++i) {
          start[i] = i / threadCnt * zipfianKeys.size();
        }

        for (Distribution distribution: {uniform, exponential}) {
          ostringstream oss;
          oss << "SetSep parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
              << (distribution == exponential ? "Zipfian" : "uniform");
          Clocker plookup(oss.str());

          for (int i = 0; i < threadCnt; ++i) {
            threads[i] = std::thread(
              [](const SetSep<Key, Val, VL> *dp, uint32_t start, const vector<Key> *zipfianKeys,
                 uint32_t lookupCnt) {
                int stupid = 0;

                int ii = start;
                do {
                  const Key &k = zipfianKeys->at(ii);
                  Val val;
                  dp->lookUp(k, val);
                  stupid += val;

                  if (ii == lookupCnt - 1) ii = -1;
                  ++ii;
                } while (ii != start);
                printf("%d\b", stupid & 7);
              }, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
          }

          for (int i = 0; i < threadCnt; ++i) {
            threads[i].join();
          }
        }
      }
    }

//    if (VL == 20 && nn <= 1048576) {
      uint32_t updateCnt = 5000;
      Clocker c("SetSep apply " + to_string(updateCnt) + " updates");
      uint32_t halfSize = (keys.size() / 2);
      for (int updates = 0; updates < updateCnt; ++updates) {
        if (updates % 3 == 0) { // delete
          cp.remove(keys[rand() % halfSize]);
        } else if (updates % 3 == 1) { // modify
          cp.updateMapping(keys[rand() % halfSize], rand());
        } else {// insert
          cp.insert(rand(), rand());
        }
      }
//    }
  } catch (exception &e) {
    cout << e.what() << endl;
  }
}

template<int VL, class Val>
void testOthello(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Othello CP construction");
  ControlPlaneOthello<Key, Val, VL> cp(nn, true, keys, values);
  construction.stop();
  
  Clocker exp("Othello export");
  DataPlaneOthello<Key, Val, VL> dp(cp);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Othello parallel lookup " << threadCnt << " threads " << lookupCnt
            << " keys " << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread(
            [](const DataPlaneOthello<Key, Val, VL> *dp, uint32_t start, const vector<Key> *zipfianKeys,
               uint32_t lookupCnt) {
              int stupid = 0;
              
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              printf("%d\b", stupid & 7);
            }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
  
  if (VL == 20) {
    for (int i = nn * 2 / 3; i < nn; ++i) {
      cp.remove(keys[i]);
    }
    
    DataPlaneOthello<Key, Val, VL> dp(cp);
    
    vector<OthelloChange<Key>> oChanges;
    
    Clocker gen("Othello generate " + to_string(lookupCnt) + " updates");
// prepare many updates. modification : insertion : deletion = 1:1:1
    for (int i = 0; i < lookupCnt; ++i) {
      if (i % 3 == 0) {  // delete
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          if (cp.isMember(k)) break;
        }
        cp.remove(k);
        
        uint32_t ha, hb;
        cp.getIndices(k, ha, hb);
        
        OthelloChange<Key> oc;
        oc.type = 'D';
        oc.marks[0] = cp.isEmpty(ha) ? ha : -1;
        oc.marks[1] = cp.isEmpty(hb) ? hb : -1;
        oChanges.push_back(oc);
        Counter::count_("Othello deletion update message length",
                        1 + (oc.marks[0] == -1 ? 0 : 4) + (oc.marks[1] == -1 ? 0 : 4));
        Counter::count_("Othello deletion update message");
      } else if (i % 3 == 1) { // modify
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          if (cp.isMember(k)) break;
        }
        Val v = rand();
        
        OthelloChange<Key> oc;
        oc.type = 'M';
        int64_t tmp = cp.updateMapping(k, v);
        uint32_t ha, hb;
        cp.getIndices(k, ha, hb);
        oc.xorTemplate = uint64_t(tmp < 0 ? -tmp : tmp);
        oc.cc = cp.getHalfTree(k, tmp > 0, false);
        oChanges.push_back(oc);
        
        Counter::count_("Othello modification update message length",
                        1 + oc.cc.size() * 4 + (VL + 7) / 8);
        Counter::count_("Othello modification update message");
      } else { // insert
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          if (!cp.isMember(k)) break;
        }
        Val v = rand();
        
        OthelloChange<Key> oc;
        oc.type = 'A';
        uint32_t ha, hb;
        cp.getIndices(k, ha, hb);
        
        oc.marks[0] = cp.isEmpty(ha) ? ha : -1;
        oc.marks[1] = cp.isEmpty(hb) ? hb : -1;
        int64_t tmp = cp.insert(k, v);
        
        oc.xorTemplate = uint64_t(tmp < 0 ? -tmp : tmp);
        oc.cc = cp.getHalfTree(k, tmp > 0, false);
        oChanges.push_back(oc);
        
        Counter::count_("Othello addition update message length",
                        1 + oc.cc.size() * 4 + (VL + 7) / 8 +
                        (oc.marks[0] == -1 ? 0 : 4) + (oc.marks[1] == -1 ? 0 : 4));
        Counter::count_("Othello addition update message");
      }
    }
    gen.stop();
    
    {
      DataPlaneOthello<Key, Val, VL> dp(cp);
      
      Clocker c("Othello apply " + to_string(lookupCnt) + " updates");
      
      for (OthelloChange<Key> oc: oChanges)
        if (oc.type == 'D') {
          if (oc.marks[0] >= 0) {
            dp.setEmpty(oc.marks[0]);
          }
          
          if (oc.marks[1] >= 0) {
            dp.setEmpty(oc.marks[1]);
          }
        } else if (oc.type == 'M') {
          dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
        } else {
          dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
          
          if (oc.marks[0] >= 0) {
            dp.setTaken(oc.marks[0]);
          }
          
          if (oc.marks[1] >= 0) {
            dp.setTaken(oc.marks[1]);
          }
        }
    }
    
    for (Distribution distribution: {uniform, exponential}) {
      for (int updateps = 25; updateps <= 1600; updateps *= 4) {
        for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
          thread threads[threadCnt];
          uint32_t start[threadCnt];
          bool running[threadCnt];
          
          for (int i = 0; i < threadCnt; ++i) {
            start[i] = i / threadCnt * zipfianKeys.size();
            running[i] = true;
          }
          
          ostringstream oss;
          oss << "Othello parallel lookup " << threadCnt << " threads " << lookupCnt
              << " keys " << (distribution == exponential ? "Zipfian" : "uniform") << ", under " << updateps
              << " updates per second";
          Clocker plookup(oss.str());
          
          timeval startTime;
          gettimeofday(&startTime, nullptr);
          uint updates = 0;
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i] = std::thread([](const DataPlaneOthello<Key, Val, VL> *dp,
                                        uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt,
                                        bool *running) {
              int stupid = 0;
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              *running = false;
              printf("%d\b", stupid & 7);
            }, &dp, start[i], &zipfianKeys, lookupCnt, &running[i]);
          }
          
          uint32_t failCnt = 0;
          while (true) {
            bool active = false;
            for (int i = 0; i < threadCnt; ++i) {
              active |= running[i];
            }
            if (!active || updates >= oChanges.size()) break;
            
            OthelloChange<Key> oc = oChanges[updates];
            if (oc.type == 'D') {
              if (oc.marks[0] >= 0) {
                dp.setEmpty(oc.marks[0]);
              }
              
              if (oc.marks[1] >= 0) {
                dp.setEmpty(oc.marks[1]);
              }
            } else if (oc.type == 'M') {
              dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
            } else {
              dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
              
              if (oc.marks[0] >= 0) {
                dp.setTaken(oc.marks[0]);
              }
              
              if (oc.marks[1] >= 0) {
                dp.setTaken(oc.marks[1]);
              }
            }
            
            updates++;
            timeval now;
            gettimeofday(&now, nullptr);
            
            uint64_t diff = diff_us(now, startTime);
            uint64_t shouldDiff = double(updates) / updateps * 1E6;
            if (shouldDiff > diff) {
              this_thread::sleep_for(std::chrono::microseconds(shouldDiff - diff));
            } else {
              if (++failCnt >= 100) {
                cout << ("cannot meet time requirement") << endl;
                break;
              }
            }
          }
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i].join();
          }
          
          if (failCnt >= 100) break;
        }
      }
    }
  }
}

void print_arr(std::vector<double>& v, std::string s)
{
        std::ofstream logs;
	logs.open("/home/ubuntu/Ludo/FIB_logs/Ludo_FIB"+fname_global+".txt", std::ios::app);

	std::sort(v.begin(), v.end());
	uint32_t len = v.size();
	//cout<<"&&&&& "<<s<<endl;
	if (len > 0){
		printf("FOR %s Arr len: %i, min: %f, 25th: %f, avg: %f, 50th: %f, 75th: %f, 90th: %f, 95th: %f, max: %f", s.c_str(), v.size(), v[0], v[len/4], std::accumulate(v.begin(), v.end(),0.0)/(float)len, v[len/2], v[3*len/4], v[9*len/10], v[95*len/100], v[len-1]);
		if (s.find("Ludo") != string::npos)
			logs<<s.substr(s.size()-7)<<", "<<std::accumulate(v.begin(), v.end(),0.0)/(float)len<<endl;
	}
	else
		printf("FOR %s Arr len:  IS ZERO!!!", s.c_str());
	logs.close();
}


template<int VL, class Val>
void testLudo(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  // Clocker is being used for logging time: output file has info on when a task starts, ends.
	Clocker cp_dp_constr("MPC CP+DP construction");
	Clocker construction("MPC construction");
  
  Clocker cpBuild("CP build");
  ControlPlaneMinimalPerfectCuckoo<Key, Val, VL> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
 

  Clocker cpPrepare("CP prepare for DP");
  cp.prepareToExport();
  cpPrepare.stop();
  construction.stop();
  
  Clocker exp("MPC export");
  DataPlaneMinimalPerfectCuckoo<Key, Val, VL> dp(cp);
  exp.stop();
  cp_dp_constr.stop();
  
  std::cout << "~~~~~In testLudo: VL:" << VL << ", #cores:" << cores << ", lookUp len: " << keys.size() << ", keyValSz: " << nn << ", vals Len:" << values.size() << std::endl;
  
  if (VL == 4 || VL == 20 || VL == 32) {
  Clocker ludoClock("actual_lookup_Ludo_start");
	  std::cout << "~~~~~In testLudo: VL:" << VL << ", #cores:" << cores << std::endl;
    cores = 1; // LIMITING ||alism for comparison with ASA.
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "MPC parallel lookup " << threadCnt << " threads " << keys.size() << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread([](const DataPlaneMinimalPerfectCuckoo<Key, Val, VL> *dp, const ControlPlaneMinimalPerfectCuckoo<Key, Val, VL> *cp, uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt, Distribution distr, vector<Val> vals, uint64_t nn1) {
        // Clocker plookup(oss.str());
           Clocker plookup(std::string(distr == exponential ? "MPC_Zipfian" : "MPC_uniform"));
	std::vector<double> per_lookup_times;
			 
	     int stupid = 0;
	    int wrong_Ct =0;
	    std::cout << "I AM A LUDO-LOOKUP thread. Iterating over keys, start: " << start << std::endl;
            int ii = start;
            do {
              const Key &k = zipfianKeys->at(ii);
              struct timespec start_rt, end_rt;
	      clock_gettime(CLOCK_REALTIME, &start_rt);
	      Val val;
              // dp->lookUp(k, val);
              
	        cp->lookUp(k, val);
		clock_gettime(CLOCK_REALTIME, &end_rt);
		per_lookup_times.push_back(diff_ns_RT(end_rt,start_rt));	

	      stupid += val;
	       if ( (distr != exponential) && (val != vals[ii%nn1]) )
	      	wrong_Ct += 1; 
	      	// std::cout << "LUDO made an error! val:" << val << ", actual: " << values[ii%nn] << std::endl;
              if (ii == lookupCnt - 1) ii = -1;
              ++ii;
            } while (ii != start);
            printf("Stupid:%d\b", stupid & 7);
		plookup.stop();
		print_arr(per_lookup_times, std::string("LudoLookup: Distrib: ") + std::string(distr == exponential ? "Zipfian" : "uniform"));
	    	
		std::cout << "LUDO: done : wrong ct:" << wrong_Ct << " out of " << lookupCnt << std::endl;
          }, &dp, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, keys.size(), distribution, values, nn);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
    ludoClock.stop();
  }
  
  if (VL == 20) {
    for (int i = nn * 2 / 3; i < nn; ++i) {
      cp.remove(keys[i]);
    }
    
    DataPlaneMinimalPerfectCuckoo<Key, Val, VL> dp(cp);
    
    vector<pair<vector<MPC_PathEntry>, Val>> insertPaths;
    vector<pair<uint32_t, Val>> modifications;
    
    Clocker gen("MPC generate " + to_string(lookupCnt) + " updates");
    // prepare many updates. modification : insertion : deletion = 1:1:1
    for (int i = 0; i < lookupCnt; ++i) {
      if (i % 3 == 0) {  // delete
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          Val tmp;
          if (cp.lookUp(k, tmp)) {
            break;
          }
        }
        cp.remove(k);
        
        Counter::count_("MPC deletion update message length", 0);
        Counter::count_("MPC deletion update message");
      } else if (i % 3 == 1) { // modify
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          
          Val tmp;
          if (cp.lookUp(k, tmp)) {
            break;
          }
        }
        Val v = rand();
        
        cp.updateMapping(k, v);
        pair<uint32_t, uint32_t> tmp = cp.locate(k);
        modifications.emplace_back((tmp.first << 2) + tmp.second, v);
        
        Counter::count_("MPC modification update message length", 1 + 4 + (VL + 7) / 8);
        Counter::count_("MPC modification update message");
      } else { // insert
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          
          Val tmp;
          if (!cp.lookUp(k, tmp)) {
            break;
          }
        }
        Val v = rand();
        vector<MPC_PathEntry> path;
        cp.insert(k, v, &path);
        insertPaths.emplace_back(path, v);
        
        uint32_t s = 0;
        for (auto e: path) {
          s += e.locatorCC.size() * 4;
        }
        
        Counter::count_("MPC addition update message length", 1 + (VL + 7) / 8 + path.size() * (4 + 1 + 1) + s);
        Counter::count_("MPC addition update message");
      }
    }
    gen.stop();
    
    {
      DataPlaneMinimalPerfectCuckoo<Key, Val, VL> dp(cp);
      
      Clocker c("MPC apply " + to_string(lookupCnt) + " updates");
      
      for (int updates = 0; updates < lookupCnt; ++updates) {
        if (updates % 3 == 0) { // delete
          // empty
        } else if (updates % 3 == 1) { // modify
          pair<uint32_t, Val> tmp = modifications.at(updates / 3);
          dp.applyUpdate(tmp.first, tmp.second);
        } else {// insert
          pair<vector<MPC_PathEntry>, Val> tmp = insertPaths.at(updates / 3);
          dp.applyInsert(tmp.first, tmp.second);
        }
      }
    }
    
    for (Distribution distribution: {uniform, exponential})
      for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
        for (int updateps = 25; updateps <= 1600; updateps *= 4) {
          thread threads[threadCnt];
          uint32_t start[threadCnt];
          bool running[threadCnt];
          
          for (int i = 0; i < threadCnt; ++i) {
            start[i] = i / threadCnt * zipfianKeys.size();
            running[i] = true;
          }
          
          ostringstream oss;
          oss << "MPC parallel lookup " << threadCnt << " threads " << lookupCnt
              << " keys " << (distribution == exponential ? "Zipfian" : "uniform") << ", under " << updateps
              << " updates per second";
          Clocker plookup(oss.str());
          
          timeval startTime;
          gettimeofday(&startTime, nullptr);
          uint updates = 0;
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i] = std::thread([](const DataPlaneMinimalPerfectCuckoo<Key, Val, VL> *dp,
                                        uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt,
                                        bool *running) {
              int stupid = 0;
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              *running = false;
              printf("%d\b", stupid & 7);
            }, &dp, start[i], &zipfianKeys, lookupCnt, running + i);
          }
          
          uint32_t failCnt = 0;
          while (true) {
            bool active = false;
            for (int i = 0; i < threadCnt; ++i) {
              active |= running[i];
            }
            if (!active || modifications.empty() || insertPaths.empty()) break;
            
            if (updates % 3 == 0) { // delete
              // empty
            } else if (updates % 3 == 1) { // modify
              pair<uint32_t, Val> tmp = modifications.at(updates / 3);
              dp.applyUpdate(tmp.first, tmp.second);
            } else {// insert
              pair<vector<MPC_PathEntry>, Val> tmp = insertPaths.at(updates / 3);
              dp.applyInsert(tmp.first, tmp.second);
            }
            
            updates++;
            timeval now;
            gettimeofday(&now, nullptr);
            
            uint64_t diff = diff_us(now, startTime);
            uint64_t shouldDiff = double(updates) / updateps * 1E6;
            if (shouldDiff > diff) {
              this_thread::sleep_for(std::chrono::microseconds(shouldDiff - diff));
            } else {
              if (++failCnt >= 100) {
                cout << ("cannot meet time requirement") << endl;
                break;
              }
            }
          }
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i].join();
          }
          
          if (failCnt >= 100) break;
        }
      }
  }
}

template<int VL, class Val>
void testDPH(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("DPH construction");
  DPH<Val, sizeof(Key) * 8> cp;
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  Counter::count_("DPH memory", cp.memInBytes());
  construction.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "DPH parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread([](const DPH<Val, sizeof(Key) * 8> *cp,
                                      uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt) {
            int stupid = 0;
            int ii = start;
            do {
              const Key &k = zipfianKeys->at(ii);
              Val val;
              cp->lookUp(k, val);
              stupid += val;
              
              if (ii == lookupCnt - 1) ii = -1;
              ++ii;
            } while (ii != start);
            printf("%d\b", stupid & 7);
          }, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
}

void analyse_ASA_timings(std::vector<double>& ht_times, std::vector<double>& bf_times1, std::vector<std::vector<double>>& bf_itimes, std::ofstream& logs)
{
	std::cout << "ANALYSING TIMINGs: \n";
	  printf("#Lookups in HT: %i, #Lookups in BFs: %i", ht_times.size(), bf_times1.size());
	// analyse timings:
	  std::vector<double> total_lookup_times (ht_times.begin(), ht_times.end());
	  print_arr(ht_times, "HT wallTime (ns) :");
	  for (int i = 0 ; i < bf_times1.size(); i++)
	  {
		  double ti = bf_times1[i];
		  ti += *std::max_element(bf_itimes[i].begin(), bf_itimes[i].end());
		  if (i % 2900 == 5)
			  std::cout << "i:" << i << ", lookup1 time: " << bf_times1[i] << ", adding max_bf_lookup:" << ti  << std::endl;
		  total_lookup_times.push_back(ti);
	  }
	  double total_time = std::accumulate(total_lookup_times.begin(), total_lookup_times.end(), 0.0);
	  std::cout << "TOTAL time (ns) for ASA:" << total_time << " avg per query: " << total_time/total_lookup_times.size() << std::endl;
	  print_arr(total_lookup_times, "Total lookuptime: "); 
	  logs << total_time/total_lookup_times.size() << ", ";
	// clear all arrs:
	ht_times.clear();
	bf_times1.clear();
	bf_itimes.clear();
}

template<int VL, class Val>
void testFIB(vector<Key> &keys, vector<Val> &values, uint64_t nn, std::string fname, int fib_model_flag, int fib_mem_flag, vector<Key> &zipfianKeys) {
  Fib<VL,Val> fib (fib_model_flag, fib_mem_flag);
  cout<<"Called testFIB, starting population of fib>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;;
  uint64_t memory = 1024*4*8;
  printf("CALLING populate with nn %i, lookupKeySz: %i, model flag: %i, mem flag: %i", nn, keys.size(), fib_model_flag, fib_mem_flag);
  Clocker cp_dp_construction("ASA Construction");
  fib.populate(keys, values, nn); // memory
  cp_dp_construction.stop();
  cout<<"Called testFIB, done population of fib>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;
  
  std::ofstream logs;
  logs.open("/home/ubuntu/Ludo/FIB_logs/ASA_FIB"+fname+".txt", std::ios::app);
  for (Distribution distribution: {uniform, exponential})
  {
	  Clocker total_fwd_time("ASA FIB: Total Lookup WORKLOAD:" + std::string(distribution == exponential ? "Zipfian" : "uniform") );
  	  int wrong_count = 0;
	  std::vector<Key>& lookup_keys = (distribution == exponential ? zipfianKeys : keys);
	  for (int i = 0; i < lookup_keys.size(); i++)
	  {
		  if (i<5)
			  cout << "k,v:" << lookup_keys[i] << "," << fib.actual_keys_vals[lookup_keys[i]] << std::endl;
		  Val v = fib.forward(lookup_keys[i]);
		  if (v != fib.actual_keys_vals[lookup_keys[i]])
			  wrong_count++;
	  }
	  total_fwd_time.stop();
  
  	  cout<< "DONE with lookups! Workload-Distr:" << (distribution == exponential ? "Zipfian" : "uniform") << "Wrong Count: "<<wrong_count<<" total lookUp keys: "<<lookup_keys.size() <<", Accuracy score: " << 1.0 - (wrong_count/(float)lookup_keys.size()) <<endl;

	logs << fib_model_flag << ", " << fib_mem_flag << ", " << std::string(distribution == exponential ? "Zipfian" : "uniform") << ", " << (float)(1.0 - (wrong_count/(float)lookup_keys.size())) << ", ";
          analyse_ASA_timings(fib.ht_lookup_time, fib.bf_lookup1_time, fib.bf_ilookup_times, logs);
        logs <<fib.total_memory<<endl;
  }
  logs.close(); 
}


template<int VL, class Val>
void testBloom(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Bloom construction");
  ControlPlaneBloomFiltable<Key, Val> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  construction.stop();
  
  Clocker exp("Bloom export");
  DataPlaneBloomFiltable<Key, Val> dp(16, cp);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Bloom parallel lookup " << threadCnt << " threads " << lookupCnt
            << " keys " << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread(
            [](const DataPlaneBloomFiltable<Key, Val> *dp, uint32_t start, const vector<Key> *zipfianKeys,
               uint32_t lookupCnt) {
              int stupid = 0;
              
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              printf("%d\b", stupid & 7);
            }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
}

template<int VL, class Val>
void testPKCuckoo(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Partial key Cuckoo construction");
  Clocker cpBuild("CP build");
  ControlPlaneCuckooFiltable<Key, Val, uint8_t, 0> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  construction.stop();
  
  Clocker exp("Partial key Cuckoo export");
  DataPlaneCuckooFiltable<Key, Val, uint8_t, 0> dp(*cp.level1, *cp.level2);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Partial key Cuckoo parallel lookup " << threadCnt << " threads "
            << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread([](const DataPlaneCuckooFiltable<Key, Val, uint8_t, 0> *dp, uint32_t start,
                                      const vector<Key> *zipfianKeys, uint32_t lookupCnt) {
            int stupid = 0;
            
            int ii = start;
            do {
              const Key &k = zipfianKeys->at(ii);
              Val val;
              dp->lookUp(k, val);
              stupid += val;
              
              if (ii == lookupCnt - 1) ii = -1;
              ++ii;
            } while (ii != start);
            printf("%d\b", stupid & 7);
          }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
}

template<int VL, class Val>
void testCuckoo(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Cuckoo construction");
  Clocker cpBuild("CP build");
  ControlPlaneCuckooMap<Key, Val, uint8_t> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  construction.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Cuckoo parallel lookup " << threadCnt << " threads "
            << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread(
            [](const ControlPlaneCuckooMap<Key, Val, uint8_t> *dp, uint32_t start, const vector<Key> *zipfianKeys,
               uint32_t lookupCnt) {
              int stupid = 0;
              
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              printf("%d\b", stupid & 7);
            }, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
    
    if (VL == 20 && nn < 1048576) {
      Clocker c("Cuckoo apply " + to_string(lookupCnt) + " updates");
      uint32_t halfSize = (keys.size() / 2);
      for (int updates = 0; updates < lookupCnt; ++updates) {
        if (updates % 3 == 0) { // delete
          cp.remove(keys[rand() % halfSize]);
        } else if (updates % 3 == 1) { // modify
          cp.updateMapping(keys[rand() % halfSize], rand());
        } else {// insert
          cp.insert(rand(), rand());
        }
      }
    }
  }
}

std::pair< std::vector<Key>, std::vector<uint32_t> > read_fib_json(std::string fpath)
{
	
	std::ifstream file(fpath);
        Json::Value rawJson;
        Json::Reader reader;

	reader.parse(file, rawJson);
	
	int totalEntries = rawJson.size();

	std::vector<Key> keys;
	std::vector<uint32_t> values;

	std::cout << "read " << totalEntries << " entries \n";

	int copy_ct = 0;
	std::unordered_set<Key> unique_keys;
	for (int entry = 0; entry < totalEntries; ++entry)
	{
		Key key  = ( (uint32_t) rawJson[entry]["ip1"].asUInt() * pow(2,24)) + ((uint32_t) rawJson[entry]["ip2"].asUInt() * pow(2,16)) + ((uint32_t) rawJson[entry]["ip3"].asUInt() * pow(2,8)) + ((uint32_t) rawJson[entry]["ip4"].asUInt() * 1);
		
		if (unique_keys.find(key) != unique_keys.end())
			copy_ct += 1;
			// std::cout << "WEIRD!!! THis key is already there!!! \n";
		else
		{
			unique_keys.insert(key);
			keys.push_back(( (uint32_t) rawJson[entry]["ip1"].asUInt() * pow(2,24)) + ((uint32_t) rawJson[entry]["ip2"].asUInt() * pow(2,16)) + ((uint32_t) rawJson[entry]["ip3"].asUInt() * pow(2,8)) + ((uint32_t) rawJson[entry]["ip4"].asUInt() * 1));
			values.push_back((uint32_t) rawJson[entry]["nextHopInt"].asUInt());	
		}
		//if (entry%500 == 7)
		// printf("%i %i %i %i : %i", rawJson[entry]["ip1"], rawJson[entry]["ip2"], rawJson[entry]["ip3"], rawJson[entry]["ip4"], rawJson[entry]["nextHopInt"]);
	/*	
	       	std::cout << rawJson[entry]["ip1"] << ", " << rawJson[entry]["ip1"].asUInt() << ", " << (uint32_t) rawJson[entry]["ip1"].asUInt() * pow(2,24)<< ", ip2:" << rawJson[entry]["ip2"] << ", " << rawJson[entry]["ip3"]<< ", " << rawJson[entry]["ip4"]<< ": " << rawJson[entry]["nextHopInt"] << "\n"; 
		std::cout << rawJson[entry]["ip4"] << ", " << rawJson[entry]["ip4"].asUInt() << ", " << (uint32_t) rawJson[entry]["ip4"].asUInt() * 1 << " | ";
		std::cout << "FINAL KEY: " << keys[entry] << std::endl; 
	*/
	 }
	 std::cout << "Unique keys:" << unique_keys.size() << std::endl;
	
	/* RANDOM FIB data:
	   std::vector<Key> keys (10000);
	std::vector<uint32_t> values (10000);
	for (int i = 0; i < 10000; i++)
	{
		keys[i] = i+5;
		values[i] = i%200; // i%77 + i%90;
	}
	*/
	return std::make_pair(keys, values);
}


template<int VL, class Val>
void test(std::string fname = "", int fib_model_flag=0, int fib_mem_flag = 0) {
  for (int repeat = 7; repeat < 8; repeat += 1)
  {
	  std::cout << "starting repeat = " << repeat << std::endl;
	 for (uint64_t nn = 32768; nn <= 32769; nn *= 2) // (1U << 30)
      try {
        ostringstream oss;
        
        oss << "value length " << VL << ", key set size " << nn << ", repeat#" << repeat
            << ", version#" << version;
       printf("value length? %i key set sz: %i repeat: %i, version: %i \n", VL, nn, repeat, version); 
        string logName = "../dist/logs/" + oss.str() + ".log";
        
        ifstream testLog(logName);
        string lastLine, tmp;
        
        while (getline(testLog, tmp)) {
          if (tmp.size() && tmp[0] == '|') lastLine = tmp;
        }
        testLog.close();
        
        if (lastLine.size() >= 3 && lastLine[2] == '-')
	{
		printf("Last Line > 3 and [2] = -, %s continuing [thi smeans it wont run test] \n", lastLine.c_str());
		// continue;
	}	
        
        TeeOstream tos(logName);
        Clocker clocker(oss.str(), &tos);
       
       // LFSRGen : seed, max, skip ct inputs.	
        LFSRGen<Key> keyGen(0x1234567801234567ULL, max((uint64_t) lookupCnt, nn), 0);
        LFSRGen<Val> valueGen(0x1234567887654321ULL, 1000, 0);
        
	// keys is the array of keys to be looked up: represents workload.
	// fill the first nn vals by LFSR KeyGen, repeat the rest.
        // vector<Key> keys(max((uint64_t) lookupCnt, nn));
        // vector<Val> values(nn);
        
        /*
	for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          Val v;
          valueGen.gen(&v);
	  // std::cout << "key:" << keys[i] << ", valuegen generated: " << v << " || ";
          values[i] = v & (uint(-1) >> (32 - VL)); // emptying out 32-VL bits
	  // std::cout << "val[" << i << "]: " << values[i] << std::endl;
	}
	*/

	std::pair< std::vector<Key>, std::vector<uint32_t> > keys_vals = read_fib_json("../../cisco_data/" + fname + ".json");
	uint64_t new_nn = keys_vals.first.size();
	uint64_t new_lookupCnt = 8 * new_nn;

	uint64_t ludo_space = (3.76 + 1.05*VL) * new_nn;
	std::cout << "FOR router" << fname << " ludo space:" << ludo_space << std::endl;

	vector<Key> keys(max((uint64_t) new_lookupCnt, new_nn));
	vector<Val> values( keys_vals.second.begin(), keys_vals.second.end() );
	
	// Uniformly distributed keys: [each key occurs lookupCnt/nn times]
        for (uint64_t i = 0; i < max((uint64_t) new_lookupCnt, new_nn); i++) {
          	keys[i] = keys_vals.first[i%new_nn];
		// keys[i] = keys[i % nn];
        }
        
        vector<Key> zipfianKeys;
        zipfianKeys.reserve(new_lookupCnt);
        
          InputBase::distribution = exponential;
          InputBase::bound = new_nn;
          uint seed = Hasher32<string>()(logName);
          InputBase::setSeed(seed);
	  
	  vector<uint32_t> per_ind_count (new_nn);

	  // fill zipfian keys vec [of keys to be looked up] based on zipfian.
          for (int i = 0; i < new_lookupCnt; ++i) {
          	uint32_t idx = InputBase::rand();
          	// std::cout << "seed: "<< seed << ": pushing idx" << idx << ", ";
	  	zipfianKeys.push_back(keys_vals.first[idx]);
		per_ind_count[idx] += 1;
          }
	  // for (int i = 0; i < new_nn; i++)
		  // std::cout << "Ct for i=" << i << " is " << per_ind_count[i] << "\n";
  
        // if (nn < 268435456)
          // testSetSep<VL, Val>(keys, values, nn, zipfianKeys);
	  fname_global = fname;
  	  testFIB<VL, Val>(keys,values, new_nn, fname, fib_model_flag, fib_mem_flag, zipfianKeys);
//        testCuckoo<VL, Val>(keys, values, nn, zipfianKeys);
//        testOthello<VL, Val>(keys, values, nn, zipfianKeys);
        
	  try {
            testLudo<VL, Val>(keys,values, new_nn, zipfianKeys);
        } catch (exception &e) {
          cout << e.what() << endl;
          break;
        }

//        if (nn < 4194304) {
//          testDPH<VL, Val>(keys, values, nn, zipfianKeys);
//        }

//        if (VL == 4) {
//          testBloom<VL, Val>(keys, values, nn, zipfianKeys);
//        }

//        testPKCuckoo<VL, Val>(keys, values, nn, zipfianKeys);
        
        return;
      } catch (exception &e) {
        cerr << e.what() << endl;
      }

  }  
}

// returns opt fpr's
template<int VL, class Val>
std::pair<std::unordered_map<Val, double>, double> get_opt_bf_sizes(vector<Key> &keys, vector<Val> &values, uint64_t nn, uint64_t memory, bool is_hh_bf)
{
	// find #keys for each value, find HOT keys
	std::map<Val, uint64_t> per_val_key_ct;
	for (uint64_t i = 0; i < nn; i++)
		per_val_key_ct[ values[i] ] += 1;
	// HAWT keys:
	std::unordered_set<Val> hawt_vals;
	uint64_t total_hh_keys = 0;
	for (auto &i : per_val_key_ct)
	{
		cout << "Value(nh): " << i.first << ", Key ct: " << i.second << endl;
		if (i.second > 0.013 * nn )
		{
			hawt_vals.insert(i.first);
			total_hh_keys += i.second;
			cout << "HAWT!! \n";
		}
	}
	// Si = per_val_key_ct[hot_val]
	double nln_sum = 0.0;
	double n_sum = 0.0;
	for (auto &hv: hawt_vals)
	{
		nln_sum += per_val_key_ct[hv] * log(per_val_key_ct[hv]) ;
		n_sum += per_val_key_ct[hv];
		std::cout << "val:" << hv << " sz:" << per_val_key_ct[hv] << "nln sum: " << nln_sum << ", nsum: " << n_sum << std::endl;
	}
	// if there's BFh at the top:
	double w_h = 1.0;
	if (is_hh_bf)
	{
		// Give less wt to fh, since Sh is bigger than Si's.
		w_h = 1.0/hawt_vals.size(); // 0.020833;
		nln_sum += total_hh_keys * log(total_hh_keys / w_h);
		n_sum += total_hh_keys;
		std::cout << "After adding HH-BF: nln sum: " << nln_sum << ", nsum: " << n_sum << std::endl;
	}	
	
	// double lambda = log( (memory*log(2)*log(2) + nln_sum) / n_sum );
	// std::cout << "lambda: " << lambda << std::endl;
	double muu = exp( -1.0 * (memory + nln_sum) / n_sum );
	std::cout << "muu: " << muu << std::endl;

	int total_iters = 0; // for logging
	bool some_fpr_invalid = true; // loop exit condn
	std::unordered_map<Val, double> fpr_iter; // current set of fprs for val
	std::unordered_set<Val> remaining_vals = hawt_vals; // which vals have a BF
	for (auto &hv: hawt_vals)
		fpr_iter[hv] = (per_val_key_ct[hv] * muu);
	double hh_fpr_iter =  (total_hh_keys * muu / w_h);
	bool remaining_hh_bf = (is_hh_bf);

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
			double nln_sum_i = (remaining_hh_bf) ? total_hh_keys*log(total_hh_keys) : 0.0;
			double n_sum_i = (remaining_hh_bf) ? total_hh_keys : 0.0;
			for (auto &hv: remaining_vals)
			{
				nln_sum_i += per_val_key_ct[hv] * log(per_val_key_ct[hv]);
				n_sum_i += per_val_key_ct[hv];
			}

			muu = exp( -1.0 * (memory + nln_sum_i) / (n_sum_i) );
			for (auto &hv: remaining_vals)
				fpr_iter[hv] = muu * per_val_key_ct[hv];
			hh_fpr_iter = remaining_hh_bf ? (muu * total_hh_keys)/w_h : -1.0;
		}

	}

	std::unordered_map<Val, double> per_bf_fpr;
	for (auto &hv: remaining_vals)
	{
		per_bf_fpr[hv] = fpr_iter[hv];
		std::cout << "For val=" << hv << " BF fpr: " << per_bf_fpr[hv] << std::endl;
	}
	std::cout << hh_fpr_iter << " : HH FPR" << std::endl;
	return std::make_pair(per_bf_fpr, hh_fpr_iter); // pow(0.5, (lambda - log(total_hh_keys) )/log(2) ) 
}

int main(int argc, char **argv) {
  commonInit();

	// some key, values to test get_opt_bf_sizes  
	string fname = "../../cisco_data/" + string(argv[1]) + ".json";
	// std::pair< std::vector<Key>, std::vector<uint32_t> > keys_vals = read_fib_json(fname);
	// get_opt_bf_sizes<32,uint32_t>(keys_vals.first, keys_vals.second, keys_vals.first.size(), 1024*16, false);	
	
	for (int i = 0; i < 1; ++i) test<32, uint32_t>(string(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]));
  if (argc == 1) {
    	printf("STARTING expt value sz = 4 \n");
    // for (int i = 0; i < 100; ++i) test<8, uint8_t>();
    // for (int i = 0; i < 100; ++i) test<12, uint16_t>();
    // for (int i = 0; i < 100; ++i) test<16, uint16_t>();
    // for (int i = 0; i < 100; ++i) test<20, uint32_t>();
   // test<32>();
  	return 0;
  }
 
 /* 
  switch (atoi(argv[1])) {
    case 4:
      for (int i = 0; i < 100; ++i) test<4, uint8_t>();
      break;
    case 8:
      for (int i = 0; i < 100; ++i) test<8, uint8_t>();
      break;
    case 12:
      for (int i = 0; i < 100; ++i) test<12, uint16_t>();
      break;
    case 16:
      for (int i = 0; i < 100; ++i) test<16, uint16_t>();
      break;
    case 20:
      for (int i = 0; i < 100; ++i) test<20, uint32_t>();
      break;
//    case 16:
//      test<16>();
//    case 32:
//      test<32>();
  }
  */
  
  return 0;
}
