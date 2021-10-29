#ifndef BIR_LOOPNEST_H
#define BIR_LOOPNEST_H

#include "bir/LoopAxis.h"

namespace bir {

class LoopNest {
 public:
  std::vector<LoopAxis*> loops;
  std::unordered_map<std::string, LoopAxis*> name_loop_map;
  std::unordered_map<LoopNest*, int> loop_level_map;

  template <typename... Args> 
  LoopNest(Args... loops) {
    for (auto loop : {loops...}) {
      int level = this->loops.size();
      loop_level_map[loop] = level;
      name_loop_map[loop->getName()] = loop;
      this->loops.push_back(loop);
    }
  }
};

}

#endif
