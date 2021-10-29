#ifndef BIR_FUNCTION_H
#define BIR_FUNCTION_H

#include <map>
#include "bir/NamedObjectContainer.h"
#include "bir/BasicBlock.h"

namespace bir {

class Module;

class Function : public NamedObjectContainer<Function, BasicBlock>,
  public NamedObject<Function, Module> {
 public:
  Function(const std::string &Name, Module *Parent);
  using block_iterator = NamedObjectContainer<Function, BasicBlock>::iterator;
  using const_block_iterator = NamedObjectContainer<Function, BasicBlock>::const_iterator;
  using NamedObjectContainer<Function, BasicBlock>::size;
  using NamedObjectContainer<Function, BasicBlock>::empty;
  using NamedObjectContainer<Function, BasicBlock>::begin;
  using NamedObjectContainer<Function, BasicBlock>::end;
  using NamedObjectContainer<Function, BasicBlock>::getElementByName;

  BasicBlock & addBasicBlock(const std::string &Name);
  void removeBasicBlock(BasicBlock &Block);
  BasicBlock *getBasicBlockByName(const std::string &Name);
  // loop
  void dump() const;
  std::vector<MemoryLocation*> mem_locs;
  void addMemoryLocation(MemoryLocation* m) {mem_locs.push_back(m);}
  std::vector<MemoryLocation*> MemoryLocations() {return mem_locs;}
};

}  // namespace bir

#endif  // BIR_FUNCTION_H