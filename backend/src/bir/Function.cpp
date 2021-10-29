#include <fstream>
#include <iostream>
#include "bir/Function.h"
#include "bir/BasicBlock.h"
#include "bir/Instruction.h"
#include "bir/NamedObjectContainer.h"

#include "nlohmann/json.hpp"

using namespace bir;


template<>
std::atomic<unsigned> NamedObject<Function, Module>::IdCounter(0);

template<>
std::vector<NamedObject<Function, Module>*> NamedObject<Function, Module>::Id2Obj{};

Function::Function(const std::string &Name, Module *Parent) : NamedObject<Function, Module>(Name, Parent){};

BasicBlock & Function::addBasicBlock(const std::string &Name) {
  BasicBlock &Block = NamedObjectContainer<Function, BasicBlock>::insertElement<BasicBlock>(end(), Name);
  return Block;  
}

void Function::removeBasicBlock(BasicBlock &Block) {
  NamedObjectContainer<Function, BasicBlock>::removeElement(Block);
}

BasicBlock* Function::getBasicBlockByName(const std::string &Name) {
  return NamedObjectContainer<Function, BasicBlock>::getElementByName(Name);
}