#ifndef BIR_MODULE_H
#define BIR_MODULE_H

#include <map>
#include <boost/range/iterator_range.hpp>
#include "bir/Function.h"
#include "bir/NamedObjectContainer.h"
#include "Arch.h"

namespace bir {

class Module : public NamedObjectContainer<Module, Function> {
 public:
  Arch* arch;

  using function_iterator = NamedObjectContainer<Module, Function>::iterator;
  using const_function_iterator = NamedObjectContainer<Module, Function>::const_iterator;

  Module() {
    arch = new Arch();
  }
  Module(Module &&other);
  Module &operator=(Module &&other);
  void dump() const;
  Function &addFunction(const std::string &Name);
  void removeFunction(Function &Func);
  Function *getFunctionByName(const std::string &Name);
  Arch* getArchModel() {return arch;}
  void load(const std::string &inputFileName);
  void loadJson(std::ifstream &inputStream);
};  

}  // namespace bir

#endif  // BIR_MODULE_H