#ifndef BIR_NAMEDNamedObjectCONTAINER_H
#define BIR_NAMEDNamedObjectCONTAINER_H

#include <cassert>
#include <cstdint>
#include <map>
#include <vector>
#include <atomic>
#include <llvm/ADT/ilist.h>
#include <boost/range/iterator_range.hpp>

namespace bir {
  
template <typename, typename> class NamedObject;
template <typename, typename> class NamedObjectContainer;

class NamedObjectBase {
 public:
  std::string Name;
  const unsigned Id;
  uint32_t PrivateIndex;

  NamedObjectBase(const std::string &Name, unsigned Id) : Name(Name), Id(Id), PrivateIndex(0) {}
  const std::string &getName() const {return Name;}
  unsigned getId() const {return Id;}
  uint32_t getPrivateIndex() const {return PrivateIndex;}
  void setPrivateIndex(uint32_t i) {PrivateIndex = i;}

  // noncopyable
  NamedObjectBase(const NamedObjectBase&) = delete;
  NamedObjectBase& operator=(const NamedObjectBase&) = delete;
};

/*
 A NamedObject can be stored in a NamedObjectContainer, has a name and a
 pointer to its parent (the container holding it)
*/
template <typename C, typename P>
class NamedObject : public llvm::ilist_node_with_parent<C, P>, public NamedObjectBase {
 public:
  P *Parent;
  void setParent(P *Parent) {this->Parent = Parent;}
  static std::atomic<unsigned> IdCounter;
  static std::vector<NamedObject<C, P>*> Id2Obj;

  using llvm::ilist_node_with_parent<C, P>::getIterator;
  using llvm::ilist_node_with_parent<C, P>::getPrevNode;
  using llvm::ilist_node_with_parent<C, P>::getNextNode;

  friend class NamedObjectContainer<P, C>;

  NamedObject (const std::string &Name, P *Parent) : NamedObjectBase(Name, IdCounter++), Parent(Parent) {
    Id2Obj.push_back(this);
  }

  static C *id2obj(unsigned id) {return static_cast<C*>(Id2Obj[id]);} 
  static unsigned getIdCount() {return IdCounter;}
  P *getParent() {return Parent;}
  const P *getParent() const {return Parent;}

  void moveBefore(NamedObject<C, P> &MovePos) {
    NamedObjectContainer<P, C> *MoveParent = MovePos.getParent();
    MoveParent->splice(MovePos.getIterator(), *getParent(), getIterator());
  }

  void moveAfter(NamedObject<C, P> &MovePos) {
    NamedObjectContainer<P, C> *MoveParent = MovePos.getParent();
    MoveParent->splice(++MovePos.getIterator(), *getParent(), getIterator());
  }
};


template <typename C, typename T>
class NamedObjectContainer {
 public:
  using ListType = typename llvm::ilist<T>;
  using iterator = typename ListType::iterator;
  using const_iterator = typename ListType::const_iterator;

  ListType Elements;
  std::map<std::string, T*> Symboltable;

  template <typename SUB_OF_T, typename... Args>
  SUB_OF_T &insertElement(iterator where, const std::string &Name,
                           Args... ConstructorArgs) {
    SUB_OF_T *Elt = new SUB_OF_T(Name, static_cast<C*>(this),
                                 std::forward<Args>(ConstructorArgs)...);                          
    Elements.insert(where, Elt);
    Symboltable.insert({Name, Elt});
    return *Elt;                             
  }

  NamedObjectContainer() = default;

  // noncopyable
  NamedObjectContainer(const NamedObjectContainer&) = delete;
  NamedObjectContainer& operator=(const NamedObjectContainer&) = delete;

  // movable
  NamedObjectContainer(NamedObjectContainer &&other)
    : Elements(std::move(other.Elements)),
      Symboltable(std::move(other.Symboltable)) {}

  NamedObjectContainer & operator=(NamedObjectContainer &&other) {
    Elements.clear();
    std::swap(Elements, other.Elements);
    Symboltable.clear();
    std::swap(Symboltable, other.Symboltable);
    return *this;
  }
          
  iterator begin() {return Elements.begin();}
  iterator end() {return Elements.end();}
  const_iterator begin() const {return Elements.begin();}
  const_iterator end() const {return Elements.end();}
  size_t size() const {return Elements.size();}
  bool empty() const {return Elements.empty();}

  boost::iterator_range<iterator> elements() {
    return boost::iterator_range<iterator>(Elements.begin(), Elements.end()); 
  }

  boost::iterator_range<const_iterator> elements() const {
    return boost::iterator_range<const_iterator>(Elements.begin(), Elements.end()); 
  }

  T *getElementByName(std::string Name) {
    auto it = Symboltable.find(Name);
    if (it == Symboltable.end()) return nullptr;
    else it->second;
  }

  void splice(iterator where, NamedObjectContainer<C, T> &other, iterator pos) {
    Elements.splice(where, other.Elements, pos);
    if (this != &other) {
      T &NamedObject = *pos;
      other.Symboltable.erase(NamedObject.getName());
      Symboltable.insert({NamedObject.getName(), &NamedObject});
      NamedObject.setParent(static_cast<C*>(this));  
    }  
  }

  static ListType NamedObjectContainer<C, T>::*getSublistAccess(T*) {
    return &NamedObjectContainer<C, T>::Elements;
  }

  void removeElement(NamedObject<T, C> & object) {
    Symboltable.erase(object.getName());
    Elements.erase(object.getIterator());
  }
};

}  // namespace bir

#endif
