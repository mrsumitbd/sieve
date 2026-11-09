#include <vector>
#include <string>
#include <map>

namespace outer {
namespace inner {

/** A documented class. */
class Widget : public Base, private Other {
public:
  Widget() {}
  ~Widget() {}
  Widget(const Widget& o) : v_(o.v_) {}
  int value() const { return v_; }
  Widget& operator=(const Widget& o) { v_ = o.v_; return *this; }
  bool operator==(const Widget& o) const { return v_ == o.v_; }
  operator int() const { return v_; }
  template <typename T> T convert() const { return static_cast<T>(v_); }
  void declared_only();
  class Nested { public: void inNested() {} };
  struct Plain { int f() { return 1; } };
private:
  int v_ = 0;
};

class Fwd;
void takes(class Fwd* p) {}
void takes_ref(const class Fwd& p) {}

template <typename T, int N>
class Array { public: T at(int i) { return data[i]; } T data[N]; };

template <typename T>
class Array<T*, 4> { public: T first() { return nullptr; } };

template <typename T>
T maximum(T a, T b) { return a > b ? a : b; }

int* pointer_return() { return nullptr; }
const std::string& ref_return(const std::string& s) { return s; }
int** double_pointer() { return nullptr; }

void Widget::declared_only() {}
int Widget::Nested::nope() { return 0; }

void with_local_class() {
  class Local { public: void l() {} };
  auto lam = [](int x) { return x; };
}

}  // namespace inner
}  // namespace outer

struct NotAClass { int g() { return 1; } };
union U { int a; };

int main(int argc, char** argv) { return 0; }
