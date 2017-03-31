#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

// General definition for a static for loop required for instantiation of templates
template <int from, int to>
struct for_
{
  template <template <int> class Fn>
  static void
  run()
  {
    Fn<from>::run();
    for_<from + 1, to>::template run<Fn>();
  }
};

template <int to>
struct for_<to, to>
{
  template <template <int> class Fn>
  static void
  run()
  {
  }
};
/////////////////////////////////////////////////////////////////////////////////////

#endif
