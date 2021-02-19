#ifndef DNE_LOGGING_H
#define DNE_LOGGING_H

inline void report(std::string const &log){
  LOG(INFO) << log;
  std::cout << log << std::endl;
}

#endif //DNE_LOGGING_H
