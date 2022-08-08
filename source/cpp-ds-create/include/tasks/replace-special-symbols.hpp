/* Symbols for replacing special symbols in translated datasets (header). */

#ifndef REPLACE_SPECIAL_SYMBOLS_HPP
#define REPLACE_SPECIAL_SYMBOLS_HPP

#include <json/json.h>
#include <vector>
#include <boost/program_options.hpp>

namespace DutchKBQADSCreate {
    namespace po = boost::program_options;
    Json::Value json_with_symbols_replaced(Json::Value json,
                                           const std::map<std::string, std::string>& replace_map);
    Json::Value json_with_html_entities_replaced(Json::Value json);
    void replace_special_symbols_in_dataset_file(const po::variables_map &vm);
}

#endif  /* REPLACE_SPECIAL_SYMBOLS_HPP */
