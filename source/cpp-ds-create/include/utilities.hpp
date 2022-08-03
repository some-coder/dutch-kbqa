/* Various utility symbols (header). */

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <unordered_map>
#include <string>
#include <filesystem>
#include <json/json.h>

namespace DutchKBQADSCreate {
    namespace fs = std::filesystem;
    enum NaturalLanguage {
        ENGLISH,
        DUTCH
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::NaturalLanguage> string_to_natural_language_map = {
        {"en", DutchKBQADSCreate::NaturalLanguage::ENGLISH},
        {"nl", DutchKBQADSCreate::NaturalLanguage::DUTCH}
    };
    enum LCQuADSplit {
        TRAIN,
        TEST
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::LCQuADSplit> string_to_lc_quad_split_map = {
        {"train", DutchKBQADSCreate::LCQuADSplit::TRAIN},
        {"test", DutchKBQADSCreate::LCQuADSplit::TEST}
    };
    const DutchKBQADSCreate::fs::path root_dir =
        DutchKBQADSCreate::fs::canonical(DutchKBQADSCreate::fs::path(".",
                                                                     DutchKBQADSCreate::fs::path::format::generic_format));
    const DutchKBQADSCreate::fs::path resources_dir =
        DutchKBQADSCreate::fs::canonical(DutchKBQADSCreate::root_dir /
                                         "../../resources/");
    Json::Value json_loaded_from_resources_file(const std::string &file_name);
    void save_json_to_resources_file(const Json::Value &json,
                                     const std::string &file_name);
    void append_json_to_resources_file(const Json::Value &json,
                                       const std::string &file_name);
}

#endif  /* UTILITIES_HPP */
