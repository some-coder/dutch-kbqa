/* Various utility symbols (header). */

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <unordered_map>
#include <string>
#include <filesystem>
#include <json/json.h>

namespace DutchKBQADSCreate {
    #if __APPLE__
        namespace fs = std::__fs::filesystem;
    #else
        namespace fs = std::filesystem;
    #endif
    enum NaturalLanguage {
        ENGLISH,
        DUTCH
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::NaturalLanguage> string_to_natural_language_map = {
        {"en", DutchKBQADSCreate::NaturalLanguage::ENGLISH},
        {"nl", DutchKBQADSCreate::NaturalLanguage::DUTCH}
    };
    std::string string_from_natural_language(const NaturalLanguage &language);
    enum LCQuADSplit {
        TRAIN,
        TEST
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::LCQuADSplit> string_to_lc_quad_split_map = {
        {"train", DutchKBQADSCreate::LCQuADSplit::TRAIN},
        {"test", DutchKBQADSCreate::LCQuADSplit::TEST}
    };
    std::string string_from_lc_quad_split(const LCQuADSplit &split);
    const DutchKBQADSCreate::fs::path root_dir = DutchKBQADSCreate::fs::canonical(
            DutchKBQADSCreate::fs::path(".",
                                           DutchKBQADSCreate::fs::path::format::generic_format));
    const DutchKBQADSCreate::fs::path dataset_dir =
        DutchKBQADSCreate::fs::canonical(DutchKBQADSCreate::root_dir /
                                         "../../resources/dataset");
    const DutchKBQADSCreate::fs::path supplements_dir =
            DutchKBQADSCreate::dataset_dir / "supplements";
    void create_directory_if_absent(const DutchKBQADSCreate::fs::path &dir_path);
    Json::Value json_loaded_from_dataset_file(const std::string &file_name);
    void save_json_to_dataset_file(const Json::Value &json,
                                   const std::string &file_name);
    void append_json_to_dataset_file(const Json::Value &json,
                                     const std::string &file_name);
    std::string string_with_regex_characters_escaped(const std::string &non_escaped);
}

#endif  /* UTILITIES_HPP */
