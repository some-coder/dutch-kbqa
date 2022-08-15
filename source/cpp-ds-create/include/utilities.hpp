/* Various utility symbols (header). */

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <unordered_map>
#include <string>
#include <filesystem>
#include <set>
#include <json/json.h>

namespace DutchKBQADSCreate {
    /**
     * @brief An index range. The first entry in this pair is the starting
     *   index; the second entry is the ending index. Both ends are inclusive.
     */
    using index_range = std::pair<int, int>;

    #if __APPLE__
        namespace fs = std::__fs::filesystem;
    #else
        namespace fs = std::filesystem;
    #endif

    /**
     * @brief A natural language. Contrast with formal languages, such as
     *   SPARQL.
     */
    enum NaturalLanguage {
        ENGLISH,
        DUTCH
    };
    /**
     * @brief A map that associates strings to natural languages.
     */
    const std::unordered_map<std::string, DutchKBQADSCreate::NaturalLanguage> string_to_natural_language_map = {
        {"en", DutchKBQADSCreate::NaturalLanguage::ENGLISH},
        {"nl", DutchKBQADSCreate::NaturalLanguage::DUTCH}
    };
    std::string string_from_natural_language(const NaturalLanguage &language);

    /**
     * @brief A dataset split within the LC-QuAD 2.0 dataset. Note that a
     *   common third subset, namely 'validation' (or 'development') is not
     *   included.
     */
    enum LCQuADSplit {
        TRAIN,
        TEST
    };
    /**
     * @brief A map that associates strings to LC-QuAD 2.0 dataset splits.
     */
    const std::unordered_map<std::string, DutchKBQADSCreate::LCQuADSplit> string_to_lc_quad_split_map = {
        {"train", DutchKBQADSCreate::LCQuADSplit::TRAIN},
        {"test", DutchKBQADSCreate::LCQuADSplit::TEST}
    };
    std::string string_from_lc_quad_split(const LCQuADSplit &split);

    /**
     * @brief A symbol type on WikiData. Either an entity or a property.
     */
    enum WikiDataSymbol {
        ENTITY,
        PROPERTY
    };
    WikiDataSymbol wiki_data_symbol_for_entity_or_property(const std::string &ent_or_prp);

    /**
     * @brief The root directory of the project that this C++ dataset
     *   processing is part of. As such, it is NOT the root directory of this
     *   C++ dataset processing project.
     */
    const DutchKBQADSCreate::fs::path root_dir = DutchKBQADSCreate::fs::canonical(
        DutchKBQADSCreate::fs::path(".",
                                       DutchKBQADSCreate::fs::path::format::generic_format)
    );
    /**
     * @brief The directory in which datasets are saved. Not identical to the
     *   resources directory; that is the parent directory of the `dataset_dir`.
     */
    const DutchKBQADSCreate::fs::path dataset_dir = DutchKBQADSCreate::fs::canonical(DutchKBQADSCreate::root_dir /
                                                                                     "../../resources/dataset");
    /**
     * @brief The directory in which supplemental data is stored with which
     *   derived datasets of LC-QuAD 2.0 are made.
     */
    const DutchKBQADSCreate::fs::path supplements_dir = DutchKBQADSCreate::dataset_dir / "supplements";

    bool dataset_file_exists(const fs::path &file);
    void create_directory_if_absent(const DutchKBQADSCreate::fs::path &dir_path);
    Json::Value json_loaded_from_dataset_file(const std::string &file_name);
    void save_json_to_dataset_file(const Json::Value &json,
                                   const std::string &file_name);
    void append_json_to_dataset_file(const Json::Value &json,
                                     const std::string &file_name);

    std::string string_with_regex_characters_escaped(const std::string &non_escaped);
    std::set<std::string> string_set_from_string_vec(const std::vector<std::string> &vec);
    std::optional<index_range> index_bounds_of_substring_in_string(const std::string &str, const std::string &sub_str);
    bool substring_replacement_success(std::string &str,
                                       const std::string &original,
                                       const std::string &replacement);
}

#endif  /* UTILITIES_HPP */
