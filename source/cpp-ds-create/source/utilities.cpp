/* Various utility symbols. */

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include "utilities.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns the natural language as a string.
 *
 * @param language The natural language to express in string form.
 * @return Said natural language, but in string form.
 */
std::string DutchKBQADSCreate::string_from_natural_language(const NaturalLanguage &language) {
    switch (language) {
        case NaturalLanguage::ENGLISH:
            return "en";
        case NaturalLanguage::DUTCH:
            return "nl";
        default:
            throw std::invalid_argument("This natural language is not supported!");
    }
}

/**
 * @brief Returns the LC-QuAD 2.0 dataset `split` as a string.
 *
 * @param split The dataset split.
 * @return Said split, but in string form.
 */
std::string DutchKBQADSCreate::string_from_lc_quad_split(const LCQuADSplit &split) {
    switch (split) {
        case LCQuADSplit::TRAIN:
            return "train";
        case LCQuADSplit::TEST:
            return "test";
        default:
            throw std::invalid_argument("This split type is not supported!");
    }
}

/**
 * @brief Determines which type of WikiData symbol an entity or property
 *   belongs to.
 *
 * This function throws an `invalid_argument` error if `ent_or_prp` is neither
 * an entity nor a property.
 *
 * @param ent_or_prp The entity or property.
 */
WikiDataSymbol DutchKBQADSCreate::wiki_data_symbol_for_entity_or_property(const std::string &ent_or_prp) {
    assert(!ent_or_prp.empty());
    switch (ent_or_prp[0]) {
        case 'Q':
            return WikiDataSymbol::ENTITY;
        case 'P':
            return WikiDataSymbol::PROPERTY;
        default:
            throw std::invalid_argument(std::string("\"") +
                                        ent_or_prp +
                                        "\" is not an entity or property!");
    }
}

/**
 * @brief Determines whether `file` exists under the project root's
 *   `resources/dataset` directory.
 *
 * @param file The file to check existence for. The path to its location should
 *   be expressed relative to `resources/dataset`.
 * @return The question's answer.
 */
bool DutchKBQADSCreate::dataset_file_exists(const fs::path &file) {
    return fs::exists(dataset_dir / file);
}

/**
 * @brief Creates the directory specified by `path` similar to how `mkdir` is
 *   used on UNIX-like machines, if it does not already exist.
 *
 * If the directory already exists, this method No-Ops silently.
 *
 * @param dir_path The directory path. The parent directory must already exist.
 */
void DutchKBQADSCreate::create_directory_if_absent(const fs::path &dir_path) {
    std::error_code ec;
    if (fs::exists(dir_path)) {
        return;  /* no need to create already existing directory */
    }
    const bool creation_succeeded = fs::create_directory(dir_path, ec);
    if (!creation_succeeded) {
        throw std::runtime_error(std::string("Couldn't create directory.") +
                                 "Error: " +
                                 std::to_string(ec.value()) +
                                 ".");
    }
}

/**
 * @brief Returns JSON data loaded from a file stores in the project root
 *   `resources/dataset/` directory.
 * 
 * @param file_name The name of the file to load in `resources/dataset/`.
 *   Exclude `.json`.
 * @return The JSON data.
 */
Json::Value DutchKBQADSCreate::json_loaded_from_dataset_file(const std::string &file_name) {
    std::ifstream file(dataset_dir / (file_name + ".json"),
                       std::ifstream::binary);
    Json::Value json;
    file >> json;
    return json;
}

/**
 * @brief Saves the supplied JSON data to the resources directory, under
 *   `file_name`.
 * 
 * @param json The JSON data to save to disk.
 * @param file_name The file in the project root's `resources` directory to save
 *   to. Exclude `.json`.
 */
void DutchKBQADSCreate::save_json_to_dataset_file(const Json::Value &json,
                                                  const std::string &file_name) {
    Json::StyledStreamWriter writer;
    std::ofstream file;
    file.open(dataset_dir / (file_name + ".json"),
              std::ofstream::trunc);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("JSON save file \"") +
                                 file_name +
                                 "\" won't open!");
    }
    writer.write(file, json);
}

/**
 * @brief Appends the contents of `json` to the JSON content already present in
 *   the file `file_name`, present in the project root's `resources` directory.
 * 
 * @param json The JSON to append.
 * @param file_name The name of the file in the project root's `resources`
 *   directory to which to append. Exclude `.json`.
 */
void DutchKBQADSCreate::append_json_to_dataset_file(const Json::Value &json,
                                                    const std::string &file_name) {
    std::ifstream file(dataset_dir / (file_name + ".json"),
                       std::ifstream::binary);
    Json::Value file_json;  /* JSON already present in `file`. */
    /* Try to get the JSON content of `file`. If it fails, create the file. */
    try {
        file >> file_json;
    } catch (Json::Exception::exception &exception) {
        save_json_to_dataset_file(json, file_name);
        return;
    }
    Json::Value updated_file_json;  /* `file_json` plus `json`. */
    if (file_json.isArray() && json.isArray()) {
        /* Update case 1: Append two JSON arrays. */
        updated_file_json = Json::arrayValue;
        for (const auto &entry : file_json) {
            updated_file_json.append(entry);
        }
        for (const auto &entry : json) {
            updated_file_json.append(entry);
        }
    } else {
        /* Update case 2: Approach both JSON files as objects. */
        for (const auto &member : file_json.getMemberNames()) {
            updated_file_json[member] = file_json[member];
        }
        for (const auto &member : json.getMemberNames()) {
            updated_file_json[member] = json[member];
        }
    }
    save_json_to_dataset_file(updated_file_json, file_name);
}

/* The following characters are reserved in regular expressions. */
const std::vector<char> regex_characters_to_escape {
    '.',
    '(', ')',
    '[', ']',
    '|',
    '{', '}',
    '*',
    '+',
    '-',
    '?',
    '^',
    '$',
    '/',
    '\\'
};

/**
 * @brief Returns the input string, but with any reserved RegEx characters
 *   replaced.
 *
 * @param non_escaped The original, non-escaped string.
 * @return The string with any RegEx-reserved characters escaped.
 */
std::string DutchKBQADSCreate::string_with_regex_characters_escaped(const std::string &non_escaped) {
    std::string escapes_added;
    for (const auto &c : non_escaped) {
        if (*std::find(regex_characters_to_escape.begin(),
                       regex_characters_to_escape.end(),
                       c) == c) {
            escapes_added += "\\";
        }
        escapes_added += c;
    }
    return escapes_added;
}

/**
 * @brief Returns a set of strings built from a vector of strings.
 *
 * @param vec The vector to obtain a set from.
 * @return The set.
 */
std::set<std::string> DutchKBQADSCreate::string_set_from_string_vec(const std::vector<std::string> &vec) {
    std::set<std::string> set;
    std::for_each(vec.begin(),
                  vec.end(),
                  [&set] (const std::string &entry) -> void { set.insert(entry); });
    return set;
}

/**
 * @brief Returns the starting and ending index of `sub_str` in `str`. Both
 *   indices are inclusive, by the description of the type `index_range`. If
 *   `sub_str` is not wholly present in `str`, null is returned instead; if
 *   multiple matches occur, only the first one's indices are provided.
 *
 * @param str The string in which to search for `sub_str`.
 * @param sub_str The substring to search for in `str`.
 * @return The starting and ending index of the first full occurrence of
 *   `sub_str` in `str`. If no full occurrence is found, null is returned
 *   instead.
 */
std::optional<index_range> DutchKBQADSCreate::index_bounds_of_substring_in_string(const std::string &str,
                                                                                  const std::string &sub_str) {
    std::string::size_type pos = str.find(sub_str, 0);
    if (pos == std::string::npos) {
        return std::nullopt;
    } else {
        return index_range(pos, pos + sub_str.size() - 1);  /* Recall: both ends inclusive. */
    }
}

/**
 * @brief Attempts to replace `original` in the string `str` by replacement
 *   `replacement`.
 *
 * @param str The string in which to replace the substring `original`.
 * @param original The substring to replace.
 * @param replacement The replacement for `original`.
 * @return Whether the replacement operation succeeded (`true`) or not
 *   (`false`).
 */
bool DutchKBQADSCreate::substring_replacement_success(std::string &str,
                                                      const std::string &original,
                                                      const std::string &replacement) {
    const std::string &from = original;
    const std::string &to = replacement;
    const std::size_t start_pos = str.find(from);
    if (start_pos == std::string::npos) {
        /* Couldn't find substring in string. Can't replace. */
        return false;
    } else {
        str.replace(start_pos, from.length(), to);
        return true;
    }
}
