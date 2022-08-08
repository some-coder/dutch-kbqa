/* Symbols for replacing special symbols in translated datasets. */

#include <regex>
#include <stdexcept>
#include "utilities.hpp"
#include "tasks/replace-special-symbols.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns a RegEx for finding the keys of a symbol replacement map.
 *
 * @param replace_map The symbol replacement map.
 * @return The RegEx.
 */
std::regex symbol_replacement_search_query(const std::map<std::string, std::string> &replace_map) {
    std::string search_query = "(";
    bool is_first = true;
    for (const auto &pair : replace_map) {
        if (is_first) {
            is_first = false;
        } else {
            search_query += "|";  /* separate capturing groups */
        }
        search_query += std::string("(") + pair.first + ")";
    }
    search_query += ")";
    return std::regex(search_query);
}

/**
 * @brief Returns the input string, but with specified symbols replaced.
 *
 * @param str The string to replace symbols in.
 * @param replace_map The symbol replacement map.
 * @param search_query A RegEx for finding the keys of `replace_map`.
 * @return The string with symbols replaced.
 */
std::string string_with_symbols_replaced(std::string str,
                                         const std::map<std::string, std::string> &replace_map,
                                         const std::regex &search_query) {
    std::string replaced = str;  /* the string, but with symbols replaced */
    for (std::sregex_iterator it(str.begin(), str.end(), search_query);
         it != std::sregex_iterator();
         ++it) {
        /* iterate over all symbols to replace */
        std::string matched_symbol = string_with_regex_characters_escaped(it->str());
        std::regex matched_symbol_query("(" + matched_symbol + ")");
        replaced = std::regex_replace(replaced, matched_symbol_query, replace_map.at(matched_symbol));
    }
    return replaced;
}

/**
 * @brief Returns the JSON data, but with the symbols specified in
 *   `replace_map` replaced.
 *
 * @param json The JSON data to replace symbols in.
 * @param replace_map A mapping from symbols to replace to their replacements.
 * @return The JSON data, but with the specified symbols replaced.
 */
Json::Value DutchKBQADSCreate::json_with_symbols_replaced(Json::Value json,
                                                          const std::map<std::string, std::string> &replace_map) {
    std::regex search_query = symbol_replacement_search_query(replace_map);
    for (const auto &key : json.getMemberNames()) {
        json[key] = string_with_symbols_replaced(json[key].asString(),
                                                 replace_map,
                                                 search_query);
    }
    return json;
}

/* These HTML character entities have been taken from the W3C's Wiki:
 *   https://www.w3.org/wiki/Common_HTML_entities_used_for_typography
 * This source was last referenced on August 3rd, 2022.
 */
const std::map<std::string, std::string> html_character_entity_map = {
    {"&quot;", "\""},
    {"&amp;", "&"},
    {"&cent;", "¢"},
    {"&pound;", "£"},
    {"&sect;", "§"},
    {"&copy;", "©"},
    {"&laquo;", "«"},
    {"&raquo;", "»"},
    {"&reg;", "®"},
    {"&deg;", "°"},
    {"&plusmn;", "±"},
    {"&para;", "¶"},
    {"&middot;", "·"},
    {"&frac12;", "½"},
    {"&ndash;", "–"},
    {"&mdash;", "—"},
    {"&lsquo;", "‘"},
    {"&rsquo;", "’"},
    {"&sbquo;", "‚"},
    {"&ldquo;", "“"},
    {"&rdquo;", "”"},
    {"&bdquo;", "„"},
    {"&dagger;", "†"},
    {"&bull;", "•"},
    {"&hellip;", "…"},
    {"&prime;", "′"},
    {"&euro;", "€"},
    {"&trade;", "™"},
    {"&asymp;", "≈"},
    {"&ne;", "≠"},
    {"&le;", "≤"},
    {"&ge;", "≥"},
    {"&lt;", "<"},
    {"&gt;", ">"}
};

const std::regex html_entity_query("((&#[0-9]{1,4};)|(&[a-z]+;))");
const std::regex code_point_query("[0-9]{1,4}");

/**
 * A type of HTML entity.
 */
enum HTMLEntityType {
    CHARACTER,
    NUMERIC
};

/**
 * @brief Determines and returns the type of the supplied HTML entity.
 *
 * For performance, this function won't check whether `html_entity` is indeed
 * an HTML entity; it will only check what type the entity is, given that it
 * is an HTML entity.
 *
 * @param html_entity The HTML entity to determine the type of.
 * @return The HTML entity type.
 */
HTMLEntityType html_entity_type(const std::string &html_entity) {
    return html_character_entity_map.count(html_entity) == 0 ?
           HTMLEntityType::NUMERIC :
           HTMLEntityType::CHARACTER;
}

/**
 * @brief Returns the input string, but with the given HTML character entity
 *   replaced by its referent.
 *
 * @param str The string to replace in.
 * @param entity The HTML character entity to replace.
 * @return The string with the HTML character entity replaced.
 */
std::string string_with_html_character_entity_replaced(const std::string &str,
                                                       const std::string &entity) {
    return std::regex_replace(str,
                              std::regex("(" + entity + ")"),
                              html_character_entity_map.at(entity));
}

/**
 * @brief Returns the input string, but with the given HTML numeric entity
 *   replaced by its referent.
 *
 * This function will only convert entities with numeric codes between #0 and
 * #255 (both ends inclusive).
 *
 * @param str The string to replace in.
 * @param entity The HTML numeric entity to replace.
 * @return The string with the HTML numeric entity replaced.
 */
std::string string_with_html_numeric_entity_replaced(const std::string &str,
                                                     const std::string &entity) {
    std::smatch code_point_match;  /* Stores only the four digits (U+0000) of the entity. */
    std::regex_search(entity,
                      code_point_match,
                      code_point_query);
    std::string referent { static_cast<char>(std::stoi(code_point_match.str())) };
    return std::regex_replace(str,
                              std::regex("(" + entity + ")"),
                              referent);
}

/**
 * @brief Returns the input string, but with HTML entities replaced.
 *
 * @param str The string to replace HTML entities in.
 * @return The string with HTML entities replaced by the symbols they refer to.
 */
std::string string_with_html_entities_replaced(std::string str) {
    std::string replaced = str;  /* the string with HTML entities replaced */
    for (std::sregex_iterator it(str.begin(), str.end(), html_entity_query);
         it != std::sregex_iterator();
         ++it) {
        /* iterate over all HTML entities to replace */
        std::string matched_entity = it->str();
        switch (html_entity_type(matched_entity)) {
            case HTMLEntityType::CHARACTER:
                replaced = string_with_html_character_entity_replaced(replaced, matched_entity);
                break;
            case HTMLEntityType::NUMERIC:
                replaced = string_with_html_numeric_entity_replaced(replaced, matched_entity);
                break;
        }
    }
    return replaced;
}

/**
 * @brief Returns the JSON data, but with the HTML character and numeric
 *   entities replaced by their referents.
 *
 * @param json The JSON data to replace HTML entities in.
 * @return The JSON data, but with the HTML entities replaced.
 */
Json::Value DutchKBQADSCreate::json_with_html_entities_replaced(Json::Value json) {
    for (const auto &key : json.getMemberNames()) {
        json[key] = string_with_html_entities_replaced(json[key].asString());
    }
    return json;
}

/**
 * @brief Replaces various special symbols in the designated file.
 *
 * @param vm The variables map with which to determine which file to replace
 *   special symbols in, and where to save results to.
 */
void DutchKBQADSCreate::replace_special_symbols_in_dataset_file(const po::variables_map &vm) {
    if (vm.count("load-file-name") == 0) {
        throw std::invalid_argument(std::string(R"(The "--load-file-name" flag )") +
                                    "is required.");
    } else if (vm.count("save-file-name") == 0) {
        throw std::invalid_argument(std::string(R"(The "--save-file-name" flag )") +
                                    "is required.");
    }
    std::string load_file_name = vm["load-file-name"].as<std::string>();
    std::string save_file_name = vm["save-file-name"].as<std::string>();
    Json::Value json = json_loaded_from_dataset_file(load_file_name);
    std::map<std::string, std::string> replace_map {
            {string_with_regex_characters_escaped("_"), " " },
            {string_with_regex_characters_escaped("{"), "" },
            {string_with_regex_characters_escaped("}"), "" }
    };
    json = json_with_symbols_replaced(json, replace_map);
    json = json_with_html_entities_replaced(json);
    save_json_to_dataset_file(json, save_file_name);
}
