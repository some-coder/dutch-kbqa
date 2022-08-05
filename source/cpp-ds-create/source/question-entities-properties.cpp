/* Symbols for relating LC-QuAD 2.0 questions to WikiData entities and properties. */

#include <regex>
#include <set>
#include <iostream>
#include "question-entities-properties.hpp"
#include "utilities.hpp"

using namespace DutchKBQADSCreate;

const char *wikidata_ent_prp_regex = "[QP][0-9]+";

/**
 * @brief Returns the entities and properties discoverable in `question`.
 *
 * @param question The LC-QuAD 2.0 question to collect entities and properties
 *   for.
 * @return The collected WikiData entities and properties.
 */
std::set<std::string> entities_and_properties_of_question(const Json::Value &question) {
    const std::string sparql = question["sparql_wikidata"].asString();
    std::set<std::string> s;
    const std::regex ent_prp_query(wikidata_ent_prp_regex);
    for (std::sregex_iterator it(sparql.begin(),
                              sparql.end(),
                              ent_prp_query);
         it != std::sregex_iterator();
         ++it) {
        s.insert(it->str());
    }
    return s;
}

/**
 * @brief Returns a mapping from questions in `ds_split` to WikiData entities
 *   and properties discovered in those questions' SPARQL answer formulations.
 *
 * @param ds_split The LC-QuAD 2.0 dataset split to make a mapping of.
 * @return The map.
 */
q_ent_prp_map DutchKBQADSCreate::question_entities_properties_map(const Json::Value &ds_split) {
    q_ent_prp_map m;
    for (const auto &question : ds_split) {
        const int uid = question["uid"].asInt();
        const auto ent_prp = entities_and_properties_of_question(question);
        m.insert({ uid, ent_prp });
    }
    return m;
}

/**
 * @brief Returns the question-to-entities-and-properties map, but converted to
 *   a JSON value.
 *
 * @param m The original question-to-entities-and-properties map.
 * @return The map, but converted to a JSON object.
 */
Json::Value json_from_question_entities_properties_map(const q_ent_prp_map &m) {
    Json::Value json;
    for (const auto &q_ent_prp_pair : m) {
        /* First, add all entities and properties to a JSON array... */
        Json::Value ent_prp_array = Json::arrayValue;
        for (const auto &ent_or_prp : q_ent_prp_pair.second) {
            ent_prp_array.append(ent_or_prp);
        }
        /* ...and second, relate this array to a question UID. */
        json[std::to_string(q_ent_prp_pair.first)] = ent_prp_array;
    }
    return json;
}

/**
 * @brief Saves the question-to-entities-and-properties map `m` to disk.
 *
 * @param m The map to save to disk. The file will be saved in the project
 *   root's `resources` subdirectory.
 * @param split The LC-QuAD 2.0 dataset split on which `m` is based. Will
 *   influence the file name.
 */
void DutchKBQADSCreate::save_question_entities_properties_map(const q_ent_prp_map &m,
                                                              const LCQuADSplit &split) {
    Json::Value json = json_from_question_entities_properties_map(m);
    const std::string file_name = string_from_lc_quad_split(split) +
                                  "_ent_prp_map";
    save_json_to_resources_file(json,
                                file_name);
}

/**
 * @brief Creates and saves a question-to-entities-and-properties map for
 *   questions of an LC-QuAD 2.0 dataset split.
 *
 * @param vm The variables map with which to determine which LC-QuAD 2.0
 *   dataset split to work on.
 */
void DutchKBQADSCreate::generate_question_entities_properties_map(const po::variables_map &vm) {
    if (vm.count("split") == 0) {
        throw std::invalid_argument(std::string(R"(The "--split" flag is required.)"));
    }
    const std::string split_str = vm["split"].as<std::string>();
    const LCQuADSplit split = string_to_lc_quad_split_map.at(split_str);
    const Json::Value ds_split = json_loaded_from_resources_file(split_str);
    const q_ent_prp_map m = question_entities_properties_map(ds_split);
    save_question_entities_properties_map(m, split);
}
