/* Symbols for retrieving labels for WikiData entities and properties. */

#include "entity-property-labels.hpp"
#include "question-entities-properties.hpp"
#include "utilities.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns the set of entities and properties present in the
 *   question-to-entities-and-properties map of `split`.
 *
 * @param split The split to retrieve the unique entities and properties of.
 * @return The set.
 */
std::set<std::string> DutchKBQADSCreate::unique_entities_and_properties_of_split(const LCQuADSplit &split) {
    Json::Value json = loaded_question_entities_properties_map(split);
    std::set<std::string> ent_prp_set;
    for (const auto &member : json.getMemberNames()) {
        for (const auto &ent_or_prp : json[member]) {
            ent_prp_set.insert(ent_or_prp.asString());
        }
    }
    return ent_prp_set;
}

std::string entity_and_property_labels_file_name(const LCQuADSplit &split,
                                                 const NaturalLanguage &language) {
    return string_from_lc_quad_split(split) +
           "-" +
           string_from_natural_language(language) +
           "_labels";
}

//Json::Value DutchKBQADSCreate::saved_entity_and_property_labels(const Json::Value &json,
//                                                                const LCQuADSplit &split,
//                                                                const NaturalLanguage &language) {
//
//}

//Json::Value DutchKBQADSCreate::entity_and_property_labels(const LCQuADSplit &split,
//                                                          const NaturalLanguage &language) {
//
//}
