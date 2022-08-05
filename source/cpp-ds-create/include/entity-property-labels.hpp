/* Symbols for retrieving labels for WikiData entities and properties (header). */

#ifndef ENTITY_PROPERTY_LABELS_HPP
#define ENTITY_PROPERTY_LABELS_HPP

#include <set>
#include <json/json.h>
#include "utilities.hpp"

namespace DutchKBQADSCreate {
    using ent_prp_partitioning = std::vector<std::set<std::string>>;
    std::set<std::string> unique_entities_and_properties_of_split(const LCQuADSplit &split);
    Json::Value saved_entity_and_property_labels(const Json::Value &json,
                                                 const LCQuADSplit &split,
                                                 const NaturalLanguage &language);
    Json::Value loaded_entity_and_property_labels(const LCQuADSplit &split,
                                                  const NaturalLanguage &language);
    std::set<std::string> entities_and_properties_requiring_labeling(const LCQuADSplit &split,
                                                                     const NaturalLanguage &language);
    DutchKBQADSCreate::ent_prp_partitioning entity_property_partitioning(const std::set<std::string> &ent_prp_set,
                                                                         int part_size);
    void label_entity_property_partitions(const LCQuADSplit &split,
                                          const NaturalLanguage &language);
}

#endif  /* ENTITY_PROPERTY_LABELS_HPP */
