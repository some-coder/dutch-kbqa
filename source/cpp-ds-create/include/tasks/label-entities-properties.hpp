/* Symbols for retrieving labels for WikiData entities and properties (header). */

#ifndef LABEL_ENTITIES_PROPERTIES_HPP
#define LABEL_ENTITIES_PROPERTIES_HPP

#include <set>
#include <json/json.h>
#include <boost/program_options.hpp>
#include "utilities.hpp"

namespace DutchKBQADSCreate {
    namespace po = boost::program_options;
    using ent_prp_partitioning = std::vector<std::set<std::string>>;
    using ent_prp_label_map = std::map<std::string, std::vector<std::string>>;

    std::set<std::string> unique_entities_and_properties_of_split(const LCQuADSplit &split);
    void save_entity_and_property_labels(const Json::Value &json,
                                         const LCQuADSplit &split,
                                         const NaturalLanguage &language);
    Json::Value loaded_json_entity_and_property_labels(const LCQuADSplit &split,
                                                       const NaturalLanguage &language);
    DutchKBQADSCreate::ent_prp_label_map loaded_entity_and_property_labels(const LCQuADSplit &split,
                                                                           const NaturalLanguage &language);
    DutchKBQADSCreate::ent_prp_label_map entity_and_property_labels_subset(
        const std::set<std::string> &ent_prp_set,
        const DutchKBQADSCreate::ent_prp_label_map &all
    );
    std::set<std::string> entities_and_properties_requiring_labeling(const LCQuADSplit &split,
                                                                     const NaturalLanguage &language);
    DutchKBQADSCreate::ent_prp_partitioning entity_property_partitioning(const std::set<std::string> &ent_prp_set,
                                                                         int part_size);
    void label_entities_and_properties(const DutchKBQADSCreate::po::variables_map &vm);
}

#endif  /* LABEL_ENTITIES_PROPERTIES_HPP */
