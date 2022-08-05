/* Symbols for relating LC-QuAD 2.0 questions to WikiData entities and properties (header). */

#ifndef QUESTION_ENTITIES_PROPERTIES_HPP
#define QUESTION_ENTITIES_PROPERTIES_HPP

#include <json/json.h>
#include <map>
#include <boost/program_options.hpp>
#include "utilities.hpp"

namespace DutchKBQADSCreate {
    using q_ent_prp_map = std::map<int, std::set<std::string>>;
    namespace po = boost::program_options;
    DutchKBQADSCreate::q_ent_prp_map question_entities_properties_map(const Json::Value &ds_split);
    void save_question_entities_properties_map(const DutchKBQADSCreate::q_ent_prp_map &m,
                                               const LCQuADSplit &split);
    void generate_question_entities_properties_map(const po::variables_map &vm);
}

#endif  /* QUESTION_ENTITIES_PROPERTIES_HPP */
