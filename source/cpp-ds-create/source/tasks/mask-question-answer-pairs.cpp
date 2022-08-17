/* Symbols for masking question-answer pairs. */

#include <iostream>
#include <utility>
#include <cassert>
#include <regex>
#include "tasks/mask-question-answer-pairs.hpp"
#include "tasks/collect-entities-properties.hpp"
#include "tasks/label-entities-properties.hpp"
#include "utilities.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Constructs an LC-QuAD 2.0 question-answer pair.
 *
 * @param uid The UID of the question-answer pair.
 * @param question The question.
 * @param answer The answer.
 */
DutchKBQADSCreate::QuestionAnswerPair::QuestionAnswerPair(int uid,
                                                          std::string question,
                                                          std::string answer) :
                                                          q(std::move(question)), a(std::move(answer)) {
    this->uid = uid;
}


/**
 * @brief Constructs resultant statistics of matching a label to a string.
 *
 * @param label The label to use for matching.
 * @param match_bounds The index bounds of the match with respect ot the
 *   question the matching operation was performed in.
 * @param ent_or_prp The entity or property associated with this label.
 */
DutchKBQADSCreate::LabelMatch::LabelMatch(const std::string &label,
                                          const index_range &match_bounds,
                                          const std::string &ent_or_prp) {
    this->ent_or_prp = ent_or_prp;
    this->label = label;
    this->match_bounds = match_bounds;
}


std::optional<index_range> DutchKBQADSCreate::LabelMatch::match_label_in_sentence(const std::string &label,
                                                                                  const std::string &sentence) {
    std::string inner_re = label;
    inner_re = std::regex_replace(inner_re, std::regex("(\\[)"), "\\[");
    inner_re = std::regex_replace(inner_re, std::regex("(\\])"), "\\]");
    std::regex re("(" + inner_re + ")");
    std::smatch label_re_match;
    if (std::regex_search(sentence, label_re_match, re)) {
        int start_idx = static_cast<int>(label_re_match.position(0));
        return index_range(start_idx, start_idx + label_re_match.length(0) - 1);
    } else {
        return std::nullopt;
    }
}


/**
 * @brief Compares this label match against another, determining whether this
 *   match appears earlier in the matched-against string than `other`.
 *
 * In the case that this match and `other` both share the same starting index,
 * precedence is determined by considering whether the ending index of this
 * string lies strictly before that of `other`.
 *
 * @param other The other label match.
 * @return The question's answer.
 */
bool DutchKBQADSCreate::LabelMatch::appears_earlier_in_string(const LabelMatch &first, const LabelMatch &second) {
    if (first.match_bounds.first == second.match_bounds.first) {
        return first.match_bounds.second < second.match_bounds.second;
    } else {
        return first.match_bounds.first < second.match_bounds.first;
    }
}


/**
 * @brief Returns the best-matched label with respect to some question, or null
 *   if none of the labels is satisfactory.
 *
 * @param matches Label matches for the question. Possibly empty. Note that
 *   some label "matches" may not have matched at all; these will have their
 *   `match_bounds` field set to the special `std::pair` value of
 *   `{ no_label_match_pos, no_label_match_pos }`.
 * @return The best match, or null if no satisfactory match could be found.
 */
std::optional<LabelMatch> DutchKBQADSCreate::LabelMatch::best_label_match(const std::vector<LabelMatch> &matches) {
    for (const auto &candidate : matches) {
        if (candidate.match_bounds.first != no_label_match_pos &&
            candidate.match_bounds.second != no_label_match_pos) {
            return candidate;
        }
    }
    return std::nullopt;
}

/**
 * @brief Sorts the supplied series of label matches such, such that label
 *   matches that matched earlier in the matched-with string appear earlier
 *   in the sorted series.
 *
 * @param matches The possibly unsorted series of label matches.
 */
void DutchKBQADSCreate::LabelMatch::sorted_label_matches(std::vector<LabelMatch> &matches) {
    std::sort(matches.begin(), matches.end(), LabelMatch::appears_earlier_in_string);
}

/**
 * @brief Determines whether collisions exist within the series of label
 *   matches `matches`.
 *
 * @param matches The series of label matches.
 * @return The question's answer.
 */
bool DutchKBQADSCreate::LabelMatch::collision_present_in_label_matches(std::vector<LabelMatch> matches) {
    LabelMatch::sorted_label_matches(matches);
    if (matches.size() < 2) {
        return false;  /* No collisions possible. */
    }
    for (std::size_t idx = 0; idx < (matches.size() - 1); ++idx) {
        if (matches[idx].match_bounds.second >= matches[idx + 1].match_bounds.first) {
            return true;
        }
    }
    return false;
}

/**
 * @brief A suffix to append to the base 'translated questions' file name in
 *   order to get the version that has various artifacts removed.
 */
const std::string variant_suffix = "replaced-no-errors";

/**
 * @brief Returns the translated questions as a JSON object loaded from disk.
 *
 * @param split The LC-QuAD 2.0 dataset split to target.
 * @param language The natural language in which the questions must have been
 *   translated.
 * @return The translated questions as a JSON object.
 */
Json::Value translated_questions_json(const LCQuADSplit &split,
                                      const NaturalLanguage &language) {
    std::string file_name = string_from_lc_quad_split(split) +
                            "-" +
                            string_from_natural_language(language) +
                            "-" +
                            variant_suffix;
    Json::Value json = json_loaded_from_dataset_file(file_name);
    return json;
}

/**
 * @brief Returns the original LC-QuAD 2.0 dataset split (which includes both
 *   questions and answers) as a JSON object loaded from disk.
 *
 * @param split The LC-QuAD 2.0 dataset split to target.
 * @return The translated questions as a JSON object.
 */
Json::Value original_questions_and_answers_json(const LCQuADSplit &split) {
    std::string file_name = string_from_lc_quad_split(split) +
                            "-" +
                            string_from_natural_language(NaturalLanguage::ENGLISH);
    Json::Value json = json_loaded_from_dataset_file(file_name);
    return json;
}

/**
 * @brief Returns the question-answer pairs of the requested LC-QuAD 2.0
 *   dataset split and natural language.
 *
 * The questions are formulated in `language`; the answers are those from the
 * `sparql_wikidata` field of the original LC-QuAD 2.0 dataset.
 *
 * @param split The LC-QuAD 2.0 dataset split.
 * @param language The natural language.
 * @return A series of question-answer pairs.
 */
std::vector<QuestionAnswerPair> DutchKBQADSCreate::question_answer_pairs(const LCQuADSplit &split,
                                                                         const NaturalLanguage &language) {
    std::vector<QuestionAnswerPair> pairs;
    const Json::Value trl_q = translated_questions_json(split, language);
    const Json::Value ori_qa = original_questions_and_answers_json(split);
    for (const auto &qa : ori_qa) {
        int uid = qa["uid"].asInt();
        const std::string uid_str = std::to_string(uid);
        pairs.emplace_back(QuestionAnswerPair(uid,
                                              trl_q[uid_str].asString(),
                                              qa["sparql_wikidata"].asString()));
    }
    return pairs;
}

/**
 * @brief Returns the label to use for this combination of question and entity
 *   or property, or null if no appropriate label can be found.
 *
 * @param question The question.
 * @param ent_or_prp The entity or property.
 * @param labels The labels for `ent_or_prp`.
 * @param map A mapping from entities and properties to labels. The entities
 *   and properties that previously have already received a label.
 * @return The label to use, or a null value if no appropriate label could be
 *   found.
 */
ent_or_prp_chosen_label DutchKBQADSCreate::selected_label_for_entity_or_property(const std::string &question,
                                                                                 const std::string &ent_or_prp,
                                                                                 const std::vector<std::string> &labels,
                                                                                 const ent_prp_chosen_label_map &map) {
    if (labels.empty()) {
        /* No labels exist for this entity or property. Skip. */
        return std::nullopt;
    }
    std::vector<LabelMatch> label_matches;
    for (const auto &label : labels) {
        std::optional<index_range> match_bounds = LabelMatch::match_label_in_sentence(label, question);
        if (!match_bounds.has_value()) {
            std::pair<int, int> empty_match_bounds = { no_label_match_pos, no_label_match_pos };
            label_matches.emplace_back(label, empty_match_bounds, ent_or_prp);
        } else {
            label_matches.emplace_back(label, match_bounds.value(), ent_or_prp);
        }
    }
    std::optional<LabelMatch> best = LabelMatch::best_label_match(label_matches);
    if (best.has_value()) {
        return std::pair<std::string, LabelMatch>(ent_or_prp, best.value());
    } else {
        return std::nullopt;
    }
}

/**
 * @brief Returns selected (substrings of) labels for each entity and property
 *   present in a questions, or null if one or more entities or properties
 *   could not be assigned an appropriate label.
 *
 * @param question The question.
 * @param entities_properties The question's entities and properties.
 * @param ent_prp_labels A mapping from entities and properties to zero or more
 *   labels.
 * @return For each entity and property of the question, a single (substring of a)
 *   label, representing the selected label. Null is returned if one or more
 *   entities or properties could not be assigned a satisfactory label.
 */
ent_prp_chosen_label_map DutchKBQADSCreate::selected_labels_for_entities_and_properties(
        const std::string &question,
        const std::set<std::string> &entities_properties,
        const DutchKBQADSCreate::ent_prp_label_map &ent_prp_labels) {
    ent_prp_chosen_label_map map = std::map<std::string, LabelMatch>();
    for (const auto &ent_or_prp : entities_properties) {
        /* Try to associate entities and properties to appropriate labels. */
        const std::vector<std::string> &ent_or_prp_labels = ent_prp_labels.at(ent_or_prp);
        ent_or_prp_chosen_label label = selected_label_for_entity_or_property(question,
                                                                              ent_or_prp,
                                                                              ent_or_prp_labels,
                                                                              map);
        if (label.has_value()) {
            map->insert(label.value());
        } else {
            /* If even one entity or property cannot be label-matched, then we
             * discard this question. Thus, return null for the complete
             * question. */
            return std::nullopt;
        }
    }
    /* All entities and properties meant to be included have gotten an
     * appropriate label assigned to them. Return. */
    return map;
}

/**
 * @brief Masks a single label within the supplied question, given the current
 *   states of the entity- and property counters and the already-existent
 *   mask names for entities and properties masked earlier on in the masking
 *   process.
 *
 * @param q The question in which to replace.
 * @param match The label match to replace for.
 * @param ent_counter The entity counter.
 * @param prp_counter The property counter.
 * @param mask_map A mapping from entities and properties to mask names (or
 *   simply "masks"). Used to determine whether the current label match already
 *   has a mask name assigned to it.
 */
void DutchKBQADSCreate::mask_single_entity_or_property_in_question(std::string &q,
                                                                   const LabelMatch &match,
                                                                   int &ent_counter,
                                                                   int &prp_counter,
                                                                   ent_prp_mask_map &mask_map) {
    std::string replacement;
    const auto potential_replacement = mask_map.find(match.ent_or_prp);
    if (potential_replacement == mask_map.end()) {
        /* No already-existing mask name. Create a new one. */
        switch (wiki_data_symbol_for_entity_or_property(match.ent_or_prp)) {
            case WikiDataSymbol::ENTITY:
                replacement = std::string("Q") + std::to_string(ent_counter);
                ent_counter++;
                break;
            case WikiDataSymbol::PROPERTY:
                replacement = std::string("P") + std::to_string(prp_counter);
                prp_counter++;
                break;
        }
        mask_map.insert({ match.ent_or_prp, replacement });
    } else {
        /* Mask name already exists. Use it. */
        replacement = potential_replacement->second;
    }
    q = std::regex_replace(q, std::regex("(" + match.label + ")"), replacement);
}

/**
 * @brief Masks a single label within the supplied answer, given the current
 *   state of the already-existent mask names for entities and properties
 *   masked earlier on in the masking process.
 *
 * @param q The question in which to replace.
 * @param match The label match to replace for.
 * @param mask_map A mapping from entities and properties to mask names (or
 *   simply "masks"). Used to determine whether the current label match already
 *   has a mask name assigned to it.
 */
void DutchKBQADSCreate::mask_single_entity_or_property_in_answer(std::string &a,
                                                                 const LabelMatch &match,
                                                                 ent_prp_mask_map &mask_map) {
    const auto potential_replacement = mask_map.find(match.ent_or_prp);
    if (potential_replacement == mask_map.end()) {
        throw std::runtime_error(std::string("Logical error: ") +
                                 "mask map is missing for \"" +
                                 match.ent_or_prp +
                                 "\" (" +
                                 match.label +
                                 ")!");
    }
    a = std::regex_replace(a, std::regex("(" + match.ent_or_prp + ")"), potential_replacement->second);
}

/**
 * @brief Masks a single question-answer pair.
 *
 * @param qa_pair The question-answer pair to mask.
 * @param entities_properties The unique entities and properties present within
 *   this question-answer pair.
 * @param ent_prp_labels A mapping from entities and properties to labels.
 * @return The masked equivalent of `qa_pair`.
 */
std::optional<QuestionAnswerPair> DutchKBQADSCreate::masked_question_answer_pair(
        const QuestionAnswerPair &qa_pair,
        const std::set<std::string> &entities_properties,
        const ent_prp_label_map &ent_prp_labels) {
    ent_prp_chosen_label_map labels_map = selected_labels_for_entities_and_properties(qa_pair.q,
                                                                                      entities_properties,
                                                                                      ent_prp_labels);
    if (!labels_map.has_value()) {
        /* One or more entities and/or properties haven't gotten an appropriate
         * label assigned to them; masking cannot be performed. */
        return std::nullopt;
    }
    std::vector<LabelMatch> label_matches;
    for (const auto &ent_or_prp : entities_properties) {
        label_matches.push_back(labels_map.value().at(ent_or_prp));
    }
    if (LabelMatch::collision_present_in_label_matches(label_matches)) {
        return std::nullopt;
    }
    std::string replaced_q = qa_pair.q;
    std::string replaced_a = qa_pair.a;
    int ent_counter = 1;
    int prp_counter = 1;
    ent_prp_mask_map mask_map;
    for (const auto &label_match : label_matches) {
        mask_single_entity_or_property_in_question(replaced_q,
                                                   label_match,
                                                   ent_counter,
                                                   prp_counter,
                                                   mask_map);
        mask_single_entity_or_property_in_answer(replaced_a, label_match, mask_map);
    }
    return QuestionAnswerPair(qa_pair.uid, replaced_q, replaced_a);
}

/**
 * @brief Masks all question-answer pairs present in the LC-QuAD 2.0 dataset
 *   split-natural language pair and returns the results as a JSON object.
 *
 * @param split The LC-QuAD 2.0 dataset split to target.
 * @param language The natural language to target. Is the natural language of
 *   the translation, not that of the original LC-QuAD 2.0 dataset.
 * @param quiet Whether to report on progress (`false`) or not (`true`).
 * @return The masked question-answer pairs as a JSON object.
 */
Json::Value DutchKBQADSCreate::masked_question_answer_pairs(const LCQuADSplit &split,
                                                            const NaturalLanguage &language,
                                                            bool quiet) {
    Json::Value json;
    const std::vector<QuestionAnswerPair> qa_pairs = question_answer_pairs(split, language);
    const q_ent_prp_map questions_entities_properties = loaded_question_entities_properties_map(split);
    const ent_prp_label_map ent_prp_labels = loaded_entity_and_property_labels(split, language);
    std::size_t counter = 0;
    for (const auto &qa_pair : qa_pairs) {
        const std::set<std::string> &question_entities_properties = questions_entities_properties.at(qa_pair.uid);
        const ent_prp_label_map q_ent_prp_labels = entity_and_property_labels_subset(question_entities_properties,
                                                                                     ent_prp_labels);
        const std::optional<QuestionAnswerPair> masked = masked_question_answer_pair(qa_pair,
                                                                                     question_entities_properties,
                                                                                     q_ent_prp_labels);
        if (masked.has_value()) {
            Json::Value json_masked_qa_pair;
            json_masked_qa_pair["q"] = masked.value().q;
            json_masked_qa_pair["a"] = masked.value().a;
            json[std::to_string(qa_pair.uid)] = json_masked_qa_pair;
        }
        if (!quiet) {
            printf("\rMasking question-answer pairs... (%6.2lf%%)",
                   (static_cast<double>(counter) / static_cast<double>(qa_pairs.size())) * 100.);
            std::cout << std::flush;
        }
        counter++;
    }
    if (!quiet) {
        std::cout << std::endl << "Done." << std::endl;
    }
    return json;
}

/**
 * @brief Saves the masked question-answer pairs to disk.
 *
 * @param json A JSON object in which the keys are UIDs and the values are
 *   sub-objects containing the translated questions and the original
 *   SPARQL WikiData answer queries; both have their entities and properties
 *   masked.
 * @param split The LC-QuAD 2.0 dataset split of which `json` stores the labels.
 * @param language The natural language of the labels of `json`.
 */
void DutchKBQADSCreate::save_masked_question_answer_pairs_json(const Json::Value &json,
                                                               const LCQuADSplit &split,
                                                               const NaturalLanguage &language) {
    const std::string file_name = string_from_lc_quad_split(split) +
                                  "-" +
                                  string_from_natural_language(language) +
                                  "-" +
                                  "replaced-no-errors-masked";
    save_json_to_dataset_file(json, file_name);
}

/**
 * @brief Masks entities and properties in translated question +
 *   original-language answer pairs of an LC-QuAD 2.0 dataset split, and
 *   saves these masked pairs to disk.
 *
 * @param vm The variables map with which to determine which dataset split
 *   and translation natural language to use in the masking operation.
 */
void DutchKBQADSCreate::mask_question_answer_pairs(const po::variables_map &vm) {
    const std::vector<std::string> required_flags = { "split",
                                                      "language",
                                                      "quiet" };
    for (const auto &required_flag : required_flags) {
        if (vm.count(required_flag) == 0) {
            throw std::invalid_argument(std::string("The \"--") +
                                        required_flag +
                                        "\" flag is required.");
        }
    }
    const LCQuADSplit split = string_to_lc_quad_split_map.at(vm["split"].as<std::string>());
    const NaturalLanguage language = string_to_natural_language_map.at(vm["language"].as<std::string>());
    const bool quiet = vm["quiet"].as<bool>();
    Json::Value json = masked_question_answer_pairs(split, language, quiet);
    std::cout << "Saving... ";
    save_masked_question_answer_pairs_json(json, split, language);
    std::cout << "Done." << std::endl;
}
