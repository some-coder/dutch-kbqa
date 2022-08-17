/* Symbols for masking question-answer pairs (header). */

#ifndef MASK_QUESTION_ANSWER_PAIRS_HPP
#define MASK_QUESTION_ANSWER_PAIRS_HPP

#include <boost/program_options.hpp>
#include <map>
#include <optional>
#include "tasks/collect-entities-properties.hpp"
#include "tasks/label-entities-properties.hpp"
#include "suffix-trees/longest-common-substring.hpp"
#include "utilities.hpp"

/* Forward-declare the `LabelMatch` structure for usage in type definitions. */
namespace DutchKBQADSCreate {
    struct LabelMatch;
}

namespace DutchKBQADSCreate {
    namespace po = boost::program_options;
    /**
     * @brief An entity or property associated with a (substring of a) label. If
     *   no label could be matched (see `selected_label_for_entity_or_property`
     *   below), a null value is stored instead.
     */
    using ent_or_prp_chosen_label = std::optional<std::pair<std::string, LabelMatch>>;
    /**
     * @brief A mapping from entities and properties to associated (substrings
     *   of) labels. If null is stored, this indicates that one or more
     *   entities or properties meant for inclusion could not be associated
     *   to an appropriate label. For situations in which this happens, see
     *   the function `selected_label_for_entity_or_property`.
     */
    using ent_prp_chosen_label_map = std::optional<std::map<std::string, LabelMatch>>;
    /**
     * @brief A mapping from entities and properties to masks for them within
     *   a to-be-masked question-answer pair.
     */
    using ent_prp_mask_map = std::map<std::string, std::string>;

    /**
     * @brief An LC-QuAD 2.0 question-answer pair. The question's natural
     *   language need not be the language used in LC-QuAD 2.0. Moreover,
     *   both the question and answer may have their entities and properties
     *   masked.
     */
    struct QuestionAnswerPair {
        int uid;
        const std::string q;
        const std::string a;

        QuestionAnswerPair(int uid, std::string question, std::string answer);
    };

    /**
     * @brief A special value indicating that no label match could be found.
     */
    const int no_label_match_pos = -1;

    /**
     * @brief The result of trying to match a label against a question.
     *   'Matching' here means: trying to find the largest possible
     *   commonly-shared substring between the label and question. This
     *   structure, which is the result of that match, stores various
     *   statistics of the match.
     */
    struct LabelMatch {
        /**
         * @brief The entity or property to which this label belongs to.
         */
        std::string ent_or_prp;
        /**
         * @brief The original, complete label, including portions that have
         *   been removed in finding the longest common substring.
         */
        std::string label;
        /**
         * @brief The index boundaries of this label match within the question.
         */
        index_range match_bounds;
        /* @brief The absolute number of label characters that got matched in
         *   the question.
         */
        std::string::size_type chars_matched;
        /* @brief The fraction of label characters that got matched in the
         *   question.
         */
        double fraction_matched;

        LabelMatch(const std::string &label,
                   const index_range &match_bounds,
                   const std::string &ent_or_prp);
        static std::optional<index_range> match_label_in_sentence(const std::string &label,
                                                                  const std::string &sentence);
        static bool appears_earlier_in_string(const LabelMatch &first, const LabelMatch &second);
        static std::optional<LabelMatch> best_label_match(const std::vector<LabelMatch> &matches);
        static void sorted_label_matches(std::vector<LabelMatch> &matches);
        static bool collision_present_in_label_matches(std::vector<LabelMatch> matches);
    };

    std::vector<DutchKBQADSCreate::QuestionAnswerPair> question_answer_pairs(const LCQuADSplit &split,
                                                                             const NaturalLanguage &language);
    DutchKBQADSCreate::ent_or_prp_chosen_label selected_label_for_entity_or_property(
        const std::string &question,
        const std::string &ent_or_prp,
        const std::vector<std::string> &labels,
        const ent_prp_chosen_label_map &map,
        double fraction_match_threshold
    );
    DutchKBQADSCreate::ent_prp_chosen_label_map selected_labels_for_entities_and_properties(
        const std::string &question,
        const std::set<std::string> &entities_properties,
        const ent_prp_label_map &ent_prp_labels,
        double fraction_match_threshold
    );
    void mask_single_entity_or_property_in_question(std::string &q,
                                                    const LabelMatch &match,
                                                    int &ent_counter,
                                                    int &prp_counter,
                                                    ent_prp_mask_map &mask_map);
    void mask_single_entity_or_property_in_answer(std::string &a,
                                                  const LabelMatch &match,
                                                  ent_prp_mask_map &mask_map);
    std::optional<DutchKBQADSCreate::QuestionAnswerPair> masked_question_answer_pair(
        const QuestionAnswerPair &qa_pair,
        const std::set<std::string> &entities_properties,
        const ent_prp_label_map &ent_prp_labels,
        double fraction_match_threshold
    );
    Json::Value masked_question_answer_pairs(const LCQuADSplit &split,
                                             const NaturalLanguage &language,
                                             double fraction_match_threshold,
                                             bool quiet);
    void save_masked_question_answer_pairs_json(const Json::Value &json,
                                                const LCQuADSplit &split,
                                                const NaturalLanguage &language);
    void mask_question_answer_pairs(const po::variables_map &vm);
}

#endif
