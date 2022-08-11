/* Symbols for obtaining longest common substrings between pairs of strings. */

#include <cassert>
#include "suffix-trees/longest-common-substring.hpp"
#include "suffix-trees/unicode-string.hpp"
#include "suffix-trees/suffix-tree.hpp"

using namespace DutchKBQADSCreate::SuffixTrees;

/**
 * @brief Determines whether this explicit state is a leaf of its Ukkonen
 *   suffix tree.
 *
 * @param es The explicit state to check.
 * @return The question's answer.
 */
bool DutchKBQADSCreate::SuffixTrees::is_leaf_state(ExplicitState *es) {
    return es->transitions_start() == es->transitions_end();
}

/**
 * @brief Determines what substring type the given explicit- and leaf state of
 *   some Ukkonen suffix tree belongs to.
 *
 * @param leaf_state_range The index range that the substring of the explicit
 *   state forms: its starting- and ending index into the string of the Ukkonen
 *   suffix tree.
 * @param sep_end_range The index range that the query string forms, together
 *   with its separator- and ending character. (These are commonly '#' and
 *   '$'.)
 * @return The substring type of the explicit- and leaf state.
 */
SubstringType DutchKBQADSCreate::SuffixTrees::leaf_state_substring_type(index_range leaf_state_range,
                                                                        index_range sep_end_range) {
    return leaf_state_range.first <= sep_end_range.first ? SubstringType::FIRST : SubstringType::SECOND;
}

/**
 * @brief Updates an explicit state's substring type while it is still being
 *   determined. (Thus the 'preliminary' in this method's name.)
 *
 * @param old_type The old classification of the explicit state.
 * @param child_type The new classification of the explicit state.
 * @return The revised substring type.
 */
SubstringType DutchKBQADSCreate::SuffixTrees::updated_preliminary_state_substring_type(SubstringType old_type,
                                                                                       SubstringType child_type) {
    switch (old_type) {
        case UNDETERMINED:
            /* 'Undetermined' substring types should always be overridden. */
            return child_type;
        case FIRST:
        case SECOND:
            /* If the substring type was `FIRST` or `SECOND`, and the same
             * child type is found, then leave the type as-is; otherwise
             * we should combine `FIRST` and `SECOND` into
             * `FIRST_AND_SECOND`. */
            return old_type != child_type ? FIRST_AND_SECOND : old_type;
        case FIRST_AND_SECOND:
            /* If the type already was `FIRST_AND_SECOND`, it cannot become any
             * other way. */
            return FIRST_AND_SECOND;
        default:
            throw std::logic_error(std::string("Reached a non-") +
                                   "implemented substring type case!");
    }
}

/**
 * @brief Determines the substring type that the given explicit state
 *   represents. The three types are: belongs to the first string only, belongs
 *   to the second string only, and belongs to both the first and second
 *   string.
 *
 * @param es The explicit state for which to determine the substring type.
 * @param length The length of the substring that `es` represents.
 * @param lcs_length The currently longest path length; the length of the
 *   longest common substring (LCS).
 * @param lcs_start_index The starting index of the currently longest path
 *   (and thereby that of the LCS).
 * @param sep_end_range The index range that the query string forms, together
 *   with its separator- and ending character. (These are commonly '#' and
 *   '$'.)
 * @return The substring type of explicit state `es`.
 * @warning This method is recursive. As such, if fed sufficiently large input,
 *   it may overflow the program's stack memory.
 */
SubstringType DutchKBQADSCreate::SuffixTrees::state_substring_type(ExplicitState *es,
                                                                   int length,
                                                                   int *lcs_length,
                                                                   int *lcs_start_index,
                                                                   index_range sep_end_range) {
    SubstringType es_type, child_type;
    int *right_ptr;
    es_type = SubstringType::UNDETERMINED;
    for (auto it = es->transitions_start(); it != es->transitions_end(); ++it) {
        /* Get the index of the right pointer. (Recall: the indices are inclusive.) */
        right_ptr = weak_int_ptr_from_variant(it->second.first.second);
        if (is_leaf_state(it->second.second.get())) {
            /* Base case. */
            child_type = leaf_state_substring_type({ it->second.first.first, *right_ptr },
                                                   sep_end_range);
        } else {
            /* Recursive case. */
            child_type = state_substring_type(it->second.second.get(),
                                              length + (*right_ptr - it->second.first.first + 1),
                                              lcs_length,
                                              lcs_start_index,
                                              sep_end_range);
        }
        es_type = updated_preliminary_state_substring_type(es_type, child_type);
        if ((es_type == FIRST_AND_SECOND) && (child_type == FIRST_AND_SECOND)) {
            int total_length = length + (*right_ptr - it->second.first.first + 1);
            if (*lcs_length < total_length) {
                *lcs_length = total_length;
                *lcs_start_index = *right_ptr - total_length + 1;
            }
        }
    }
    assert(es_type != UNDETERMINED);
    return es_type;
}

/**
 * @brief Searches for a separator-ending symbol combination that can be used
 *   to separate and terminate a string concatenation of `first` and `second`.
 *
 * To be able to do so, both symbols must not already occur in either string.
 *
 * @param first The first string.
 * @param second The second string.
 * @return The symbol combination if such a combination is found. Otherwise, a
 *   null value.
 */
std::optional<separator_end_pair> workable_separator_end_symbol_pair(const std::string &first,
                                                                     const std::string &second) {
    for (const auto &pair : separator_end_pairs) {
        if ((first.find(pair.first) == std::string::npos) &&
            (first.find(pair.second) == std::string::npos) &&
            (second.find(pair.first) == std::string::npos) &&
            (second.find(pair.second) == std::string::npos)) {
            return pair;
        }
    }
    return std::nullopt;
}

/**
 * @brief Determines what the longest common substring is between two strings
 *   `first` and `second`. If there is no commonality, a null value is returned.
 *
 * If multiple longest common substrings can be selected, this function returns
 * only the first one.
 *
 * @param first The first string.
 * @param second The second string.
 * @return The longest common substring of `first` and `second`. A null value
 *   is returned if `first` and `second` do not share any symbol.
 */
std::optional<std::string> DutchKBQADSCreate::SuffixTrees::longest_common_substring(const std::string &first,
                                                                                    const std::string &second) {
    std::optional<separator_end_pair> sep_end = workable_separator_end_symbol_pair(first, second);
    if (!sep_end.has_value()) {
        /* Early exit: Cannot start the Ukkonen suffix tree procedure. */
        return std::nullopt;
    }
    std::string concat = first + sep_end->first + second + sep_end->second;
    SuffixTree tree(concat);
    tree.construct();

    int max_length, substring_start_idx, sep_idx, end_idx;
    UnicodeString uni_concat(concat);
    max_length = 0;
    substring_start_idx = 0;
    sep_idx = uni_concat.index_of_code_point(sep_end->first).value();
    end_idx = uni_concat.index_of_code_point(sep_end->second).value();
    state_substring_type(tree.root,
                         0,
                         &max_length,
                         &substring_start_idx,
                         { sep_idx + 1, end_idx + 1 });
    if (max_length - 1 >= 0) {
        UnicodeString lcs = uni_concat.substring(substring_start_idx - 1,
                                                 substring_start_idx + max_length - 1);
        return UnicodeString::basic_string_from_unicode_string(lcs);
    } else {
        return std::nullopt;
    }
}
