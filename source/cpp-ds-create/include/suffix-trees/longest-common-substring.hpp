/* Symbols for obtaining longest common substrings between pairs of strings (header). */

#ifndef LONGEST_COMMON_SUBSTRING_HPP
#define LONGEST_COMMON_SUBSTRING_HPP

#include "explicit-state.hpp"

namespace DutchKBQADSCreate::SuffixTrees {
    /**
     * @brief An index range. The first entry in this pair is the starting
     *   index; the second entry is the ending index. Both ends are inclusive.
     */
    using index_range = std::pair<int, int>;
    /**
     * @brief A character pair. The first entry stores a symbol to separate two
     *   strings; the second stores a symbol to terminate the second string
     *   with.
     */
    using separator_end_pair = std::pair<char, char>;

    /**
     * @brief A series of separator-end pairs. These symbol pairs are used to
     *   respectively separate and terminate two concatenated strings using
     *   two singular symbols.
     *
     * This series is needed, because one or both strings may already contain
     * the separator or ending symbol; both need to be unique to be able to
     * distinguish them.
     */
    const std::vector<separator_end_pair> separator_end_pairs {
        { '_', '*' },
        { '_', '$' },
        { '#', '$' },
        { '&', '~' }
    };
    /**
     * @brief A classification of a string as a certain substring with respect
     *   to two strings. It either is unclassified, or belongs to one (or both)
     *   of the latter two strings.
     */
    enum SubstringType {
        UNDETERMINED,  /* The state's substring has not yet been classified. */
        FIRST,  /* The state's substring belongs only to the first string. */
        SECOND,  /* The state's substring belongs only to the second string. */
        FIRST_AND_SECOND  /* The state's substring belongs to both the first and second string. */
    };

    bool is_leaf_state(ExplicitState *es);
    SubstringType leaf_state_substring_type(index_range leaf_state_range, index_range sep_end_range);
    SubstringType updated_preliminary_state_substring_type(SubstringType old_type, SubstringType child_type);
    SubstringType state_substring_type(ExplicitState *es,
                                       int length,
                                       int *lcs_length,
                                       int *lcs_start_index,
                                       index_range sep_end_range);
    std::optional<std::string> longest_common_substring(const std::string &first, const std::string &second);
}

#endif  /* LONGEST_COMMON_SUBSTRING_HPP */
