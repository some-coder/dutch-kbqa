/* Symbols for constructing Ukkonen suffix trees (header). */

#ifndef SUFFIX_TREE_HPP
#define SUFFIX_TREE_HPP

#include "explicit-state.hpp"
#include "unicode-string.hpp"

namespace DutchKBQADSCreate::SuffixTrees {
    /**
     * @brief A pair consisting of a left- and right-pointer. It's 'explicit'
     *   because the right index is not a pointer to an index, but simply the
     *   index itself.
     *
     * Contrast with `explicit-state.hpp`'s `left_right_pointer_pair`.
     */
    using explicit_left_right_pointer_pair = std::pair<int, int>;
    /**
     * @brief A canonised reference pair. Only the left-pointer is given.
     *
     * For information on what 'canonisation' means in the context of Ukkonen
     * suffix trees, see page 253 of Ukkonen (1995) and documentation on
     * canonisation of `ReferencePair`s and `SuffixTree`s.
     */
    using canon_reference_pair = std::pair<ExplicitState*, int>;

    /**
     * @brief A pair consisting of an explicit state in a Ukkonen suffix tree
     *   and a substring of the UTF32-encoded Unicode string on which said
     *   suffix tree is based, which represents the path spelled out from the
     *   explicit state to some child explicit state.
     *
     * See page 253 of Ukkonen (1995). There, reference pairs are introduced as
     * pairs `(s, w)`. The substring `w` is not stored as-is, but is rather
     * referred to by means of a starting and ending pointer pair, the familiar
     * `(k, p)` pairs also seen in `explicit-state.hpp`.
     */
    class ReferencePair {
    public:
        /**
         * @brief The explicit state of the reference pair.
         */
        ExplicitState *state;
        /**
         * @brief The left pointer into a UTF32-encoded Unicode string,
         *   representing the index of the first character of `w` (inclusive).
         */
        int left_ptr;
        /**
         * @brief The right pointer into a UTF32-encoded Unicode string,
         *   representing the index of the last character of `w` (inclusive).
         */
        int right_ptr;
        explicit ReferencePair(ExplicitState *state, explicit_left_right_pointer_pair pair);
        canon_reference_pair canonised(UnicodeString uni_str);
    };

    /**
     * @brief A Ukkonen suffix tree (Ukkonen, 1995).
     *
     * The name is derived not from the tree itself, but from the way it is
     * constructed: it's done linearly with respect to the length of the
     * string.
     */
    class SuffixTree {
    private:
        /**
         * @brief The UTF32-encoded Unicode string on which this Ukkonen suffix
         *   tree is based.
         */
        UnicodeString uni_str;
        /**
         * @brief The auxiliary state of this Ukkonen suffix tree.
         */
        std::unique_ptr<ExplicitState> auxiliary;
        /**
         * @brief A right pointer (inclusive) into the UTF32-encoded Unicode
         *   string on which this suffix tree is based. Used to grow the leaves
         *   of this tree in constant time during suffix tree construction.
         */
        std::unique_ptr<int> leaf_right_ptr;
    public:
        /**
         * @brief The root of this Ukkonen suffix tree.
         *
         * Contrast with `auxiliary`, the auxiliary state of the tree.
         */
        ExplicitState *root;
        explicit SuffixTree(const std::string &str);
        std::pair<bool, ExplicitState*> test_and_split(ReferencePair pair, utf8::uint32_t code_point);
        canon_reference_pair update(ReferencePair pair);
        void construct();
        void print();
    };
}

#endif  /* SUFFIX_TREE_HPP */

