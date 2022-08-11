/* Symbols for constructing Ukkonen suffix trees. */

#include <iostream>
#include "suffix-trees/suffix-tree.hpp"

using namespace DutchKBQADSCreate::SuffixTrees;

/**
 * @brief Constructs a reference pair.
 *
 * @param state The state, the first entry in the reference pair.
 * @param pair An explicit left- and right-pointer pair. Used to construct the
 *   second entry of the reference pair.
 */
ReferencePair::ReferencePair(ExplicitState *state, explicit_left_right_pointer_pair pair) {
    this->state = state;
    this->left_ptr = pair.first;
    this->right_ptr = pair.second;
}

/**
 * @brief Canonises a reference pair and returns the result.
 *
 * Canonisation ensures that this reference pair is canonical. A reference pair
 * is canonical when its explicit state is the closest ancestor of the child
 * explicit state pointed to.
 *
 * A trivial example: For every non-auxiliary and -root explicit state, there
 * exists a reference pair from the root to said state. However, in a majority
 * of situations, there exists some non-auxiliary and -root explicit state
 * that is 'closer' to the child explicit state, in the sense that the
 * length of its substring `w` (or, its `p - k + 1`) is strictly lower. This
 * method searches for the explicit state that minimises this notion of
 * 'closeness' while ensuring that the reference pair stores an explicit state
 * (and not an implicit one).
 *
 * See also page 257 of Ukkonen (1995), procedure `canonize`.
 *
 * @param uni_str The UTF32-encoded Unicode string on which this reference pair
 *   is based.
 * @return The canonised reference pair. Note: only the left pointer is
 *   returned for the reference pair's second entry.
 */
canon_reference_pair ReferencePair::canonised(UnicodeString uni_str) {
    ExplicitState *out_state;
    int out_left_ptr;
    if (this->right_ptr < this->left_ptr) {
        return { this->state, this->left_ptr };
    } else {
        /* Find the right state transition. */
        out_state = this->state;
        out_left_ptr = this->left_ptr;
        weak_state_transition transition = out_state->weakly_get_transition(uni_str.code_point_at(out_left_ptr - 1));
        int left_ptr_prime = transition.first.first;
        int right_ptr_prime;
        if (std::holds_alternative<std::unique_ptr<int>>(transition.first.second)) {
            right_ptr_prime = *std::get<std::unique_ptr<int>>(transition.first.second);
        } else {
            right_ptr_prime = *std::get<int*>(transition.first.second);
        }
        /* Keep walking down until we have a truly canonised reference pair. */
        while ((right_ptr_prime - left_ptr_prime) <= (this->right_ptr - out_left_ptr)) {
            out_left_ptr += right_ptr_prime - left_ptr_prime + 1;
            out_state = transition.second;
            if (out_left_ptr <= (this->right_ptr)) {
                transition = out_state->weakly_get_transition(uni_str.code_point_at(out_left_ptr - 1));
                left_ptr_prime = transition.first.first;
                if (std::holds_alternative<std::unique_ptr<int>>(transition.first.second)) {
                    right_ptr_prime = *std::get<std::unique_ptr<int>>(transition.first.second);
                } else {
                    right_ptr_prime = *std::get<int*>(transition.first.second);
                }
            }
        }
        return { out_state, out_left_ptr };
    }
}

/**
 * @brief Constructs a Ukkonen suffix tree.
 *
 * @param str A UTF32-encoded Unicode string from which to build the tree.
 */
SuffixTree::SuffixTree(const std::string &str) : uni_str(str) {
    this->uni_str = UnicodeString(str);
    this->auxiliary = std::make_unique<AuxiliaryState>(UnicodeString(str));
    this->root = this->auxiliary->state_transition_if_present(this->uni_str.code_point_at(0)).value();
    this->leaf_right_ptr = std::make_unique<int>(0);
}

/**
 * @brief Tests whether the provided canonical reference pair is an endpoint of
 *   the suffix tree. Importantly, the `pair`'s state is made explicit if not
 *   already so, and returned as second argument of the returned pair.
 *
 * See also page 256 of Ukkonen (1995), procedure `test-and-split`.
 *
 * @param pair The canonical reference pair.
 * @param code_point The code point that serves as the first symbol in the
 *   transition to check for.
 * @return A pair. The first entry contains the question's answer; the second
 *   entry contains the state of the reference pair.
 */
std::pair<bool, ExplicitState*> SuffixTree::test_and_split(ReferencePair pair, utf8::uint32_t code_point) {
    int left_ptr_prime;
    if (pair.left_ptr <= pair.right_ptr) {
        utf8::uint32_t left_cp = this->uni_str.code_point_at(pair.left_ptr - 1);
        weak_state_transition transition = pair.state->weakly_get_transition(left_cp);
        left_ptr_prime = transition.first.first;
        utf8::uint32_t next_cp = this->uni_str.code_point_at(left_ptr_prime + pair.right_ptr - pair.left_ptr);
        if (code_point == next_cp) {
            return { true, pair.state };  /* Line 3. */
        } else {
            right_pointer right_ptr = std::make_unique<int>(left_ptr_prime + pair.right_ptr - pair.left_ptr);
            ExplicitState *es = pair.state->internal_split(this->uni_str,
                                                           left_ptr_prime,
                                                           std::move(right_ptr));
            return { false, es };  /* Line 6. */
        }
    } else {
        return { pair.state->has_transition(code_point), pair.state };  /* Line 9. */
    }
}

/**
 * @brief Transforms this Ukkonen suffix tree into one that has the next
 *   code point of the source string included in it.
 *
 * That is, we move from `STree(T_{i - 1})` to `STree(T_{i})` with this method.
 * See also procedure `update` on page 256 of Ukkonen (1995).
 *
 * @param pair The reference pair that represents the 'active point' of the
 *   overarching suffix tree-generating procedure. See page 254 of Ukkonen
 *   (1995) for more.
 * @return A canonical reference pair. The new active point.
 */
canon_reference_pair SuffixTree::update(ReferencePair pair) {
    ExplicitState *state_s, *old_root, *r;
    utf8::uint32_t t_i = this->uni_str.code_point_at(pair.right_ptr - 1);
    int k;
    bool end_point;
    state_s = pair.state;
    k = pair.left_ptr;
    old_root = this->root;
    auto end_point_r = this->test_and_split(ReferencePair(state_s,
                                                          { k, pair.right_ptr - 1 }), t_i);
    end_point = end_point_r.first;
    r = end_point_r.second;
    while (!end_point) {
        right_pointer right_ptr = this->leaf_right_ptr.get();
        std::unique_ptr<ExplicitState> r_prime = std::make_unique<ExplicitState>(r);
        r->set_transition(this->uni_str,
                          pair.right_ptr,
                          std::move(right_ptr),
                          std::move(r_prime));
        if (old_root != this->root) {
            old_root->set_suffix_link(r);
        }
        old_root = r;
        auto s_k_pair = ReferencePair(state_s->get_suffix_link(),
                                      { k, pair.right_ptr - 1 }).canonised(this->uni_str);
        state_s = s_k_pair.first;
        k = s_k_pair.second;
        end_point_r = this->test_and_split(ReferencePair(state_s,
                                                         { k, pair.right_ptr - 1 }), t_i);
        end_point = end_point_r.first;
        r = end_point_r.second;
    }
    if (old_root != this->root) {
        old_root->set_suffix_link(state_s);
    }
    return { state_s, k };
}

/**
 * @brief Constructs a complete Ukkonen suffix tree from the UTF32-encoded
 *   string it was initialised with.
 *
 * See for more details 'algorithm 2' at page 257 of Ukkonen (1995).
 */
void SuffixTree::construct() {
    ExplicitState *state;
    int left_ptr, i;
    state = this->root;
    left_ptr = 1;
    i = 0;
    while ((i + 1) <= this->uni_str.length) {
        i++;
        (*(this->leaf_right_ptr))++;
        canon_reference_pair update_pair = this->update(ReferencePair(state, { left_ptr, i }));
        state = update_pair.first;
        left_ptr = update_pair.second;
        canon_reference_pair canon_pair = ReferencePair(state,
                                                        { left_ptr, i }).canonised(this->uni_str);
        state = canon_pair.first;
        left_ptr = canon_pair.second;
    }
}

/**
 * @brief Prints this Ukkonen suffix tree to standard output.
 */
void SuffixTree::print() {
    std::cout << "SUFFIX TREE" << std::endl;
    this->auxiliary->print(this->uni_str, 0);
    std::cout << std::endl;
    this->root->print(this->uni_str, 0);
}
