/* Symbols for modelling 'explicit states' in Ukkonen's suffix tree algorithm. */

#include <iostream>
#include <algorithm>
#include "suffix-trees/explicit-state.hpp"

using namespace DutchKBQADSCreate::SuffixTrees;

int ExplicitState::explicit_states_id_counter = 0;

/**
 * @brief Constructs a new explicit state for a Ukkonen suffix tree.
 *
 * @param parent The parent state in the tree from which this explicit state
 *   descends.
 */
ExplicitState::ExplicitState(ExplicitState *parent) : id(explicit_states_id_counter) {
    ExplicitState::explicit_states_id_counter++;
    this->parent = parent;
    this->transitions = state_transitions();
    this->suffix_link = nullptr;
}

/**
 * @brief Destructs an explicit state for a Ukkonen suffix tree.
 */
ExplicitState::~ExplicitState() = default;

/**
 * @brief Returns the ID of this explicit state.
 *
 * @return The ID.
 */
int ExplicitState::get_id() const {
    return this->id;
}

/**
 * @brief Adds a transition from this explicit state to some descendant `child`
 *   explicit state.
 *
 * @param uni_str The UTF32-encoded Unicode string which the Ukkonen suffix
 *   tree (and, consequently, also this explicit state) is based on.
 * @param left_ptr The left pointer for the transition.
 * @param right_ptr The right pointer for the transition.
 * @param child The child explicit state; the destination of the transition.
 */
void ExplicitState::set_transition(UnicodeString uni_str,
                                   int left_ptr,
                                   right_pointer right_ptr,
                                   std::unique_ptr<ExplicitState> child) {
    utf8::uint32_t code_point = uni_str.code_point_at(left_ptr - 1);  /* 1- to 0-based indexing. */
    if (this->transitions.count(code_point) > 0) {
        throw std::logic_error("You're trying to overwrite an existing transition!");
    }
    this->transitions.insert(
        std::pair<utf8::uint32_t, state_transition>(
            code_point,
            state_transition(left_right_pointer_pair(left_ptr, std::move(right_ptr)),
                             std::move(child))
        )
    );
}

/**
 * @brief Sets this explicit state's connection to another explicit state,
 *   which represents the same substring as this explicit state, except that
 *   the first code point has been removed. That is, it is a 'minimal suffix'.
 *
 * @param next_on_path The explicit state that represents a minimal suffix of
 *   the current state's substring. 'Path' here refers to the boundary path,
 *   discussed in Ukkonen (1995), page 252.
 */
void ExplicitState::set_suffix_link(ExplicitState *next_on_path) {
    this->suffix_link = next_on_path;
}

/**
 * @brief Returns the suffix link of this explicit state.
 *
 * @return The suffix link. If this state does not have a suffix link (yet),
 *   then `nullptr` is returned.
 */
ExplicitState* ExplicitState::get_suffix_link() {
    return this->suffix_link;
}

/**
 * @brief Determines whether this explicit state has a `code_point`-transition.
 *
 * @param code_point The code point to check for the existence of a transition
 *   for.
 * @return The question's answer.
 */
bool ExplicitState::has_transition(utf8::uint32_t code_point) {
    return this->transitions.count(code_point) > 0;
}

/**
 * @brief Returns the requested explicit state `code_point`-transition in a
 *   'weak', C-style pointer form.
 *
 * @param code_point The code point that forms the start of the substring of
 *   the transition.
 * @return The weak state transition.
 */
weak_state_transition ExplicitState::weakly_get_transition(utf8::uint32_t code_point) {
    int *right_ptr;
    if (std::holds_alternative<std::unique_ptr<int>>(this->transitions[code_point].first.second)) {
        /* The right pointer is a unique pointer (a 'strong' or smart pointer). */
        right_ptr = std::get<std::unique_ptr<int>>(this->transitions[code_point].first.second).get();
    } else {
        /* The right pointer is a raw pointer (a 'weak' or C-style pointer). */
        right_ptr = std::get<int*>(this->transitions[code_point].first.second);
    }
    ExplicitState *es_ptr = this->transitions[code_point].second.get();
    return { std::pair<int, int*>(this->transitions[code_point].first.first, right_ptr), es_ptr };
}

/**
 * @brief Breaks up the direct transition from this explicit state (`s`) to
 *   another explicit state (`s'`) by introducing `r`, thus yielding two
 *   transitions: `s` to `r`, and `r` to `s'`.
 *
 * For more information on this algorithm, see lines 2 up until 6 of the
 * 'test-and-split' algorithm of Ukkonen (1995).
 *
 * @param uni_str The UTF32-encoded Unicode string on which the Ukkonen suffix
 *   tree is based (and of which this explicit state is a part).
 * @param left_ptr The left pointer of the transition to `s'`.
 * @param right_ptr The right pointer of the transition to `s'`.
 * @return A 'weak' (C-style) pointer to the newly-created explicit state, `r`.
 */
ExplicitState *ExplicitState::internal_split(UnicodeString uni_str,
                                             int left_ptr,
                                             right_pointer right_ptr) {
    int k, p, k_prime;
    k = left_ptr;

    /* (1/4) Get the old `t_k`-transition from this explicit state (called `s`) to `s'`.*/
    utf8::uint32_t t_k = uni_str.code_point_at(k - 1);  /* 1- to 0-based indexing. */
    state_transition s_prime_transition = std::move(this->transitions[t_k]);

    /* (2/4) Create a new, intermediate transition destination for `s`, called `r`. */
    k_prime = s_prime_transition.first.first;
    if (std::holds_alternative<std::unique_ptr<int>>(right_ptr)) {
        p = *(std::get<std::unique_ptr<int>>(right_ptr));
    } else {
        p = *(std::get<int*>(right_ptr));
    }
    std::unique_ptr<ExplicitState> r = std::make_unique<ExplicitState>(this);

    /* (3/4) Link `r` to `s'`. */
    r->set_transition(uni_str,
                      k_prime + p - k + 1,
                      std::move(s_prime_transition.first.second),
                      std::move(s_prime_transition.second));

    /* (4/4) Overwrite the old connection from `s` to `s'` by now transitioning to `r` instead. */
    this->transitions.erase(t_k);
    this->set_transition(uni_str,
                         k_prime,
                         right_pointer(std::make_unique<int>(k_prime + p - k)),
                         std::move(r));

    return this->transitions[t_k].second.get();
}

/**
 * @brief Returns the state transition beginning with symbol `code_point`, if
 *   it exists.
 *
 * @param code_point The first symbol of the transition's substring.
 * @return The transition, if it exists. Otherwise, a null value is returned.
 */
std::optional<ExplicitState*> ExplicitState::state_transition_if_present(utf8::uint32_t code_point) {
    if (this->transitions.count(code_point) == 0) {
        return nullptr;
    } else {
        return this->weakly_get_transition(code_point).second;
    }
}

/**
 * @brief Returns an iterator over this explicit state's transitions, beginning
 *   at the first transition.
 *
 * @return The iterator.
 */
state_transitions::iterator ExplicitState::transitions_start() {
    return this->transitions.begin();
}

/**
 * @brief Returns an iterator marking the end of this explicit state's
 *   transitions.
 *
 * @return The iterator.
 */
state_transitions::iterator ExplicitState::transitions_end() {
    return this->transitions.end();
}

/**
 * Represents this explicit state as a string.
 *
 * @return The string representation.
 */
std::string ExplicitState::as_string() const {
    return std::string("ExplicitState(") +
           std::to_string(this->get_id()) +
           ")";
}

const std::string single_indent = "  ";  /* Two spaces, not a tab. */

/**
 * @brief Returns a string with a `number` level of indents.
 *
 * @param number The number of indentations to apply. A strictly non-negative
 *   integer.
 * @return The indentation string.
 */
std::string indent_string(int number) {
    std::string indent_str;
    for (int i = 0; i < number; i++) {
        indent_str.append(single_indent);
    }
    return indent_str;
}

/**
 * @brief Prints this explicit state to standard output.
 *
 * @param uni_str The UTF32-encoded Unicode string on which this explicit state
 *   is based.
 * @param num_indents The number of indents to apply during formatting. A
 *   strictly non-negative integer.
 */
void ExplicitState::print(UnicodeString uni_str, int num_indents) {
    const std::string indent_str = indent_string(num_indents);
    std::cout << indent_str << this->as_string() << std::endl;
    for (const auto &transition : this->transitions) {
        int trn_left_ptr, *trn_right_ptr;
        trn_left_ptr = transition.second.first.first - 1;  /* Indices cancel out: -1 + 1 = 0. */
        if (std::holds_alternative<std::unique_ptr<int>>(transition.second.first.second)) {
            /* Transition destination is encoded as a smart, C++-style pointer. */
            trn_right_ptr = std::get<std::unique_ptr<int>>(transition.second.first.second).get();
        } else {
            /* Transition destination is encoded as a weak, C-style pointer. */
            trn_right_ptr = std::get<int*>(transition.second.first.second);
        }
        UnicodeString trn_substr = uni_str.substring(trn_left_ptr, *trn_right_ptr);
        std::cout << indent_str << single_indent;
        std::cout << "(" << transition.second.first.first << ", " << *trn_right_ptr << ") ";
        std::cout << "(" << UnicodeString::basic_string_from_unicode_string(trn_substr) << ") ";
        std::cout << transition.second.second->as_string() << std::endl;
        transition.second.second->print(uni_str, num_indents + 1);
    }
}

/**
 * @brief Constructs a new auxiliary state for a Ukkonen suffix tree.
 *
 * @param uni_str The UTF32-encoded Unicode string on which the tree is based.
 */
AuxiliaryState::AuxiliaryState(UnicodeString uni_str) : ExplicitState(nullptr) {
    this->root = std::make_unique<ExplicitState>(this);
    int j = -1;
    for (int idx = 0; idx < -static_cast<int>(uni_str.length); idx++) {
        const utf8::uint32_t cp = uni_str.code_point_at(idx);
        this->left_right_pointer_pair_integers_per_code_point[cp] = { cp, std::make_unique<int>(j) };
        j--;
    }
    this->root->set_suffix_link(this);
}

/**
 * @brief Destructs an auxiliary state of a Ukkonen suffix tree.
 */
AuxiliaryState::~AuxiliaryState() = default;

/**
 * @brief
 *
 * @param code_point
 * @return
 */
int *AuxiliaryState::weak_pointer_for_code_point(utf8::uint32_t code_point) {
    for (const auto &cp_pointer_pair : this->left_right_pointer_pair_integers_per_code_point) {
        if (cp_pointer_pair.first == code_point) {
            return cp_pointer_pair.second.get();
        }
    }
    return nullptr;
}

/**
 * @brief Determines whether the auxiliary state has a transition starting with
 *   the supplied code point.
 *
 * @param code_point The code point to check for.
 * @return The question's answer.
 */
bool AuxiliaryState::has_transition(utf8::uint32_t code_point) {
    return this->weak_pointer_for_code_point(code_point) != nullptr;
}

/**
 * @brief Returns the requested explicit state `code_point`-transition in a
 *   'weak', C-style pointer form, departing from the auxiliary state.
 *
 * @param code_point The code point that forms the start of the substring of
 *   the transition.
 * @return The weak state transition.
 */
weak_state_transition AuxiliaryState::weakly_get_transition(utf8::uint32_t code_point) {
    int *pointer = this->weak_pointer_for_code_point(code_point);
    if (pointer == nullptr) {
        throw std::logic_error("Encountered null pointer in getting transition. That's invalid!");
    }
    return weak_state_transition({ left_right_pointer_pair(*pointer, pointer),
                                   this->root.get() });
}

/**
 * @brief Returns the state transition beginning with symbol `code_point`, if
 *   it exists.
 *
 * @param code_point The first symbol of the transition's substring.
 * @return The transition, if it exists. Otherwise, a null value is returned.
 */
std::optional<ExplicitState*> AuxiliaryState::state_transition_if_present(utf8::uint32_t code_point) {
    if (this->has_transition(code_point)) {
        return this->weakly_get_transition(code_point).second;
    } else {
        return nullptr;
    }
}

/**
 * @brief Prints this auxiliary state to standard output.
 *
 * @param uni_str The UTF32-encoded Unicode string on which this auxiliary state
 *   is based.
 * @param num_indents The number of indents to apply during formatting. A
 *   strictly non-negative integer.
 */
void AuxiliaryState::print(UnicodeString uni_str, int num_indents) {
    const std::string indent_str = indent_string(num_indents);
    std::cout << num_indents << this->as_string() << std::endl;
    for (const auto &pair : this->left_right_pointer_pair_integers_per_code_point) {
        std::cout << indent_str;
        std::cout << "(" << *(pair.second) << ", " << *(pair.second) << ") ";
        std::cout << "(" << UnicodeString::basic_string_from_unicode_code_point(pair.first) << ") ";
        std::cout << (this->root)->as_string() << std::endl;
    }
}
