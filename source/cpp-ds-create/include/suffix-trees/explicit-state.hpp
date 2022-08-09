/* Symbols for modelling 'explicit states' in Ukkonen's suffix tree algorithm (header). */

#ifndef EXPLICIT_STATE_HPP
#define EXPLICIT_STATE_HPP

#include <variant>
#include <memory>
#include <map>
#include "utf8.h"
#include "suffix-trees/unicode-string.hpp"

/* Forward-declare the class `ExplicitState` for use in various type definitions. */
namespace DutchKBQADSCreate::SuffixTrees {
    class ExplicitState;
}

namespace DutchKBQADSCreate::SuffixTrees {
    /**
     * @brief A right pointer in a Ukkonen suffix tree.
     *
     * For more information, see `left_right_pointer_pair`.
     */
    using right_pointer = std::variant<std::unique_ptr<int>, int*>;
    /**
     * @brief A left- with right pointer pair, pointing to the starting and
     *   ending index of a substring of a string represented by a Ukkonen
     *   suffix tree, respectively.
     *
     * Both the left and right pointers are inclusive. See page 253 of Ukkonen
     * (1995). There, the left pointer is `k`, and the right one `p`.
     *
     * The generalised transition function used in suffix trees doesn't make
     * single-symbol transitions like the 'regular' transition function of
     * suffix tries would do; it skips minimally one, but possibly multiple,
     * symbols, thus skipping a substring of the string represented in the
     * suffix tree. Copying and storing substrings all the time is inefficient.
     * Instead, we create 'views' into one copy of the string. Where the view
     * must start and end are established by the left and right pointer,
     * respectively.
     */
    using left_right_pointer_pair = std::pair<int, right_pointer>;
    /**
     * @brief A single generalised transition towards a pointed-to successor
     *   explicit state. The left-right pointer pair represents what substring
     *   is appended onto the departure state's string while transitioning.
     *   This method is 'weak' in the sense that you get a weak pointer to the
     *   successor explicit state.
     */
    using weak_state_transition = std::pair<left_right_pointer_pair , ExplicitState*>;
    /**
     * @brief A single generalised transition towards a pointed-to successor
     *   explicit state. The left-right pointer pair represents what substring
     *   is appended onto the departure state's string while transitioning.
     *   This method is 'not weak' ('strong') in the sense that you get a
     *   unique pointer to the successor explicit state.
     */
    using state_transition = std::pair<left_right_pointer_pair , std::unique_ptr<ExplicitState>>;
    /**
     * @brief The generalised transition function's transitions, limited to
     *   contain only transitions departing from this explicit state.
     *
     * This map stores 'a-transitions' as Ukkonen would call them (1995, p.
     * 253). (We could speak of '`code_point`-transitions', instead.) Put
     * simply, given the first symbol of a substring towards a next explicit
     * state, this map will yield you a pointer to said state. Only one such
     * state can exists per code point.
     */
    using state_transitions = std::map<utf8::uint32_t, state_transition>;

    /**
     * @brief A state in a Ukkonen suffix tree that is actually stored in
     *   memory, thereby making it 'explicit'.
     *
     * See the definitions of 'explicit state' and 'implicit state' on pages
     * 252 and 253 of Ukkonen (1995). We need to distinguish between ex- and
     * implicit states in Ukkonen suffix trees, as precisely the leaving-out of
     * certain 'mundane' (i.e. non-branching, non-leaf) states reduces the time
     * complexity of constructing the tree from quadratic to linear in the
     * number of 'symbols' of the string on which the tree is based. (In our
     * case, we always refer to 'code points' as symbols, as we deal with
     * UTF32-encoded strings.)
     */
    class ExplicitState {
        /**
         * @brief A simple 'global' counter that increments every time a new
         *   explicit state is created, ensuring each such state gets assigned
         *   a new, unique ID.
         */
        static int explicit_states_id_counter;
    private:
        /**
         * @brief A unique identifier for this explicit state.
         *
         * This field is not 'part of Ukkonen's suffix tree algorithm;
         * it's here purely for convenience.
         */
        const int id;
        /**
         * @brief A pointer to this state's parent state.
         *
         * It is important to keep the `parent` of a state distinct from its
         * `suffix_link`. The former refers to the state that program-wise
         * generated this state; the latter refers to a suffix of this state,
         * which may be somewhere completely else in the tree, layout-wise.
         */
        ExplicitState *parent;
        state_transitions transitions;
        /**
         * @brief A pointer to a state that represents this state, but with
         *   the first code point removed.
         *
         * By removing the first code point, we obtain a 'minimal suffix' of
         * this state: all last-most code points are included, except one
         * (i.e., the first code point). This 'link to a (minimal) suffix
         * state' explains the name.
         *
         * The suffix link is conceptually identical to two other notions:
         * (1) the output of the suffix function, if you supply this state to
         * it, and (2) 'failure transitions' mentioned in other papers. See
         * page 250 of Ukkonen (1995).
         */
        ExplicitState *suffix_link;
    public:
        explicit ExplicitState(ExplicitState *parent);
        virtual ~ExplicitState();
        void set_transition(UnicodeString uni_str,
                            int left_ptr,
                            right_pointer right_ptr,
                            std::unique_ptr<ExplicitState> child);
        void set_suffix_link(ExplicitState *next_on_path);
        ExplicitState *get_suffix_link();
        virtual bool has_transition(utf8::uint32_t code_point);
        virtual weak_state_transition weakly_get_transition(utf8::uint32_t code_point);
        ExplicitState *internal_split(UnicodeString uni_str,
                                      int left_ptr,
                                      right_pointer right_ptr);
        virtual std::optional<ExplicitState*> tee_transition(utf8::uint32_t code_point);
        state_transitions::iterator transitions_start();
        state_transitions::iterator transitions_end();
        friend std::ostream &operator<<(std::ostream &os, const ExplicitState &es);
        virtual void print(UnicodeString uni_str, int num_indents);
    };

    /**
     * @brief The auxiliary state, a special state in a Ukkonen suffix tree.
     *
     * The auxiliary state 'precedes' the suffix tree's root in the sense that,
     * if we apply the suffix function to the root, we end up in the auxiliary
     * state. Similarly, if we apply the transition function to the auxiliary
     * state, then, for any within-string code point, we end up in the root
     * state.
     *
     * The auxiliary state is introduced on page 250 of Ukkonen (1995); there,
     * it is represented by the symbol for orthogonality (`\\bot` in LaTeX).
     */
    class AuxiliaryState : public ExplicitState {
    private:
        /**
         * @brief A pointer to the suffix tree's root state.
         */
        std::unique_ptr<ExplicitState> root;
        /**
         * @brief A sequence that represents a one-to-one correspondence
         *   between the symbols (code points) of a string represented by the
         *   Ukkonen suffix tree and left (and right) pointers for transitions
         *   from the auxiliary state to the root state.
         *
         * The entries in this vector correspond to the `j`s on page 253 and
         * page 257, algorithm 2, line 2 of Ukkonen (1995).
         *
         * Note that Ukkonen requires that 'a-transitions' be unique: "Each `s`
         * [...] for each `a \\in \\sigma` (1995, p. 253). But what about
         * transitions from the auxiliary state? You could argue that
         * multiply-occurring symbols (code points) in the string should be
         * 'squashed'. Here, however, we leave them all distinct.
         */
        std::vector<std::unique_ptr<int>> left_right_pointer_pair_integers_per_code_point;
    public:
        explicit AuxiliaryState(UnicodeString uni_str);
        ~AuxiliaryState() override;
        bool has_transition(utf8::uint32_t code_point) override;
        weak_state_transition weakly_get_transition(utf8::uint32_t code_point) override;
        std::optional<ExplicitState*> tee_transition(utf8::uint32_t code_point) override;
        void print(UnicodeString uni_str, int num_indents) override;
    };
}

#endif  /* EXPLICIT_STATE_HPP */
