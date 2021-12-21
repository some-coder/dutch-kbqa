#include "suffix-tree/explicit-state.h"
#include <iostream>


int ExplicitState::ID_COUNTER = 0;


/**
 * Constructs an explicit state; implicit states are represented via reference pairs.
 *
 * @param parent The closest ancestor to this explicit state. It may be that the state has no parent.
 */
ExplicitState::ExplicitState(ExplicitState *parent) : id(ID_COUNTER) {
	ExplicitState::ID_COUNTER++;
	this->parent = parent;
	this->transitions =
			std::map<uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>>();
	this->suffix_link = nullptr;  /* TODO: Not root? */
}


ExplicitState::~ExplicitState() = default;


/**
 * Sets a transition from the current explicit state to another explicit state.
 *
 * @param uni_str The unicode string on which the explicit states are based.
 * @param left_ptr The left index into `uni_str`. Inclusive. The first index is 1 (not 0).
 * @param right_ptr The right index into `uni_str`. Inclusive. Is maximally (inclusively) the string's length.
 * @param child The child state to connect to.
 */
void ExplicitState::set_transition(
		UnicodeString uni_str,
		int left_ptr,
		RightPointer right_ptr,
		std::unique_ptr<ExplicitState> child) {
	utf8::uint32_t utf_32_char = uni_str.entry(left_ptr - 1);  /* 1-indexing to 0-indexing */
	if (this->transitions.count(utf_32_char) > 0) {
		throw std::logic_error("You're trying to overwrite an existing transition!");
	}
	this->transitions.insert(
		std::pair<utf8::uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>>(
			utf_32_char,
			std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>(
				std::pair<int, RightPointer>(left_ptr, std::move(right_ptr)),
				std::move(child)
			)
		)
	);
}


void ExplicitState::set_suffix_link(ExplicitState *next_on_path) {
	this->suffix_link = next_on_path;
}


ExplicitState* ExplicitState::get_suffix_link() {
	return this->suffix_link;
}


bool ExplicitState::has_transition(utf8::uint32_t letter) {
	return this->transitions.count(letter) > 0;
}


/**
 * Find the transition starting with `letter`, but without transferring memory ownership of the child.
 *
 * @param letter The letter with which the searched-for transition starts.
 * @return The weak-form transition.
 */
std::pair<std::pair<int, int*>, ExplicitState*> ExplicitState::weakly_get_transition(utf8::uint32_t letter) {
	int *right_ptr;
	if (std::holds_alternative<std::unique_ptr<int>>(this->transitions[letter].first.second)) {
		/* the right pointer is a unique pointer (a smart pointer) */
		right_ptr = std::get<std::unique_ptr<int>>(this->transitions[letter].first.second).get();
	} else {
		/* the right pointer is a raw pointer (a C-style pointer) */
		right_ptr = std::get<int*>(this->transitions[letter].first.second);
	}
	ExplicitState *es_ptr = this->transitions[letter].second.get();
	return { std::pair<int, int*>(this->transitions[letter].first.first, right_ptr), es_ptr };
}


/**
 * Splits the transition from this state to another state by introducing an intermediate state in between them.
 *
 * @param uni_str The unicode string on which the explicit state is based.
 * @param left_ptr The left, inclusive range into the `uni_str`.
 * @param right_ptr The right, inclusive range into the `uni_str`.
 * @return A raw pointer to the newly created, intermediate state.
 */
ExplicitState *ExplicitState::internal_split(UnicodeString uni_str, int left_ptr, RightPointer right_ptr) {
	int k, p, k_prime;
	k = left_ptr;

	/* get old transition destination of `s` (this explicit state), `s'` */
	utf8::uint32_t t_k = uni_str.entry(k - 1);  /* 1-indexing to 0-indexing */
	std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>> s_prime_transition =
		std::move(this->transitions[t_k]);
//	this->transitions.erase(t_k);  /* TODO: Necessary? */

	/* create new, intermediate transition destination for `s`, called `r` */
	k_prime = s_prime_transition.first.first;
	if (std::holds_alternative<std::unique_ptr<int>>(right_ptr)) {
		p = *(std::get<std::unique_ptr<int>>(right_ptr));
	} else {
		p = *(std::get<int*>(right_ptr));
	}
	/* `p_prime` would require continual `std::move`s, which is why we don't list it here */
	std::unique_ptr<ExplicitState> r = std::make_unique<ExplicitState>(this);

	/* link `r` to `s'` */
	r->set_transition(
		uni_str,
		k_prime + p - k + 1,
		std::move(s_prime_transition.first.second),
		std::move(s_prime_transition.second));

	/* overwrite old connection from `s` to `s'` by now transitioning to `r` instead */
	this->transitions.erase(t_k);
	this->set_transition(
		uni_str,
		k_prime,
		RightPointer(std::make_unique<int>(k_prime + p - k)),
		std::move(r));

	/* return the internal split's result: `r` */
	return this->transitions[t_k].second.get();
}


std::optional<ExplicitState*> ExplicitState::t_transition(utf8::uint32_t letter) {
	if (this->transitions.count(letter) == 0) {
		return nullptr;
	} else {
		auto t = this->weakly_get_transition(letter);
		return t.second;
	}
}


std::map<uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>>::iterator
		ExplicitState::transitions_start() {
	return this->transitions.begin();
}


std::map<uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>>::iterator
		ExplicitState::transitions_end() {
	return this->transitions.end();
}


std::ostream& operator<<(std::ostream& os, const ExplicitState& es) {
	os << "ExplicitState(" << es.id << ")";
	return os;
}


AuxiliaryState::AuxiliaryState(UnicodeString uni_str) : ExplicitState(nullptr) {
	this->root = std::make_unique<ExplicitState>(this);
	auto uniques = uni_str.unique_symbols();
	for (int i = 0; i < (int)uniques.size(); i++) {
		this->j_map[uniques[i]] = std::make_unique<int>(-i - 1);
	}
	this->root->set_suffix_link(this);
}


AuxiliaryState::~AuxiliaryState() = default;


bool AuxiliaryState::has_transition(utf8::uint32_t letter) {
	return this->j_map.count(letter) > 0;
}


std::pair<std::pair<int, int*>, ExplicitState*> AuxiliaryState::weakly_get_transition(utf8::uint32_t letter) {
	std::pair<int, int*> p(*(this->j_map[letter]), this->j_map[letter].get());
	return { p, this->root.get() };
}


std::optional<ExplicitState*> AuxiliaryState::t_transition(utf8::uint32_t letter) {
	if (this->has_transition(letter)) {
		return std::get<1>(this->weakly_get_transition(letter));
	} else {
		std::cout << "Don't have this value!" << std::endl;
		return nullptr;
	}
}


void ExplicitState::print(UnicodeString uni_str, int num_indents) {
	std::string indent_str;
	for (int i = 0; i < num_indents; i++) {
		indent_str.append("  ");
	}
	std::cout << indent_str << *this << std::endl;
	for (auto &entry : this->transitions) {
		std::cout << indent_str << "  ";
		int *second_ptr;
		if (std::holds_alternative<std::unique_ptr<int>>(entry.second.first.second)) {
			second_ptr = std::get<std::unique_ptr<int>>(entry.second.first.second).get();
		} else {
			second_ptr = std::get<int*>(entry.second.first.second);
		}
		std::cout << "(" << entry.second.first.first << ", " << *second_ptr << ") ";
		std::cout << "(" << UnicodeString::basic_string_from_unicode_string(
			uni_str.substring(entry.second.first.first - 1, *second_ptr)  /* Indexes cancel out: -1 + 1 = 0 */
		) << ") ";
		std::cout << *(entry.second.second) << std::endl;
		entry.second.second->print(uni_str, num_indents + 1);
	}
}


void AuxiliaryState::print(UnicodeString uni_str, int num_indents) {
	std::string indent_str;
	for (int i = 0; i < num_indents; i++) {
		indent_str.append("  ");
	}
	std::cout << indent_str << *this << std::endl;
	for (auto &entry : this->j_map) {
		std::cout << indent_str;
		std::cout << "(" << *(entry.second) << ", " << *(entry.second) << ") ";
		std::cout << "(" << UnicodeString::basic_string_from_unicode_char(entry.first) << ") ";
		std::cout << *(this->root) << std::endl;
	}
}
