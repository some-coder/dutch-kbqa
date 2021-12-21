#include "suffix-tree/suffix-tree.h"

#include <utility>
#include <iostream>


ReferencePair::ReferencePair(ExplicitState *s, std::pair<int, int> k_p_pair) {
	this->s = s;
	this->k = k_p_pair.first;
	this->p = k_p_pair.second;
}


/**
 * Canonises a reference pair.
 *
 * Canonisation ensures that the ancestor of the state that is represented by the
 * reference pair is the closest ancestor that exists for it. 'Closest' here means:
 * a minimum amount of index-steps need to be taken to go from the ancestor to the
 * state referred to.
 *
 * @param uni_str The unicode string. Used for transitions.
 * @return The canonised reference pair; only the closest ancestor and the left-pointer are given.
 */
std::pair<ExplicitState *, int> ReferencePair::canonised(UnicodeString uni_str) {
	ExplicitState *out_s;
	int out_k;
	if (this->p < this->k) {
		return { this->s, this->k };
	} else {
		/* find the `t_k` transition */
		out_k = this->k;
		out_s = this->s;
		auto tr = out_s->weakly_get_transition(uni_str.entry(out_k - 1));  /* 1-indexing to 0-indexing */
		int k_prime = tr.first.first;
		int p_prime = *tr.first.second;
		/* keep walking down until we have a truly canonised reference pair */
		while ((p_prime - k_prime) <= (this->p - out_k)) {
			out_k = out_k + p_prime - k_prime + 1;
			out_s = tr.second;
			if (out_k <= (this->p)) {
				tr = out_s->weakly_get_transition(uni_str.entry(out_k - 1));  /* 1-indexing to 0-indexing */
				k_prime = tr.first.first;
				p_prime = *tr.first.second;
			}
		}
		return { out_s, out_k };
	}
}


/**
 * Constructs a suffix tree.
 *
 * @param uni_str A unicode string to build a tree from. Only read from; it is not modified.
 */
SuffixTree::SuffixTree(const std::string& str) : uni_str(str) {
	this->uni_str = UnicodeString(str);
	this->auxiliary = std::make_unique<AuxiliaryState>(UnicodeString(str));
	this->root = this->auxiliary->t_transition(this->uni_str.entry(0)).value();
	this->leaf_right_ptr = std::make_unique<int>(0);
}


std::pair<bool, ExplicitState *> SuffixTree::test_and_split(ReferencePair rp, utf8::uint32_t t) {
	int k_prime;
	if (rp.k <= rp.p) {
		utf8::uint32_t t_k = this->uni_str.entry(rp.k - 1);  /* 1-indexing to 0-indexing */
		auto tr = rp.s->weakly_get_transition(t_k);
		k_prime = tr.first.first;
		utf8::uint32_t letter = this->uni_str.entry(k_prime + rp.p - rp.k);  /* 1-indexing to 0-indexing */
		if (t == letter) {
			return { true, rp.s };
		} else {
			RightPointer right_ptr = std::make_unique<int>(k_prime + rp.p - rp.k);
			ExplicitState *es_ptr =
				rp.s->internal_split(this->uni_str, k_prime, std::move(right_ptr));
			return { false, es_ptr };
		}
	} else {
		return { rp.s->has_transition(t), rp.s };
	}
}


std::pair<ExplicitState *, int> SuffixTree::update(ReferencePair rp) {
	ExplicitState *state_s, *old_root, *r;
	utf8::uint32_t t_i = this->uni_str.entry(rp.p - 1);  /* 1-indexing to 0-indexing */
	int k;
	bool end_point;
	state_s = rp.s;
	k = rp.k;
	old_root = this->root;
	auto end_point_r = this->test_and_split(ReferencePair(state_s, { k, rp.p - 1 }), t_i);
	end_point = end_point_r.first;
	r = end_point_r.second;
	while (!end_point) {
		RightPointer right_ptr = this->leaf_right_ptr.get();
		std::unique_ptr r_prime = std::make_unique<ExplicitState>(r);
		r->set_transition(
			this->uni_str,
			rp.p,
			std::move(right_ptr),
			std::move(r_prime));
		if (old_root != this->root) {
			/* TODO: May be unstable: smart pointer addresses tend to change. */
			old_root->set_suffix_link(r);
		}
		old_root = r;
		auto s_k_rp =
			ReferencePair(state_s->get_suffix_link(), { k, rp.p - 1 }).canonised(this->uni_str);
		state_s = s_k_rp.first;
		k = s_k_rp.second;
		end_point_r = this->test_and_split(ReferencePair(state_s, { k, rp.p - 1 }), t_i);
		end_point = end_point_r.first;
		r = end_point_r.second;
	}
	if (old_root != this->root) {
		old_root->set_suffix_link(state_s);
	}
	return { state_s, k };
}


void SuffixTree::construct() {
	ExplicitState *s;
	int k, i;
	s = this->root;
	k = 1;  /* TODO: Off by one? Ukkonen uses 1 instead. */
	i = 0;
	while ((i + 1) <= this->uni_str.length) {
		i++;
		(*(this->leaf_right_ptr))++;  /* TODO: Correct? */
		auto up_pair = this->update(ReferencePair(s, { k, i }));
		s = up_pair.first;
		k = up_pair.second;
		auto ca_pair = ReferencePair(s, { k, i }).canonised(this->uni_str);
		s = ca_pair.first;
		k = ca_pair.second;
	}
}


void SuffixTree::print() {
	std::cout << "SUFFIX TREE" << std::endl;
	this->auxiliary->print(this->uni_str, 0);
	std::cout << std::endl;
	this->root->print(this->uni_str, 0);
}
