#ifndef SUFFIX_TREE_H
#define SUFFIX_TREE_H


#include "unicode-string.h"
#include "explicit-state.h"


class ReferencePair {

	public:
		ExplicitState *s;
		int k;
		int p;
		explicit ReferencePair(ExplicitState *s, std::pair<int, int> k_p_pair);
		std::pair<ExplicitState*, int> canonised(UnicodeString uni_str);

};


class SuffixTree {

	private:
		UnicodeString uni_str;
		std::unique_ptr<ExplicitState> auxiliary;  /* the auxiliary state, `bottom` */
		std::unique_ptr<int> leaf_right_ptr;

	public:
		ExplicitState *root;
		explicit SuffixTree(const std::string& str);
		std::pair<bool, ExplicitState*> test_and_split(ReferencePair rp, utf8::uint32_t t);
		std::pair<ExplicitState*, int> update(ReferencePair rp);
		void construct();
		void print();

};


#endif  /* SUFFIX_TREE_H */
