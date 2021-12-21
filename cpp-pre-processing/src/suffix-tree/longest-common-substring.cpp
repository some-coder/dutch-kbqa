#include "suffix-tree/longest-common-substring.h"


bool is_leaf_state(ExplicitState *e) {
	return e->transitions_start() == e->transitions_end();
}


SubStringType leaf_state_sub_string_type(std::pair<int, int> leaf_state_range, std::pair<int, int> sep_end) {
	return leaf_state_range.first <= sep_end.first ? SubStringType::FIRST : SubStringType::SECOND;
}


SubStringType updated_preliminary_state_sub_string_type(SubStringType old_type, SubStringType child_type) {
	switch (old_type) {
		case UNDETERMINED:
			return child_type;
		case FIRST:
		case SECOND:
			return old_type != child_type ? FIRST_AND_SECOND : old_type;
		default:
			return FIRST_AND_SECOND;
	}
}


/**
 * Finds the state's sub-string type: first string, second string, or encompassing both.
 *
 * @note This method is recursive. As such, it can lead to a stack overflow at large suffix tree sizes.
 * @param e The state for which to determine the sub-string type.
 * @param length The length of the path to `e` from the suffix tree's root in terms of number of symbols.
 * @param lcs_length The currently longest path length; the length of the longest-common substring (LCS).
 * @param lcs_start_index The start index of the currently longest path, and that of the LCS.
 * @param sep_end The indices of the separator- and ending character. Commonly associated with `#` and `$`.
 *   Note that, like all indices in the suffix tree, the two `sep_end` indices need to be given in 1-indexing,
 *   NOT the usual 0-indexing!
 * @return The sub-string type of the state `e`.
 */
SubStringType state_sub_string_type(
		ExplicitState *e,
		int length,
		int *lcs_length,
		int *lcs_start_index,
		std::pair<int, int> sep_end) {
	SubStringType e_type, child_type;
	int *right_ptr;
	e_type = SubStringType::UNDETERMINED;
	for (auto it = e->transitions_start(); it != e->transitions_end(); it++) {
		/* get index of right pointer (remember, index is inclusive) */
		if (std::holds_alternative<std::unique_ptr<int>>(it->second.first.second)) {
			right_ptr = std::get<std::unique_ptr<int>>(it->second.first.second).get();
		} else {
			right_ptr = std::get<int*>(it->second.first.second);
		}

		/* Base Case (top) and Recursive Case (bottom). The algorithm is DFS, as the logic succeeds the routine call. */
		if (is_leaf_state(it->second.second.get())) {
			child_type = leaf_state_sub_string_type(
				std::pair<int, int>(it->second.first.first, *right_ptr),
				sep_end);
		} else {
			child_type = state_sub_string_type(
				it->second.second.get(),
				length + (*right_ptr - it->second.first.first + 1),
				lcs_length,
				lcs_start_index,
				sep_end);
		}

		e_type = updated_preliminary_state_sub_string_type(e_type, child_type);

		if ((e_type == FIRST_AND_SECOND) && (child_type == FIRST_AND_SECOND)) {
			int total_length = length + (*right_ptr - it->second.first.first + 1);
			if (*lcs_length < total_length) {
				/* If the path to this child state improves upon the currently longest length
				 * of a path, then make this known across the DFS' branches.
				 *   The start index can be determined by offsetting the end of this path
				 * by the length of the path.
				 *    We require the child also to be of the type `FIRST_AND_SECOND`; the
				 * GeeksForGeeks implementation implicitly also relies on this requirement,
				 * but because in their implementation transition lengths are computed
				 * from children to parents instead of the other way around (Ukkonen's),
				 * this does not surface in the code. */
				*lcs_length = total_length;
				*lcs_start_index = *right_ptr - total_length + 1;
			}
		}
	}
	return e_type;  /* if this ever returns `UNDETERMINED`, something is wrong */
}

