#ifndef LONGEST_COMMON_SUBSTRING_H
#define LONGEST_COMMON_SUBSTRING_H


#include "explicit-state.h"


typedef enum SubStringType {
	UNDETERMINED,  /* the state's substring has not yet been classified */
	FIRST,  /* the state's substring belongs only to the first string */
	SECOND,  /* the state's substring belongs only to the second string */
	FIRST_AND_SECOND  /* the state's substring belongs to both the first and second string */
} SubStringType;


bool is_leaf_state(ExplicitState *e);

SubStringType leaf_state_sub_string_type(std::pair<int, int> leaf_state_range, std::pair<int, int> sep_end);

SubStringType updated_preliminary_state_sub_string_type(SubStringType old_type, SubStringType child_type);

SubStringType state_sub_string_type(
	ExplicitState *e,
	int length,
	int *lcs_length,
	int *lcs_start_index,
	std::pair<int, int> sep_end);


#endif  /* LONGEST_COMMON_SUBSTRING_H */
