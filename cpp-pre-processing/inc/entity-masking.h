#ifndef ENTITY_MASKING_H
#define ENTITY_MASKING_H


#include "entity-linking.h"
#include <vector>
#include <tuple>
#include <optional>
#include <map>


std::map<int, std::pair<std::string, std::string>> raw_qa_pairs(Mode mode);

std::map<int, std::vector<std::string>> uid_to_q_p_map(Mode mode);

std::map<std::string, std::vector<std::string>> q_p_to_lexemes_map(Mode mode);

std::map<std::string, std::vector<std::string>> qa_pair_q_p_map(
	int uid,
	std::map<int, std::vector<std::string>> uid_q_p_map,
	std::map<std::string, std::vector<std::string>> q_p_lexemes_map
);

std::map<std::string, std::string> q_p_values_to_masks(std::vector<std::string> q_p_values);

std::pair<int, int> lexeme_metrics(const std::string& lexeme);

std::vector<std::tuple<std::pair<int, int>, std::string, std::string>> descending_lexeme_list(
	const std::vector<std::string>& lexemes,
	const std::string& q_p_value
);

std::optional<std::vector<std::tuple<std::pair<int, int>, std::string, std::string>>> lexemes_to_use(
	std::string *raw_q,
	std::map<std::string, std::string> *q_p_values_masks,
	std::map<std::string, std::vector<std::string>> *q_p_lexeme_map
);

bool lexemes_collide(std::vector<std::pair<int, int>> lexeme_bounds);

std::optional<std::string> masked_question(
	std::string *raw_q,
	std::map<std::string, std::string>* q_p_values_masks,
	std::map<std::string, std::vector<std::string>>* q_p_lexeme_map
);

std::string masked_answer(std::string *raw_a, std::map<std::string, std::string>* q_p_values_masks);

std::optional<std::tuple<int, std::string, std::string>> masked_qa_pair(
	int uid,
	std::string raw_q,
	std::string raw_a,
	std::map<std::string, std::vector<std::string>> *q_p_map
);

std::map<int, std::pair<std::string, std::string>> masked_qa_pairs(Mode mode);

void save_masked_qa_pairs_to_disk(
	const std::map<int, std::pair<std::string, std::string>>& qa_pairs,
	const std::string& file_name);


#endif  /* ENTITY_MASKING_H */
