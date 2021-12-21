#include <cstdlib>
#include <iostream>
#include "entity-masking.h"


int main() {
//	Language lang = DUTCH;
//	Mode mode = TRAIN;
//
//	auto sen_u_ids = sentences_with_u_ids(lang, mode);
//	auto q_p_values = uid_q_p_values(mode);
//	auto m = uid_longest_common_substrings(sen_u_ids, q_p_values, lang, mode);
//	save_to_disk(m, lang, mode);

//	Json::Value obj = json_from_file("data_test_nl");
//	std::cout << "Replacing symbols..." << std::endl;
//	obj = symbols_replaced(
//		obj,
//		{ '_', '{', '}' },
//		{
//			std::pair<char, std::string>('_', " "),
//			std::pair<char, std::string>('{', ""),
//			std::pair<char, std::string>('}', "")
//		});
//	std::cout << "HTML decoding..." << std::endl;
//	for (auto &entry : obj.getMemberNames()) {
//		obj[entry] = html_decoded(obj[entry].asString());
//	}
//	std::cout << "Writing to file..." << std::endl;
//	json_to_file(obj, "data_test_nl_fixed");
//	std::cout << "Done." << std::endl;

	auto mqp = masked_qa_pairs(Mode::TEST);
	save_masked_qa_pairs_to_disk(mqp, "final_qa_pairs_test");

	return EXIT_SUCCESS;
}
