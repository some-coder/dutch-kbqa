#include "entity-linking.h"
#include "raw.h"
#include "suffix-tree/unicode-string.h"
#include "suffix-tree/suffix-tree.h"
#include "suffix-tree/longest-common-substring.h"
#include <iostream>


const std::string DATA_LOCATION;
const std::vector<std::pair<char, char>> SEPARATORS(
	{
		std::pair<char, char>('_', '*'),
		std::pair<char, char>('_', '$'),
		std::pair<char, char>('#', '$'),
		std::pair<char, char>('&', '~')
	});


std::map<int, std::string> sentences_with_u_ids(Language lang, Mode mode) {
	const std::string location =
		DATA_LOCATION + "data" +
		(mode != TRAIN ? "_test" : "") +
		(lang != ENGLISH ? "_nl" : "");
	std::cout << "Reading sentences with UIDs..." << std::endl;
	Json::Value input = json_from_file(location);
	std::cout << "Done." << std::endl;
	std::map<int, std::string> sen_u_ids;
	int uid;
	std::string sen;
	for (auto &entry : input) {
		uid = entry["uid"].asInt();
		sen = entry["question"].asString();
		sen_u_ids.insert({ uid, sen });
	}
	return sen_u_ids;
}


std::map<int, std::vector<std::string>> uid_q_p_values(Mode mode) {
	const std::string location =
		DATA_LOCATION + "uid_to_q_or_p" +
		(mode != TRAIN ? "_test" : "");
	std::cout << "Reading UIDs to Q- and P-values map..." << std::endl;
	Json::Value q_p_values = json_from_file(location);
	std::cout << "Done." << std::endl;
	std::map<int, std::vector<std::string>> uid_to_q_p_map;
	int uid;
	for (auto &key : q_p_values.getMemberNames()) {
		uid = std::stoi(key);
		std::vector<std::string> vec;
		for (auto &q_p_value : q_p_values[key]) {
			vec.push_back(q_p_value.asString());
		}
		uid_to_q_p_map.insert({ uid, vec });
	}
	return uid_to_q_p_map;
}


std::map<std::string, std::vector<std::string>> q_p_value_lexemes(Language lang, Mode mode) {
	const std::string location =
		DATA_LOCATION + "q_p_map" +
		(mode != TRAIN ? "_test" : "") +
		(lang != ENGLISH ? "_nl" : "");
	std::cout << "Reading Q- and P-values map to lexemes..." << std::endl;
	Json::Value q_p_lexemes = json_from_file(location);
	std::cout << "Done." << std::endl;
	std::map<std::string, std::vector<std::string>> q_p_to_lexeme_map;
	std::string q_p_value;
	for (auto &key : q_p_lexemes.getMemberNames()) {
		q_p_value = key;
		std::vector<std::string> vec;
		for (auto &lexeme : q_p_lexemes[key]) {
			vec.push_back(lexeme.asString());
		}
		q_p_to_lexeme_map.insert({ q_p_value, vec });
	}
	return q_p_to_lexeme_map;
}


std::map<std::string, std::string> q_p_value_longest_common_substrings(
		const std::string& sen,
		const std::vector<std::string>& lexemes) {
	std::map<std::string, std::string> lcs;
	std::vector<std::string> post_fixes;
	post_fixes.reserve(SEPARATORS.size());
	for (auto &pair : SEPARATORS) {
		post_fixes.push_back(pair.first + sen + pair.second);
	}
	for (auto &lexeme : lexemes) {
		int i, middle_sign, end_sign;
		std::string suffix_tree_string;
		for (i = 0; i < (int)SEPARATORS.size(); i++) {
			auto first_index = sen.find(SEPARATORS[i].first);
			auto second_index = sen.find(SEPARATORS[i].second);
			if ((first_index == std::string::npos) && (second_index == std::string::npos)) {
				/* the selected pair is safe to use, so use it */
				break;
			}
			/* one or both separator characters are already in use in the string */
		}
		if (i == (int)SEPARATORS.size()) {
			throw std::runtime_error("String " + sen + " has no usable separator pair!");
		}
		suffix_tree_string = lexeme + post_fixes[i];
		UnicodeString uni_str(suffix_tree_string);
		middle_sign = uni_str.symbol_index(SEPARATORS[i].first).value();
		end_sign = uni_str.symbol_index(SEPARATORS[i].second).value();

		SuffixTree tree(suffix_tree_string);
		tree.construct();

		int max_length = 0;
		int sub_string_start_index = 0;
		state_sub_string_type(
			tree.root,
			0,
			&max_length,
			&sub_string_start_index,
			{ middle_sign + 1, end_sign + 1 });

		if ((sub_string_start_index + max_length - 1) >= sub_string_start_index) {
			auto str = UnicodeString::basic_string_from_unicode_string(
					uni_str.substring(sub_string_start_index - 1, sub_string_start_index + max_length - 1));
			lcs.insert({lexeme, str});
		} else {
			lcs.insert({lexeme, ""});  /* no match */
		}
	}
	return lcs;
}


std::map<int, std::map<std::string, std::map<std::string, std::string>>> uid_longest_common_substrings(
		const std::map<int, std::string>& sen_u_ids,
		std::map<int, std::vector<std::string>> uid_q_ps,
		Language lang,
		Mode mode) {
	std::map<int, std::map<std::string, std::map<std::string, std::string>>> output;
	auto q_p_lexeme_map = q_p_value_lexemes(lang, mode);
	int max_size = (int)sen_u_ids.size();
	int i = 0;
	for (auto &uid_sen_pair : sen_u_ids) {
		printf("%5d / %5d (%6.3lf%%)\n", i + 1, max_size, ((double)(i + 1) / max_size) * 1e2);
		std::map<std::string, std::map<std::string, std::string>> q_p_map;
		for (auto &q_p_value : uid_q_ps[uid_sen_pair.first]) {
			std::map<std::string, std::string> lexeme_lcs_map =
				q_p_value_longest_common_substrings(uid_sen_pair.second, q_p_lexeme_map[q_p_value]);
			q_p_map.insert({ q_p_value, lexeme_lcs_map });
		}
		output.insert({ uid_sen_pair.first, q_p_map });
		i++;
	}
	return output;
}


void save_to_disk(
		const std::map<int, std::map<std::string, std::map<std::string, std::string>>>& uid_lcs,
		Language lang,
		Mode mode) {
	Json::Value output;
	std::cout << "Writing output file..." << std::endl;
	for (auto &uid_lcs_pair : uid_lcs) {
		/* map for this set of questions associated LCS values */
		Json::Value lcs_for_uid;
		for (auto &q_p_lcs_pair: uid_lcs_pair.second) {
			/* map for this question (identified by its UID) Q- and P-values to LCSs. */
			Json::Value lcs_for_q_p_value;
			for (auto &lexeme_lcs_pair: q_p_lcs_pair.second) {
				/* map for this WikiData Q- or P-value individual lexemes to their longest common substrings */
				Json::Value lcs_for_lexeme;
				lcs_for_q_p_value[lexeme_lcs_pair.first] = lexeme_lcs_pair.second;
			}
			lcs_for_uid[q_p_lcs_pair.first] = lcs_for_q_p_value;
		}
		output[std::to_string(uid_lcs_pair.first)] = lcs_for_uid;
	}
	const std::string location =
		DATA_LOCATION + "lcs" +
		(mode != TRAIN ? "_test" : "") +
		(lang != ENGLISH ? "_nl" : "");
	json_to_file(output, location, false);
	std::cout << "Done." << std::endl;
}
