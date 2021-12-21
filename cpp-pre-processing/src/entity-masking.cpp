#include "entity-masking.h"
#include <json/json.h>
#include <iostream>
#include <regex>
#include <algorithm>
#include "raw.h"


/**
 * Obtains a mapping from UIDs to raw, non-masked QA pairs stored on disk.
 *
 * @param mode The dataset mode.
 * @return The mapping.
 */
std::map<int, std::pair<std::string, std::string>> raw_qa_pairs(Mode mode) {
	std::string q_location =
		mode == TRAIN ? "data_nl_fixed_errors_removed" : "data_test_nl_fixed_errors_removed";
	std::string a_location =
		mode == TRAIN ? "data" : "data_test";

	Json::Value q_obj = json_from_file(q_location);
	Json::Value a_obj = json_from_file(a_location);

	std::map<int, std::pair<std::string, std::string>> raw_map;
	std::string q, a;

	for (auto &entry : a_obj) {
		int uid = entry["uid"].asInt();
		q = q_obj[std::to_string(uid)].asString();
		a = entry["sparql_wikidata"].asString();
		raw_map.insert({uid, std::pair<std::string, std::string>({q, a})});
	}

	return raw_map;
}

/**
 * Loads in a mapping from QA pair UIDs to WikiData Q- and P-values, based on a JSON file from disk.
 *
 * @param mode The dataset mode. Either training or test.
 * @return The mapping.
 */
std::map<int, std::vector<std::string>> uid_to_q_p_map(Mode mode) {
	std::string location = mode == TRAIN ? "uid_to_q_or_p" : "uid_to_q_or_p_test";
	Json::Value obj = json_from_file(location);
	std::map<int, std::vector<std::string>> uid_q_p_map;
	for (auto &key : obj.getMemberNames()) {
		int uid = std::stoi(key);
		uid_q_p_map[uid] = std::vector<std::string>();
		for (auto &entry : obj[key]) {
			uid_q_p_map[uid].push_back(entry.asString());
		}
	}
	return uid_q_p_map;
}


/**
 * Yields a mapping from WikiData Q- and P-values to a set of lexemes that said value represents.
 *
 * @note The code automatically loads in the mapping for Dutch, not English.
 * @param mode The dataset mode.
 * @return The mapping.
 */
std::map<std::string, std::vector<std::string>> q_p_to_lexemes_map(Mode mode) {
	std::string location = mode == TRAIN ? "q_p_map_nl" : "q_p_map_test_nl";
	Json::Value obj = json_from_file(location);
	std::map<std::string, std::vector<std::string>> q_p_lexemes_map;
	for (auto &key : obj.getMemberNames()) {
		q_p_lexemes_map[key] = std::vector<std::string>();
		for (auto &entry : obj[key]) {
			q_p_lexemes_map[key].push_back(entry.asString());
		}
	}
	return q_p_lexemes_map;
}


std::map<std::string, std::vector<std::string>> qa_pair_q_p_map(
		int uid,
		std::map<int, std::vector<std::string>> uid_q_p_map,
		std::map<std::string, std::vector<std::string>> q_p_lexemes_map) {
	std::map<std::string, std::vector<std::string>> q_p_map_for_qa_pair;
	for (auto &q_p_value : uid_q_p_map[uid]) {
		q_p_map_for_qa_pair[q_p_value] = std::vector<std::string>();
		for (auto &lexeme : q_p_lexemes_map[q_p_value]) {
			q_p_map_for_qa_pair[q_p_value].push_back(lexeme);
		}
	}
	return q_p_map_for_qa_pair;
}


/**
 * Yields a mapping from WikiData Q- and P-values to masks.
 *
 * @param q_p_values The Q- and P-values to work with.
 * @return The mapping from Q- and P-values to masks.
 */
std::map<std::string, std::string> q_p_values_to_masks(std::vector<std::string> q_p_values) {
	std::map<std::string, std::string> q_p_values_masks;
	int q_count = 0;  /* entities */
	int p_count = 0;  /* relations */
	for (auto &q_p_value : q_p_values) {
		char symbol;
		if (q_p_value[0] == 'Q') {
			q_count++;
			symbol = 'Q';
		} else {
			p_count++;
			symbol = 'P';
		}
		q_p_values_masks.insert({
			q_p_value,
			(symbol == 'Q' ? symbol + std::to_string(q_count) : symbol + std::to_string(p_count))
		});
	}
	return q_p_values_masks;
}


/**
 * Computes two metrics for the given lexeme.
 *
 * @param lexeme The lexeme to evaluate.
 * @return The two metrics, bundled as a 2-tuple (pair).
 */
std::pair<int, int> lexeme_metrics(const std::string& lexeme) {
	return { 0, lexeme.length() };  /* TODO: Find a better metric, perhaps? */
}


/**
 * Sorts the lexemes descendingly and returns it as a vector list.
 *
 * @param lexemes The lexemes.
 * @param q_p_value The Q- or P-value for the lexemes.
 * @return The list. Consists of 4-tuples of two metrics, the lexeme, and a Q- or P-value.
 */
std::vector<std::tuple<std::pair<int, int>, std::string, std::string>> descending_lexeme_list(
		const std::vector<std::string>& lexemes,
		const std::string& q_p_value) {
	std::vector<std::tuple<std::pair<int, int>, std::string, std::string>> list;
	list.reserve(lexemes.size());
	for (auto &lexeme : lexemes) {
		list.emplace_back(lexeme_metrics(lexeme), lexeme, q_p_value);
	}
	return list;
}


/**
 * Yields a selection among all lexemes available for this QA pair.
 *
 * @param raw_q The raw question to select lexemes for.
 * @param q_p_values_masks A mapping from Q- or P-values to masks.
 * @param q_p_lexeme_map A mapping from Q- or P-values to a series of lexemes for said values.
 * @return The selection of lexemes: index bounds plus lexemes and the Q- or P-value, if this is possible.
 */
std::optional<std::vector<std::tuple<std::pair<int, int>, std::string, std::string>>> lexemes_to_use(
		std::string *raw_q,
		std::map<std::string, std::string> *q_p_values_masks,
		std::map<std::string, std::vector<std::string>> *q_p_lexeme_map) {
	std::vector<std::tuple<std::pair<int, int>, std::string, std::string>> to_use_lexemes;
	for (auto &key_value : *q_p_values_masks) {
		/* find Q- and P-value lexemes that fit the `raw_q` */
		auto lex_list = descending_lexeme_list(
			(*q_p_lexeme_map)[key_value.first],
			key_value.first);
		int lex_idx = 0;
		for (; lex_idx < (int)lex_list.size(); ++lex_idx) {
			std::smatch lex_match;
			std::string regex_input = std::regex_replace(std::get<1>(lex_list[lex_idx]), std::regex("(\\[)"), "\\[");
			regex_input = std::regex_replace(regex_input, std::regex("(\\])"), "\\]");  /* TODO: Replace in `lex_list`? */
			std::regex re("(" + regex_input + ")");
			if (std::regex_search(*raw_q, lex_match, re)) {
				int start_index = (int)lex_match.position(0);
				std::pair<int, int> p({ start_index, start_index + lex_match.length(0) });
				to_use_lexemes.emplace_back(
					p, std::get<1>(
					lex_list[lex_idx]),
					key_value.first);
				break;  /* on to the next Q- or P-value to fit */
			}
		}
		if (lex_idx == (int)lex_list.size()) {
			return std::nullopt;
		}
	}
	return to_use_lexemes;
}


/**
 * Determines whether the provided set of lexemes collide, that is, whether their bounds overlap.
 *
 * @param lexeme_bounds The bounds of the lexemes.
 * @return The question's answer.
 */
bool lexemes_collide(std::vector<std::pair<int, int>> lexeme_bounds) {
	std::sort(lexeme_bounds.begin(), lexeme_bounds.end(), [] (const auto &x, const auto &y) {
		if (x.first == y.first) {
			return x.second < y.second;
		} else {
			return x.first < y.first;
		}
	});
	int i = 0;
	for (; i < ((int)lexeme_bounds.size() - 1); i++) {
		if (lexeme_bounds[i].second > lexeme_bounds[i + 1].first) {
			return true;
		}
	}
	return false;
}


/**
 * Derives the masked version of the answer from the raw question.
 *
 * @note TODO: Will this actually work?
 * @param raw_q The raw question.
 * @param q_p_values_masks A mapping from Q- and P-values to replace by masked variants.
 * @param q_p_lexeme_map A mapping from Q- and P-values to lexemes for said values.
 * @return The masked question, if this can be constructed.
 */
std::optional<std::string> masked_question(
		std::string* raw_q,
		std::map<std::string, std::string>* q_p_values_masks,
		std::map<std::string, std::vector<std::string>> *q_p_lexeme_map) {
	std::string modified = *raw_q;
	auto lex_list = lexemes_to_use(raw_q, q_p_values_masks, q_p_lexeme_map);
	if (!lex_list.has_value()) {
		return std::nullopt;
	}
	/* collect bounds for lexeme collision check */
	std::vector<std::pair<int, int>> bounds;
	bounds.reserve(lex_list.value().size());
	for (auto &entry : lex_list.value()) {
		bounds.push_back(std::get<0>(entry));
	}
	if (lexemes_collide(bounds)) {
		/* replacement won't work if we have collisions among lexemes */
		return std::nullopt;
	}
	for (auto &entry : lex_list.value()) {
		/* finally, replace all entries by masks */
		std::regex re("(" + std::get<1>(entry) + ")");
		modified = std::regex_replace(modified, re, (*q_p_values_masks)[std::get<2>(entry)]);
	}
	return modified;
}


/**
 * Derives the masked version of the answer from the raw answer.
 *
 * @note TODO: Will this actually work?
 * @param raw_a The raw answer.
 * @param q_p_values_masks A mapping from Q- and P-values to replace by masked variants.
 * @return The masked answer.
 */
std::string masked_answer(std::string* raw_a, std::map<std::string, std::string>* q_p_values_masks) {
	std::string modified = *raw_a;
	for (auto &key_value : *q_p_values_masks) {
		std::regex re("(" + key_value.first + ")");
		modified = std::regex_replace(modified, re, key_value.second);
	}
	return modified;
}


/**
 * Yields a single masked QA pair.
 *
 * @param uid The UID of the QA pair.
 * @param raw_q The raw, non-masked question.
 * @param raw_a The raw, non-masked answer.
 * @param q_p_map A mapping from WikiData Q- and P-values to a series of lexemes.
 * @return The masked QA pair, if it can be constructed. The first entry is the UID.
 */
std::optional<std::tuple<int, std::string, std::string>> masked_qa_pair(
		int uid,
		std::string raw_q,
		std::string raw_a,
		std::map<std::string, std::vector<std::string>> *q_p_map) {
	std::vector<std::string> q_p_values;
	q_p_values.reserve(q_p_map->size());
	for (auto &key_value : *q_p_map) {
		q_p_values.push_back(key_value.first);
	}

	std::map<std::string, std::string> q_p_masks = q_p_values_to_masks(q_p_values);
	auto possible_q = masked_question(&raw_q, &q_p_masks, q_p_map);
	if (!possible_q.has_value()) {
		return std::nullopt;
	}
	std::string q = possible_q.value();
	std::string a = masked_answer(&raw_a, &q_p_masks);

	return std::tuple<int, std::string, std::string>({ uid, q, a });
}


/**
 * Creates masked versions of existing raw QA pairs, saved on disk.
 *
 * @param mode The dataset mode.
 * @return The masked versions of the QA pairs.
 */
std::map<int, std::pair<std::string, std::string>> masked_qa_pairs(Mode mode) {
	auto raw_qas = raw_qa_pairs(mode);
	auto uid_q_p_map = uid_to_q_p_map(mode);
	auto q_p_lexemes = q_p_to_lexemes_map(mode);
	std::map<int, std::pair<std::string, std::string>> masked_qas;
	int solved = 0, not_solved = 0;
	for (auto &key_value : raw_qas) {
		auto focused_q_p_lexeme_map = qa_pair_q_p_map(
			key_value.first,
			uid_q_p_map,
			q_p_lexemes);
		auto opt_masked_qa_pair = masked_qa_pair(
			key_value.first,
			key_value.second.first,
			key_value.second.second,
			&focused_q_p_lexeme_map);
		if (opt_masked_qa_pair.has_value()) {
			auto result = opt_masked_qa_pair.value();
			masked_qas[std::get<0>(result)] = { std::get<1>(result), std::get<2>(result) };
			solved++;
		} else {
			not_solved++;
		}
		int total = solved + not_solved;
		if (total % 20 == 0) {
			/* update statistics once per 20 steps */
			printf(
					"\r[PercentCorrect=%6.2lf%%] [Total=%5d] [Progress=%6.2lf%%]",
					((double)solved / total) * 1e2,
					total,
					((double)total / (double)raw_qas.size()) * 1e2);
			fflush(stdout);
		}
	}
	return masked_qas;
}


/**
 * Saves the masked QA pairs to disk as a separate JSON file.
 *
 * @param qa_pairs The masked QA pairs to save.
 * @param file_name The name to assign to the file. Without file extension or directory specification.
 */
void save_masked_qa_pairs_to_disk(
		const std::map<int, std::pair<std::string, std::string>>& qa_pairs,
		const std::string& file_name) {
	Json::Value obj;
	for (auto &key_value : qa_pairs) {
		Json::Value sub_obj;
		sub_obj["q"] = key_value.second.first;
		sub_obj["a"] = key_value.second.second;
		obj[std::to_string(key_value.first)] = sub_obj;
	}
	json_to_file(obj, file_name);
}
