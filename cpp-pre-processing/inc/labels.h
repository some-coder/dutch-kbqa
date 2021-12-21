#ifndef LABELS_H
#define LABELS_H


#include <map>
#include <vector>
#include <set>
#include <json/json.h>


#define WIKIDATA_URL "https://query.wikidata.org/sparql?query="


void save_uid_to_q_or_p_value_map_to_disk(const std::string &in_file_name, const std::string &out_file_name);
std::set<std::string> unique_q_or_p_values(const std::string &file_name);
std::string query(std::vector<std::string> q_or_p_values, const std::string &language_label);
std::string url_encode(const std::string &value);
Json::Value labels_for_q_or_p_values(std::vector<std::string> q_or_p_values, const std::string &language_label);
Json::Value q_or_p_values_labels(std::vector<std::string> q_or_p_values, const std::string &language_label);
void update_q_p_labels_json(
		std::vector<std::string> unique_q_p,
		const std::string &file_name,
		int min,
		int max,
		const std::string &language_label);


#endif // LABELS_H
