#include "../inc/labels.h"

#include <regex>
#include <stdexcept>
#include <iostream>
#include <iomanip>


#include <json/json.h>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <utility>

#include "raw.h"


void save_uid_to_q_or_p_value_map_to_disk(const std::string &in_file_name, const std::string &out_file_name) {
	Json::Value file_json = json_from_file(in_file_name);
	std::regex q_or_p_reg_exp("[QP][0-9]+");
	Json::Value out_json;
	for (const auto & entry : file_json) {
		std::string sparql = entry["sparql_wikidata"].asString();
		auto sparql_start = std::sregex_iterator(sparql.begin(), sparql.end(), q_or_p_reg_exp);
		auto sparql_end = std::sregex_iterator();
		out_json[entry["uid"].asString()] = Json::arrayValue;
		for (std::sregex_iterator i = sparql_start; i != sparql_end; i++) {
			std::smatch match = *i;
			std::string match_str = match.str();
			out_json[entry["uid"].asString()].append(match_str);
		}
	}
	json_to_file(out_json, out_file_name);
}


std::set<std::string> unique_q_or_p_values(const std::string &file_name) {
	Json::Value j_val = json_from_file(file_name);
	std::set<std::string> q_p_set = std::set<std::string>();
	for (const auto &entry : j_val) {
		for (const auto &sub_entry : entry) {
			q_p_set.insert(sub_entry.asString());
		}
	}
	return q_p_set;
}


std::string query(std::vector<std::string> q_or_p_values, const std::string &language_label) {
	std::string q = "SELECT * WHERE {\n";
	const char *label_types[2] = {"rdfs:label", "skos:altLabel"};
	for (long unsigned int index = 0; index < q_or_p_values.size(); index++) {
		for (int label_index= 0; label_index < 2; label_index++) {
			q += (index == 0 && label_index == 0 ? "\t{\n" : "{\n");
			q += "\t\tSELECT DISTINCT ?id ?label WHERE {\n";
			q += "\t\t\tBIND(" + std::to_string(index) + " AS ?id) .\n";
			q += "\t\t\twd:" + q_or_p_values[index] + " " + label_types[label_index] + " ?label .\n";
			q += "\t\t\tfilter(lang(?label) = \"" + language_label + "\") .\n";
			q += "\t\t}\n";
			q += (index == q_or_p_values.size() - 1 && label_index == 1) ? "\t}\n" : "\t} UNION ";
		}
	}
	q += "}";
	return q;
}


std::string url_encode(const std::string &value) {
	std::ostringstream escaped;
	escaped.fill('0');
	escaped << std::hex;

	for (char c : value) {
			if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
			/* Retain alpha-numeric and other accepted characters... */
			escaped << c;
			continue;
		} else {
			/* ...code point-encode the rest. */
			escaped << std::uppercase;
			escaped << '%' << std::setw(2) << int((unsigned char) c);
		}
	}

	return escaped.str();
}


Json::Value labels_for_q_or_p_values(std::vector<std::string> q_or_p_values, const std::string &language_label) {
	std::vector<std::string> labels = std::vector<std::string>();
	curlpp::Easy request;
	std::list<std::string> header;
	header.emplace_back("Accept: application/json");
	header.emplace_back("User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0");
	request.setOpt(new curlpp::Options::HttpHeader(header));
	std::string q = query(std::move(q_or_p_values), language_label);
	std::string url = WIKIDATA_URL + url_encode(q);
	request.setOpt(new curlpp::Options::Url(url));
	request.setOpt(new curlpp::Options::WriteFunction(write_data));
	request.perform();
	std::stringstream result;
	result << request;
	Json::Value val;
	result >> val;
	return val["results"]["bindings"];
}


Json::Value q_or_p_values_labels(std::vector<std::string> q_or_p_values, const std::string &language_label) {
	Json::Value non_formatted = labels_for_q_or_p_values(q_or_p_values, language_label);
	Json::Value formatted;
	for (const auto &q_or_p_value : q_or_p_values) {
		/* initialise empty arrays */
		formatted[q_or_p_value] = Json::arrayValue;
	}
	int q_p_value_index = 0;
	for (auto & i : non_formatted) {
		if (std::stoi(i["id"]["value"].asString()) > q_p_value_index) {
			q_p_value_index++;
		}
		formatted[q_or_p_values[q_p_value_index]].append(i["label"]["value"].asString());
	}
	return formatted;
}


void update_q_p_labels_json(
		std::vector<std::string> unique_q_p,
		const std::string &file_name,
		int min,
		int max,
		const std::string &language_label) {
	if (min < 0 || (long unsigned int)max > unique_q_p.size() || min >= max) {
		throw std::runtime_error("Bounds are off!");
	}
	std::vector<std::string> slice = std::vector<std::string>();
	for (int i = min; i < max; i++) {
		slice.push_back(unique_q_p[i]);
	}
	Json::Value j_val = q_or_p_values_labels(slice, language_label);
	json_append(j_val, file_name);
}
