#ifndef FIX_DETAILS_H
#define FIX_DETAILS_H


#include <json/json.h>


Json::Value symbols_replaced(
		Json::Value obj,
		const std::vector<char>& symbols_to_remove,
		const std::map<char, std::string>& rep_map);

std::string html_decoded(std::string raw);


#endif  /* FIX_DETAILS_H */
