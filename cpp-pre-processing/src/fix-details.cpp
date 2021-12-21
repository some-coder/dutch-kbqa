#include "fix-details.h"
#include <regex>
#include <iostream>


/* Adapted from the W3C" 'Common HTML entities used for typography'. */
const std::map<std::string, std::string> html_decode_map({
	 std::pair<std::string, std::string>("&quot;", "\""),
	 std::pair<std::string, std::string>("&amp;", "&"),
	std::pair<std::string, std::string>("&cent;", "¢"),
    std::pair<std::string, std::string>("&pound;", "£"),
    std::pair<std::string, std::string>("&sect;", "§"),
    std::pair<std::string, std::string>("&copy;", "©"),
    std::pair<std::string, std::string>("&laquo;", "«"),
    std::pair<std::string, std::string>("&raquo;", "»"),
    std::pair<std::string, std::string>("&reg;", "®"),
    std::pair<std::string, std::string>("&deg;", "°"),
    std::pair<std::string, std::string>("&plusmn;", "±"),
    std::pair<std::string, std::string>("&para;", "¶"),
    std::pair<std::string, std::string>("&middot;", "·"),
    std::pair<std::string, std::string>("&frac12;", "½"),
    std::pair<std::string, std::string>("&ndash;", "–"),
    std::pair<std::string, std::string>("&mdash;", "—"),
    std::pair<std::string, std::string>("&lsquo;", "‘"),
    std::pair<std::string, std::string>("&rsquo;", "’"),
    std::pair<std::string, std::string>("&sbquo;", "‚"),
    std::pair<std::string, std::string>("&ldquo;", "“"),
    std::pair<std::string, std::string>("&rdquo;", "”"),
    std::pair<std::string, std::string>("&bdquo;", "„"),
    std::pair<std::string, std::string>("&dagger;", "†"),
    std::pair<std::string, std::string>("&Dagger;", "‡"),
    std::pair<std::string, std::string>("&bull;", "•"),
    std::pair<std::string, std::string>("&hellip;", "…"),
    std::pair<std::string, std::string>("&prime;", "′"),
    std::pair<std::string, std::string>("&Prime;", "″"),
    std::pair<std::string, std::string>("&euro;", "€"),
    std::pair<std::string, std::string>("&trade;", "™"),
    std::pair<std::string, std::string>("&asymp;", "≈"),
    std::pair<std::string, std::string>("&ne;", "≠"),
    std::pair<std::string, std::string>("&le;", "≤"),
    std::pair<std::string, std::string>("&ge;", "≥"),
    std::pair<std::string, std::string>("&lt;", "<"),
    std::pair<std::string, std::string>("&gt;", ">")
});


/**
 * Removes the specified characters from the JSON file's entries.
 *
 * @note We assume the JSON file is a mapping from UIDs to strings.
 * @param obj The object whose entries certain characters must be removed from.
 * @param symbols_to_remove The symbols to remove from the JSON file's entries.
 * @param rep_map The replacement mapping.
 * @return The modified JSON object.
 */
Json::Value symbols_replaced(
		Json::Value obj,
		const std::vector<char>& symbols_to_remove,
		const std::map<char, std::string>& rep_map) {
	std::smatch str_match;
	std::string search_characters = "[";
    for (auto &symbol : symbols_to_remove) {
		search_characters += symbol;
    }
	search_characters += ']';
    std::regex reg_exp(search_characters);
    for (auto &key : obj.getMemberNames()) {
		std::string value = obj[key].asString();
		std::string modified = value;
		for (std::sregex_iterator it = std::sregex_iterator(value.begin(), value.end(), reg_exp); it != std::sregex_iterator(); ++it) {
			/* iterate over all symbols to replace */
			str_match = *it;
			std::string match = str_match.str();
			std::regex single_reg_exp((match == "}" || match == "{" ? "(\\" : "(") + match + ")");
			modified = std::regex_replace(modified, single_reg_exp, rep_map.at(str_match.str()[0]));
		}
		obj[key] = modified;
    }
    return obj;
}


std::string html_decoded(std::string raw) {
    std::regex reg_exp("(&#[0-9]+;)|(&[a-z]+;)");
    std::regex ascii_exp("[0-9]+");
    std::smatch ascii_match;
    std::string modified = raw;  /* will be modified in loop below */
    for (std::sregex_iterator it = std::sregex_iterator(raw.begin(), raw.end(), reg_exp); it != std::sregex_iterator(); ++it) {
        std::smatch mat = *it;
        std::string str_mat = mat.str();
        std::regex single_reg_exp("(" + str_mat + ")");
        if (html_decode_map.count(str_mat) > 0) {
            /* it's a pre-defined HTML code (not an ASCII-code based HTML code) */
            modified = std::regex_replace(modified, single_reg_exp, html_decode_map.at(str_mat));
        } else {
            /* it's an ASCII-code based HTML code (e.g. single quote is ASCII code point 39) */
            std::regex_search(str_mat, ascii_match, ascii_exp);
            std::string code_point;  /* equivalent to initialising it with "" */
            code_point += (char)stoi(ascii_match.str());  /* generally unsafe, but warranted here */
            modified = std::regex_replace(modified, single_reg_exp, code_point);
        }
    }
    return modified;
}
