#ifndef UNICODE_STRING_H
#define UNICODE_STRING_H


#include "utf8.h"

#include <string>
#include <vector>
#include <optional>


/**
 * A convenience class for working with Unicode strings.
 */
class UnicodeString {

	private:
		std::vector<utf8::uint32_t> vec;

	public:
		int length;
		explicit UnicodeString(std::string str);
		explicit UnicodeString(const std::vector<utf8::uint32_t>& vector);
		UnicodeString substring(int start_index, int end_index);
		utf8::uint32_t entry(int index);
		static std::basic_string<char> basic_string_from_unicode_string(UnicodeString uni_str);
		static std::basic_string<char> basic_string_from_unicode_char(utf8::uint32_t uni_char);
		std::vector<utf8::uint32_t> unique_symbols();
		std::optional<int> symbol_index(utf8::uint32_t symbol);

};


#endif /* UNICODE_STRING_H */
