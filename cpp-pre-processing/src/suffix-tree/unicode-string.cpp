#include "suffix-tree/unicode-string.h"

#include <limits>
#include <unordered_set>
#include <algorithm>


UnicodeString::UnicodeString(std::string str) {
	if (!utf8::is_valid(str.begin(), str.end())) {
		throw std::logic_error("The string you input is not properly UTF-8-encoded!");
	}

	utf8::iterator<std::string::const_iterator> it(str.begin(), str.begin(), str.end());
	utf8::iterator<std::string::const_iterator> end_it(str.end(), str.begin(), str.end());

	this->vec = std::vector<utf8::uint32_t>();
	while (it != end_it) {
		this->vec.push_back(*it);
		it++;
	}
	if (this->vec.size() >= (long unsigned int)std::numeric_limits<int>::max()) {
		throw std::logic_error("Unicode strings of this length are not supported right now!");
	}
	this->length = (int)this->vec.size();
}


UnicodeString::UnicodeString(const std::vector<utf8::uint32_t>& vector) {
	this->vec = std::vector<utf8::uint32_t>(vector);
	if (vector.size() >= (long unsigned int)std::numeric_limits<int>::max()) {
		throw std::logic_error("Unicode strings of this length are not supported right now!");
	}
	this->length = (int)this->vec.size();
}


UnicodeString UnicodeString::substring(int start_index, int end_index) {
	std::vector<utf8::uint32_t> new_vec(
			this->vec.begin() + start_index,
			this->vec.begin() + end_index,
			this->vec.get_allocator());
	return UnicodeString(new_vec);
}


utf8::uint32_t UnicodeString::entry(int index) {
	return this->vec[index];
}


std::basic_string<char> UnicodeString::basic_string_from_unicode_string(UnicodeString uni_str) {
	std::u32string utf_32_str(uni_str.vec.begin(), uni_str.vec.end(), uni_str.vec.get_allocator());
	return std::basic_string<char>(utf8::utf32to8(utf_32_str));
}


std::basic_string<char> UnicodeString::basic_string_from_unicode_char(utf8::uint32_t uni_char) {
	std::u32string utf_32_str({uni_char});
	return std::basic_string<char>(utf8::utf32to8(utf_32_str));
}


std::vector<utf8::uint32_t> UnicodeString::unique_symbols() {
	std::unordered_set<utf8::uint32_t> set{};
	std::for_each(this->vec.begin(), this->vec.end(), [&set](const auto& p) {
		set.insert(p);
	});
	return { set.begin(), set.end() };
}


/**
 * Yields the index of the first occurrence of `symbol` in the Unicode string.
 *
 * @param symbol The symbol to search for. It may be that it is not found in the string.
 * @return The 0-based index of the symbol in the string, if it exists in it. Otherwise, nothing is returned.
 */
std::optional<int> UnicodeString::symbol_index(utf8::uint32_t symbol) {
	auto it = std::find(this->vec.begin(), this->vec.end(), symbol);
	if (it != this->vec.end()) {
		return std::distance(this->vec.begin(), it);
	} else {
		return std::nullopt;
	}
}
